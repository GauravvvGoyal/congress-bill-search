#!/usr/bin/env python3
"""
GovInfo Congressional Bills Ingestion Script

Fetches bill text (BILLS) and metadata (BILLSTATUS) from GovInfo API
and stores them in DuckDB with text fragmentation for search.

Usage:
    export GOVINFO_API_KEY=your_key_here
    export CONGRESS=119 
    export LIMIT=500  # optional
    python ingest_govinfo.py
"""

import os
import re
import asyncio
import httpx
import duckdb
import hashlib
import logging
from datetime import date, datetime
from typing import Optional, Tuple, List, Iterator, Dict, Any
from lxml import etree as ET

# Configuration
GOVINFO_API_KEY = os.environ.get("GOVINFO_API_KEY")
if not GOVINFO_API_KEY:
    raise ValueError("GOVINFO_API_KEY environment variable is required")

DB_PATH = os.environ.get("CONGRESS_DB", "congress_119.duckdb")
CONGRESS = int(os.environ.get("CONGRESS", "119"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = no limit

BASE_URL = "https://api.govinfo.gov"
HEADERS = {"User-Agent": "Congressional-Bill-Search/1.0"}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_package_id(package_id: str) -> Optional[Tuple[int, str, int, str]]:
    """Parse package ID like BILLS-119hr5159ih into components."""
    match = re.match(r"BILLS-(\d+)([a-z]+)(\d+)([a-z]+)$", package_id)
    if not match:
        return None
    congress, bill_type, number, version = match.groups()
    return int(congress), bill_type, int(number), version


def hash_text(text: str) -> str:
    """Create SHA-256 hash of text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GovInfoClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            headers=HEADERS,
            timeout=httpx.Timeout(60.0, connect=30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def fetch_json(self, endpoint: str, **params) -> Dict[str, Any]:
        """Fetch JSON from GovInfo API with retry logic."""
        params["api_key"] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self.session.get(url, params=params)
            
            # Handle 503 with Retry-After (ZIP generation)
            if response.status_code == 503 and "Retry-After" in response.headers:
                retry_after = int(response.headers.get("Retry-After", "30"))
                logger.info(f"API busy, retrying after {retry_after}s")
                await asyncio.sleep(retry_after)
                response = await self.session.get(url, params=params)
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {endpoint}: {e}")
            raise

    async def fetch_content(self, url: str) -> bytes:
        """Fetch raw content from URL."""
        try:
            response = await self.session.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            logger.error(f"Error fetching content from {url}: {e}")
            raise

    async def iter_packages(self, collection: str, congress: int) -> Iterator[str]:
        """Iterate through all packages for a collection and congress."""
        # Wide date range to catch all bills for the congress
        start_date = f"{congress-1}-12-01"
        end_date = f"{congress+1}-01-31"
        offset = "*"
        
        while True:
            try:
                data = await self.fetch_json(
                    f"/published/{start_date}/{end_date}",
                    collection=collection,
                    congress=str(congress),
                    pageSize=1000,
                    offsetMark=offset
                )
                
                packages = data.get("packages", [])
                if not packages:
                    break
                
                for package in packages:
                    yield package["packageId"]
                
                # Check for next page
                next_page = data.get("nextPage")
                if not next_page:
                    break
                    
                offset = next_page["offsetMark"]
                
            except Exception as e:
                logger.error(f"Error fetching packages: {e}")
                break

    async def get_related_billstatus(self, bills_package_id: str) -> Optional[str]:
        """Find BILLSTATUS package related to a BILLS package."""
        try:
            data = await self.fetch_json(f"/packages/{bills_package_id}/related")
            for related in data.get("relatedPackages", []):
                if related.get("collectionCode") == "BILLSTATUS":
                    return related["packageId"]
        except Exception as e:
            logger.warning(f"Could not fetch related docs for {bills_package_id}: {e}")
        
        return None


class BillFragmenter:
    """Fragment bill text for search indexing."""
    
    BILL_TAGS = {"section", "subsection", "paragraph", "subparagraph", "clause"}
    
    @staticmethod
    def fragments_from_xml(xml_content: bytes) -> List[Tuple[int, str, str, str, str]]:
        """Extract fragments from Bill-DTD XML."""
        try:
            root = ET.fromstring(xml_content)
            fragments = []
            seq = 0
            
            for elem in root.iter():
                local_name = ET.QName(elem).localname.lower()
                if local_name in BillFragmenter.BILL_TAGS:
                    seq += 1
                    
                    # Extract heading
                    heading = (elem.get("heading") or "").strip()
                    if not heading:
                        # Look for header child element
                        header_elem = elem.find(".//header")
                        if header_elem is not None:
                            heading = "".join(header_elem.itertext()).strip()
                    
                    # Extract text content
                    text = " ".join("".join(elem.itertext()).split())
                    if not text.strip():
                        continue
                    
                    # Create identifier
                    identifier = elem.get("id") or heading or f"frag-{seq}"
                    
                    # Create content hash
                    content_hash = hash_text(text)
                    
                    fragments.append((seq, heading, identifier, text, content_hash))
                    
            return fragments
            
        except ET.XMLSyntaxError as e:
            logger.error(f"XML parsing error: {e}")
            return []
    
    @staticmethod
    def fragments_from_text(text_content: str, chunk_size: int = 1800) -> List[Tuple[int, str, str, str, str]]:
        """Fragment plain text into chunks."""
        text = text_content.strip()
        if not text:
            return []
        
        fragments = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                seq = i // chunk_size + 1
                heading = f"Chunk {seq}"
                identifier = f"chunk-{seq}"
                content_hash = hash_text(chunk)
                fragments.append((seq, heading, identifier, chunk, content_hash))
        
        return fragments


class BillIngester:
    """Ingest bills and metadata into DuckDB."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    async def ingest_bill(self, client: GovInfoClient, package_id: str) -> bool:
        """Ingest a single bill package."""
        try:
            # Get package summary
            summary = await client.fetch_json(f"/packages/{package_id}/summary")
            download = summary.get("download", {})
            
            # Find best available format (prefer XML)
            content_url = (
                download.get("xmlLink") or 
                download.get("txtLink") or 
                download.get("htmLink")
            )
            
            if not content_url:
                logger.warning(f"No downloadable content for {package_id}")
                return False
            
            # Parse package ID
            parsed = parse_package_id(package_id)
            if not parsed:
                logger.warning(f"Could not parse package ID: {package_id}")
                return False
            
            congress, bill_type, number, version = parsed
            
            # Insert basic bill record
            title = summary.get("title", "")
            origin_chamber = "house" if bill_type.startswith("h") else "senate"
            
            self.conn.execute("""
                INSERT OR REPLACE INTO bills (
                    bill_id, congress, bill_type, number, origin_chamber, title
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, [package_id, congress, bill_type, number, origin_chamber, title])
            
            # Insert version record
            text_format = content_url.split(".")[-1].lower()
            self.conn.execute("""
                INSERT OR REPLACE INTO bill_versions (
                    bill_id, version_code, published, text_fmt, content_url
                ) VALUES (?, ?, ?, ?, ?)
            """, [package_id, version, date.today(), text_format, content_url])
            
            # Download and fragment content
            content = await client.fetch_content(content_url)
            fragments = []
            
            if text_format == "xml":
                fragments = BillFragmenter.fragments_from_xml(content)
            
            if not fragments:
                # Fall back to text chunking
                try:
                    text_content = content.decode("utf-8", "ignore")
                    fragments = BillFragmenter.fragments_from_text(text_content)
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode content for {package_id}")
                    return False
            
            # Insert fragments
            if fragments:
                fragment_rows = []
                for seq, heading, path, text, content_hash in fragments:
                    fragment_id = int(hash_text(f"{package_id}:{path}")[:16], 16)
                    fragment_rows.append((
                        fragment_id, package_id, version, seq, heading, 
                        path, 0, 0, text, content_hash
                    ))
                
                self.conn.executemany("""
                    INSERT OR IGNORE INTO fragments (
                        fragment_id, bill_id, version_code, seq_in_version,
                        heading, path, char_start, char_end, text, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, fragment_rows)
            
            # Enrich with BILLSTATUS metadata
            await self.enrich_with_billstatus(client, package_id)
            
            logger.info(f"Ingested {package_id}: {len(fragments)} fragments")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting {package_id}: {e}")
            return False

    async def enrich_with_billstatus(self, client: GovInfoClient, package_id: str):
        """Enrich bill with BILLSTATUS metadata."""
        try:
            billstatus_id = await client.get_related_billstatus(package_id)
            if not billstatus_id:
                return
            
            # Get BILLSTATUS summary
            summary = await client.fetch_json(f"/packages/{billstatus_id}/summary")
            download = summary.get("download", {})
            xml_link = download.get("xmlLink")
            
            if not xml_link:
                return
            
            # Download and parse BILLSTATUS XML
            content = await client.fetch_content(xml_link)
            try:
                root = ET.fromstring(content)
                ns = {"b": "http://www.gpo.gov/billstatus"}
                
                # Extract policy area and subjects
                policy_area = root.findtext(".//b:policyArea/b:policyAreaTerm", namespaces=ns)
                subjects = [
                    elem.text for elem in root.findall(".//b:billSubjects//b:term", namespaces=ns)
                    if elem.text
                ]
                
                # Update bills table
                self.conn.execute("""
                    UPDATE bills 
                    SET policy_area = ?, subjects = ?
                    WHERE bill_id = ?
                """, [policy_area, subjects, package_id])
                
                # Extract and insert actions
                actions = []
                for action in root.findall(".//b:actions//b:item", namespaces=ns):
                    action_date = action.findtext("./b:actionDate", namespaces=ns)
                    description = action.findtext("./b:text", namespaces=ns)
                    chamber = (action.findtext("./b:committee/b:chamber", namespaces=ns) or "").lower()
                    code = int(action.findtext("./b:actionCode", namespaces=ns) or "0")
                    
                    if action_date and description:
                        actions.append((package_id, action_date, chamber, code, description))
                
                if actions:
                    self.conn.executemany("""
                        INSERT OR IGNORE INTO bill_actions (
                            bill_id, action_date, chamber, code, description
                        ) VALUES (?, ?, ?, ?, ?)
                    """, actions)
                    
            except ET.XMLSyntaxError as e:
                logger.warning(f"Could not parse BILLSTATUS XML for {package_id}: {e}")
                
        except Exception as e:
            logger.warning(f"Error enriching {package_id} with BILLSTATUS: {e}")


async def main():
    """Main ingestion function."""
    logger.info(f"Starting ingestion for Congress {CONGRESS}")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Limit: {LIMIT if LIMIT > 0 else 'No limit'}")
    
    async with GovInfoClient(GOVINFO_API_KEY) as client:
        with BillIngester(DB_PATH) as ingester:
            count = 0
            success_count = 0
            
            async for package_id in client.iter_packages("BILLS", CONGRESS):
                if LIMIT > 0 and count >= LIMIT:
                    break
                
                success = await ingester.ingest_bill(client, package_id)
                if success:
                    success_count += 1
                
                count += 1
                
                if count % 10 == 0:
                    logger.info(f"Processed {count} packages, {success_count} successful")
            
            logger.info(f"Ingestion complete: {success_count}/{count} packages successful")


if __name__ == "__main__":
    asyncio.run(main())
