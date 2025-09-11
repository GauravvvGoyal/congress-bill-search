#!/usr/bin/env python3
"""
Embedding and Indexing Script

Generates embeddings for bill fragments using TEI (Text Embeddings Inference)
and builds HNSW vector search index in DuckDB.

Usage:
    # Start TEI services first:
    docker compose up -d
    
    # Run embedding:
    export TEI_URL=http://localhost:8080
    export EMBED_DIM=256
    python embed_and_index.py
"""

import os
import math
import logging
import time
from typing import List, Iterator, Tuple, Optional
import duckdb
import httpx
import numpy as np

# Configuration
DB_PATH = os.environ.get("CONGRESS_DB", "congress_119.duckdb")
TEI_URL = os.environ.get("TEI_URL", "http://localhost:8080")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "256"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = no limit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using TEI service."""
    
    def __init__(self, tei_url: str, embed_dim: int = 256):
        self.tei_url = tei_url
        self.embed_dim = embed_dim
        self.client = httpx.Client(timeout=httpx.Timeout(120.0))
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
    
    def wait_for_service(self, max_wait: int = 300) -> bool:
        """Wait for TEI service to be ready."""
        logger.info(f"Waiting for TEI service at {self.tei_url}")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self.client.get(f"{self.tei_url}/health", timeout=10.0)
                if response.status_code == 200:
                    logger.info("TEI service is ready")
                    return True
            except Exception:
                pass
            
            time.sleep(5)
        
        logger.error(f"TEI service not ready after {max_wait}s")
        return False
    
    def l2_normalize(self, vec: List[float]) -> np.ndarray:
        """L2 normalize vector for cosine similarity."""
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return (v / norm).astype(np.float32)
    
    def crop_matryoshka(self, vec: List[float], target_dim: int) -> np.ndarray:
        """Crop Matryoshka embedding to target dimension."""
        v = np.asarray(vec, dtype=np.float32)
        if len(v) < target_dim:
            # Pad with zeros if vector is shorter than target
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:len(v)] = v
            return padded
        return v[:target_dim]
    
    def batch_texts(self, texts: List[str], batch_size: int) -> Iterator[List[str]]:
        """Batch texts for efficient processing."""
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        embeddings = []
        
        for batch in self.batch_texts(texts, BATCH_SIZE):
            try:
                # Call TEI embeddings endpoint (OpenAI compatible)
                response = self.client.post(
                    f"{self.tei_url}/v1/embeddings",
                    json={
                        "model": "tei",
                        "input": list(batch)
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                data = response.json()
                
                for embedding_data in data["data"]:
                    vec = embedding_data["embedding"]
                    
                    # Apply Matryoshka cropping and normalization
                    vec = self.crop_matryoshka(vec, self.embed_dim)
                    vec = self.l2_normalize(vec)
                    
                    embeddings.append(vec)
                
                logger.info(f"Generated {len(batch)} embeddings")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Add None embeddings for failed batch
                embeddings.extend([None] * len(batch))
        
        return embeddings


class VectorIndexer:
    """Build and manage vector search indexes."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        
    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def get_pending_fragments(self, limit: int = 0) -> List[Tuple[int, str]]:
        """Get fragments that need embeddings."""
        query = "SELECT fragment_id, text FROM fragments WHERE embedding IS NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        
        result = self.conn.execute(query).fetchall()
        logger.info(f"Found {len(result)} fragments needing embeddings")
        return result
    
    def update_embeddings(self, fragment_ids: List[int], embeddings: List[Optional[np.ndarray]]):
        """Update fragment embeddings in database."""
        if not fragment_ids or not embeddings:
            return
        
        success_count = 0
        
        # Begin transaction
        self.conn.execute("BEGIN")
        
        try:
            for frag_id, embedding in zip(fragment_ids, embeddings):
                if embedding is not None:
                    # Convert numpy array to list for DuckDB
                    embedding_list = embedding.tolist()
                    self.conn.execute(
                        "UPDATE fragments SET embedding = ? WHERE fragment_id = ?",
                        [embedding_list, frag_id]
                    )
                    success_count += 1
            
            self.conn.execute("COMMIT")
            logger.info(f"Updated {success_count}/{len(fragment_ids)} embeddings")
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error updating embeddings: {e}")
            raise
    
    def build_vector_index(self):
        """Build HNSW vector search index."""
        try:
            logger.info("Building HNSW vector index...")
            
            # Check if index already exists
            try:
                self.conn.execute("SELECT name FROM duckdb_indexes() WHERE index_name = 'frags_hnsw_cos'")
                if self.conn.fetchone():
                    logger.info("HNSW index already exists, dropping and rebuilding")
                    self.conn.execute("DROP INDEX frags_hnsw_cos")
            except:
                pass
            
            # Create HNSW index with cosine distance
            self.conn.execute("""
                CREATE INDEX frags_hnsw_cos ON fragments USING HNSW (embedding)
                WITH (metric='cosine')
            """)
            
            logger.info("HNSW vector index built successfully")
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise
    
    def get_embedding_stats(self) -> dict:
        """Get statistics about embeddings in the database."""
        try:
            stats = {}
            
            # Total fragments
            result = self.conn.execute("SELECT COUNT(*) FROM fragments").fetchone()
            stats['total_fragments'] = result[0] if result else 0
            
            # Embedded fragments
            result = self.conn.execute("SELECT COUNT(*) FROM fragments WHERE embedding IS NOT NULL").fetchone()
            stats['embedded_fragments'] = result[0] if result else 0
            
            # Pending fragments
            stats['pending_fragments'] = stats['total_fragments'] - stats['embedded_fragments']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {}


def main():
    """Main embedding and indexing function."""
    logger.info("Starting embedding and indexing process")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"TEI URL: {TEI_URL}")
    logger.info(f"Embedding dimension: {EMBED_DIM}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    
    with EmbeddingGenerator(TEI_URL, EMBED_DIM) as embedder:
        # Wait for TEI service to be ready
        if not embedder.wait_for_service():
            logger.error("TEI service not available, exiting")
            return 1
        
        with VectorIndexer(DB_PATH) as indexer:
            # Get initial stats
            initial_stats = indexer.get_embedding_stats()
            logger.info(f"Initial stats: {initial_stats}")
            
            if initial_stats.get('pending_fragments', 0) == 0:
                logger.info("No pending fragments to embed")
            else:
                # Get fragments needing embeddings
                fragments = indexer.get_pending_fragments(LIMIT)
                
                if not fragments:
                    logger.info("No fragments found")
                    return 0
                
                fragment_ids, texts = zip(*fragments)
                
                # Generate embeddings
                logger.info(f"Generating embeddings for {len(texts)} fragments...")
                embeddings = embedder.generate_embeddings(list(texts))
                
                # Update database
                indexer.update_embeddings(list(fragment_ids), embeddings)
            
            # Build vector index
            logger.info("Building vector search index...")
            indexer.build_vector_index()
            
            # Get final stats
            final_stats = indexer.get_embedding_stats()
            logger.info(f"Final stats: {final_stats}")
            
    logger.info("Embedding and indexing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
