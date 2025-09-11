#!/usr/bin/env python3
"""
Congressional Bill Search Engine

Hybrid search combining BM25 full-text search with vector similarity 
and optional cross-encoder reranking.

Usage:
    python search.py "definition of covered AI system"
    python search.py "climate change mitigation" --congress 119 --limit 10
    python search.py "healthcare reform" --rerank --bill-type hr
"""

import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
import duckdb
import httpx
import numpy as np

# Configuration
DB_PATH = os.environ.get("CONGRESS_DB", "congress_119.duckdb")
TEI_EMBED_URL = os.environ.get("TEI_EMBED_URL", "http://localhost:8080")
TEI_RERANK_URL = os.environ.get("TEI_RERANK_URL", "http://localhost:8081")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "256"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BillSearchEngine:
    """Hybrid search engine for congressional bills."""
    
    def __init__(self, db_path: str, tei_embed_url: str, tei_rerank_url: str):
        self.db_path = db_path
        self.tei_embed_url = tei_embed_url
        self.tei_rerank_url = tei_rerank_url
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.http_client = httpx.Client(timeout=httpx.Timeout(60.0))
        
    def __enter__(self):
        self.conn = duckdb.connect(self.db_path, read_only=True)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        self.http_client.close()
    
    def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for search query."""
        try:
            response = self.http_client.post(
                f"{self.tei_embed_url}/v1/embeddings",
                json={
                    "model": "tei",
                    "input": [query]
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data["data"][0]["embedding"]
            
            # Apply same preprocessing as fragments
            vec = np.asarray(embedding, dtype=np.float32)
            if len(vec) > EMBED_DIM:
                vec = vec[:EMBED_DIM]  # Matryoshka crop
            
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            return vec
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None
    
    def bm25_search(self, query: str, limit: int = 500, 
                   congress: Optional[int] = None,
                   bill_type: Optional[str] = None,
                   chamber: Optional[str] = None) -> List[int]:
        """BM25 full-text search prefilter."""
        try:
            # Build WHERE clause for filters
            where_conditions = ["fts_main_fragments.match_bm25(fragment_id, ?) IS NOT NULL"]
            params = [query]
            
            if congress:
                where_conditions.append("b.congress = ?")
                params.append(congress)
            
            if bill_type:
                where_conditions.append("b.bill_type = ?")
                params.append(bill_type)
                
            if chamber:
                where_conditions.append("b.origin_chamber = ?")
                params.append(chamber)
            
            where_clause = " AND ".join(where_conditions)
            
            sql = f"""
            SELECT f.fragment_id
            FROM fragments f
            JOIN bills b ON f.bill_id = b.bill_id
            WHERE {where_clause}
            ORDER BY fts_main_fragments.match_bm25(f.fragment_id, ?) DESC
            LIMIT ?
            """
            
            params.append(query)  # For ORDER BY
            params.append(limit)
            
            result = self.conn.execute(sql, params).fetchall()
            return [row[0] for row in result]
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def vector_search(self, query_embedding: np.ndarray, 
                     fragment_ids: List[int], 
                     limit: int = 25) -> List[Dict[str, Any]]:
        """Vector similarity search on prefiltered fragments."""
        if not fragment_ids:
            return []
        
        try:
            # Convert numpy array to list for DuckDB
            query_vec = query_embedding.tolist()
            
            # Create placeholders for fragment IDs
            id_placeholders = ",".join(["?"] * len(fragment_ids))
            
            sql = f"""
            SELECT 
                f.fragment_id,
                f.bill_id,
                f.version_code,
                f.heading,
                f.text,
                f.path,
                b.congress,
                b.bill_type,
                b.number,
                b.title,
                b.policy_area,
                array_cosine_distance(f.embedding, ?::FLOAT[{EMBED_DIM}]) AS similarity_score
            FROM fragments f
            JOIN bills b ON f.bill_id = b.bill_id
            WHERE f.fragment_id IN ({id_placeholders})
              AND f.embedding IS NOT NULL
            ORDER BY similarity_score ASC
            LIMIT ?
            """
            
            params = [query_vec] + fragment_ids + [limit]
            result = self.conn.execute(sql, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [
                'fragment_id', 'bill_id', 'version_code', 'heading', 'text',
                'path', 'congress', 'bill_type', 'number', 'title', 
                'policy_area', 'similarity_score'
            ]
            
            return [dict(zip(columns, row)) for row in result]
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      top_k: int = 25) -> List[Dict[str, Any]]:
        """Cross-encoder reranking using TEI rerank endpoint."""
        if not results or len(results) <= 1:
            return results
        
        try:
            # Prepare texts for reranking
            texts = []
            for result in results:
                # Combine heading and text for better context
                text = result['heading'] + " " + result['text'] if result['heading'] else result['text']
                texts.append(text.strip())
            
            # Call TEI rerank endpoint
            response = self.http_client.post(
                f"{self.tei_rerank_url}/rerank",
                json={
                    "query": query,
                    "texts": texts,
                    "raw_scores": False
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            rerank_data = response.json()
            
            # Sort results by rerank scores
            reranked_results = []
            for item in rerank_data["results"]:
                idx = item["index"]
                score = item["relevance_score"]
                
                if 0 <= idx < len(results):
                    result = results[idx].copy()
                    result['rerank_score'] = score
                    reranked_results.append(result)
            
            # Sort by rerank score (descending)
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original results if reranking fails
            return results[:top_k]
    
    def hybrid_search(self, query: str, 
                     limit: int = 25,
                     prefilter_limit: int = 500,
                     use_rerank: bool = False,
                     congress: Optional[int] = None,
                     bill_type: Optional[str] = None,
                     chamber: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search: BM25 prefilter + vector search + optional rerank."""
        logger.info(f"Searching for: '{query}'")
        
        # Step 1: BM25 prefilter
        logger.info("Running BM25 prefilter...")
        fragment_ids = self.bm25_search(
            query, 
            limit=prefilter_limit,
            congress=congress,
            bill_type=bill_type,
            chamber=chamber
        )
        logger.info(f"BM25 found {len(fragment_ids)} candidate fragments")
        
        if not fragment_ids:
            logger.warning("No results from BM25 search")
            return []
        
        # Step 2: Generate query embedding
        logger.info("Generating query embedding...")
        query_embedding = self.generate_query_embedding(query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        
        # Step 3: Vector similarity search
        logger.info("Running vector similarity search...")
        results = self.vector_search(query_embedding, fragment_ids, limit * 2)
        logger.info(f"Vector search returned {len(results)} results")
        
        if not results:
            logger.warning("No results from vector search")
            return []
        
        # Step 4: Optional reranking
        if use_rerank:
            logger.info("Running cross-encoder reranking...")
            results = self.rerank_results(query, results, limit)
            logger.info(f"Reranking returned {len(results)} results")
        else:
            results = results[:limit]
        
        return results
    
    def format_results(self, results: List[Dict[str, Any]], show_scores: bool = True) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."
        
        output = []
        output.append(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            bill_display = f"{result['bill_type'].upper()}{result['number']}"
            congress_display = f"({result['congress']}th Congress)"
            
            output.append(f"{i}. {bill_display} {congress_display}")
            output.append(f"   Title: {result['title']}")
            
            if result.get('heading'):
                output.append(f"   Section: {result['heading']}")
            
            if result.get('policy_area'):
                output.append(f"   Policy Area: {result['policy_area']}")
            
            # Show scores if requested
            if show_scores:
                scores = []
                if 'similarity_score' in result:
                    scores.append(f"Vector: {result['similarity_score']:.4f}")
                if 'rerank_score' in result:
                    scores.append(f"Rerank: {result['rerank_score']:.4f}")
                if scores:
                    output.append(f"   Scores: {', '.join(scores)}")
            
            # Show text preview
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            output.append(f"   Text: {text_preview}")
            output.append("")
        
        return "\n".join(output)


def main():
    """Command-line interface for bill search."""
    parser = argparse.ArgumentParser(description="Search congressional bills")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--congress", type=int, help="Filter by congress number")
    parser.add_argument("--bill-type", help="Filter by bill type (hr, s, etc.)")
    parser.add_argument("--chamber", choices=["house", "senate"], help="Filter by chamber")
    parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("--prefilter-limit", type=int, default=500, help="BM25 prefilter limit")
    parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranking")
    parser.add_argument("--no-scores", action="store_true", help="Hide similarity scores")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    with BillSearchEngine(args.db, TEI_EMBED_URL, TEI_RERANK_URL) as search_engine:
        try:
            results = search_engine.hybrid_search(
                query=args.query,
                limit=args.limit,
                prefilter_limit=args.prefilter_limit,
                use_rerank=args.rerank,
                congress=args.congress,
                bill_type=args.bill_type,
                chamber=args.chamber
            )
            
            formatted_results = search_engine.format_results(
                results, 
                show_scores=not args.no_scores
            )
            
            print(formatted_results)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
