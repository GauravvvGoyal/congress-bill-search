#!/usr/bin/env python3
"""
Example usage script for Congressional Bill Search System

This script demonstrates various search capabilities and use cases.
"""

import os
import time
from search import BillSearchEngine

# Configuration
DB_PATH = os.environ.get("CONGRESS_DB", "congress_119.duckdb")
TEI_EMBED_URL = "http://localhost:8080"
TEI_RERANK_URL = "http://localhost:8081"


def example_basic_search():
    """Basic search example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Search")
    print("=" * 60)
    
    with BillSearchEngine(DB_PATH, TEI_EMBED_URL, TEI_RERANK_URL) as engine:
        results = engine.hybrid_search(
            "artificial intelligence safety",
            limit=5
        )
        
        print(engine.format_results(results))


def example_filtered_search():
    """Search with filters."""
    print("=" * 60)
    print("EXAMPLE 2: Filtered Search")
    print("=" * 60)
    
    with BillSearchEngine(DB_PATH, TEI_EMBED_URL, TEI_RERANK_URL) as engine:
        results = engine.hybrid_search(
            "climate change",
            limit=5,
            congress=119,
            bill_type="hr",  # House resolutions only
            chamber="house"
        )
        
        print(engine.format_results(results))


def example_with_reranking():
    """Search with cross-encoder reranking."""
    print("=" * 60)
    print("EXAMPLE 3: Search with Reranking")
    print("=" * 60)
    
    with BillSearchEngine(DB_PATH, TEI_EMBED_URL, TEI_RERANK_URL) as engine:
        # First without reranking
        print("WITHOUT RERANKING:")
        results_no_rerank = engine.hybrid_search(
            "definition of covered system",
            limit=3,
            use_rerank=False
        )
        print(engine.format_results(results_no_rerank))
        
        print("\n" + "-" * 40 + "\n")
        
        # Then with reranking
        print("WITH RERANKING:")
        results_rerank = engine.hybrid_search(
            "definition of covered system", 
            limit=3,
            use_rerank=True
        )
        print(engine.format_results(results_rerank))


def example_search_comparison():
    """Compare different search strategies."""
    print("=" * 60)
    print("EXAMPLE 4: Search Strategy Comparison")
    print("=" * 60)
    
    query = "healthcare data privacy"
    
    with BillSearchEngine(DB_PATH, TEI_EMBED_URL, TEI_RERANK_URL) as engine:
        # BM25 only (using large prefilter, small final limit)
        print("BM25-FOCUSED RESULTS (prefilter=100, limit=3):")
        results_bm25 = engine.hybrid_search(
            query,
            limit=3,
            prefilter_limit=100,
            use_rerank=False
        )
        for i, result in enumerate(results_bm25[:3], 1):
            print(f"{i}. {result['bill_type'].upper()}{result['number']} - {result['title'][:80]}...")
        
        print("\nVECTOR-FOCUSED RESULTS (prefilter=500, limit=3):")
        results_vector = engine.hybrid_search(
            query,
            limit=3, 
            prefilter_limit=500,
            use_rerank=False
        )
        for i, result in enumerate(results_vector[:3], 1):
            print(f"{i}. {result['bill_type'].upper()}{result['number']} - {result['title'][:80]}...")
        
        print("\nRERANKED RESULTS:")
        results_rerank = engine.hybrid_search(
            query,
            limit=3,
            prefilter_limit=500, 
            use_rerank=True
        )
        for i, result in enumerate(results_rerank[:3], 1):
            print(f"{i}. {result['bill_type'].upper()}{result['number']} - {result['title'][:80]}...")


def example_semantic_vs_keyword():
    """Compare semantic vs keyword-heavy queries."""
    print("=" * 60)
    print("EXAMPLE 5: Semantic vs Keyword Queries")
    print("=" * 60)
    
    with BillSearchEngine(DB_PATH, TEI_EMBED_URL, TEI_RERANK_URL) as engine:
        # Keyword-heavy query
        print("KEYWORD QUERY: 'tax credit renewable energy solar wind'")
        results_keyword = engine.hybrid_search(
            "tax credit renewable energy solar wind",
            limit=3
        )
        for i, result in enumerate(results_keyword, 1):
            print(f"{i}. {result['bill_type'].upper()}{result['number']} - {result['title'][:80]}...")
        
        print("\n" + "-" * 40 + "\n")
        
        # Semantic query
        print("SEMANTIC QUERY: 'incentives for clean energy adoption'")
        results_semantic = engine.hybrid_search(
            "incentives for clean energy adoption",
            limit=3
        )
        for i, result in enumerate(results_semantic, 1):
            print(f"{i}. {result['bill_type'].upper()}{result['number']} - {result['title'][:80]}...")


def example_performance_timing():
    """Measure search performance."""
    print("=" * 60)
    print("EXAMPLE 6: Performance Timing")
    print("=" * 60)
    
    query = "national defense authorization"
    
    with BillSearchEngine(DB_PATH, TEI_EMBED_URL, TEI_RERANK_URL) as engine:
        # Time without reranking
        start = time.time()
        results = engine.hybrid_search(query, limit=10, use_rerank=False)
        no_rerank_time = time.time() - start
        
        print(f"Hybrid search (no rerank): {no_rerank_time:.3f}s - {len(results)} results")
        
        # Time with reranking
        start = time.time()
        results_rerank = engine.hybrid_search(query, limit=10, use_rerank=True)
        rerank_time = time.time() - start
        
        print(f"Hybrid search (with rerank): {rerank_time:.3f}s - {len(results_rerank)} results")
        print(f"Reranking overhead: +{rerank_time - no_rerank_time:.3f}s")


def main():
    """Run all examples."""
    print("Congressional Bill Search System - Examples")
    print("=" * 60)
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        print("Please run the ingestion and embedding steps first:")
        print("1. duckdb congress_119.duckdb -c '.read bootstrap_duckdb.sql'")
        print("2. python ingest_govinfo.py")  
        print("3. python embed_and_index.py")
        return
    
    try:
        example_basic_search()
        example_filtered_search() 
        example_with_reranking()
        example_search_comparison()
        example_semantic_vs_keyword()
        example_performance_timing()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nTips:")
        print("- Make sure TEI services are running: docker compose up -d")
        print("- Check service health: curl http://localhost:8080/health")
        print("- Verify database has data: duckdb congress_119.duckdb -c 'SELECT COUNT(*) FROM fragments'")


if __name__ == "__main__":
    main()
