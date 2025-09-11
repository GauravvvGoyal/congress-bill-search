# Congressional Bill Search System

A high-quality bill text and metadata search system using:
- **GovInfo API** for authoritative bill text (BILLS) and metadata (BILLSTATUS)
- **DuckDB** for storage with BM25 FTS and HNSW vector search  
- **Local embeddings** via TEI + nomic-embed-text-v1.5 (256d Matryoshka)
- **Hybrid search** (BM25 prefilter → vector rerank → optional cross-rerank)

## Quick Start

1. **Get GovInfo API Key**
   ```bash
   # Register at https://api.govinfo.gov/docs/
   export GOVINFO_API_KEY=your_key_here
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start TEI Services**
   ```bash
   docker compose up -d
   
   # Wait for models to download and services to start
   # Check with: curl http://localhost:8080/health
   ```

4. **Initialize Database**
   ```bash
   duckdb congress_119.duckdb -c ".read bootstrap_duckdb.sql"
   ```

5. **Ingest Bills (119th Congress sample)**
   ```bash
   export CONGRESS=119 LIMIT=100
   python ingest_govinfo.py
   ```

6. **Generate Embeddings**
   ```bash
   python embed_and_index.py
   ```

7. **Search!**
   ```bash
   python search.py "definition of covered AI system"
   python search.py "climate change mitigation" --rerank --limit 5
   ```

## System Architecture

```
GovInfo API (BILLS + BILLSTATUS)
       ↓
   DuckDB Storage
   ├─ BM25 FTS (fragments.text)
   └─ HNSW VSS (fragments.embedding)
       ↓
   Hybrid Search Pipeline:
   1. BM25 prefilter (500 candidates)
   2. Vector similarity (cosine distance)  
   3. Optional cross-encoder rerank (Top-25)
```

## Configuration

### Environment Variables

- `GOVINFO_API_KEY` - Required GovInfo API key
- `CONGRESS` - Congress number to ingest (default: 119)
- `CONGRESS_DB` - Database path (default: congress_119.duckdb)  
- `LIMIT` - Limit ingestion count (default: 0 = no limit)
- `TEI_URL` / `TEI_EMBED_URL` - TEI embeddings endpoint (default: http://localhost:8080)
- `TEI_RERANK_URL` - TEI reranking endpoint (default: http://localhost:8081)
- `EMBED_DIM` - Embedding dimensions (default: 256, supports 64-768)

### TEI Services

The `docker-compose.yml` runs two TEI instances:

- **Port 8080**: `nomic-ai/nomic-embed-text-v1.5` for embeddings
- **Port 8081**: `BAAI/bge-reranker-v2-m3` for reranking

For GPU support, uncomment the `deploy` sections in docker-compose.yml.

## Usage Examples

### Basic Search
```bash
python search.py "healthcare reform"
```

### Filtered Search  
```bash
python search.py "artificial intelligence" --congress 119 --bill-type hr --limit 5
```

### With Reranking
```bash
python search.py "climate change adaptation" --rerank
```

### Programmatic Usage
```python
from search import BillSearchEngine

with BillSearchEngine("congress_119.duckdb", 
                     "http://localhost:8080", 
                     "http://localhost:8081") as engine:
    results = engine.hybrid_search(
        "definition of covered AI system",
        limit=10,
        use_rerank=True
    )
    for result in results:
        print(f"{result['bill_id']}: {result['title']}")
```

## Data Coverage

- **Bills**: XML text from 113th Congress → current (2013+)
- **Metadata**: BILLSTATUS with actions, subjects, policy areas
- **Updates**: Current congress updated every ~4 hours via GovInfo

## Performance & Sizing

- **256-dim embeddings**: ~1KB per fragment  
- **3M fragments**: ~3GB for vectors + HNSW overhead
- **Search latency**: BM25 + vector ~100-500ms, +rerank ~1-2s
- **Memory**: HNSW index must fit in RAM when loaded

## Sharding Strategy

Use one database per Congress for manageable sizing:
- `congress_113.duckdb`, `congress_114.duckdb`, etc.  
- Query multiple databases for cross-congress search
- HNSW index rebuilds are fast per-congress

## Extensions & Improvements

1. **Subject search**: Index `subjects` array as FTS side table
2. **Deduplication**: Content hash across versions (IH→RH→ENR)  
3. **USLM integration**: Add enrolled bills/laws corpus
4. **Caching**: Redis for frequent queries
5. **API**: FastAPI wrapper for REST endpoints

## Troubleshooting

### TEI Services Not Starting
```bash
# Check logs
docker compose logs tei-embed
docker compose logs tei-rerank

# Verify health  
curl http://localhost:8080/health
curl http://localhost:8081/health
```

### HNSW Index Issues
```bash
# Experimental persistence flag required
# If index corrupts, rebuild:
duckdb congress_119.duckdb -c "DROP INDEX IF EXISTS frags_hnsw_cos"
python embed_and_index.py  # Rebuilds index
```

### Memory Issues
```bash
# Reduce embedding dimensions
export EMBED_DIM=128  # vs default 256

# Or reduce prefilter
python search.py "query" --prefilter-limit 200  # vs default 500
```

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code  
```bash
black *.py
```

### Add New Congress
```bash
export CONGRESS=120 CONGRESS_DB=congress_120.duckdb
duckdb congress_120.duckdb -c ".read bootstrap_duckdb.sql"
python ingest_govinfo.py
python embed_and_index.py
```

## License

MIT License - see LICENSE file for details.
