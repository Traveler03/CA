# Sherlock English Wikipedia FAISS RAG

Local English Wikipedia retrieval stack for large-scale card/RAG construction.

Assets:

- `data/external/wikipedia-en-2026-07-01-faiss/ivfpq.faiss`
  - 17,473,199 passages
  - 1024-dimensional Qwen3-Embedding-4B vectors
  - compressed IVFPQ index; low RAM, lower recall
- `data/external/wikipedia-en-2026-07-01-faiss/hnsw_sq.faiss`
  - high-recall HNSW/SQ index; larger RAM footprint
- `data/external/wikipedia-en-2026-07-01-faiss/ids.txt`
  - FAISS row id to passage id
- `data/external/wikipedia-en-2026-07-01-faiss/offsets.sqlite`
  - passage id to JSONL file offset
- `data/external/wikipedia-en-2026-07-01-passages/*.jsonl`
  - passage text: `id`, `title`, `section`, `text`
- `data/external/models/Qwen3-Embedding-4B`
  - local query embedding model

## Download / resume

Small compressed index plus passages and model:

```bash
python scripts/wiki_faiss/download_sherlock_wiki.py \
  --output-root data/external \
  --workers 4 \
  --insecure
```

High-recall HNSW index:

```bash
python scripts/wiki_faiss/download_sherlock_wiki.py \
  --output-root data/external \
  --no-passages \
  --no-model \
  --hnsw \
  --workers 2 \
  --insecure
```

The downloader uses `curl -C -`, so interrupted downloads resume.

## Run local service

Fast compressed index:

```bash
python scripts/wiki_faiss/serve_sherlock_wiki.py \
  --host 127.0.0.1 \
  --port 8897 \
  --index ivfpq.faiss
```

High-recall index:

```bash
python scripts/wiki_faiss/serve_sherlock_wiki.py \
  --host 127.0.0.1 \
  --port 8897 \
  --index hnsw_sq.faiss
```

Health:

```bash
curl -s http://127.0.0.1:8897/health | python -m json.tool
```

Single search:

```bash
curl -s -X POST http://127.0.0.1:8897/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"cyclic subgroup symmetric group S_10 order elements","top_k":5}'
```

Batch search:

```bash
curl -s -X POST http://127.0.0.1:8897/search_batch \
  -H 'Content-Type: application/json' \
  -d '{"queries":["cyclic subgroup","ice albedo feedback"],"top_k":5}'
```

## Bulk JSONL search

Input JSONL must contain a query field:

```json
{"id":"q1","query":"cyclic subgroup symmetric group S_10 order elements"}
```

Run:

```bash
python scripts/wiki_faiss/bulk_search_sherlock_wiki.py \
  --input input.jsonl \
  --output output.wiki.jsonl \
  --query-field query \
  --id-field id \
  --top-k 8 \
  --batch-size 32
```

Output JSONL:

```json
{"id":"q1","query":"...","wiki_results":[...],"wiki_latency_s":0.01,"error":null}
```

## Notes for experiments

- The Sherlock indexes were built with Qwen3-Embedding-4B query/document vector space. Do not query them with `text-embedding-*` or `compass-embedding-*`.
- For large construction, keep the HTTP service alive; otherwise each CLI invocation reloads the FAISS index, ids, and model.
- `ivfpq.faiss` is useful for cheap smoke and high-throughput approximate retrieval.
- `hnsw_sq.faiss` should be the default for final large-scale construction once downloaded.
- The corpus is body prose only; infobox/table facts are under-represented.
