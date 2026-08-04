from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ca_mem.embedding import HashingTextEmbedder
from src.utils.jsonl import read_jsonl


def _embed_query_with_wikipag_service(query: str, *, service_url: str, timeout_s: float) -> np.ndarray:
    endpoint = service_url.rstrip("/") + "/embed_query"
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(endpoint, json={"query": query})
        response.raise_for_status()
        payload = response.json()
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError("embed_query returned no embedding")
    array = np.asarray([float(item) for item in vector], dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array if norm == 0.0 else array / norm


def query_cards(
    index_dir: Path,
    query: str,
    *,
    top_k: int,
    subject: str | None = None,
    min_score: float | None = None,
    service_url: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    rows = list(read_jsonl(index_dir / "runtime_card_index.jsonl"))
    matrix = np.load(index_dir / "runtime_card_index.npy")
    if matrix.shape[0] != len(rows):
        raise RuntimeError(f"index row mismatch: matrix={matrix.shape[0]} rows={len(rows)}")
    meta_path = index_dir / "runtime_card_index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    backend = meta.get("embedding_backend") or "hash"
    if backend == "wikipag_service":
        resolved_url = service_url or meta.get("embedding_service_url") or "http://127.0.0.1:8897"
        query_vec = _embed_query_with_wikipag_service(query, service_url=str(resolved_url), timeout_s=timeout_s)
    else:
        embedder = HashingTextEmbedder()
        query_vec = embedder.embed([query])[0]
    scores = (matrix @ query_vec).astype(float)
    order = np.argsort(-scores).tolist()
    results = []
    for idx in order:
        row = rows[idx]
        score = float(scores[idx])
        if subject and row.get("subject") != subject:
            continue
        if min_score is not None and score < min_score:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "card_id": row.get("card_id"),
                "subject": row.get("subject"),
                "concept_id": row.get("concept_id"),
                "concept": row.get("concept"),
                "card": row.get("payload"),
            }
        )
        if len(results) >= top_k:
            break
    return {
        "query": query,
        "top_k": top_k,
        "subject": subject,
        "min_score": min_score,
        "embedding_backend": backend,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve top-k compact runtime cards from a local card index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--subject", help="Optional subject filter; recommended when querying a combined multi-subject index.")
    parser.add_argument("--min-score", type=float, help="Optional cosine threshold for dropping weak card matches.")
    parser.add_argument("--service-url")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            query_cards(
                args.index_dir,
                args.query,
                top_k=args.top_k,
                subject=args.subject,
                min_score=args.min_score,
                service_url=args.service_url,
                timeout_s=args.timeout_s,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
