from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ca_mem.embedding import HashingTextEmbedder
from src.utils.jsonl import read_jsonl


def query_cards(index_dir: Path, query: str, *, top_k: int) -> dict[str, Any]:
    rows = list(read_jsonl(index_dir / "runtime_card_index.jsonl"))
    matrix = np.load(index_dir / "runtime_card_index.npy")
    if matrix.shape[0] != len(rows):
        raise RuntimeError(f"index row mismatch: matrix={matrix.shape[0]} rows={len(rows)}")
    embedder = HashingTextEmbedder()
    query_vec = embedder.embed([query])
    scores = (matrix @ query_vec[0]).astype(float)
    order = np.argsort(-scores)[:top_k].tolist()
    results = []
    for rank, idx in enumerate(order, start=1):
        row = rows[idx]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "card_id": row.get("card_id"),
                "subject": row.get("subject"),
                "concept_id": row.get("concept_id"),
                "concept": row.get("concept"),
                "card": row.get("payload"),
            }
        )
    return {"query": query, "top_k": top_k, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve top-k compact runtime cards from a local card index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(query_cards(args.index_dir, args.query, top_k=args.top_k), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
