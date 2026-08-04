from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_FAISS_DIR = Path("data/external/wikipedia-en-2026-07-01-faiss")
DEFAULT_PASSAGES_DIR = Path("data/external/wikipedia-en-2026-07-01-passages")
DEFAULT_MODEL_DIR = Path("data/external/models/Qwen3-Embedding-4B")


def load_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def load_file_map(db_path: Path) -> dict[int, str]:
    con = sqlite3.connect(str(db_path))
    try:
        return {int(idx): str(path) for idx, path in con.execute("select idx, path from files")}
    finally:
        con.close()


class PassageStore:
    def __init__(self, *, db_path: Path, passages_dir: Path) -> None:
        self.db_path = db_path
        self.passages_dir = passages_dir
        self.con = sqlite3.connect(str(db_path), check_same_thread=False)
        self.file_map = {int(idx): str(path) for idx, path in self.con.execute("select idx, path from files")}
        self._handles: dict[int, Any] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()
            self.con.close()

    def fetch(self, passage_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.con.execute("select f, off from pos where id = ?", (passage_id,)).fetchone()
            if row is None:
                return {"id": passage_id, "missing": "offset_not_found"}
            file_idx, offset = int(row[0]), int(row[1])
            rel_path = self.file_map[file_idx]
            handle = self._handles.get(file_idx)
            if handle is None:
                path = self.passages_dir / rel_path
                if not path.exists():
                    return {"id": passage_id, "missing": f"passage_file_not_found:{path}"}
                handle = path.open("rb")
                self._handles[file_idx] = handle
            handle.seek(offset)
            line = handle.readline()
        payload = json.loads(line)
        return payload

    def fetch_many(self, passage_ids: list[str]) -> list[dict[str, Any]]:
        return [self.fetch(passage_id) for passage_id in passage_ids]


def fetch_passage(db_path: Path, passages_dir: Path, file_map: dict[int, str], passage_id: str) -> dict[str, Any]:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("select f, off from pos where id = ?", (passage_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"id": passage_id, "missing": "offset_not_found"}
    file_idx, offset = int(row[0]), int(row[1])
    rel_path = file_map[file_idx]
    path = passages_dir / rel_path
    if not path.exists():
        return {"id": passage_id, "missing": f"passage_file_not_found:{path}"}
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    payload = json.loads(line)
    return payload


def encode_queries(model: SentenceTransformer, queries: list[str], *, target_dim: int) -> np.ndarray:
    kwargs: dict[str, Any] = {
        "prompt_name": "query",
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    try:
        vector = model.encode(queries, truncate_dim=target_dim, **kwargs)
    except TypeError:
        vector = model.encode(queries, **kwargs)
        if vector.shape[1] > target_dim:
            vector = vector[:, :target_dim]
            vector = vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (len(queries), target_dim):
        raise ValueError(f"query embedding shape {vector.shape} does not match FAISS index dim {target_dim}")
    return vector


def encode_query(model: SentenceTransformer, query: str, *, target_dim: int) -> np.ndarray:
    return encode_queries(model, [query], target_dim=target_dim)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local Sherlock-Comms Wikipedia FAISS index.")
    parser.add_argument("query")
    parser.add_argument("--faiss-dir", type=Path, default=DEFAULT_FAISS_DIR)
    parser.add_argument("--passages-dir", type=Path, default=DEFAULT_PASSAGES_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--index", default="ivfpq.faiss")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    index_path = args.faiss_dir / args.index
    ids_path = args.faiss_dir / "ids.txt"
    db_path = args.faiss_dir / "offsets.sqlite"

    index = faiss.read_index(str(index_path))
    if hasattr(faiss, "extract_index_ivf"):
        try:
            faiss.extract_index_ivf(index).nprobe = args.nprobe
        except Exception:
            pass

    ids = load_ids(ids_path)
    passage_store = PassageStore(db_path=db_path, passages_dir=args.passages_dir)
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    query_vec = encode_query(model, args.query, target_dim=index.d)
    scores, hits = index.search(query_vec, args.top_k)

    results = []
    for rank, (score, hit) in enumerate(zip(scores[0].tolist(), hits[0].tolist()), start=1):
        if hit < 0:
            continue
        passage_id = ids[hit]
        passage = passage_store.fetch(passage_id)
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "faiss_id": int(hit),
                "passage_id": passage_id,
                "title": passage.get("title"),
                "section": passage.get("section"),
                "text": passage.get("text"),
                "missing": passage.get("missing"),
            }
        )

    payload = {
        "query": args.query,
        "index": str(index_path),
        "index_dim": int(index.d),
        "top_k": args.top_k,
        "latency_s": time.perf_counter() - started,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"query: {args.query}")
        print(f"index: {index_path} dim={index.d} latency_s={payload['latency_s']:.3f}")
        for result in results:
            print(f"\n#{result['rank']} score={result['score']:.4f} {result.get('title')} / {result.get('section') or '<lead>'}")
            if result.get("missing"):
                print(result["missing"])
            else:
                text = (result.get("text") or "").replace("\n", " ")
                print(text[:900] + ("..." if len(text) > 900 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
