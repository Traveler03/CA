from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import faiss
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from scripts.wiki_faiss.query_sherlock_wiki import (
    DEFAULT_FAISS_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_PASSAGES_DIR,
    PassageStore,
    encode_queries,
    encode_query,
    load_ids,
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=100)


class SearchBatchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=512)
    top_k: int = Field(default=8, ge=1, le=100)


class EmbedQueryRequest(BaseModel):
    query: str


def build_app(
    *,
    faiss_dir: Path,
    passages_dir: Path,
    model_dir: Path,
    index_name: str,
    nprobe: int,
    device: str,
) -> FastAPI:
    app = FastAPI(title="Sherlock Wikipedia FAISS Search", version="0.1.0")
    state: dict[str, Any] = {}
    lock = threading.Lock()

    @app.on_event("startup")
    def load_assets() -> None:
        started = time.perf_counter()
        index_path = faiss_dir / index_name
        index = faiss.read_index(str(index_path))
        try:
            faiss.extract_index_ivf(index).nprobe = nprobe
        except Exception:
            pass
        state["index_path"] = str(index_path)
        state["index"] = index
        state["ids"] = load_ids(faiss_dir / "ids.txt")
        state["passage_store"] = PassageStore(db_path=faiss_dir / "offsets.sqlite", passages_dir=passages_dir)
        state["model"] = SentenceTransformer(str(model_dir), device=device)
        state["loaded_s"] = time.perf_counter() - started

    @app.get("/health")
    def health() -> dict[str, Any]:
        index = state.get("index")
        return {
            "ok": bool(index),
            "index": state.get("index_path"),
            "index_dim": int(index.d) if index else None,
            "ntotal": int(index.ntotal) if index else None,
            "loaded_s": state.get("loaded_s"),
        }

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        started = time.perf_counter()
        index = state["index"]
        model = state["model"]
        with lock:
            query_vec = encode_query(model, request.query, target_dim=index.d)
        scores, hits = index.search(query_vec, request.top_k)
        results = []
        for rank, (score, hit) in enumerate(zip(scores[0].tolist(), hits[0].tolist()), start=1):
            if hit < 0:
                continue
            passage_id = state["ids"][hit]
            passage = state["passage_store"].fetch(passage_id)
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
        return {"query": request.query, "latency_s": time.perf_counter() - started, "results": results}

    @app.post("/search_batch")
    def search_batch(request: SearchBatchRequest) -> dict[str, Any]:
        started = time.perf_counter()
        index = state["index"]
        model = state["model"]
        with lock:
            query_vecs = encode_queries(model, request.queries, target_dim=index.d)
        scores, hits = index.search(query_vecs, request.top_k)
        batch_results = []
        for query, query_scores, query_hits in zip(request.queries, scores.tolist(), hits.tolist()):
            item_started = time.perf_counter()
            results = []
            for rank, (score, hit) in enumerate(zip(query_scores, query_hits), start=1):
                if hit < 0:
                    continue
                passage_id = state["ids"][hit]
                passage = state["passage_store"].fetch(passage_id)
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
            batch_results.append({"query": query, "latency_s": time.perf_counter() - item_started, "results": results})
        return {
            "count": len(request.queries),
            "top_k": request.top_k,
            "latency_s": time.perf_counter() - started,
            "results": batch_results,
        }

    @app.post("/embed_query")
    def embed_query(request: EmbedQueryRequest) -> dict[str, Any]:
        started = time.perf_counter()
        index = state["index"]
        model = state["model"]
        with lock:
            vector = encode_query(model, request.query, target_dim=index.d)
        return {
            "query": request.query,
            "dimension": int(index.d),
            "latency_s": time.perf_counter() - started,
            "embedding": vector[0].astype(float).tolist(),
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local Sherlock-Comms Wikipedia FAISS index.")
    parser.add_argument("--faiss-dir", type=Path, default=DEFAULT_FAISS_DIR)
    parser.add_argument("--passages-dir", type=Path, default=DEFAULT_PASSAGES_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--index", default="ivfpq.faiss")
    parser.add_argument("--nprobe", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8897)
    args = parser.parse_args()

    import uvicorn

    app = build_app(
        faiss_dir=args.faiss_dir,
        passages_dir=args.passages_dir,
        model_dir=args.model_dir,
        index_name=args.index,
        nprobe=args.nprobe,
        device=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
