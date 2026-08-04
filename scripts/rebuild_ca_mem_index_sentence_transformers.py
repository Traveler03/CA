from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ca_mem.embedding import key_for_node
from src.ca_mem.schemas import MemoryNode
from src.utils.jsonl import read_jsonl


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copytree_file(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _encode_batches(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    truncate_dim: int | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    started = time.perf_counter()
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if truncate_dim:
            kwargs["truncate_dim"] = truncate_dim
        vec = model.encode(batch, **kwargs)
        vec = np.asarray(vec, dtype=np.float32)
        chunks.append(vec)
        done = min(start + len(batch), total)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        print(
            json.dumps(
                {
                    "event": "qwen_index_progress",
                    "done": done,
                    "total": total,
                    "elapsed_s": round(elapsed, 3),
                    "rows_per_s": round(rate, 2),
                    "dim": int(vec.shape[1]) if vec.ndim == 2 else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not chunks:
        return np.zeros((0, int(truncate_dim or 0)), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32, copy=False)


def rebuild_index(
    source_bank_dir: Path,
    output_bank_dir: Path,
    *,
    model_dir: Path,
    device: str,
    batch_size: int,
    truncate_dim: int | None,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    source_bank = source_bank_dir / "bank.jsonl"
    if not source_bank.exists():
        raise FileNotFoundError(source_bank)
    output_bank_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "bank.jsonl",
        "card_to_memory_trace.jsonl",
        "concept_cards.jsonl",
        "concept_card_to_usage_trace.jsonl",
        "rejected.jsonl",
    ]:
        src = source_bank_dir / name
        if src.exists():
            _copy_or_link(src, output_bank_dir / name)

    source_snapshots = source_bank_dir / "snapshots"
    output_snapshots = output_bank_dir / "snapshots"
    if source_snapshots.exists():
        if output_snapshots.exists():
            shutil.rmtree(output_snapshots)
        shutil.copytree(source_snapshots, output_snapshots, copy_function=_copytree_file)

    nodes = [MemoryNode.from_dict(row) for row in read_jsonl(source_bank)]
    node_keys = [key_for_node(node) for node in nodes]

    load_started = time.perf_counter()
    model = SentenceTransformer(str(model_dir), device=device)
    load_s = time.perf_counter() - load_started
    print(json.dumps({"event": "qwen_model_loaded", "model_dir": str(model_dir), "device": device, "load_s": round(load_s, 3)}, ensure_ascii=False), flush=True)

    encode_started = time.perf_counter()
    matrix = _encode_batches(model, node_keys, batch_size=batch_size, truncate_dim=truncate_dim)
    encode_s = time.perf_counter() - encode_started

    index_path = output_bank_dir / "build_index.npy"
    np.save(index_path, matrix)

    source_manifest_path = source_bank_dir / "bank_manifest.json"
    manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "embedding_backend": "sentence_transformers",
            "embedding_model": model_dir.name,
            "embedding_model_path": str(model_dir),
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else truncate_dim,
            "embedding_dtype": "float32",
            "normalize_embeddings": True,
            "document_prompt_name": None,
            "query_prompt_name": "query",
            "build_index_path": str(index_path),
            "source_bank_dir": str(source_bank_dir),
            "final_node_count": len(nodes),
            "candidate_count": len(nodes),
            "bank_version": len(nodes),
            "qwen_index_build": {
                "device": device,
                "batch_size": batch_size,
                "truncate_dim": truncate_dim,
                "model_load_s": load_s,
                "encode_s": encode_s,
                "rows_per_s": (len(nodes) / encode_s) if encode_s > 0 else None,
            },
        }
    )
    (output_bank_dir / "bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild an existing CA-Mem bank index with a local SentenceTransformer model.")
    parser.add_argument("--source-bank-dir", required=True)
    parser.add_argument("--output-bank-dir", required=True)
    parser.add_argument("--model-dir", default="data/external/models/Qwen3-Embedding-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--truncate-dim", type=int, default=1024)
    args = parser.parse_args(argv)
    manifest = rebuild_index(
        Path(args.source_bank_dir),
        Path(args.output_bank_dir),
        model_dir=Path(args.model_dir),
        device=args.device,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
