#!/usr/bin/env python3
"""Build resumable low-resource Wikipedia passage and FAISS indexes.

The output layout intentionally matches ``MultilingualFaissRetriever``::

    <root>/20231101.<lang>/passages/*.jsonl
    <root>/20231101.<lang>/faiss/{ids.txt,offsets.sqlite,hnsw.faiss}

Input Parquet files come directly from the user-selected ``wikimedia/wikipedia``
snapshot.  Each completed stage has a manifest; an interrupted passage stage is
regenerated, while an interrupted embedding stage resumes from the last durable
FAISS/ids checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wikipag_low_resource_20231101.json"))
    parser.add_argument("--stage", choices=("all", "download", "passages", "index", "verify"), default="all")
    parser.add_argument("--languages", nargs="+", help="Subset of configured languages.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=8192)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset", "revision", "languages", "snapshot_prefix", "output_root", "model_dir", "embedding_dimension"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("complete") is True


def snapshot_name(config: dict[str, Any], language: str) -> str:
    return f"{config['snapshot_prefix']}{language}"


def language_root(config: dict[str, Any], language: str) -> Path:
    return Path(config["output_root"]) / snapshot_name(config, language)


def source_files(config: dict[str, Any], language: str) -> list[str]:
    from huggingface_hub import HfApi

    prefix = snapshot_name(config, language) + "/"
    entries = HfApi().list_repo_tree(
        config["dataset"], repo_type="dataset", revision=config["revision"], recursive=True, expand=True
    )
    return sorted(entry.path for entry in entries if entry.path.startswith(prefix) and entry.path.endswith(".parquet"))


def download(config: dict[str, Any], language: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    root = language_root(config, language)
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in source_files(config, language):
        local = hf_hub_download(
            repo_id=config["dataset"], repo_type="dataset", revision=config["revision"], filename=filename, local_dir=source_dir
        )
        downloaded.append(Path(local))
        print(json.dumps({"event": "downloaded", "language": language, "path": str(local)}, ensure_ascii=False), flush=True)
    write_json(root / "source_manifest.json", {
        "complete": True, "dataset": config["dataset"], "revision": config["revision"],
        "snapshot": snapshot_name(config, language), "files": [str(p.relative_to(root.resolve())) for p in downloaded],
    })
    return downloaded


def split_passages(text: str, max_chars: int, overlap_chars: int) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return
    start = 0
    length = len(normalized)
    while start < length:
        stop = min(length, start + max_chars)
        if stop < length:
            boundary = max(normalized.rfind(". ", start + max_chars // 2, stop), normalized.rfind("。", start + max_chars // 2, stop))
            if boundary > start:
                stop = boundary + 1
        chunk = normalized[start:stop].strip()
        if chunk:
            yield chunk
        if stop >= length:
            return
        start = max(start + 1, stop - overlap_chars)


def iter_articles(parquet_paths: list[Path]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    for parquet_path in parquet_paths:
        reader = pq.ParquetFile(parquet_path)
        names = set(reader.schema.names)
        expected = {"id", "title", "text"}
        if not expected.issubset(names):
            raise ValueError(f"{parquet_path} missing expected columns: {sorted(expected - names)}")
        for batch in reader.iter_batches(batch_size=1024, columns=["id", "title", "text"]):
            for row in batch.to_pylist():
                yield row


def prepare_passages(config: dict[str, Any], language: str) -> None:
    root = language_root(config, language)
    done_path = root / "passages_manifest.json"
    if completed(done_path):
        print(json.dumps({"event": "passages_already_complete", "language": language}), flush=True)
        return
    source_manifest = root / "source_manifest.json"
    if not completed(source_manifest):
        download(config, language)
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    parquet_paths = [root / item for item in source_payload["files"]]
    passages_dir = root / "passages"
    faiss_dir = root / "faiss"
    # A partial passage DB cannot safely be resumed article-by-article. It is
    # derived data, so regenerate this stage while preserving downloaded source.
    if passages_dir.exists():
        for path in passages_dir.glob("*.jsonl"):
            path.unlink()
    passages_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    db_path = faiss_dir / "offsets.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE files (idx INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
        CREATE TABLE pos (id TEXT PRIMARY KEY, f INTEGER NOT NULL, off INTEGER NOT NULL);
    """)
    passages_per_file = int(config.get("passages_per_file", 100000))
    max_chars = int(config.get("passage_max_chars", 1200))
    overlap = int(config.get("passage_overlap_chars", 200))
    file_index = -1
    in_file = passages_per_file
    output = None
    count = 0
    article_count = 0
    try:
        for article in iter_articles(parquet_paths):
            article_count += 1
            article_id = str(article["id"])
            title = re.sub(r"\s+", " ", str(article.get("title") or "")).strip()
            for chunk_number, chunk in enumerate(split_passages(str(article.get("text") or ""), max_chars, overlap)):
                if in_file >= passages_per_file:
                    if output is not None:
                        output.close()
                    file_index += 1
                    relative = f"part-{file_index:05d}.jsonl"
                    output = (passages_dir / relative).open("wb")
                    con.execute("INSERT INTO files(idx, path) VALUES (?, ?)", (file_index, relative))
                    in_file = 0
                assert output is not None
                passage_id = f"{snapshot_name(config, language)}:{article_id}:{chunk_number}"
                record = {"id": passage_id, "title": title, "section": None, "text": chunk}
                offset = output.tell()
                output.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                con.execute("INSERT INTO pos(id, f, off) VALUES (?, ?, ?)", (passage_id, file_index, offset))
                in_file += 1
                count += 1
            if article_count % 10000 == 0:
                con.commit()
                print(json.dumps({"event": "passages_progress", "language": language, "articles": article_count, "passages": count}), flush=True)
        con.commit()
    finally:
        if output is not None:
            output.close()
        con.close()
    write_json(done_path, {"complete": True, "language": language, "articles": article_count, "passages": count, "created_at": time.time()})
    print(json.dumps({"event": "passages_complete", "language": language, "passages": count}), flush=True)


def iter_passage_records(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((root / "passages").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def count_passages(root: Path) -> int:
    con = sqlite3.connect(root / "faiss" / "offsets.sqlite")
    try:
        return int(con.execute("SELECT COUNT(*) FROM pos").fetchone()[0])
    finally:
        con.close()


def encode_documents(model: Any, texts: list[str], dimension: int) -> Any:
    import numpy as np

    try:
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, truncate_dim=dimension, show_progress_bar=False)
    except TypeError:
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = vectors[:, :dimension]
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), dimension):
        raise ValueError(f"document embedding shape {vectors.shape} != {(len(texts), dimension)}")
    return vectors


def build_index(config: dict[str, Any], language: str, device: str, batch_size: int, checkpoint_every: int) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    root = language_root(config, language)
    done_path = root / "index_manifest.json"
    if completed(done_path):
        print(json.dumps({"event": "index_already_complete", "language": language}), flush=True)
        return
    if not completed(root / "passages_manifest.json"):
        prepare_passages(config, language)
    faiss_dir = root / "faiss"
    index_path = faiss_dir / "hnsw.faiss"
    ids_path = faiss_dir / "ids.txt"
    dimension = int(config["embedding_dimension"])
    total = count_passages(root)
    ids: list[str] = ids_path.read_text(encoding="utf-8").splitlines() if ids_path.exists() else []
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != dimension:
            raise ValueError(f"wrong index dimension {index.d}; expected {dimension}")
        if index.ntotal > len(ids):
            # IDs are essential to retrieval. The tail cannot be recovered from
            # HNSW, so safely restart only the derived index stage.
            index_path.unlink()
            ids_path.unlink(missing_ok=True)
            index = faiss.IndexHNSWFlat(dimension, int(config.get("hnsw_m", 32)), faiss.METRIC_INNER_PRODUCT)
            ids = []
        elif index.ntotal < len(ids):
            ids = ids[: index.ntotal]
            ids_path.write_text("".join(item + "\n" for item in ids), encoding="utf-8")
    else:
        index = faiss.IndexHNSWFlat(dimension, int(config.get("hnsw_m", 32)), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(config.get("hnsw_ef_construction", 200))
    index.hnsw.efSearch = int(config.get("hnsw_ef_search", 128))
    start_at = len(ids)
    if start_at > total:
        raise ValueError(f"ids count {start_at} exceeds passage count {total}")
    print(json.dumps({"event": "index_plan", "language": language, "total_passages": total, "already_indexed": start_at, "batch_size": batch_size}), flush=True)
    model = SentenceTransformer(str(config["model_dir"]), device=device)
    pending: list[dict[str, Any]] = []
    added_since_checkpoint = 0
    seen = 0
    with ids_path.open("a", encoding="utf-8") as ids_handle:
        for record in iter_passage_records(root):
            if seen < start_at:
                seen += 1
                continue
            pending.append(record)
            if len(pending) < batch_size:
                continue
            texts = [f"{item.get('title') or ''}\n{item['text']}" for item in pending]
            index.add(encode_documents(model, texts, dimension))
            ids_handle.write("".join(str(item["id"]) + "\n" for item in pending))
            pending.clear()
            seen += batch_size
            added_since_checkpoint += batch_size
            if added_since_checkpoint >= checkpoint_every:
                ids_handle.flush()
                os.fsync(ids_handle.fileno())
                faiss.write_index(index, str(index_path))
                added_since_checkpoint = 0
                print(json.dumps({"event": "index_progress", "language": language, "indexed": int(index.ntotal), "total": total}), flush=True)
        if pending:
            texts = [f"{item.get('title') or ''}\n{item['text']}" for item in pending]
            index.add(encode_documents(model, texts, dimension))
            ids_handle.write("".join(str(item["id"]) + "\n" for item in pending))
        ids_handle.flush()
        os.fsync(ids_handle.fileno())
    faiss.write_index(index, str(index_path))
    if int(index.ntotal) != total:
        raise RuntimeError(f"index count {index.ntotal} != passage count {total}")
    write_json(done_path, {"complete": True, "language": language, "passages": total, "dimension": dimension, "index": str(index_path), "created_at": time.time()})
    print(json.dumps({"event": "index_complete", "language": language, "passages": total, "index": str(index_path)}), flush=True)


def verify(config: dict[str, Any], language: str) -> None:
    import faiss

    root = language_root(config, language)
    faiss_dir = root / "faiss"
    expected = count_passages(root)
    index = faiss.read_index(str(faiss_dir / "hnsw.faiss"))
    ids = (faiss_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
    if expected != len(ids) or expected != index.ntotal:
        raise RuntimeError(f"verification count mismatch passages={expected} ids={len(ids)} index={index.ntotal}")
    con = sqlite3.connect(faiss_dir / "offsets.sqlite")
    try:
        for passage_id in (ids[0], ids[-1]) if ids else ():
            if con.execute("SELECT 1 FROM pos WHERE id = ?", (passage_id,)).fetchone() is None:
                raise RuntimeError(f"missing offset for {passage_id}")
    finally:
        con.close()
    payload = {"complete": True, "language": language, "passages": expected, "dimension": int(index.d), "index_bytes": (faiss_dir / "hnsw.faiss").stat().st_size, "verified_at": time.time()}
    write_json(root / "verification.json", payload)
    print(json.dumps({"event": "verified", **payload}, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    languages = args.languages or list(config["languages"])
    unexpected = sorted(set(languages) - set(config["languages"]))
    if unexpected:
        raise ValueError(f"languages not present in config: {unexpected}")
    if args.embedding_batch_size:
        config["embedding_batch_size"] = args.embedding_batch_size
    print(json.dumps({"event": "plan", "stage": args.stage, "languages": languages, "root": config["output_root"], "dimension": config["embedding_dimension"], "model_dir": config["model_dir"]}, ensure_ascii=False), flush=True)
    stages = ("download", "passages", "index", "verify") if args.stage == "all" else (args.stage,)
    for language in languages:
        for stage in stages:
            if stage == "download":
                download(config, language)
            elif stage == "passages":
                prepare_passages(config, language)
            elif stage == "index":
                build_index(config, language, args.device, int(config.get("embedding_batch_size", 128)), args.checkpoint_every)
            elif stage == "verify":
                verify(config, language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
