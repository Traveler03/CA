from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_subject_concept_smoke import embed_texts_with_wikipag_service
from src.ca_mem.embedding import HashingTextEmbedder
from src.smoke_test.io import read_jsonl, write_jsonl


JSONL_FILES = [
    "subject_profile.jsonl",
    "concept_queries.jsonl",
    "concept_passages.jsonl",
    "candidate_concepts.jsonl",
    "concept_registry.jsonl",
    "concept_evidence.jsonl",
    "merge_redirects.jsonl",
    "evidence_packs.jsonl",
    "runtime_cards.raw.jsonl",
    "runtime_card_claims.jsonl",
    "runtime_card_quality.jsonl",
    "runtime_cards.jsonl",
    "runtime_card_index.jsonl",
    "build_events.jsonl",
    "rejected_items.jsonl",
]


COUNT_KEYS = [
    "concept_count",
    "active_concept_count",
    "active_card_count",
    "runtime_card_count",
    "runtime_card_claim_count",
    "runtime_card_index_count",
    "evidence_pack_count",
    "avg_card_quality_score",
    "evidence_section_count",
    "rejected_item_count",
    "model_network_calls",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tag_row(row: dict[str, Any], *, shard: Path) -> dict[str, Any]:
    tagged = dict(row)
    tagged.setdefault("clean_shard_source_dir", str(shard))
    tagged.setdefault("source_corpus_only", True)
    return tagged


def combine_jsonl(output_dir: Path, shard_dirs: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for file_name in JSONL_FILES:
        rows: list[dict[str, Any]] = []
        for shard in shard_dirs:
            path = shard / file_name
            if path.exists():
                rows.extend(tag_row(row, shard=shard) for row in read_jsonl(path))
        write_jsonl(output_dir / file_name, rows)
        counts[file_name] = len(rows)
    return counts


def copy_component_manifests(output_dir: Path, shard_dirs: list[Path]) -> list[dict[str, Any]]:
    component_dir = output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for idx, shard in enumerate(shard_dirs):
        manifest_path = shard / "bank_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing shard manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        manifest["clean_shard_index"] = idx
        manifest["source_dir"] = str(shard)
        manifests.append(manifest)
        (component_dir / f"clean_shard_{idx}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifests


def rebuild_runtime_card_index(
    output_dir: Path,
    *,
    backend: str,
    service_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    index_path = output_dir / "runtime_card_index.jsonl"
    rows = list(read_jsonl(index_path)) if index_path.exists() else []
    texts = [str(row.get("index_text") or "") for row in rows]
    if backend == "wikipag":
        matrix, dimension = embed_texts_with_wikipag_service(texts, service_url=service_url, timeout_s=timeout_s)
        embedding_backend = "wikipag_service"
        embedding_model = "Qwen3-Embedding-4B"
    else:
        embedder = HashingTextEmbedder()
        matrix = embedder.embed(texts)
        dimension = int(matrix.shape[1]) if matrix.ndim == 2 else embedder.dim
        embedding_backend = "hash"
        embedding_model = embedder.model_name
    import numpy as np

    matrix_path = output_dir / "runtime_card_index.npy"
    np.save(matrix_path, matrix)
    meta = {
        "index_type": "numpy_dense_matrix",
        "metric": "cosine_on_normalized_embeddings",
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "embedding_service_url": service_url if backend == "wikipag" else None,
        "dimension": dimension,
        "count": len(rows),
        "matrix_path": str(matrix_path),
        "ids_path": str(index_path),
    }
    (output_dir / "runtime_card_index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def combine_evidence_sections(output_dir: Path, shard_dirs: list[Path]) -> None:
    try:
        import pandas as pd

        frames = []
        for shard in shard_dirs:
            path = shard / "evidence_sections.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if frames:
            pd.concat(frames, ignore_index=True).drop_duplicates(subset=["source_id"]).to_parquet(
                output_dir / "evidence_sections.parquet",
                index=False,
            )
    except Exception as exc:
        (output_dir / "evidence_sections.error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )


def build_manifest(
    *,
    bank_version: str,
    shard_dirs: list[Path],
    shard_manifests: list[dict[str, Any]],
    combined_counts: dict[str, int],
) -> dict[str, Any]:
    subjects: list[str] = []
    seen_subjects: set[str] = set()
    duplicate_subjects: list[str] = []
    totals = Counter()
    for manifest in shard_manifests:
        for subject in manifest.get("subject_ids", []):
            if subject in seen_subjects:
                duplicate_subjects.append(subject)
                continue
            seen_subjects.add(subject)
            subjects.append(subject)
        for key in COUNT_KEYS:
            totals[key] += int(manifest.get(key) or 0)

    return {
        "bank_version": bank_version,
        "construction_mode": "wikipag_clean_sharded_combined",
        "construction_model": next((m.get("construction_model") for m in shard_manifests if m.get("construction_model")), None),
        "embedding_model": next((m.get("embedding_model") for m in shard_manifests if m.get("embedding_model")), None),
        "source_snapshot": next((m.get("source_snapshot") for m in shard_manifests if m.get("source_snapshot")), None),
        "subjects": len(subjects),
        "subject_ids": subjects,
        "duplicate_subject_ids": duplicate_subjects,
        "shards": [{"source_dir": str(path), "subjects": len(manifest.get("subject_ids", []))} for path, manifest in zip(shard_dirs, shard_manifests)],
        "concept_count": combined_counts.get("concept_registry.jsonl", 0),
        "active_concept_count": totals["active_concept_count"],
        "active_card_count": combined_counts.get("runtime_cards.jsonl", 0),
        "runtime_card_count": combined_counts.get("runtime_cards.jsonl", 0),
        "runtime_card_claim_count": combined_counts.get("runtime_card_claims.jsonl", 0),
        "runtime_card_quality_count": combined_counts.get("runtime_card_quality.jsonl", 0),
        "runtime_card_index_count": combined_counts.get("runtime_card_index.jsonl", 0),
        "evidence_pack_count": combined_counts.get("evidence_packs.jsonl", 0),
        "avg_card_quality_score": (
            sum(float(manifest.get("avg_card_quality_score") or 0.0) for manifest in shard_manifests)
            / len(shard_manifests)
            if shard_manifests
            else 0.0
        ),
        "evidence_section_count": totals["evidence_section_count"],
        "rejected_item_count": combined_counts.get("rejected_items.jsonl", 0),
        "model_network_calls": totals["model_network_calls"],
        "combined_counts": combined_counts,
        "construction_cutoff": datetime.now(timezone.utc).isoformat(),
        "source_corpus_only": True,
        "one_card_per_subject_concept": True,
    }


def combine(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [Path(item) for item in args.shard_dirs]
    missing = [str(path) for path in shard_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard dirs: {missing}")

    combined_counts = combine_jsonl(output_dir, shard_dirs)
    shard_manifests = copy_component_manifests(output_dir, shard_dirs)
    combine_evidence_sections(output_dir, shard_dirs)
    runtime_index_meta = rebuild_runtime_card_index(
        output_dir,
        backend=args.card_index_backend,
        service_url=args.wikipag_service_url,
        timeout_s=args.embedding_timeout_s,
    )
    manifest = build_manifest(
        bank_version=args.bank_version,
        shard_dirs=shard_dirs,
        shard_manifests=shard_manifests,
        combined_counts=combined_counts,
    )
    manifest["runtime_card_index"] = runtime_index_meta
    manifest["embedding_model"] = runtime_index_meta.get("embedding_model")
    (output_dir / "bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": not manifest["duplicate_subject_ids"],
                "output_dir": str(output_dir),
                "construction_mode": manifest["construction_mode"],
                "subjects": manifest["subjects"],
                "active_concept_count": manifest["active_concept_count"],
                "active_card_count": manifest["active_card_count"],
                "runtime_card_index_count": manifest["runtime_card_index_count"],
                "runtime_card_quality_count": manifest["runtime_card_quality_count"],
                "source_corpus_only": True,
                "duplicate_subject_ids": manifest["duplicate_subject_ids"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine completed clean Wikipag Usage Bank shards.")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank-version", default="wikipag-clean-combined-v0.1")
    parser.add_argument("--card-index-backend", choices=["wikipag", "hash"], default="wikipag")
    parser.add_argument("--wikipag-service-url", default="http://127.0.0.1:8897")
    parser.add_argument("--embedding-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    manifest = combine(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
