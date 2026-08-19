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

from src.smoke_test.io import read_jsonl, write_jsonl
from scripts.export_concept_card_runtime_bank import write_runtime_bank


JSONL_FILES = [
    "concept_registry.jsonl",
    "concept_relations.jsonl",
    "concept_evidence.jsonl",
    "concept_index.jsonl",
    "usage_cards.jsonl",
    "usage_card_claims.jsonl",
    "usage_index.jsonl",
    "usage_consolidation.jsonl",
    "build_events.jsonl",
    "rejected_items.jsonl",
]


COUNT_KEYS = [
    "concept_count",
    "active_concept_count",
    "active_card_count",
    "usage_claim_count",
    "usage_index_count",
    "usage_faiss_index_count",
    "evidence_section_count",
    "rejected_item_count",
    "model_network_calls",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tag_row(row: dict[str, Any], *, shard: Path) -> dict[str, Any]:
    tagged = dict(row)
    tagged.setdefault("clean_shard_source_dir", str(shard))
    tagged.setdefault("benchmark_content_accessed", False)
    tagged.setdefault("uses_gold_answer", False)
    tagged.setdefault("can_be_used_for_clean_global_mmlu_eval", True)
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


def copy_usage_indexes(output_dir: Path, shard_dirs: list[Path]) -> None:
    dst_root = output_dir / "usage_indexes"
    for shard in shard_dirs:
        src_root = shard / "usage_indexes"
        if not src_root.exists():
            continue
        for subject_dir in src_root.iterdir():
            if not subject_dir.is_dir():
                continue
            dst = dst_root / subject_dir.name
            shutil.copytree(subject_dir, dst, dirs_exist_ok=True)


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
    runtime_bank: dict[str, Any] | None = None,
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
        "active_card_count": combined_counts.get("usage_cards.jsonl", 0),
        "usage_claim_count": combined_counts.get("usage_card_claims.jsonl", 0),
        "usage_index_count": combined_counts.get("usage_index.jsonl", 0),
        "usage_faiss_index_count": totals["usage_faiss_index_count"],
        "evidence_section_count": totals["evidence_section_count"],
        "rejected_item_count": combined_counts.get("rejected_items.jsonl", 0),
        "model_network_calls": totals["model_network_calls"],
        "combined_counts": combined_counts,
        "runtime_bank": runtime_bank,
        "construction_cutoff": datetime.now(timezone.utc).isoformat(),
        "benchmark_content_accessed": False,
        "uses_gold_answer": False,
        "can_be_used_for_clean_global_mmlu_eval": True,
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
    copy_usage_indexes(output_dir, shard_dirs)
    combine_evidence_sections(output_dir, shard_dirs)
    runtime_bank = None
    if getattr(args, "write_runtime_bank", True):
        runtime_bank = write_runtime_bank(
            input_dir=output_dir,
            build_embedding_index=bool(getattr(args, "build_runtime_embedding_index", True)),
            embedding_model_dir=Path(getattr(args, "embedding_model_dir", "data/external/models/Qwen3-Embedding-4B")),
            embedding_device=str(getattr(args, "embedding_device", "cuda")),
            embedding_truncate_dim=int(getattr(args, "embedding_truncate_dim", 1024)),
            embedding_batch_size=int(getattr(args, "embedding_batch_size", 64)),
        )
    manifest = build_manifest(
        bank_version=args.bank_version,
        shard_dirs=shard_dirs,
        shard_manifests=shard_manifests,
        combined_counts=combined_counts,
        runtime_bank=runtime_bank,
    )
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
                "benchmark_content_accessed": False,
                "duplicate_subject_ids": manifest["duplicate_subject_ids"],
                "runtime_bank": runtime_bank,
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
    parser.add_argument("--write-runtime-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-runtime-embedding-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("data/external/models/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-truncate-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    manifest = combine(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
