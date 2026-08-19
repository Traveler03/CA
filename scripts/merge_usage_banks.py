from __future__ import annotations

import argparse
import json
import shutil
import sys
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


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "bank_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def tag_row(row: dict[str, Any], *, component: str, source_dir: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    tagged = dict(row)
    tagged.setdefault("bank_component", component)
    tagged.setdefault("component_source_dir", str(source_dir))
    manifest = manifest or {}
    construction_mode = str(manifest.get("construction_mode") or "")
    manifest_is_oracle = (
        bool(manifest.get("benchmark_content_accessed"))
        or bool(manifest.get("uses_gold_answer"))
        or is_oracle_component(construction_mode)
    )
    if is_oracle_component(component) or manifest_is_oracle:
        tagged.setdefault("benchmark_content_accessed", True)
        tagged.setdefault("uses_gold_answer", True)
        tagged.setdefault("can_be_used_for_clean_eval", False)
        tagged.setdefault("can_be_used_for_clean_global_mmlu_eval", False)
    elif component == "wikipag_clean":
        tagged.setdefault("benchmark_content_accessed", False)
        tagged.setdefault("uses_gold_answer", False)
        tagged.setdefault("can_be_used_for_clean_eval", True)
        tagged.setdefault("can_be_used_for_clean_global_mmlu_eval", True)
    return tagged


def is_oracle_component(component: str) -> bool:
    return "oracle" in component.lower()


def combine_jsonl(output_dir: Path, components: list[tuple[str, Path]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    manifests = {component: load_manifest(source_dir) for component, source_dir in components}
    for file_name in JSONL_FILES:
        rows: list[dict[str, Any]] = []
        for component, source_dir in components:
            path = source_dir / file_name
            if path.exists():
                rows.extend(
                    tag_row(row, component=component, source_dir=source_dir, manifest=manifests.get(component))
                    for row in read_jsonl(path)
                )
        write_jsonl(output_dir / file_name, rows)
        counts[file_name] = len(rows)
    return counts


def copy_component_manifests(output_dir: Path, components: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    component_dir = output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for component, source_dir in components:
        manifest = load_manifest(source_dir)
        manifest["component"] = component
        manifest["source_dir"] = str(source_dir)
        manifests.append(manifest)
        (component_dir / f"{component}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifests


def copy_usage_indexes(output_dir: Path, components: list[tuple[str, Path]]) -> None:
    dst_root = output_dir / "usage_indexes"
    for component, source_dir in components:
        src_root = source_dir / "usage_indexes"
        if not src_root.exists():
            continue
        dst = dst_root / component
        shutil.copytree(src_root, dst, dirs_exist_ok=True)


def parse_component_specs(values: list[str]) -> list[tuple[str, Path]]:
    components: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--component must be NAME=DIR, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"--component must be NAME=DIR, got: {value}")
        components.append((name, Path(raw_path)))
    return components


def build_manifest(
    *,
    bank_version: str,
    components: list[tuple[str, Path]],
    component_manifests: list[dict[str, Any]],
    combined_counts: dict[str, int],
    runtime_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subjects = sorted(
        {
            subject
            for manifest in component_manifests
            for subject in manifest.get("subject_ids", [])
            if subject
        }
    )
    return {
        "bank_version": bank_version,
        "construction_mode": "combined_oracle_plus_clean",
        "components": [
            {
                "name": component,
                "source_dir": str(source_dir),
                "construction_mode": manifest.get("construction_mode"),
                "construction_model": manifest.get("construction_model"),
                "benchmark_content_accessed": manifest.get("benchmark_content_accessed"),
                "uses_gold_answer": manifest.get("uses_gold_answer"),
                "concept_count": manifest.get("concept_count"),
                "active_concept_count": manifest.get("active_concept_count"),
                "active_card_count": manifest.get("active_card_count") or manifest.get("oracle_node_count"),
                "oracle_node_count": manifest.get("oracle_node_count"),
            }
            for (component, source_dir), manifest in zip(components, component_manifests)
        ],
        "subjects": len(subjects),
        "subject_ids": subjects,
        "concept_count": combined_counts.get("concept_registry.jsonl", 0),
        "usage_card_count": combined_counts.get("usage_cards.jsonl", 0),
        "usage_claim_count": combined_counts.get("usage_card_claims.jsonl", 0),
        "usage_index_count": combined_counts.get("usage_index.jsonl", 0),
        "combined_counts": combined_counts,
        "runtime_bank": runtime_bank,
        "benchmark_content_accessed": any(bool(manifest.get("benchmark_content_accessed")) for manifest in component_manifests),
        "can_be_used_for_clean_eval": False,
        "can_be_used_for_clean_global_mmlu_eval": False,
        "provenance_warning": "Combined bank contains oracle benchmark-derived rows. Filter bank_component=wikipag_clean for clean evaluation.",
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.component:
        components = parse_component_specs(args.component)
    else:
        if not args.oracle_dir or not args.clean_dir:
            raise ValueError("Either pass --component NAME=DIR one or more times, or pass both --oracle-dir and --clean-dir.")
        components = [
            (args.oracle_component, Path(args.oracle_dir)),
            (args.clean_component, Path(args.clean_dir)),
        ]
    missing = [str(path) for _name, path in components if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing component dirs: {missing}")
    combined_counts = combine_jsonl(output_dir, components)
    component_manifests = copy_component_manifests(output_dir, components)
    copy_usage_indexes(output_dir, components)
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
        components=components,
        component_manifests=component_manifests,
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
                "ok": True,
                "output_dir": str(output_dir),
                "construction_mode": manifest["construction_mode"],
                "subjects": manifest["subjects"],
                "concept_count": manifest["concept_count"],
                "usage_card_count": manifest["usage_card_count"],
                "benchmark_content_accessed": manifest["benchmark_content_accessed"],
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
    parser = argparse.ArgumentParser(description="Merge oracle and clean Usage Bank shards without erasing provenance.")
    parser.add_argument("--component", action="append", default=[], help="Generic component as NAME=DIR. Can be passed multiple times.")
    parser.add_argument("--oracle-dir")
    parser.add_argument("--clean-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-component", default="global_mmlu_oracle")
    parser.add_argument("--clean-component", default="wikipag_clean")
    parser.add_argument("--bank-version", default="combined-oracle-clean-v0.1")
    parser.add_argument("--write-runtime-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-runtime-embedding-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("data/external/models/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-truncate-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    manifest = merge(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
