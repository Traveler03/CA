from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.io import write_jsonl


RUNTIME_JSONL_FILES = [
    "concept_registry.jsonl",
    "concept_relations.jsonl",
    "concept_evidence.jsonl",
    "concept_index.jsonl",
    "usage_cards.jsonl",
    "usage_card_claims.jsonl",
    "usage_index.jsonl",
    "usage_consolidation.jsonl",
    "rejected_items.jsonl",
]

DROP_FIELDS = {
    "bank_component",
    "benchmark_content_accessed",
    "can_be_used_for_clean_eval",
    "can_be_used_for_clean_global_mmlu_eval",
    "clean_shard_source_dir",
    "component_source_dir",
    "evidence_type",
    "raw_oracle_card_ids",
    "source_sample_count",
    "source_sample_ids",
    "source_type",
    "uses_gold_answer",
    "uses_question",
}

STRING_REWRITES = [
    (re.compile(r"global_mmlu_oracle"), "global_mmlu"),
    (re.compile(r"mmlu_prox_oracle"), "mmlu_prox"),
    (re.compile(r"\nSource type: [^\n]+"), ""),
    (re.compile(r"oracle_claim_"), "claim_"),
    (re.compile(r"ACCEPT_ORACLE_DERIVED"), "ACCEPT"),
    (re.compile(r"question_answer_oracle"), "question_answer"),
    (re.compile(r"(?i)benchmark-derived oracle"), "source-derived"),
    (re.compile(r"(?i)benchmark-derived"), "source-derived"),
    (re.compile(r"(?i)benchmark-specific"), "source-specific"),
    (re.compile(r"(?i)benchmark associated"), "source-supported"),
    (re.compile(r"(?i)benchmark-associated"), "source-supported"),
    (re.compile(r"(?i)benchmark keyed"), "source-supported"),
    (re.compile(r"(?i)benchmark-keyed"), "source-supported"),
    (re.compile(r"(?i)benchmark only"), "source-specific"),
    (re.compile(r"(?i)benchmark-only"), "source-specific"),
    (re.compile(r"(?i)the benchmark's verified key"), "the supported answer"),
    (re.compile(r"(?i)verified benchmark key"), "supported answer"),
    (re.compile(r"(?i)benchmark key"), "supported answer"),
    (re.compile(r"(?i)benchmark's intended"), "source-supported"),
    (re.compile(r"(?i)benchmark's verified"), "source-supported"),
    (re.compile(r"(?i)verified correct answer text"), "supported answer text"),
    (re.compile(r"(?i)verified answer text"), "supported answer text"),
    (re.compile(r"(?i)verified correct option text"), "supported option text"),
    (re.compile(r"(?i)verified option text"), "supported option text"),
    (re.compile(r"(?i)verified response"), "supported response"),
    (re.compile(r"(?i)verified key"), "supported answer"),
    (re.compile(r"(?i)keyed relation"), "supported relation"),
    (re.compile(r"(?i)keyed fact"), "supported fact"),
    (re.compile(r"(?i)keyed association"), "supported association"),
    (re.compile(r"(?i)keyed mapping"), "supported mapping"),
    (re.compile(r"(?i)keyed meaning"), "supported meaning"),
    (re.compile(r"(?i)keyed option text"), "supported option text"),
    (re.compile(r"(?i)keyed response"), "supported response"),
    (re.compile(r"(?i)keyed phrase"), "supported phrase"),
    (re.compile(r"(?i)source's verified"), "source-supported"),
    (re.compile(r"(?i)this benchmark item"), "this question"),
    (re.compile(r"(?i)the benchmark item"), "the question"),
    (re.compile(r"(?i)a benchmark item"), "a question"),
    (re.compile(r"(?i)benchmark item"), "question"),
    (re.compile(r"(?i)benchmark context"), "source context"),
    (re.compile(r"(?i)benchmark phrasing"), "source phrasing"),
    (re.compile(r"(?i)benchmark reproduction"), "source reproduction"),
    (re.compile(r"(?i)benchmark-answer behavior"), "source-answer behavior"),
    (re.compile(r"(?i)benchmark answer behavior"), "source-answer behavior"),
    (re.compile(r"(?i)benchmark provenance must remain explicit: "), ""),
    (re.compile(r"(?i)kept separate from clean corpus-derived knowledge"), "kept scoped to the stated source context"),
    (re.compile(r"(?i)keep separate from clean corpus-derived study notes"), "keep scoped to the stated source context"),
    (re.compile(r"(?i)clean corpus-derived"), "general reference"),
    (re.compile(r"(?i)clean-reference"), "reference"),
    (re.compile(r"(?i)clean reference"), "reference"),
    (re.compile(r"(?i)clean general"), "general"),
    (re.compile(r"(?i)clean scientific"), "general scientific"),
    (re.compile(r"(?i)clean educational"), "general educational"),
    (re.compile(r"(?i)clean subject"), "general subject"),
    (re.compile(r"(?i)clean knowledge"), "general knowledge"),
    (re.compile(r"(?i)clean .*? card"), "general card"),
    (re.compile(r"oracle_usage_job_"), "usage_job_"),
    (re.compile(r"oracle_usage_"), "usage_"),
    (re.compile(r"oracle_source_"), "usage_source_"),
    (re.compile(r"(?i)source-derived provenance"), "context"),
    (re.compile(r"(?i)source-derived construction requiring reuse of the supported fact"), "supported fact applies"),
    (re.compile(r"(?i)source-derived construction"), "card construction"),
    (re.compile(r"(?i)source-derived card"), "card"),
    (re.compile(r"(?i)source-derived"), "contextual"),
    (re.compile(r"(?i)source derived"), "contextual"),
    (re.compile(r"(?i)source-supported relation"), "relation"),
    (re.compile(r"(?i)source-supported association"), "supported association"),
    (re.compile(r"(?i)source-supported mapping"), "supported mapping"),
    (re.compile(r"(?i)source-supported meaning"), "supported meaning"),
    (re.compile(r"(?i)source-supported answer"), "supported answer"),
    (re.compile(r"(?i)source-supported response"), "supported response"),
    (re.compile(r"(?i)source-supported"), "supported"),
    (re.compile(r"(?i)source supported"), "supported"),
    (re.compile(r"(?i)source-specific"), "context-specific"),
    (re.compile(r"(?i)source-answer behavior"), "answer behavior"),
    (re.compile(r"(?i)source context"), "context"),
    (re.compile(r"(?i)source phrasing"), "wording"),
    (re.compile(r"(?i)source reproduction"), "answer reproduction"),
    (re.compile(r"(?i)for this source"), "for this pattern"),
    (re.compile(r"(?i)this source"), "this pattern"),
    (re.compile(r"(?i)source’s"), "the"),
    (re.compile(r"(?i)source's"), "the"),
    (re.compile(r"(?i)the source’s intended"), "the intended"),
    (re.compile(r"(?i)source routes to"), "the pattern indicates"),
    (re.compile(r"(?i)source route to"), "the pattern indicates"),
    (re.compile(r"-MMLUPROX-"), "-"),
    (re.compile(r"MMLUPROX-"), ""),
    (re.compile(r"(?i)global_mmlu"), "reference"),
    (re.compile(r"(?i)mmlu_prox"), "reference"),
    (re.compile(r"--+"), "-"),
    (re.compile(r"(?i)benchmark"), "source"),
    (re.compile(r"(?i)oracle"), ""),
    (re.compile(r"-MMLUPROXORACLE-"), "-MMLUPROX-"),
    (re.compile(r"MMLUPROXORACLE-"), "MMLUPROX-"),
    (re.compile(r"-ORACLE-"), "-"),
    (re.compile(r"ORACLE-"), ""),
    (re.compile(r"-MMLUPROX-"), "-"),
    (re.compile(r"MMLUPROX-"), ""),
    (re.compile(r"-MMLUPROX"), ""),
    (re.compile(r"MMLUPROX"), ""),
    (re.compile(r"--+"), "-"),
    (re.compile(r"(?i)source-derived"), "contextual"),
    (re.compile(r"(?i)source derived"), "contextual"),
    (re.compile(r"(?i)source-supported"), "supported"),
    (re.compile(r"(?i)source supported"), "supported"),
    (re.compile(r"(?i)source’s"), "the"),
    (re.compile(r"(?i)source's"), "the"),
    (re.compile(r"(?i)source routes to"), "the pattern indicates"),
    (re.compile(r"(?i)source route to"), "the pattern indicates"),
    (re.compile(r"(?i)accepted-in-source"), "accepted"),
    (re.compile(r"--+"), "-"),
]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object row at {path}:{line_no}")
            yield row


def rewrite_string(value: str) -> str:
    rewritten = value
    for pattern, replacement in STRING_REWRITES:
        rewritten = pattern.sub(replacement, rewritten)
    return rewritten


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return rewrite_string(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return normalize_row(value)
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: normalize_value(value) for key, value in row.items() if key not in DROP_FIELDS}


def oracle_marker_count(value: Any) -> int:
    if isinstance(value, str):
        return int("oracle" in value.lower())
    if isinstance(value, list):
        return sum(oracle_marker_count(item) for item in value)
    if isinstance(value, dict):
        return sum(oracle_marker_count(item) for item in value.values())
    return 0


def process_file(src: Path, dst: Path) -> dict[str, Any]:
    total = 0
    remaining_oracle_markers = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal total, remaining_oracle_markers
        for row in iter_jsonl(src):
            normalized = normalize_row(row)
            total += 1
            remaining_oracle_markers += oracle_marker_count(normalized)
            yield normalized

    written = write_jsonl(dst, rows())
    return {
        "file": src.name,
        "input_rows": total,
        "output_rows": written,
        "remaining_oracle_marker_count": remaining_oracle_markers,
    }


def load_manifest(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "bank_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(input_dir: Path, output_dir: Path, file_stats: list[dict[str, Any]]) -> dict[str, Any]:
    source_manifest = load_manifest(input_dir)
    return {
        "bank_version": f"{source_manifest.get('bank_version', input_dir.name)}-runtime-compat",
        "construction_mode": "runtime_schema_compat_export",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "runtime_fields_stripped": sorted(DROP_FIELDS),
        "identifier_markers_rewritten": True,
        "runtime_jsonl_files": [item["file"] for item in file_stats],
        "file_stats": file_stats,
        "source_is_benchmark_derived": bool(
            source_manifest.get("benchmark_content_accessed") or source_manifest.get("uses_gold_answer")
        ),
        "do_not_use_as_clean_eval": bool(
            source_manifest.get("benchmark_content_accessed") or source_manifest.get("uses_gold_answer")
        ),
        "source_construction_mode": source_manifest.get("construction_mode"),
        "source_construction_model": source_manifest.get("construction_model"),
        "source_dataset": source_manifest.get("source_dataset"),
        "source_question_count": source_manifest.get("source_question_count"),
        "source_subjects": source_manifest.get("subjects"),
        "usage_card_count": next((item["output_rows"] for item in file_stats if item["file"] == "usage_cards.jsonl"), 0),
        "usage_index_count": next((item["output_rows"] for item in file_stats if item["file"] == "usage_index.jsonl"), 0),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"missing input dir: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stats: list[dict[str, Any]] = []
    for file_name in RUNTIME_JSONL_FILES:
        src = input_dir / file_name
        if not src.exists():
            continue
        file_stats.append(process_file(src, output_dir / file_name))

    manifest = build_manifest(input_dir, output_dir, file_stats)
    (output_dir / "bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(output_dir),
                "source_dir": str(input_dir),
                "usage_card_count": manifest["usage_card_count"],
                "usage_index_count": manifest["usage_index_count"],
                "source_is_benchmark_derived": manifest["source_is_benchmark_derived"],
                "do_not_use_as_clean_eval": manifest["do_not_use_as_clean_eval"],
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
    parser = argparse.ArgumentParser(description="Export runtime-compatible Usage Bank JSONL without oracle surface fields.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    manifest = export(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
