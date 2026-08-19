from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "concept_registry.jsonl",
    "concept_relations.jsonl",
    "evidence_sections.parquet",
    "usage_cards.jsonl",
    "usage_card_claims.jsonl",
    "usage_index.jsonl",
    "build_events.jsonl",
    "rejected_items.jsonl",
    "bank_manifest.json",
    "summary.json",
]
ALLOW_EMPTY_FILES = {
    "concept_relations.jsonl",
    "rejected_items.jsonl",
}

FORBIDDEN_MODEL_PATTERNS = [
    re.compile(r"qwen" + r"3\.5", re.IGNORECASE),
    re.compile(r"qwen" + r"-?3\.5", re.IGNORECASE),
]

BENCHMARK_KEYS = {"question", "options", "answer", "gold", "sample_id", "prediction"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path}")
    elif path.stat().st_size <= 0:
        errors.append(f"empty file: {path}")


def scan_forbidden_model_strings(root: Path, errors: list[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".faiss", ".parquet", ".npz", ".sqlite"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_MODEL_PATTERNS:
            if pattern.search(text):
                errors.append(f"forbidden legacy chat-model reference in {path}")
                break


def check_no_benchmark_keys(rows: list[dict[str, Any]], file_name: str, errors: list[str]) -> None:
    for idx, row in enumerate(rows[:1000], start=1):
        bad = sorted(BENCHMARK_KEYS & set(row))
        if bad:
            errors.append(f"benchmark-like keys {bad} in {file_name}:{idx}")
            return


def audit(root: Path, *, expected_model: str) -> dict[str, Any]:
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        if rel in ALLOW_EMPTY_FILES:
            if not (root / rel).exists():
                errors.append(f"missing file: {root / rel}")
        else:
            assert_file(root / rel, errors)

    summary = read_json(root / "summary.json") if (root / "summary.json").exists() else {}
    manifest = read_json(root / "bank_manifest.json") if (root / "bank_manifest.json").exists() else {}
    subject_ids = [str(item) for item in (manifest.get("subject_ids") or [summary.get("subject")]) if item]
    usage_index_dirs = [root / "usage_indexes" / subject for subject in subject_ids]
    for usage_index_dir in usage_index_dirs:
        for rel in ["usage_index.jsonl", "usage_index.faiss", "usage_index_ids.jsonl", "usage_index_meta.json"]:
            assert_file(usage_index_dir / rel, errors)

    if manifest.get("construction_model") != expected_model:
        errors.append(f"construction_model mismatch: {manifest.get('construction_model')} != {expected_model}")
    if summary.get("model") != expected_model:
        errors.append(f"summary model mismatch: {summary.get('model')} != {expected_model}")
    if manifest.get("benchmark_content_accessed") is not False:
        errors.append("bank_manifest benchmark_content_accessed is not false")
    if summary.get("active_card_count", 0) <= 0:
        errors.append("no active usage cards")
    if summary.get("active_concept_count", 0) <= 0:
        errors.append("no active concepts")

    scan_forbidden_model_strings(root, errors)

    concept_rows = read_jsonl(root / "concept_registry.jsonl") if (root / "concept_registry.jsonl").exists() else []
    usage_rows = read_jsonl(root / "usage_cards.jsonl") if (root / "usage_cards.jsonl").exists() else []
    claim_rows = read_jsonl(root / "usage_card_claims.jsonl") if (root / "usage_card_claims.jsonl").exists() else []
    index_rows = read_jsonl(root / "usage_index.jsonl") if (root / "usage_index.jsonl").exists() else []
    for file_name, rows in [
        ("concept_registry.jsonl", concept_rows),
        ("usage_cards.jsonl", usage_rows),
        ("usage_card_claims.jsonl", claim_rows),
        ("usage_index.jsonl", index_rows),
    ]:
        check_no_benchmark_keys(rows, file_name, errors)

    active_usage_ids = {row.get("usage_id") for row in usage_rows if row.get("status") == "active"}
    rejected_core = [
        row
        for row in claim_rows
        if row.get("usage_id") in active_usage_ids
        and row.get("field") in {"concept_boundary", "decision_procedure", "trigger_conditions", "verification_rules"}
        and row.get("decision") == "REJECT"
    ]
    if rejected_core:
        errors.append(f"active cards have rejected core claims: {len(rejected_core)}")

    for usage_index_dir in usage_index_dirs:
        if not usage_index_dir.exists() or not (usage_index_dir / "usage_index.faiss").exists():
            continue
        try:
            import faiss

            index = faiss.read_index(str(usage_index_dir / "usage_index.faiss"))
            ids = read_jsonl(usage_index_dir / "usage_index_ids.jsonl")
            meta = read_json(usage_index_dir / "usage_index_meta.json")
            if int(index.ntotal) != len(ids):
                errors.append(f"faiss ntotal {index.ntotal} != ids {len(ids)}")
            subject_rows = [row for row in index_rows if row.get("subject") == usage_index_dir.name]
            if int(index.ntotal) != len(subject_rows):
                errors.append(f"{usage_index_dir.name}: faiss ntotal {index.ntotal} != usage_index rows {len(subject_rows)}")
            if int(meta.get("count") or -1) != len(subject_rows):
                errors.append(f"{usage_index_dir.name}: usage_index_meta count {meta.get('count')} != usage_index rows {len(subject_rows)}")
        except Exception as exc:
            errors.append(f"failed to read FAISS usage index {usage_index_dir}: {type(exc).__name__}: {exc}")

    if (root / "evidence_sections.parquet").exists():
        try:
            import pandas as pd

            df = pd.read_parquet(root / "evidence_sections.parquet")
            if len(df) <= 0:
                errors.append("evidence_sections.parquet has no rows")
        except Exception as exc:
            errors.append(f"failed to read evidence_sections.parquet: {type(exc).__name__}: {exc}")

    return {
        "ok": not errors,
        "errors": errors,
        "summary": {
            "subject": summary.get("subject"),
            "model": summary.get("model"),
            "active_concept_count": summary.get("active_concept_count"),
            "active_card_count": summary.get("active_card_count"),
            "usage_index_count": summary.get("usage_index_count"),
            "usage_faiss_index_count": summary.get("usage_faiss_index_count"),
            "benchmark_content_accessed": manifest.get("benchmark_content_accessed"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a subject-driven Usage Bank build directory.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-model", default="gpt-5.4")
    args = parser.parse_args(argv)
    result = audit(args.output_dir, expected_model=args.expected_model)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
