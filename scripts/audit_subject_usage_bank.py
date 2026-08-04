from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "subject_profile.jsonl",
    "concept_queries.jsonl",
    "concept_passages.jsonl",
    "candidate_concepts.jsonl",
    "concept_registry.jsonl",
    "concept_evidence.jsonl",
    "evidence_packs.jsonl",
    "runtime_cards.raw.jsonl",
    "runtime_card_claims.jsonl",
    "runtime_cards.jsonl",
    "runtime_card_index.jsonl",
    "runtime_card_index.npy",
    "runtime_card_index_meta.json",
    "evidence_sections.parquet",
    "build_events.jsonl",
    "rejected_items.jsonl",
    "bank_manifest.json",
    "summary.json",
]
ALLOW_EMPTY_FILES = {
    "rejected_items.jsonl",
}

FORBIDDEN_MODEL_PATTERNS = [
    re.compile(r"qwen" + r"3\.5", re.IGNORECASE),
    re.compile(r"qwen" + r"-?3\.5", re.IGNORECASE),
]

ASSESSMENT_ARTIFACT_KEYS = {"sample_id", "prediction", "reference_response", "labeled_alternatives"}


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


def check_no_assessment_artifact_keys(rows: list[dict[str, Any]], file_name: str, errors: list[str]) -> None:
    for idx, row in enumerate(rows[:1000], start=1):
        bad = sorted(ASSESSMENT_ARTIFACT_KEYS & set(row))
        if bad:
            errors.append(f"assessment-artifact keys {bad} in {file_name}:{idx}")
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
    if manifest.get("construction_model") != expected_model:
        errors.append(f"construction_model mismatch: {manifest.get('construction_model')} != {expected_model}")
    if summary.get("model") != expected_model:
        errors.append(f"summary model mismatch: {summary.get('model')} != {expected_model}")
    if manifest.get("source_corpus_only") is not True:
        errors.append("bank_manifest source_corpus_only is not true")
    if summary.get("active_card_count", 0) <= 0:
        errors.append("no active usage cards")
    if summary.get("active_concept_count", 0) <= 0:
        errors.append("no active concepts")

    scan_forbidden_model_strings(root, errors)

    concept_rows = read_jsonl(root / "concept_registry.jsonl") if (root / "concept_registry.jsonl").exists() else []
    runtime_rows = read_jsonl(root / "runtime_cards.jsonl") if (root / "runtime_cards.jsonl").exists() else []
    claim_rows = read_jsonl(root / "runtime_card_claims.jsonl") if (root / "runtime_card_claims.jsonl").exists() else []
    index_rows = read_jsonl(root / "runtime_card_index.jsonl") if (root / "runtime_card_index.jsonl").exists() else []
    for file_name, rows in [
        ("concept_registry.jsonl", concept_rows),
        ("runtime_cards.jsonl", runtime_rows),
        ("runtime_card_claims.jsonl", claim_rows),
        ("runtime_card_index.jsonl", index_rows),
    ]:
        check_no_assessment_artifact_keys(rows, file_name, errors)

    active_card_ids = {row.get("card_id") for row in runtime_rows if row.get("status") == "active"}
    rejected_core = [
        row
        for row in claim_rows
        if row.get("card_id") in active_card_ids
        and row.get("slot") in {"definition", "trigger", "rule"}
        and row.get("decision") == "REJECT"
    ]
    if rejected_core:
        errors.append(f"active cards have rejected core claims: {len(rejected_core)}")

    try:
        import numpy as np

        matrix = np.load(root / "runtime_card_index.npy")
        meta = read_json(root / "runtime_card_index_meta.json")
        if int(matrix.shape[0]) != len(index_rows):
            errors.append(f"runtime_card_index rows {matrix.shape[0]} != ids {len(index_rows)}")
        if int(meta.get("count") or -1) != len(index_rows):
            errors.append(f"runtime_card_index_meta count {meta.get('count')} != rows {len(index_rows)}")
    except Exception as exc:
        errors.append(f"failed to read runtime card index: {type(exc).__name__}: {exc}")

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
            "runtime_card_count": summary.get("runtime_card_count"),
            "runtime_card_index_count": summary.get("runtime_card_index_count"),
            "source_corpus_only": manifest.get("source_corpus_only"),
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
