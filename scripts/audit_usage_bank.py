from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.io import read_jsonl


JSONL_COUNT_FIELDS = {
    "concept_registry.jsonl": ["concept_count", "active_concept_count"],
    "usage_cards.jsonl": ["usage_card_count", "active_card_count", "oracle_node_count"],
    "usage_card_claims.jsonl": ["usage_claim_count"],
    "usage_index.jsonl": ["usage_index_count"],
    "rejected_items.jsonl": ["rejected_item_count"],
}


def is_oracle_component(component: str) -> bool:
    return "oracle" in component.lower()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def load_manifest(bank_dir: Path) -> dict[str, Any]:
    path = bank_dir / "bank_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_counts(bank_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    combined_counts = manifest.get("combined_counts") or {}
    for file_name, manifest_keys in JSONL_COUNT_FIELDS.items():
        actual = line_count(bank_dir / file_name)
        expected_values = {
            key: manifest.get(key)
            for key in manifest_keys
            if manifest.get(key) is not None
        }
        if file_name in combined_counts:
            expected_values[f"combined_counts.{file_name}"] = combined_counts[file_name]
        ok = all(int(value) == actual for value in expected_values.values())
        checks.append(
            {
                "file": file_name,
                "actual_rows": actual,
                "expected": expected_values,
                "ok": ok,
            }
        )
    return checks


def audit_usage_cards(bank_dir: Path) -> dict[str, Any]:
    path = bank_dir / "usage_cards.jsonl"
    result: dict[str, Any] = {
        "exists": path.exists(),
        "rows": 0,
        "bank_component_counts": {},
        "benchmark_flag_counts": {},
        "uses_gold_answer_counts": {},
        "component_flag_errors": [],
    }
    if not path.exists():
        return result

    component_counts: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(path), start=1):
        result["rows"] += 1
        component = str(row.get("bank_component") or "missing")
        component_counts[component] += 1
        benchmark = row.get("benchmark_content_accessed")
        uses_gold = row.get("uses_gold_answer")
        benchmark_counts[str(benchmark)] += 1
        gold_counts[str(uses_gold)] += 1

        if is_oracle_component(component) and (benchmark is not True or uses_gold is not True):
            if len(errors) < 20:
                errors.append({"line": idx, "component": component, "benchmark_content_accessed": benchmark, "uses_gold_answer": uses_gold})
        if component == "wikipag_clean" and (benchmark is not False or uses_gold is not False):
            if len(errors) < 20:
                errors.append({"line": idx, "component": component, "benchmark_content_accessed": benchmark, "uses_gold_answer": uses_gold})

    result["bank_component_counts"] = dict(component_counts)
    result["benchmark_flag_counts"] = dict(benchmark_counts)
    result["uses_gold_answer_counts"] = dict(gold_counts)
    result["component_flag_errors"] = errors
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    bank_dir = Path(args.bank_dir)
    manifest = load_manifest(bank_dir)
    count_checks = audit_counts(bank_dir, manifest)
    usage_card_audit = audit_usage_cards(bank_dir)
    ok = bool(manifest) and all(item["ok"] for item in count_checks) and not usage_card_audit["component_flag_errors"]
    result = {
        "ok": ok,
        "bank_dir": str(bank_dir),
        "manifest_exists": bool(manifest),
        "construction_mode": manifest.get("construction_mode"),
        "subjects": manifest.get("subjects"),
        "count_checks": count_checks,
        "usage_cards": usage_card_audit,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Usage Bank counts and provenance flags.")
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = audit(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
