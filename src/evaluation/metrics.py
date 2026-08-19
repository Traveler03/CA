from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class AccuracyRow:
    method: str
    language: str
    total: int
    parsed: int
    correct: int
    accuracy: float


def summarize_accuracy(rows: list[dict[str, Any]]) -> list[AccuracyRow]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["language"])].append(row)
    out: list[AccuracyRow] = []
    for (method, language), items in sorted(grouped.items()):
        total = len(items)
        parsed = sum(1 for item in items if item.get("parsed_valid"))
        correct = sum(1 for item in items if item.get("correct"))
        out.append(
            AccuracyRow(
                method=method,
                language=language,
                total=total,
                parsed=parsed,
                correct=correct,
                accuracy=correct / total if total else 0.0,
            )
        )
    return out


def macro_average(summary: list[AccuracyRow]) -> dict[str, float]:
    by_method: dict[str, list[AccuracyRow]] = defaultdict(list)
    for row in summary:
        by_method[row.method].append(row)
    return {
        method: sum(row.accuracy for row in rows) / len(rows)
        for method, rows in sorted(by_method.items())
        if rows
    }


def help_harm_net(rows: list[dict[str, Any]], baseline_method: str = "zero_shot") -> dict[str, dict[str, int]]:
    by_sample: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        by_sample[(row["language"], row["sample_id"])][row["method"]] = bool(row.get("correct"))
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"help": 0, "harm": 0, "net": 0})
    for method_map in by_sample.values():
        if baseline_method not in method_map:
            continue
        base = method_map[baseline_method]
        for method, correct in method_map.items():
            if method == baseline_method:
                continue
            if correct and not base:
                out[method]["help"] += 1
            elif base and not correct:
                out[method]["harm"] += 1
    for values in out.values():
        values["net"] = values["help"] - values["harm"]
    return dict(out)

