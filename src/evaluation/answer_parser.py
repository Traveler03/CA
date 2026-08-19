from __future__ import annotations

import json
import re
from dataclasses import dataclass


VALID_LABELS = {"A", "B", "C", "D"}


@dataclass
class ParsedAnswer:
    answer: str | None
    valid: bool
    source: str


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.I | re.S)
ANSWER_FIELD_RE = re.compile(r'"answer"\s*:\s*"([ABCD])"', re.I)
LABEL_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:)\s*([ABCD])\b", re.I),
    re.compile(r"\boption\s+([ABCD])\b", re.I),
    re.compile(r"^\s*([ABCD])\s*$", re.I),
    re.compile(r"\b([ABCD])\b", re.I),
]


def parse_answer(text: str) -> ParsedAnswer:
    text = (text or "").strip()
    if not text:
        return ParsedAnswer(None, False, "empty")

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer.strip().upper() in VALID_LABELS:
                return ParsedAnswer(answer.strip().upper(), True, "json")

    field = ANSWER_FIELD_RE.search(text)
    if field:
        return ParsedAnswer(field.group(1).upper(), True, "json_field_regex")

    for pattern in LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return ParsedAnswer(match.group(1).upper(), True, "label_regex")

    return ParsedAnswer(None, False, "unparsed")


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in JSON_FENCE_RE.finditer(text))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    return candidates

