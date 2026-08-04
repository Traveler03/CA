from __future__ import annotations

import hashlib
import json
import re
from ast import literal_eval
from pathlib import Path
from typing import Any


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "which",
    "what",
    "when",
    "where",
    "true",
    "false",
    "statement",
    "following",
}


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    raw = json.dumps([str(part) for part in parts], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def normalize_name(value: str) -> str:
    value = re.sub(r"[_/]+", " ", value.lower())
    value = re.sub(r"[^a-z0-9\s-]+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def keyword_candidates(text: str, *, limit: int) -> list[str]:
    tokens = [normalize_name(tok) for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)]
    tokens = [tok for tok in tokens if tok and tok not in STOPWORDS]
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts, key=lambda tok: (-counts[tok], tokens.index(tok), tok))
    return ranked[:limit]


def parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def is_time_sensitive(row: dict[str, Any]) -> bool:
    return any(item.strip().lower() == "yes" for item in parse_listish(row.get("time_sensitive")))


def is_culturally_sensitive(row: dict[str, Any]) -> bool:
    return str(row.get("cultural_sensitivity_label", "")).strip() not in {"", "-"}


def output_path(output_dir: str | Path, filename: str) -> Path:
    return Path(output_dir) / filename
