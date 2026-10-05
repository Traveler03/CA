"""Read-only same-article evidence retrieval and conservative passage filtering.

The lexical ranking below is a candidate selector, never an entailment test.
No FAISS index or embedding model is loaded by this backend.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from src.construction.application_cards import Passage, PassageGroup, digest, write_json


MISSING_MATH = (
    r"\b(?:requirement|complexity|cost)\s+(?:of\s*)?[,.;)]",
    r"\b(?:real|complex) number\s+[,;]",
    r"\bif\s+is\s+(?:a |positive|negative)",
    r"\bfor\s+and\s+(?:real|complex)",
    r"\b(?:defined|given|expressed) as:\s*\n\s*(?:This|The|In|For)\b",
)
USAGE_TERMS = re.compile(r"\b(?:calculate|calculated|divide|multiply|sum|total|ratio|procedure|"
                         r"steps|compute|if|when|using|therefore|implies|example)\b", re.I)
BOUNDARY_TERMS = re.compile(r"\b(?:unless|except|exception|only|not|cannot|however|"
                            r"limitation|assumption|instead|requires|unlike)\b", re.I)


def quality_issues(passage: Passage) -> list[str]:
    """High-precision signs of source damage; absence of flags is not a certificate."""
    issues = []
    if len(passage.text.strip()) < 80:
        issues.append("too_short")
    if "\ufffd" in passage.text or any(ord(c) < 32 and c not in "\n\r\t" for c in passage.text):
        issues.append("damaged_encoding")
    if "(disambiguation)" in passage.title.lower():
        issues.append("disambiguation")
    for pattern in MISSING_MATH:
        match = re.search(pattern, passage.text)
        if match:
            issues.append(f"missing_formula: {match.group(0)!r}")
    return issues


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class SourcePreparer:
    def __init__(self, *, offsets_db: Path | None = None, corpus_root: Path | None = None,
                 source_id_prefix: str | None = None, max_candidates: int = 64,
                 max_passages: int = 5):
        if not 1 <= max_passages <= 5 or not 1 <= max_candidates <= 256:
            raise ValueError("preparation limits must be max_passages=1..5, max_candidates=1..256")
        if any(x is not None for x in (offsets_db, corpus_root, source_id_prefix)) and not all(
                x is not None for x in (offsets_db, corpus_root, source_id_prefix)):
            raise ValueError("local retrieval requires offsets_db, corpus_root and source_id_prefix together")
        self.offsets_db = offsets_db.resolve() if offsets_db is not None else None
        self.corpus_root = corpus_root.resolve() if corpus_root is not None else None
        self.source_id_prefix = source_id_prefix
        self.max_candidates = max_candidates
        self.max_passages = max_passages
        self.files: dict[int, Path] = {}
        self.protocol: dict[str, Any] = {
            "backend": "local_same_article_lexical" if self.offsets_db else "supplied_passages_only",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_candidates_per_article": max_candidates, "max_passages": max_passages,
            "source_id_prefix": source_id_prefix,
        }
        if self.offsets_db:
            with closing(self._connect()) as connection:
                for index, relative in connection.execute("SELECT idx, path FROM files ORDER BY idx"):
                    path = (self.corpus_root / relative).resolve()
                    if not path.is_relative_to(self.corpus_root):
                        raise ValueError("corpus file is outside the configured read-only corpus root")
                    self.files[index] = path
            self.protocol.update(offsets_db=fingerprint(self.offsets_db),
                                 corpus_files=[fingerprint(p) for p in self.files.values()])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.offsets_db.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _read(self, identity: str, file_index: int, offset: int) -> dict[str, Any]:
        path = self.files[file_index]
        with path.open("rb") as stream:
            stream.seek(offset)
            line = stream.readline(1024 * 1024)
        if not line.endswith(b"\n") and len(line) == 1024 * 1024:
            raise ValueError("corpus record exceeds read limit")
        raw = json.loads(line)
        if raw["id"] != identity or not isinstance(raw["text"], str):
            raise ValueError("corpus offset and passage identity disagree")
        return {
            "passage": {"source_id": self.source_id_prefix + identity, "passage_id": identity,
                        "title": raw.get("title", ""), "text": raw["text"]},
            "origin": {"file": str(path), "offset": offset, "section": raw.get("section", ""),
                       "raw_line_sha256": hashlib.sha256(line).hexdigest(),
                       "text_sha256": hashlib.sha256(raw["text"].encode()).hexdigest()},
        }

    def retrieve(self, group: PassageGroup, output_dir: Path) -> dict[str, Any]:
        request = {"group": group.model_dump(mode="json"), "protocol": self.protocol}
        request_hash = digest(request)
        cache_path = output_dir / "cache" / "retrieval" / f"{request_hash}.json"
        if cache_path.exists():
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            if saved.get("request_sha256") != request_hash or saved.get("payload_sha256") != digest(saved["payload"]):
                raise ValueError("retrieval cache checksum mismatch")
            return saved["payload"]
        rows: dict[str, dict] = {}
        article_stats = []
        with closing(self._connect()) as connection:
            for seed in group.passages:
                if not re.fullmatch(r"[0-9]+#[0-9]+", seed.passage_id):
                    raise ValueError("same-article retrieval requires article#passage numeric IDs")
                if seed.source_id != self.source_id_prefix + seed.passage_id:
                    raise ValueError("seed source_id does not match the configured corpus prefix")
                position = connection.execute("SELECT id, f, off FROM pos WHERE id = ?", (seed.passage_id,)).fetchone()
                if position is None:
                    raise ValueError(f"seed passage not found in corpus: {seed.passage_id}")
                row = self._read(*position)
                if row["passage"]["text"] != seed.text or (seed.title and row["passage"]["title"] != seed.title):
                    raise ValueError(f"seed passage differs from raw corpus: {seed.passage_id}")
                rows[seed.passage_id] = row
            for article in sorted({p.passage_id.split("#")[0] for p in group.passages}):
                bounds = (article + "#", article + "$")
                count = connection.execute("SELECT count(*) FROM pos WHERE id >= ? AND id < ?", bounds).fetchone()[0]
                positions = connection.execute(
                    "SELECT id, f, off FROM pos WHERE id >= ? AND id < ? ORDER BY id LIMIT ?",
                    (*bounds, self.max_candidates)).fetchall()
                article_stats.append({"article_id": article, "available": count, "fetched": len(positions),
                                      "truncated": count > len(positions)})
                for position in positions:
                    if position[0] not in rows:
                        rows[position[0]] = self._read(*position)
        payload = {"candidates": list(rows.values()), "articles": article_stats}
        write_json(cache_path, {"request_sha256": request_hash, "payload_sha256": digest(payload), "payload": payload})
        return payload

    def prepare(self, group: PassageGroup, output_dir: Path) -> tuple[PassageGroup | None, dict[str, Any]]:
        if self.offsets_db:
            raw = self.retrieve(group, output_dir)
        else:
            raw = {"candidates": [{"passage": p.model_dump(mode="json"), "origin": {"input_only": True}}
                                  for p in group.passages], "articles": []}
        seeds = {(p.source_id, p.passage_id) for p in group.passages}
        seen: set[tuple[str, str]] = set()
        audit, usable = [], []
        for row in raw["candidates"]:
            payload = row["passage"]
            identity = (payload["source_id"], payload["passage_id"])
            if identity in seen:
                raise ValueError("duplicate retrieval candidate identity")
            seen.add(identity)
            # Validate the interface even for overlong sources; reject these without truncating them.
            checked = Passage.model_validate({**payload, "text": payload["text"][:12000]})
            issues = ["too_long"] if len(payload["text"]) > 12000 else quality_issues(checked)
            usage = len(set(m.group(0).lower() for m in USAGE_TERMS.finditer(checked.text)))
            boundary = len(set(m.group(0).lower() for m in BOUNDARY_TERMS.finditer(checked.text)))
            entry = {"source_id": checked.source_id, "passage_id": checked.passage_id,
                     "origin": row["origin"], "is_seed": identity in seeds, "issues": issues,
                     "usage_cues": usage, "boundary_cues": boundary, "selected": False}
            audit.append(entry)
            if not issues:
                usable.append((checked, entry))
        if not seeds.issubset(seen):
            raise ValueError("retrieval cache is missing seed passages")
        # Preserve healthy seeds, then add complementary procedure/boundary evidence.
        # Cue counts rank same-article candidates only, and make no support assertion.
        usable.sort(key=lambda pair: (not pair[1]["is_seed"],
                                      -(pair[1]["usage_cues"] + 2 * pair[1]["boundary_cues"]),
                                      pair[0].passage_id))
        selected, texts, total = [], set(), 0
        for passage, entry in usable:
            if passage.text in texts:
                entry["selection_note"] = "duplicate_text"
                continue
            if len(selected) >= self.max_passages or total + len(passage.text) > 30000:
                entry["selection_note"] = "evidence_budget"
                continue
            selected.append(passage)
            texts.add(passage.text)
            total += len(passage.text)
            entry["selected"] = True
        prepared = PassageGroup.model_validate({**group.model_dump(mode="json"),
                                               "passages": [p.model_dump(mode="json") for p in selected]}) if selected else None
        summary = {"backend": self.protocol["backend"], "candidates": len(audit),
                   "filtered_seeds": sum(e["is_seed"] and bool(e["issues"]) for e in audit),
                   "filtered_candidates": sum(bool(e["issues"]) for e in audit),
                   "selected_passages": len(selected),
                   "supplementary_passages": sum(e["selected"] and not e["is_seed"] for e in audit)}
        write_json(output_dir / "source_preparation" / f"{digest(group.group_id)}.json", {
            "group_id": group.group_id, "protocol": self.protocol, "summary": summary,
            "input_sha256": digest(group.model_dump(mode="json")), "candidates": audit,
            "articles": raw["articles"], "prepared_group": prepared.model_dump(mode="json") if prepared else None,
        })
        return prepared, summary
