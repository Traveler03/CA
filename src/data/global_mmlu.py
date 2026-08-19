from __future__ import annotations

import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.jsonl import write_jsonl


DATASET_REPO = "https://huggingface.co/datasets/CohereLabs/Global-MMLU"
DEFAULT_LANGUAGES = ["en", "bn", "sw", "te", "ne", "hi"]
VALID_ANSWERS = {"A", "B", "C", "D"}


@dataclass
class DataAudit:
    languages: list[str]
    raw_counts: dict[str, int]
    duplicate_ids: dict[str, list[str]]
    invalid_answer_ids: dict[str, list[str]]
    missing_option_ids: dict[str, list[str]]
    sample_id_sets_equal: bool
    common_id_count: int
    removed_id_count: int
    answer_mismatch_ids: list[str]
    subject_mismatch_ids: list[str]
    retained_id_count: int
    split_seed: int
    split_counts: dict[str, int]
    overlap_counts: dict[str, int]


def ensure_global_mmlu_repo(repo_dir: str | Path, languages: list[str]) -> Path:
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                DATASET_REPO,
                str(repo_dir),
            ],
            check=True,
            env={**dict(), **_git_env()},
        )
    include = ",".join(f"{lang}/*" for lang in languages)
    subprocess.run(
        ["git", "-C", str(repo_dir), "lfs", "pull", "--include", include],
        check=True,
        env=_git_env(),
    )
    return repo_dir


def copy_language_parquets(repo_dir: str | Path, output_dir: str | Path, languages: list[str]) -> None:
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    for lang in languages:
        src_dir = repo_dir / lang
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing language directory: {src_dir}")
        dst_dir = output_dir / lang
        dst_dir.mkdir(parents=True, exist_ok=True)
        for split in ["dev", "test"]:
            matches = sorted(src_dir.glob(f"{split}-*.parquet"))
            if not matches:
                raise FileNotFoundError(f"Missing {lang}/{split} parquet in {repo_dir}")
            for src in matches:
                dst = dst_dir / src.name
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)


def load_language(raw_dir: str | Path, lang: str) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    rows: list[dict[str, Any]] = []
    for parquet in sorted((raw_dir / lang).glob("*.parquet")):
        split_name = parquet.name.split("-", 1)[0]
        frame = pd.read_parquet(parquet)
        for row in frame.to_dict(orient="records"):
            normalized = normalize_row(row, lang)
            normalized["source_dataset_split"] = split_name
            rows.append(normalized)
    return rows


def normalize_row(row: dict[str, Any], lang: str) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip().upper()
    return {
        "sample_id": str(row.get("sample_id", "")).strip(),
        "language": lang,
        "subject": str(row.get("subject", "")).strip(),
        "subject_category": str(row.get("subject_category", "")).strip(),
        "question": _clean_text(row.get("question", "")),
        "options": {
            "A": _clean_text(row.get("option_a", "")),
            "B": _clean_text(row.get("option_b", "")),
            "C": _clean_text(row.get("option_c", "")),
            "D": _clean_text(row.get("option_d", "")),
        },
        "answer": answer,
        "cultural_sensitivity_label": str(row.get("cultural_sensitivity_label", "")).strip(),
        "required_knowledge": row.get("required_knowledge", "[]"),
        "time_sensitive": row.get("time_sensitive", "[]"),
        "reference": row.get("reference", "[]"),
        "culture": row.get("culture", "[]"),
        "region": row.get("region", "[]"),
        "country": row.get("country", "[]"),
        "is_annotated": bool(row.get("is_annotated", False)),
    }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def audit_and_write(
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
    manifests_dir: str | Path,
    reports_dir: str | Path,
    languages: list[str],
    split_seed: int = 20260726,
) -> DataAudit:
    processed_dir = Path(processed_dir)
    manifests_dir = Path(manifests_dir)
    reports_dir = Path(reports_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    by_lang = {lang: load_language(raw_dir, lang) for lang in languages}
    id_to_row: dict[str, dict[str, dict[str, Any]]] = {}
    raw_counts = {lang: len(rows) for lang, rows in by_lang.items()}
    duplicate_ids: dict[str, list[str]] = {}
    invalid_answer_ids: dict[str, list[str]] = {}
    missing_option_ids: dict[str, list[str]] = {}

    invalid_global: set[str] = set()
    id_sets: dict[str, set[str]] = {}
    for lang, rows in by_lang.items():
        counts = Counter(row["sample_id"] for row in rows)
        duplicates = sorted([sid for sid, count in counts.items() if count > 1])
        duplicate_ids[lang] = duplicates
        invalid_global.update(duplicates)
        id_sets[lang] = set(counts)
        invalid_answer_ids[lang] = sorted([row["sample_id"] for row in rows if row["answer"] not in VALID_ANSWERS])
        missing_option_ids[lang] = sorted(
            [
                row["sample_id"]
                for row in rows
                if not row["question"] or any(not row["options"].get(label, "") for label in ["A", "B", "C", "D"])
            ]
        )
        invalid_global.update(invalid_answer_ids[lang])
        invalid_global.update(missing_option_ids[lang])
        for row in rows:
            id_to_row.setdefault(row["sample_id"], {})[lang] = row

    common_ids = set.intersection(*id_sets.values()) if id_sets else set()
    missing_parallel = set.union(*id_sets.values()) - common_ids if id_sets else set()
    invalid_global.update(missing_parallel)

    answer_mismatch_ids: list[str] = []
    subject_mismatch_ids: list[str] = []
    for sid in sorted(common_ids):
        rows = id_to_row[sid]
        answers = {rows[lang]["answer"] for lang in languages}
        subjects = {rows[lang]["subject"] for lang in languages}
        if len(answers) > 1:
            answer_mismatch_ids.append(sid)
        if len(subjects) > 1:
            subject_mismatch_ids.append(sid)
    invalid_global.update(answer_mismatch_ids)
    invalid_global.update(subject_mismatch_ids)

    retained_ids = sorted(common_ids - invalid_global)
    split = stratified_split(retained_ids, id_to_row, split_seed=split_seed)

    assert not set(split["source"]) & set(split["dev"])
    assert not set(split["source"]) & set(split["test"])
    assert not set(split["dev"]) & set(split["test"])

    for lang in languages:
        retained_rows = [id_to_row[sid][lang] for sid in retained_ids]
        write_jsonl(processed_dir / f"{lang}.jsonl", retained_rows)
        for split_name, ids in split.items():
            write_jsonl(processed_dir / f"{lang}.{split_name}.jsonl", [id_to_row[sid][lang] for sid in ids])

    split_manifest = {
        "split_seed": split_seed,
        "proportions": {"source": 0.60, "dev": 0.10, "test": 0.30},
        "languages": languages,
        "source_ids": split["source"],
        "dev_ids": split["dev"],
        "test_ids": split["test"],
    }
    (manifests_dir / "splits.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = DataAudit(
        languages=languages,
        raw_counts=raw_counts,
        duplicate_ids={lang: vals[:200] for lang, vals in duplicate_ids.items()},
        invalid_answer_ids={lang: vals[:200] for lang, vals in invalid_answer_ids.items()},
        missing_option_ids={lang: vals[:200] for lang, vals in missing_option_ids.items()},
        sample_id_sets_equal=len({frozenset(ids) for ids in id_sets.values()}) == 1,
        common_id_count=len(common_ids),
        removed_id_count=len(common_ids) - len(retained_ids),
        answer_mismatch_ids=answer_mismatch_ids[:500],
        subject_mismatch_ids=subject_mismatch_ids[:500],
        retained_id_count=len(retained_ids),
        split_seed=split_seed,
        split_counts={name: len(ids) for name, ids in split.items()},
        overlap_counts={
            "source_dev": len(set(split["source"]) & set(split["dev"])),
            "source_test": len(set(split["source"]) & set(split["test"])),
            "dev_test": len(set(split["dev"]) & set(split["test"])),
        },
    )
    (reports_dir / "data_audit.json").write_text(json.dumps(asdict(audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render_data_report(audit)
    (reports_dir / "data_audit.md").write_text(report, encoding="utf-8")
    (reports_dir / "DATA_AND_SPLIT_REPORT.md").write_text(report, encoding="utf-8")
    return audit


def stratified_split(
    retained_ids: list[str],
    id_to_row: dict[str, dict[str, dict[str, Any]]],
    *,
    split_seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(split_seed)
    by_subject: dict[str, list[str]] = defaultdict(list)
    for sid in retained_ids:
        by_subject[id_to_row[sid]["en"]["subject"]].append(sid)

    split = {"source": [], "dev": [], "test": []}
    for subject in sorted(by_subject):
        ids = sorted(by_subject[subject])
        rng.shuffle(ids)
        n = len(ids)
        n_source = int(n * 0.60)
        n_dev = int(n * 0.10)
        if n >= 3 and n_dev == 0:
            n_dev = 1
        if n >= 2 and n_source == 0:
            n_source = 1
        if n_source + n_dev > n:
            n_dev = max(0, n - n_source)
        split["source"].extend(ids[:n_source])
        split["dev"].extend(ids[n_source : n_source + n_dev])
        split["test"].extend(ids[n_source + n_dev :])

    for name in split:
        split[name] = sorted(split[name])
    return split


def render_data_report(audit: DataAudit) -> str:
    lines = [
        "# Data and split report",
        "",
        "## Dataset",
        "",
        "- Source: `CohereLabs/Global-MMLU`",
        f"- Languages: `{', '.join(audit.languages)}`",
        f"- Split seed: `{audit.split_seed}`",
        "",
        "## Raw counts",
        "",
        "```json",
        json.dumps(audit.raw_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Parallel audit",
        "",
        f"- Sample ID sets equal across languages: `{audit.sample_id_sets_equal}`",
        f"- Common sample IDs: `{audit.common_id_count}`",
        f"- Retained sample IDs after synchronized filtering: `{audit.retained_id_count}`",
        f"- Removed common IDs: `{audit.removed_id_count}`",
        f"- Answer mismatch IDs: `{len(audit.answer_mismatch_ids)}` shown up to report cap",
        f"- Subject mismatch IDs: `{len(audit.subject_mismatch_ids)}` shown up to report cap",
        "",
        "## Split counts",
        "",
        "```json",
        json.dumps(audit.split_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Leakage assertions",
        "",
        "```json",
        json.dumps(audit.overlap_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "All splits are by `sample_id`; every language inherits the same split manifest. Source resources must use only English `source_ids`; low-resource evaluation uses held-out `test_ids`.",
        "",
    ]
    return "\n".join(lines)


def _git_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["GIT_SSL_NO_VERIFY"] = "true"
    return env

