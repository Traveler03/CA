from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.io import read_jsonl, write_jsonl


DATASET_REPO = "li-lab/MMLU-ProX"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_REPO}"
RESOLVE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
OVERLAP_SRC_PREFIXES = ("ori_mmlu-", "cot_lib-", "global_mmlu-", "mmlu-")
OPTION_FIELDS = [f"option_{idx}" for idx in range(10)]
ANSWER_LABELS = tuple("ABCDEFGHIJ")


SRC_TO_GLOBAL_SUBJECT = {
    "scibench-atkins": "college_chemistry",
    "scibench-calculus": "college_mathematics",
    "scibench-chemmc": "college_chemistry",
    "scibench-class": "college_physics",
    "scibench-diff": "college_mathematics",
    "scibench-fund": "college_physics",
    "scibench-matter": "college_chemistry",
    "scibench-quan": "college_chemistry",
    "scibench-stat": "high_school_statistics",
    "scibench-thermo": "college_physics",
    "stemez-Biology": "college_biology",
    "stemez-Business": "management",
    "stemez-Chemistry": "college_chemistry",
    "stemez-ComputerScience": "college_computer_science",
    "stemez-Economics": "high_school_microeconomics",
    "stemez-ElectricCircuits": "electrical_engineering",
    "stemez-ElectricalMachines": "electrical_engineering",
    "stemez-Electromagnetics": "electrical_engineering",
    "stemez-ElectronicCommunications": "electrical_engineering",
    "stemez-FluidMechanics": "college_physics",
    "stemez-Genetics": "medical_genetics",
    "stemez-HeatTransfer": "college_physics",
    "stemez-MachineDesign": "electrical_engineering",
    "stemez-Mechanics": "college_physics",
    "stemez-Optics": "college_physics",
    "stemez-OrganicChemistry": "college_chemistry",
    "stemez-PhysicalChemistry": "college_chemistry",
    "stemez-Physics": "college_physics",
    "stemez-Psychology": "high_school_psychology",
    "stemez-Thermodynamics": "college_physics",
    "stemez-TransportPhenomena": "college_physics",
    "theoremQA-EECS": "college_computer_science",
    "theoremQA-Finance": "professional_accounting",
    "theoremQA-Math": "college_mathematics",
    "theoremQA-Physics": "college_physics",
}


CATEGORY_TO_GLOBAL_SUBJECT = {
    "biology": "college_biology",
    "business": "management",
    "chemistry": "college_chemistry",
    "computer science": "college_computer_science",
    "economics": "high_school_microeconomics",
    "engineering": "electrical_engineering",
    "health": "professional_medicine",
    "history": "high_school_world_history",
    "law": "professional_law",
    "math": "college_mathematics",
    "other": "miscellaneous",
    "philosophy": "philosophy",
    "physics": "college_physics",
    "psychology": "high_school_psychology",
}


def _ssl_context() -> ssl.SSLContext:
    return ssl._create_unverified_context()


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, context=_ssl_context(), timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, path: Path, *, refresh: bool = False) -> bool:
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, context=_ssl_context(), timeout=300) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)
    return True


def parse_langs(value: str, available: list[str]) -> list[str]:
    value = value.strip()
    if not value or value.lower() == "all":
        return list(available)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown languages: {missing}; available={available}")
    return requested


def list_parquet_files(api_payload: dict[str, Any]) -> list[str]:
    return sorted(
        str(item.get("rfilename") or "")
        for item in api_payload.get("siblings", [])
        if str(item.get("rfilename") or "").endswith(".parquet")
    )


def available_languages(parquet_files: list[str]) -> list[str]:
    return sorted({path.split("/", 1)[0] for path in parquet_files if "/" in path})


def download_raw(
    *,
    raw_dir: Path,
    languages: list[str],
    refresh: bool,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_payload = fetch_json(API_URL)
    (raw_dir / "hf_api.json").write_text(json.dumps(api_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parquet_files = list_parquet_files(api_payload)
    selected_files = [
        path
        for path in parquet_files
        if path.split("/", 1)[0] in set(languages)
    ]
    downloaded = 0
    skipped = 0
    for rel_path in selected_files:
        changed = download_file(f"{RESOLVE_URL}/{rel_path}", raw_dir / rel_path, refresh=refresh)
        downloaded += int(changed)
        skipped += int(not changed)
    return {
        "available_languages": available_languages(parquet_files),
        "selected_languages": languages,
        "selected_file_count": len(selected_files),
        "downloaded_file_count": downloaded,
        "reused_file_count": skipped,
    }


def load_global_subject_catalog(path: Path) -> tuple[set[str], dict[str, str]]:
    rows = read_jsonl(path)
    subjects: set[str] = set()
    category_by_subject: dict[str, str] = {}
    for row in rows:
        subject = str(row.get("subject") or "").strip()
        if not subject:
            continue
        subjects.add(subject)
        category_by_subject.setdefault(subject, str(row.get("subject_category") or "").strip())
    return subjects, category_by_subject


def is_overlap_src(src: str) -> bool:
    return str(src or "").startswith(OVERLAP_SRC_PREFIXES)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def answer_label_from_row(row: dict[str, Any], options: dict[str, str]) -> str:
    raw_answer = clean_text(row.get("answer")).upper()
    if raw_answer in options:
        return raw_answer
    try:
        idx = int(row.get("answer_index"))
    except Exception:
        return ""
    if 0 <= idx < len(ANSWER_LABELS):
        return ANSWER_LABELS[idx]
    return ""


def options_from_row(row: dict[str, Any]) -> dict[str, str]:
    options: dict[str, str] = {}
    for idx, field in enumerate(OPTION_FIELDS):
        text = clean_text(row.get(field))
        if text:
            options[ANSWER_LABELS[idx]] = text
    return options


def map_subject(src: str, category: str, global_subjects: set[str]) -> tuple[str, str]:
    if src in SRC_TO_GLOBAL_SUBJECT:
        subject = SRC_TO_GLOBAL_SUBJECT[src]
        method = "src_exact"
    elif category in CATEGORY_TO_GLOBAL_SUBJECT:
        subject = CATEGORY_TO_GLOBAL_SUBJECT[category]
        method = "category_fallback"
    else:
        subject = "miscellaneous"
        method = "default_miscellaneous"
    if subject not in global_subjects:
        raise ValueError(f"mapped subject {subject!r} is not in Global-MMLU subject catalog")
    return subject, method


def normalize_split(
    *,
    parquet_path: Path,
    language: str,
    split: str,
    global_subjects: set[str],
    global_category_by_subject: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")
    output: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "raw_rows": len(records),
        "kept_rows": 0,
        "excluded_global_mmlu_overlap_rows": 0,
        "missing_question_rows": 0,
        "missing_options_rows": 0,
        "missing_answer_rows": 0,
        "source_prefix_counts": Counter(),
        "kept_source_counts": Counter(),
        "mapped_subject_counts": Counter(),
        "mapping_method_counts": Counter(),
    }
    for row in records:
        src = clean_text(row.get("src"))
        category = clean_text(row.get("category")).lower()
        prefix = src.split("-", 1)[0] if src else ""
        counts["source_prefix_counts"][prefix] += 1
        if is_overlap_src(src):
            counts["excluded_global_mmlu_overlap_rows"] += 1
            continue
        question = clean_text(row.get("question"))
        if not question:
            counts["missing_question_rows"] += 1
            continue
        options = options_from_row(row)
        if not options:
            counts["missing_options_rows"] += 1
            continue
        answer_label = answer_label_from_row(row, options)
        answer_text = options.get(answer_label, "")
        if not answer_label or not answer_text:
            counts["missing_answer_rows"] += 1
            continue
        subject, mapping_method = map_subject(src, category, global_subjects)
        sample_id = f"mmlu_prox/{language}/{split}/{row.get('question_id')}"
        output.append(
            {
                "sample_id": sample_id,
                "source_dataset": "mmlu_prox",
                "source_dataset_split": split,
                "language": language,
                "question": question,
                "options": options,
                "answer": answer_label,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "subject": subject,
                "subject_category": global_category_by_subject.get(subject, ""),
                "source_subject": src,
                "source_category": category,
                "subject_mapping_method": mapping_method,
                "question_id": int(row.get("question_id")) if pd.notna(row.get("question_id")) else None,
                "question_id_src": int(row.get("question_id_src")) if pd.notna(row.get("question_id_src")) else None,
                "answer_index": int(row.get("answer_index")) if pd.notna(row.get("answer_index")) else None,
                "excluded_global_mmlu_overlap": False,
            }
        )
        counts["kept_rows"] += 1
        counts["kept_source_counts"][src] += 1
        counts["mapped_subject_counts"][subject] += 1
        counts["mapping_method_counts"][mapping_method] += 1
    return output, counts


def counter_to_dict(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {key: counter_to_dict(item) for key, item in value.items()}
    return value


def process_languages(
    *,
    raw_dir: Path,
    processed_dir: Path,
    languages: list[str],
    global_mmlu_en: Path,
) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    global_subjects, global_category_by_subject = load_global_subject_catalog(global_mmlu_en)
    manifest: dict[str, Any] = {
        "dataset": "mmlu_prox",
        "global_subject_catalog": str(global_mmlu_en),
        "global_subject_count": len(global_subjects),
        "overlap_src_prefixes_excluded": list(OVERLAP_SRC_PREFIXES),
        "languages": {},
    }
    all_rows: list[dict[str, Any]] = []
    for language in languages:
        lang_rows: list[dict[str, Any]] = []
        lang_counts: dict[str, Any] = {
            "raw_rows": 0,
            "kept_rows": 0,
            "excluded_global_mmlu_overlap_rows": 0,
            "splits": {},
            "mapped_subject_counts": Counter(),
            "mapping_method_counts": Counter(),
            "kept_source_counts": Counter(),
        }
        for split in ("validation", "test"):
            parquet_path = raw_dir / language / f"{split}-00000-of-00001.parquet"
            if not parquet_path.exists():
                continue
            split_rows, split_counts = normalize_split(
                parquet_path=parquet_path,
                language=language,
                split=split,
                global_subjects=global_subjects,
                global_category_by_subject=global_category_by_subject,
            )
            write_jsonl(processed_dir / f"{language}.{split}.jsonl", split_rows)
            lang_rows.extend(split_rows)
            lang_counts["splits"][split] = counter_to_dict(split_counts)
            for key in ("raw_rows", "kept_rows", "excluded_global_mmlu_overlap_rows"):
                lang_counts[key] += int(split_counts.get(key) or 0)
            for key in ("mapped_subject_counts", "mapping_method_counts", "kept_source_counts"):
                lang_counts[key].update(split_counts.get(key) or {})
        lang_rows.sort(key=lambda item: (str(item.get("subject") or ""), str(item.get("sample_id") or "")))
        write_jsonl(processed_dir / f"{language}.jsonl", lang_rows)
        all_rows.extend(lang_rows)
        manifest["languages"][language] = counter_to_dict(lang_counts)
    all_rows.sort(key=lambda item: (str(item.get("language") or ""), str(item.get("subject") or ""), str(item.get("sample_id") or "")))
    write_jsonl(processed_dir / "all.jsonl", all_rows)
    manifest["total_raw_rows"] = sum(int(item.get("raw_rows") or 0) for item in manifest["languages"].values())
    manifest["total_kept_rows"] = sum(int(item.get("kept_rows") or 0) for item in manifest["languages"].values())
    manifest["total_excluded_global_mmlu_overlap_rows"] = sum(
        int(item.get("excluded_global_mmlu_overlap_rows") or 0)
        for item in manifest["languages"].values()
    )
    return counter_to_dict(manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    api_payload = fetch_json(API_URL)
    available = available_languages(list_parquet_files(api_payload))
    download_languages = parse_langs(args.download_languages, available)
    process_languages_list = parse_langs(args.process_languages, available)
    download_summary = download_raw(raw_dir=raw_dir, languages=download_languages, refresh=args.refresh)
    missing_for_process = [
        language
        for language in process_languages_list
        for split in ("validation", "test")
        if not (raw_dir / language / f"{split}-00000-of-00001.parquet").exists()
    ]
    if missing_for_process:
        raise FileNotFoundError(f"missing raw parquet for process languages: {sorted(set(missing_for_process))}")
    process_summary = process_languages(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        languages=process_languages_list,
        global_mmlu_en=Path(args.global_mmlu_en),
    )
    manifest = {
        "ok": True,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "download": download_summary,
        "process": process_summary,
        "latency_s": time.perf_counter() - started,
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and normalize MMLU-ProX, excluding original MMLU/Global-MMLU overlap.")
    parser.add_argument("--raw-dir", default="data/raw/mmlu_prox")
    parser.add_argument("--processed-dir", default="data/processed/mmlu_prox")
    parser.add_argument("--global-mmlu-en", default="data/processed/global_mmlu/en.jsonl")
    parser.add_argument("--download-languages", default="en", help='Comma-separated language configs, or "all".')
    parser.add_argument("--process-languages", default="en", help='Comma-separated language configs, or "all".')
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
