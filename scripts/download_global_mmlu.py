from __future__ import annotations

import argparse
from pathlib import Path

from src.data.global_mmlu import (
    DEFAULT_LANGUAGES,
    audit_and_write,
    copy_language_parquets,
    ensure_global_mmlu_repo,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download, normalize, audit, and split Global-MMLU.")
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--output-dir", default="data/raw/global_mmlu")
    parser.add_argument("--repo-dir", default="data/raw/global_mmlu_repo")
    parser.add_argument("--processed-dir", default="data/processed/global_mmlu")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--split-seed", type=int, default=20260726)
    args = parser.parse_args(argv)

    repo_dir = ensure_global_mmlu_repo(args.repo_dir, args.languages)
    copy_language_parquets(repo_dir, args.output_dir, args.languages)
    audit = audit_and_write(
        raw_dir=Path(args.output_dir),
        processed_dir=Path(args.processed_dir),
        manifests_dir=Path(args.manifests_dir),
        reports_dir=Path(args.reports_dir),
        languages=args.languages,
        split_seed=args.split_seed,
    )
    print(f"retained_ids={audit.retained_id_count}")
    print(f"split_counts={audit.split_counts}")
    print(f"reports=data_audit.md, DATA_AND_SPLIT_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

