from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ZERO = Path("runs/smoke_001/full_qwen3_8b_localized_zero_shot_2gpu/predictions.jsonl")
DEFAULT_PROCESSED = Path("data/processed/global_mmlu")
DEFAULT_OUTPUT = Path("artifacts/ca_mem/zero_wrong100_global_mmlu_oracle_card_ca_gpt54_v1")
DEFAULT_LANGUAGES = ["bn", "hi", "ne", "sw", "te"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_processed_by_sample(processed_dir: Path, language: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in read_jsonl(processed_dir / f"{language}.jsonl")}


def pred_key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}|{row.get('language')}|{row.get('sample_id')}"


def select_wrong_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    predictions = [
        row
        for row in read_jsonl(args.zero_shot_predictions)
        if row.get("dataset") == args.dataset
        and str(row.get("language")) in set(args.languages)
        and row.get("correct") is False
        and row.get("sample_id")
    ]
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_lang[str(row.get("language"))].append(row)
    for rows in by_lang.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    if args.per_language > 0:
        for language in args.languages:
            picked = 0
            for row in by_lang.get(language, []):
                sample_id = str(row.get("sample_id"))
                if sample_id in seen_samples:
                    continue
                selected.append(row)
                seen_samples.add(sample_id)
                picked += 1
                if picked >= args.per_language:
                    break
            if picked < args.per_language:
                raise RuntimeError(f"only selected {picked} wrong unique samples for language={language}")

    if len(selected) < args.n:
        remaining = [row for row in predictions if str(row.get("sample_id")) not in seen_samples]
        rng.shuffle(remaining)
        for row in remaining:
            selected.append(row)
            seen_samples.add(str(row.get("sample_id")))
            if len(selected) >= args.n:
                break
    return selected[: args.n]


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    selected_preds = select_wrong_rows(args)
    if len(selected_preds) != args.n:
        raise RuntimeError(f"selected {len(selected_preds)} rows, expected {args.n}")

    english_by_sample = load_processed_by_sample(args.processed_global_mmlu_dir, "en")
    localized_cache = {language: load_processed_by_sample(args.processed_global_mmlu_dir, language) for language in args.languages}

    eval_rows: list[dict[str, Any]] = []
    card_input_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for pred in selected_preds:
        language = str(pred.get("language"))
        sample_id = str(pred.get("sample_id"))
        localized = localized_cache.get(language, {}).get(sample_id)
        english = english_by_sample.get(sample_id)
        if not localized or not english:
            missing.append({"language": language, "sample_id": sample_id, "localized": bool(localized), "english": bool(english)})
            continue
        eval_row = dict(localized)
        eval_row["_dataset"] = args.dataset
        eval_row["_language"] = language
        eval_row["_zero_shot_prediction"] = pred.get("prediction")
        eval_row["_zero_shot_gold"] = pred.get("gold")
        eval_row["_zero_shot_correct"] = False
        eval_row["_zero_shot_eval_key"] = pred.get("eval_key")
        eval_rows.append(eval_row)

        card_row = dict(english)
        card_row["_target_language"] = language
        card_row["_target_sample_id"] = sample_id
        card_row["_zero_shot_prediction"] = pred.get("prediction")
        card_row["_zero_shot_eval_key"] = pred.get("eval_key")
        card_input_rows.append(card_row)

        ref = dict(pred)
        ref["eval_key_ca"] = f"{args.dataset}|{language}|{sample_id}"
        ref_rows.append(ref)
    if missing:
        raise RuntimeError(f"missing processed rows for selected predictions: {missing[:5]}")

    write_jsonl(out / "zero_wrong100_eval_input.jsonl", eval_rows)
    write_jsonl(out / "zero_wrong100_card_input_en.jsonl", card_input_rows)
    write_jsonl(out / "zero_wrong100_zero_shot_reference.jsonl", ref_rows)

    by_language = Counter(str(row.get("language")) for row in eval_rows)
    by_subject = Counter(str(row.get("subject")) for row in eval_rows)
    manifest = {
        "ok": True,
        "dataset": args.dataset,
        "n": len(eval_rows),
        "seed": args.seed,
        "languages": args.languages,
        "per_language_target": args.per_language,
        "zero_shot_predictions": str(args.zero_shot_predictions),
        "processed_global_mmlu_dir": str(args.processed_global_mmlu_dir),
        "eval_input_jsonl": str(out / "zero_wrong100_eval_input.jsonl"),
        "card_input_jsonl": str(out / "zero_wrong100_card_input_en.jsonl"),
        "zero_shot_reference_jsonl": str(out / "zero_wrong100_zero_shot_reference.jsonl"),
        "by_language": dict(sorted(by_language.items())),
        "by_subject_top20": by_subject.most_common(20),
        "note": "All selected rows are zero-shot incorrect; English card input uses the same sample_id and gold answer text.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "commands.md").write_text(build_commands(args, manifest), encoding="utf-8")
    return manifest


def build_commands(args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    out = args.output_dir
    card_dir = out / "cards_gpt54"
    bank_dir = card_dir / "banks" / "concept_card_qwen3_1024"
    ca_dir = out / "ca_eval_qwen3_8b"
    return f"""# Zero-shot wrong100 oracle-card CA experiment

## 1. Generate one strict card per English question with GPT-5.4

```bash
python scripts/run_global_mmlu_oracle_usage_bank.py \\
  --resume \\
  --input-jsonl {manifest['card_input_jsonl']} \\
  --output-dir {card_dir} \\
  --bank-version global-mmlu-zero-wrong100-en-oracle-gpt54-strict3-v1 \\
  --model gpt-5.4 \\
  --concurrency 24 \\
  --batch-size 100 \\
  --max-rows 0 \\
  --max-rows-per-subject 0 \\
  --cards-per-question 1 \\
  --max-completion-tokens 1600 \\
  --max-model-calls 100 \\
  --build-embedding-index
```

## 2. Run CA answer flow on the 100 localized zero-shot-wrong rows

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ca_concept_bank_smoke.py \\
  --resume \\
  --input-jsonl {manifest['eval_input_jsonl']} \\
  --output-dir {ca_dir} \\
  --bank-dir {bank_dir} \\
  --embedding-model-dir data/external/models/Qwen3-Embedding-4B \\
  --local-model-dir data/external/models/Qwen3-8B \\
  --local-max-model-len 16384 \\
  --local-gpu-memory-utilization 0.75 \\
  --local-batch-size 16 \\
  --max-rows 100 \\
  --candidate-top-k 20 \\
  --top-k 5 \\
  --max-model-calls 400
```

## 3. Compare against the selected zero-shot wrong baseline

```bash
python scripts/compare_zero_wrong_card_ca_eval.py \\
  --experiment-dir {out} \\
  --ca-predictions {ca_dir / 'predictions.jsonl'}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 100 Global-MMLU zero-shot-wrong rows for oracle-card CA evaluation.")
    parser.add_argument("--zero-shot-predictions", type=Path, default=DEFAULT_ZERO)
    parser.add_argument("--processed-global-mmlu-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="global_mmlu")
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--per-language", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_outputs(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
