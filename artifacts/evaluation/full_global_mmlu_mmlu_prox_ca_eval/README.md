# Full CA Evaluation Artifacts

This directory stores the current full CA evaluation outputs for Global-MMLU and MMLU-ProX.

## Scope

- Global-MMLU: `en`, `bn`, `hi`, `ne`, `sw`, `te`, 4,341 rows each.
- MMLU-ProX: `en`, `bn`, `hi`, `ne`, `sw`, `te`, 5,222 rows each.
- Total evaluated rows: 57,378.
- Runtime errors: 0 across both CA runs.

## Result Summary

See:

- `final_ca_vs_tcrag_report.md`
- `final_ca_vs_tcrag_summary.json`

Comparable low-resource results:

| scope | CA acc | tCRAG acc | delta |
|---|---:|---:|---:|
| Global-MMLU low-resource 5 languages | 62.55% | 58.99% | +3.57pp |
| MMLU-ProX low-resource 5 languages | 45.96% | 43.16% | +2.80pp |

## Prediction Files

The raw prediction JSONL outputs are gzip-compressed and split into GitHub-safe parts under `predictions/`.

Restore them with:

```bash
cat artifacts/evaluation/full_global_mmlu_mmlu_prox_ca_eval/predictions/ca_eval_global_mmlu_mmlu_prox_en_predictions.jsonl.gz.part-* > ca_eval_global_mmlu_mmlu_prox_en_predictions.jsonl.gz
gunzip -c ca_eval_global_mmlu_mmlu_prox_en_predictions.jsonl.gz > ca_eval_global_mmlu_mmlu_prox_en_predictions.jsonl

cat artifacts/evaluation/full_global_mmlu_mmlu_prox_ca_eval/predictions/ca_eval_mmlu_prox_low_resource_predictions.jsonl.gz.part-* > ca_eval_mmlu_prox_low_resource_predictions.jsonl.gz
gunzip -c ca_eval_mmlu_prox_low_resource_predictions.jsonl.gz > ca_eval_mmlu_prox_low_resource_predictions.jsonl
```

## Run Directories

- `ca_eval_full_bank_qwen3_8b_all_test/`
  - Global-MMLU six languages plus MMLU-ProX English.
- `ca_eval_full_bank_qwen3_8b_mmlu_prox_low_resource_all_test/`
  - MMLU-ProX low-resource five languages.

