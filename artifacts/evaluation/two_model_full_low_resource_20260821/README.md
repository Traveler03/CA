# Two-Model Full Low-Resource Evaluation

This directory stores the full low-resource evaluation outputs for:

- Models: `Ministral-3-8B-Instruct-2512` and `Llama-3.1-8B-Instruct`.
- Methods: zero-shot, tCRAG, CORAL-Wikipag, and CA.
- Datasets: Global-MMLU and MMLU-ProX.
- Languages: `bn`, `hi`, `ne`, `sw`, and `te`.
- Target rows: 47,815.
- Coverage: 100% for all model/method rows.
- Missing rows: 0 for all model/method rows.

## Main Results

| model | method | Global-MMLU | MMLU-ProX | Overall |
|---|---:|---:|---:|---:|
| Ministral-3-8B-Instruct-2512 | zero_shot | 54.44% | 29.83% | 41.00% |
| Ministral-3-8B-Instruct-2512 | tCRAG | 63.57% | 41.93% | 51.75% |
| Ministral-3-8B-Instruct-2512 | CORAL-Wikipag | 61.98% | 38.83% | 49.34% |
| Ministral-3-8B-Instruct-2512 | CA | 72.13% | 58.46% | 64.67% |
| Llama-3.1-8B-Instruct | zero_shot | 31.83% | 19.70% | 25.21% |
| Llama-3.1-8B-Instruct | tCRAG | 45.78% | 28.78% | 36.49% |
| Llama-3.1-8B-Instruct | CORAL-Wikipag | 46.03% | 28.74% | 36.59% |
| Llama-3.1-8B-Instruct | CA | 41.99% | 27.77% | 34.23% |

## Files

- `two_model_summary/report.md`: final human-readable table with dataset/language breakdowns.
- `two_model_summary/metrics.json`: final machine-readable metrics.
- `ministral3_8b_instruct_2512_bf16/summary_full_low_resource/`: Ministral-only summary.
- `llama3_1_8b_instruct/summary_full_low_resource/`: Llama-only summary.
- `run_control/gpu0_full_queue.nohup.log`: GPU0 queue completion log.
- `run_control/run_gpu0_full_queue.sh`: queue script used for the run.
- `predictions/*.jsonl.gz.part-*`: gzip-compressed prediction JSONL files split into GitHub-safe parts.
- `predictions.sha256`: sha256 checksums for the original uncompressed JSONL files.
- `prediction_parts.sha256`: sha256 checksums for the split compressed parts.

## Prediction Shards

Baseline shards contain one prediction row per method for `zero_shot`, `trag`, and `coral_wikipag`.
CA shards contain `ca_concept_bank_embedding_query_rewrite_rerank_check`.

| file prefix | rows before compression |
|---|---:|
| `ministral3_8b_instruct_2512_baselines_shard0` | 71,724 |
| `ministral3_8b_instruct_2512_baselines_shard1` | 71,721 |
| `ministral3_8b_instruct_2512_ca_shard0` | 23,908 |
| `ministral3_8b_instruct_2512_ca_shard1` | 23,907 |
| `llama3_1_8b_instruct_baselines_shard0` | 71,724 |
| `llama3_1_8b_instruct_baselines_shard1` | 71,721 |
| `llama3_1_8b_instruct_ca_shard0` | 23,908 |
| `llama3_1_8b_instruct_ca_shard1` | 23,907 |

Restore any prediction JSONL with:

```bash
cat artifacts/evaluation/two_model_full_low_resource_20260821/predictions/<prefix>.jsonl.gz.part-* \
  | gzip -dc > <prefix>.jsonl
```

For example:

```bash
cat artifacts/evaluation/two_model_full_low_resource_20260821/predictions/llama3_1_8b_instruct_ca_shard1.jsonl.gz.part-* \
  | gzip -dc > llama3_1_8b_instruct_ca_shard1.jsonl
```
