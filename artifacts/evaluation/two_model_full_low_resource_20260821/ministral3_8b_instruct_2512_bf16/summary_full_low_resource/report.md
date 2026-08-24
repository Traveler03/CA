# Full Low-Resource Evaluation

- target rows: 47815
- scope: Global-MMLU and MMLU-ProX, languages bn/hi/ne/sw/te

## Main

| method | Global-MMLU | MMLU-ProX | Overall | present acc | coverage | missing | valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | 54.44% | 29.83% | 41.00% | 41.00% | 100.00% | 0 | 99.80% |
| tCRAG | 63.57% | 41.93% | 51.75% | 51.75% | 100.00% | 0 | 99.63% |
| CORAL-Wikipag | 61.98% | 38.83% | 49.34% | 49.34% | 100.00% | 0 | 99.99% |
| CA | 72.13% | 58.46% | 64.67% | 64.67% | 100.00% | 0 | 99.91% |

## Dataset By Language

### global_mmlu

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 57.59% | 60.65% | 56.32% | 42.71% | 54.92% |
| tCRAG | 64.32% | 67.10% | 65.58% | 55.47% | 65.35% |
| CORAL-Wikipag | 63.86% | 65.93% | 62.96% | 53.90% | 63.26% |
| CA | 76.00% | 77.77% | 75.60% | 56.67% | 74.61% |

### mmlu_prox

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 30.01% | 30.62% | 31.69% | 25.97% | 30.85% |
| tCRAG | 42.26% | 42.44% | 42.36% | 39.91% | 42.70% |
| CORAL-Wikipag | 39.41% | 39.79% | 39.08% | 36.50% | 39.37% |
| CA | 59.27% | 59.50% | 60.32% | 52.22% | 61.01% |

## Coverage

| method | matched rows | duplicate valid rows | error rows | files |
|---|---:|---:|---:|---|
| zero_shot | 47815 | 0 | 0 | artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/baselines_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/baselines_shard1/predictions.jsonl |
| tCRAG | 47815 | 0 | 0 | artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/baselines_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/baselines_shard1/predictions.jsonl |
| CORAL-Wikipag | 47815 | 0 | 0 | artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/baselines_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/baselines_shard1/predictions.jsonl |
| CA | 47815 | 0 | 0 | artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/ca_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/ca_shard1/predictions.jsonl |
