# Full Low-Resource Evaluation

- target rows: 47815
- scope: Global-MMLU and MMLU-ProX, languages bn/hi/ne/sw/te

## Main

| method | Global-MMLU | MMLU-ProX | Overall | present acc | coverage | missing | valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | 31.83% | 19.70% | 25.21% | 25.21% | 100.00% | 0 | 98.72% |
| tCRAG | 45.78% | 28.78% | 36.49% | 36.49% | 100.00% | 0 | 99.90% |
| CORAL-Wikipag | 46.03% | 28.74% | 36.59% | 36.59% | 100.00% | 0 | 100.00% |
| CA | 41.99% | 27.77% | 34.23% | 34.23% | 100.00% | 0 | 100.00% |

## Dataset By Language

### global_mmlu

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 31.33% | 35.06% | 33.45% | 31.67% | 27.62% |
| tCRAG | 46.97% | 48.26% | 45.45% | 42.80% | 45.40% |
| CORAL-Wikipag | 47.57% | 48.93% | 45.04% | 43.56% | 45.04% |
| CA | 43.12% | 46.86% | 42.18% | 38.49% | 39.32% |

### mmlu_prox

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 19.19% | 21.66% | 22.65% | 19.02% | 15.99% |
| tCRAG | 28.63% | 29.20% | 28.86% | 27.63% | 29.57% |
| CORAL-Wikipag | 28.82% | 29.26% | 29.11% | 27.61% | 28.92% |
| CA | 27.54% | 29.87% | 29.07% | 25.37% | 27.00% |

## Coverage

| method | matched rows | duplicate valid rows | error rows | files |
|---|---:|---:|---:|---|
| zero_shot | 47815 | 0 | 0 | artifacts/model_eval_20260821/llama3_1_8b_instruct/baselines_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/llama3_1_8b_instruct/baselines_shard1/predictions.jsonl |
| tCRAG | 47815 | 0 | 0 | artifacts/model_eval_20260821/llama3_1_8b_instruct/baselines_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/llama3_1_8b_instruct/baselines_shard1/predictions.jsonl |
| CORAL-Wikipag | 47815 | 0 | 0 | artifacts/model_eval_20260821/llama3_1_8b_instruct/baselines_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/llama3_1_8b_instruct/baselines_shard1/predictions.jsonl |
| CA | 47815 | 0 | 0 | artifacts/model_eval_20260821/llama3_1_8b_instruct/ca_shard0/predictions.jsonl<br>artifacts/model_eval_20260821/llama3_1_8b_instruct/ca_shard1/predictions.jsonl |
