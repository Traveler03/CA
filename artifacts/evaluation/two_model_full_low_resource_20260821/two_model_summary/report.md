# Two-Model Full Low-Resource Evaluation

- target rows: 47815
- scope: Global-MMLU and MMLU-ProX, languages bn/hi/ne/sw/te

## Main Results

| model | method | Global-MMLU | MMLU-ProX | Overall | present acc | coverage | missing |
|---|---|---:|---:|---:|---:|---:|---:|
| Ministral-3-8B-Instruct-2512 | zero_shot | 54.44% | 29.83% | 41.00% | 41.00% | 100.00% | 0 |
| Ministral-3-8B-Instruct-2512 | tCRAG | 63.57% | 41.93% | 51.75% | 51.75% | 100.00% | 0 |
| Ministral-3-8B-Instruct-2512 | CORAL-Wikipag | 61.98% | 38.83% | 49.34% | 49.34% | 100.00% | 0 |
| Ministral-3-8B-Instruct-2512 | CA | 72.13% | 58.46% | 64.67% | 64.67% | 100.00% | 0 |
| Llama-3.1-8B-Instruct | zero_shot | 31.83% | 19.70% | 25.21% | 25.21% | 100.00% | 0 |
| Llama-3.1-8B-Instruct | tCRAG | 45.78% | 28.78% | 36.49% | 36.49% | 100.00% | 0 |
| Llama-3.1-8B-Instruct | CORAL-Wikipag | 46.03% | 28.74% | 36.59% | 36.59% | 100.00% | 0 |
| Llama-3.1-8B-Instruct | CA | 41.99% | 27.77% | 34.23% | 34.23% | 100.00% | 0 |

## By Dataset And Language

### Ministral-3-8B-Instruct-2512

#### global_mmlu

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 57.59% | 60.65% | 56.32% | 42.71% | 54.92% |
| tCRAG | 64.32% | 67.10% | 65.58% | 55.47% | 65.35% |
| CORAL-Wikipag | 63.86% | 65.93% | 62.96% | 53.90% | 63.26% |
| CA | 76.00% | 77.77% | 75.60% | 56.67% | 74.61% |

#### mmlu_prox

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 30.01% | 30.62% | 31.69% | 25.97% | 30.85% |
| tCRAG | 42.26% | 42.44% | 42.36% | 39.91% | 42.70% |
| CORAL-Wikipag | 39.41% | 39.79% | 39.08% | 36.50% | 39.37% |
| CA | 59.27% | 59.50% | 60.32% | 52.22% | 61.01% |

### Llama-3.1-8B-Instruct

#### global_mmlu

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 31.33% | 35.06% | 33.45% | 31.67% | 27.62% |
| tCRAG | 46.97% | 48.26% | 45.45% | 42.80% | 45.40% |
| CORAL-Wikipag | 47.57% | 48.93% | 45.04% | 43.56% | 45.04% |
| CA | 43.12% | 46.86% | 42.18% | 38.49% | 39.32% |

#### mmlu_prox

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero_shot | 19.19% | 21.66% | 22.65% | 19.02% | 15.99% |
| tCRAG | 28.63% | 29.20% | 28.86% | 27.63% | 29.57% |
| CORAL-Wikipag | 28.82% | 29.26% | 29.11% | 27.61% | 28.92% |
| CA | 27.54% | 29.87% | 29.07% | 25.37% | 27.00% |
