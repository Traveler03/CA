# Full tCRAG result — Qwen3-8B local

- Output dir: `runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu`
- Predictions: `runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu/predictions.jsonl`
- Compact predictions: `runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu/predictions.compact.jsonl`
- Summary: `runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu/summary.json`
- Method in code: `trag` (query translated to English, retrieve English Wiki, inject evidence, solve with local Qwen3-8B)
- Total rows: 47815
- Errors: 0

## Overall

| Scope | n | tCRAG correct | tCRAG acc | valid | valid rate | errors | zero-shot correct | zero-shot acc | net gain | delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 47815 | 24071 | 50.34% | 47806 | 99.98% | 0 | 17909 | 37.45% | +6162 | +12.89 |

## By dataset

| Dataset | n | tCRAG correct | tCRAG acc | valid | valid rate | errors | zero-shot correct | zero-shot acc | net gain | delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global_mmlu | 21705 | 12803 | 58.99% | 21701 | 99.98% | 0 | 10254 | 47.24% | +2549 | +11.74 |
| mmlu_prox | 26110 | 11268 | 43.16% | 26105 | 99.98% | 0 | 7655 | 29.32% | +3613 | +13.84 |

## By language

| Language | n | tCRAG correct | tCRAG acc | valid | valid rate | errors | zero-shot correct | zero-shot acc | net gain | delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bn | 9563 | 5023 | 52.53% | 9562 | 99.99% | 0 | 4015 | 41.98% | +1008 | +10.54 |
| hi | 9563 | 5126 | 53.60% | 9562 | 99.99% | 0 | 4175 | 43.66% | +951 | +9.94 |
| ne | 9563 | 5050 | 52.81% | 9562 | 99.99% | 0 | 3948 | 41.28% | +1102 | +11.52 |
| sw | 9563 | 3909 | 40.88% | 9558 | 99.95% | 0 | 1941 | 20.30% | +1968 | +20.58 |
| te | 9563 | 4963 | 51.90% | 9562 | 99.99% | 0 | 3830 | 40.05% | +1133 | +11.85 |

## By dataset × language

| Dataset × language | n | tCRAG correct | tCRAG acc | valid | valid rate | errors | zero-shot correct | zero-shot acc | net gain | delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global_mmlu / bn | 4341 | 2711 | 62.45% | 4340 | 99.98% | 0 | 2250 | 51.83% | +461 | +10.62 |
| global_mmlu / hi | 4341 | 2798 | 64.46% | 4341 | 100.00% | 0 | 2357 | 54.30% | +441 | +10.16 |
| global_mmlu / ne | 4341 | 2719 | 62.64% | 4341 | 100.00% | 0 | 2182 | 50.26% | +537 | +12.37 |
| global_mmlu / sw | 4341 | 1940 | 44.69% | 4339 | 99.95% | 0 | 1364 | 31.42% | +576 | +13.27 |
| global_mmlu / te | 4341 | 2635 | 60.70% | 4340 | 99.98% | 0 | 2101 | 48.40% | +534 | +12.30 |
| mmlu_prox / bn | 5222 | 2312 | 44.27% | 5222 | 100.00% | 0 | 1765 | 33.80% | +547 | +10.47 |
| mmlu_prox / hi | 5222 | 2328 | 44.58% | 5221 | 99.98% | 0 | 1818 | 34.81% | +510 | +9.77 |
| mmlu_prox / ne | 5222 | 2331 | 44.64% | 5221 | 99.98% | 0 | 1766 | 33.82% | +565 | +10.82 |
| mmlu_prox / sw | 5222 | 1969 | 37.71% | 5219 | 99.94% | 0 | 577 | 11.05% | +1392 | +26.66 |
| mmlu_prox / te | 5222 | 2328 | 44.58% | 5222 | 100.00% | 0 | 1729 | 33.11% | +599 | +11.47 |

## Latency

- Mean: 6.679s
- P50: 6.868s
- P95: 9.691s
- Max: 14.126s

## Notes

- Main merged result uses `shard0` + `shard1`; helper GPU1 output was only used as a safety fallback if a main key were missing.
- No source benchmark files or Wiki index were modified.
