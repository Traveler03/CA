# Full CA Evaluation Summary

- CA total unique evaluated rows: 57378
- CA total correct: 31725
- CA total accuracy: 55.29%
- Runtime errors: 0 in both CA runs

## Comparable To tCRAG

| scope | n | CA correct | CA acc | tCRAG correct | tCRAG acc | delta correct | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_mmlu low-resource 5 languages | 21705 | 13577 | 62.55% | 12803 | 58.99% | +774 | +3.57pp |
| mmlu_prox low-resource 5 languages | 26110 | 11999 | 45.96% | 11268 | 43.16% | +731 | +2.80pp |

## By Dataset And Language

| dataset | lang | n | CA correct | CA acc | valid | tCRAG acc | delta |
|---|---|---:|---:|---:|---:|---:|---:|
| global_mmlu | bn | 4341 | 2926 | 67.40% | 99.98% | 62.45% | +4.95pp |
| global_mmlu | en | 4341 | 3568 | 82.19% | 100.00% | n/a | n/a |
| global_mmlu | hi | 4341 | 3008 | 69.29% | 100.00% | 64.46% | +4.84pp |
| global_mmlu | ne | 4341 | 2832 | 65.24% | 100.00% | 62.64% | +2.60pp |
| global_mmlu | sw | 4341 | 2012 | 46.35% | 100.00% | 44.69% | +1.66pp |
| global_mmlu | te | 4341 | 2799 | 64.48% | 100.00% | 60.70% | +3.78pp |
| mmlu_prox | bn | 5222 | 2439 | 46.71% | 100.00% | 44.27% | +2.43pp |
| mmlu_prox | en | 5222 | 2581 | 49.43% | 100.00% | n/a | n/a |
| mmlu_prox | hi | 5222 | 2457 | 47.05% | 99.98% | 44.58% | +2.47pp |
| mmlu_prox | ne | 5222 | 2463 | 47.17% | 100.00% | 44.64% | +2.53pp |
| mmlu_prox | sw | 5222 | 2135 | 40.88% | 99.98% | 37.71% | +3.18pp |
| mmlu_prox | te | 5222 | 2505 | 47.97% | 100.00% | 44.58% | +3.39pp |
