# Qwen3-8B localized zero-shot full evaluation

- total: `47815`
- correct: `17909`
- accuracy: `0.374548`
- parse_rate: `1.000000`

## By dataset

| dataset | n | correct | accuracy | parse_rate |
|---|---:|---:|---:|---:|
| global_mmlu | 21705 | 10254 | 0.4724 | 1.0000 |
| mmlu_prox | 26110 | 7655 | 0.2932 | 1.0000 |

## By language

| language | n | correct | accuracy | parse_rate |
|---|---:|---:|---:|---:|
| bn | 9563 | 4015 | 0.4198 | 1.0000 |
| hi | 9563 | 4175 | 0.4366 | 1.0000 |
| ne | 9563 | 3948 | 0.4128 | 1.0000 |
| sw | 9563 | 1941 | 0.2030 | 1.0000 |
| te | 9563 | 3830 | 0.4005 | 1.0000 |

## By dataset-language

| dataset | language | n | correct | accuracy | parse_rate |
|---|---|---:|---:|---:|---:|
| global_mmlu | bn | 4341 | 2250 | 0.5183 | 1.0000 |
| global_mmlu | hi | 4341 | 2357 | 0.5430 | 1.0000 |
| global_mmlu | ne | 4341 | 2182 | 0.5026 | 1.0000 |
| global_mmlu | sw | 4341 | 1364 | 0.3142 | 1.0000 |
| global_mmlu | te | 4341 | 2101 | 0.4840 | 1.0000 |
| mmlu_prox | bn | 5222 | 1765 | 0.3380 | 1.0000 |
| mmlu_prox | hi | 5222 | 1818 | 0.3481 | 1.0000 |
| mmlu_prox | ne | 5222 | 1766 | 0.3382 | 1.0000 |
| mmlu_prox | sw | 5222 | 577 | 0.1105 | 1.0000 |
| mmlu_prox | te | 5222 | 1729 | 0.3311 | 1.0000 |

## Comparison to previous full English-instruction zero-shot

- old accuracy: `0.392806`
- delta localized - old: `-0.018258`
- paired overlap: `47815`
- help/harm/same/net: `3895/4768/39152/-873`
