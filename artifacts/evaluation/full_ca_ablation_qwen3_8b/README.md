# Full CA Ablation Results

Scope: full low-resource evaluation over Global-MMLU and MMLU-ProX for `bn`, `hi`, `ne`, `sw`, and `te`.

- Target rows: 47,815
- Global-MMLU rows: 21,705
- MMLU-ProX rows: 26,110
- Card bank: combined oracle plus clean Wikipag concept-card bank
- Local model: Qwen3-8B
- Embedding model: Qwen3-Embedding-4B

Main files:

- `ablation_report.md`: human-readable full ablation table.
- `ablation_summary.json`: machine-readable metrics by dataset and language.
- `<variant>/summary.json`: runner summary for each ablation variant.
- `<variant>/plan.jsonl`: runner configuration for each ablation variant.
- `predictions.sha256`: checksums for compressed prediction shards.
- `predictions/*_predictions.jsonl.gz.part-*`: gzip-compressed full prediction JSONL files split into GitHub-safe parts.

To reconstruct a prediction file:

```bash
cat predictions/ca_top10_predictions.jsonl.gz.part-* | gzip -dc > ca_top10_predictions.jsonl
```

Variants:

- `ca_no_rerank_top5`: embedding top-5 without rerank/check. This is diagnostic because the bank contains oracle-derived cards.
- `ca_random_global_top5`: random global cards.
- `ca_random_same_subject_top5`: random cards from the same subject.
- `ca_top1`: reranked top-1 card.
- `ca_top3`: reranked top-3 cards.
- `ca_top10`: reranked top-10 cards.
