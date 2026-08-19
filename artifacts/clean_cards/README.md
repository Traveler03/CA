# Clean Concept Cards

This directory stores the current clean concept-card bank used by the full CA evaluation.

## Files

- `concept_card_qwen3_1024/concept_card_bank.jsonl.gz`
  - Compressed runtime card bank.
  - Uncompressed source: `bank.jsonl`.
  - Card count: 60,653.
- `concept_card_qwen3_1024/runtime_bank_manifest.json`
  - Runtime bank metadata.

The original `build_index.npy` embedding matrix is not committed because it is a generated binary index larger than GitHub's normal file limit. Rebuild it from the JSONL bank with the Qwen3-Embedding-4B pipeline when needed.

## Restore

```bash
gunzip -c artifacts/clean_cards/concept_card_qwen3_1024/concept_card_bank.jsonl.gz > bank.jsonl
```

