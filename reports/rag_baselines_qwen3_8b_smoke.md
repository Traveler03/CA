# Qwen3-8B multilingual RAG baseline smoke

Run date: 2026-08-07

This report records the lightweight `smoke_v2` run for the three multilingual RAG baselines implemented in `scripts/run_rag_baselines_qwen3.py`.

## Scope

- Methods: `TRAG`, `DKM-RAG`, `QTT-RAG`
- Languages: `bn`, `sw`, `te`, `ne`, `hi`
- Sample size: 20 examples per language, 100 examples total
- Retriever: Qwen3-Embedding-4B FAISS indexes
- Answer / translation / refinement / quality scoring model: local Qwen3-8B-compatible model at `/tmp/qwen_models/Qwen3-8B`
- Temperature: 0
- Top-k: 5
- Evidence budget: 1200 tokens
- Per-doc evidence cap: 240 tokens
- Output directory, not committed: `runs/rag_baselines_qwen3_8b/smoke_v2/`

Raw predictions, retrieval cache, LLM cache, logs, indexes, and generated run files are intentionally not committed.

## Accuracy by language

| method | bn | hi | ne | sw | te | overall |
|---|---:|---:|---:|---:|---:|---:|
| TRAG | 30.0% | 45.0% | 15.0% | 40.0% | 35.0% | 33.0% |
| DKM-RAG | 30.0% | 35.0% | 15.0% | 30.0% | 30.0% | 28.0% |
| QTT-RAG | 25.0% | 40.0% | 20.0% | 25.0% | 35.0% | 29.0% |

## Cost by method

| method | avg_llm_calls | avg_prompt_tokens | avg_completion_tokens | avg_total_tokens | avg_evidence_tokens | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| TRAG | 2.00 | 2468.03 | 217.42 | 2685.45 | 1032.43 | 33.0% |
| DKM-RAG | 3.00 | 9771.32 | 898.87 | 10670.19 | 1163.47 | 28.0% |
| QTT-RAG | 2.69 | 7559.89 | 353.81 | 7913.70 | 1177.38 | 29.0% |

## Validation checks

- Planned examples: 100
- Estimated stage-level LLM calls: 800
- Prediction rows:
  - `trag`: 100
  - `dkm_rag`: 100
  - `qtt_rag`: 100
- Parse rate:
  - `trag`: 97%
  - `dkm_rag`: 100%
  - `qtt_rag`: 100%
- Evidence token budget:
  - `trag`: max 1197, over-budget rows 0
  - `dkm_rag`: max 1198, over-budget rows 0
  - `qtt_rag`: max 1198, over-budget rows 0

## Notes

This smoke run validates that the three baselines execute end-to-end with the required output structure and budget logging. It is not a statistically stable benchmark result; use `pilot` or `full` presets for larger runs.

`smoke_v2` also validates that QTT-RAG skips no-op translation calls when all retrieved passages are already in the query language; those rows still include an explicit zero-token translation stage in `token_logs`.
