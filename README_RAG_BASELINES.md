# Qwen3 multilingual RAG baselines

This repository includes reproducible implementations for three multilingual RAG baselines:

- `TRAG`
- `DKM-RAG`
- `QTT-RAG`

The baselines use Qwen3-Embedding-4B FAISS indexes for retrieval and a Qwen3-8B-Instruct-compatible local model for translation, refinement, quality scoring, and answer generation.

## Methods

`TRAG`

1. Translate question and options to English.
2. Retrieve top-k passages from the English Wikipedia index only.
3. Answer using the original question/options, English translation, and English evidence.

`DKM-RAG`

1. Retrieve top-k passages from the multilingual Wikipedia indexes using the original question/options.
2. Translate retrieved passages into the query language.
3. Refine translated passages for relevance with Qwen3-8B.
4. Answer using original question/options, translated passages, and refined passages.
5. If refined text conflicts with translated text, the answer prompt instructs the model to trust the translated passage.

`QTT-RAG`

1. Retrieve top-k passages from the multilingual Wikipedia indexes using the original question/options.
2. Translate non-query-language passages into the query language.
3. Score each translated passage on semantic equivalence, grammatical accuracy, and naturalness/fluency.
4. Answer using tagged passages, preferring high semantic-equivalence evidence.

For DKM-RAG and QTT-RAG, refined text and quality tags are included inside the same `max_evidence_tokens=1200` answer-time evidence budget.

## Run

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_rag_baselines_qwen3.py \
  --config configs/rag_baselines_qwen3_8b.yaml \
  --preset smoke
```

Pilot:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_rag_baselines_qwen3.py \
  --config configs/rag_baselines_qwen3_8b.yaml \
  --preset pilot
```

Full:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_rag_baselines_qwen3.py \
  --config configs/rag_baselines_qwen3_8b.yaml \
  --preset full
```

If data and model paths differ, override them on the command line:

```bash
python scripts/run_rag_baselines_qwen3.py \
  --preset smoke \
  --global-mmlu-dir /path/to/data/processed/global_mmlu \
  --mmlu-prox-dir /tmp/ca_low_resource_eval/mmlu_prox_processed \
  --model-dir /tmp/qwen_models/Qwen3-8B \
  --embedding-model-dir /path/to/Qwen3-Embedding-4B \
  --multilingual-root /tmp/ca_multilingual_wikipedia_qwen3_4b/full_20231101
```

## Output

Outputs are written under:

```text
runs/rag_baselines_qwen3_8b/<preset>/
  config.yaml
  retrieval/
  translations/
  dkm_refined/
  qtt_tags/
  predictions/
  token_logs/
  metrics/
  cache/
  manifest.json
```

The generated run directories are intentionally ignored by git. Do not commit generated data, retrieval cache, LLM cache, logs, indexes, or predictions.

Required summary tables:

```text
metrics/accuracy_by_language.csv
metrics/cost_by_method.csv
```

`cost_by_method.csv` includes:

```text
method, avg_llm_calls, avg_prompt_tokens, avg_completion_tokens, avg_total_tokens, avg_evidence_tokens, accuracy
```

## Smoke result

A 100-example smoke run was validated on 2026-08-07. The lightweight committed summary is:

```text
reports/rag_baselines_qwen3_8b_smoke.md
```

The raw `runs/rag_baselines_qwen3_8b/smoke_v2/` outputs remain local and are ignored by git.
