# Full multilingual RAG results with Qwen3-8B

Run date: 2026-08-11

This report records the full low-resource multilingual MCQ evaluation used for the current CA discussion.

## Scope

- Model: local Qwen3-8B via vLLM
- Embedding model: Qwen3-Embedding-4B
- Languages: Bengali (`bn`), Hindi (`hi`), Nepali (`ne`), Swahili (`sw`), Telugu (`te`)
- Total examples: 47,815
- Datasets: Global-MMLU plus the non-Global-MMLU portion of MMLU-ProX
- Raw predictions, caches, Wikipag indexes, and local model files are not committed.

## Overall accuracy

| method | correct | accuracy | gain vs zero-shot |
|---|---:|---:|---:|
| zero-shot | 17,909 / 47,815 | 37.45% | +0.00 |
| D-RAG adaptive | 20,592 / 47,815 | 43.07% | +5.61 |
| CORAL raw-Wikipag no-fallback | 23,366 / 47,815 | 48.87% | +11.41 |
| tRAG / tCRAG | 24,071 / 47,815 | 50.34% | +12.89 |

Current ranking:

```text
tRAG / tCRAG
> CORAL raw-Wikipag no-fallback
> D-RAG adaptive
> zero-shot
```

## Accuracy by language

| method | bn | hi | ne | sw | te |
|---|---:|---:|---:|---:|---:|
| zero-shot | 41.98% | 43.66% | 41.28% | 20.30% | 40.05% |
| D-RAG adaptive | 44.83% | 46.78% | 44.98% | 35.96% | 42.78% |
| CORAL raw-Wikipag no-fallback | 51.16% | 52.68% | 50.90% | 39.18% | 50.41% |
| tRAG / tCRAG | 52.53% | 53.60% | 52.81% | 40.88% | 51.90% |

## Method notes

### zero-shot

The solver sees only the localized question, localized choices, and subject label.

### D-RAG adaptive

This is the adaptive D-RAG-style run that retrieves multilingual Wikipag evidence and uses a score gate.

### CORAL raw-Wikipag no-fallback

This run is intentionally restricted to raw Wikipag passage retrieval:

- no offline usage-card bank;
- no concept-card retrieval;
- no D-RAG fallback;
- raw Wikipag passages are wrapped as CORAL evidence candidates;
- an adaptive direct gate chooses between the direct answer and the CORAL evidence answer.

This setting is the cleanest current CORAL result for measuring whether CA-style evidence orchestration helps without hidden card-bank leakage.

### tRAG / tCRAG

The query is translated to English and retrieves English Wikipedia evidence before answering.

## Takeaway

CORAL raw-Wikipag no-fallback is clearly stronger than zero-shot and D-RAG adaptive, but it is still behind tRAG by about 1.47 points. The next CA work should focus on converting retrieved Wikipag passages into compact, question-useful cards before answer time, while keeping the retrieval path auditable.

