# Wikipag Concept-Usage Bank Project Rules

This repository contains only the corpus-driven Wikipag construction path.

Current scope:

- Build concepts and usage cards from English Wikipag/Wikipedia passages.
- Treat local Wikipag/FAISS assets and raw source corpora as read-only.
- Do not write generated outputs into a production database.
- Do not commit generated data, model caches, logs, vector indexes, or downloaded corpus assets.
- Keep every stage resumable by writing stage outputs under a run directory.
- Cache model responses and retrieval results during construction.
- Validate JSON/JSONL interfaces before consuming downstream outputs.

Out of scope for this repository:

- Dataset/evaluation-specific construction.
- Multilingual data generation.
- Synthetic assessment generation.
- Evaluation scripts and outputs.
