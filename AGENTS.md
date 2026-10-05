# Wikipag Concept-Usage Bank Project Rules

This repository contains only the corpus-driven Wikipag construction path.

The only supported entry point is `scripts/build_wikipag_application_cards.py`.
All model stages use Qwen3.5-9B: application generation, solution, self-check,
card extraction, card review, and at most one repair followed by re-review.
Do not reintroduce the retired direct-extraction, dataset-driven, or smoke pipelines.

Current scope:

- Build concepts and usage cards from English Wikipag/Wikipedia passages.
- Synthesize English application questions, solutions, and source-constrained self-checks as intermediate card-construction records. Keep the existing five-field card body unchanged.
- Treat local Wikipag/FAISS assets and raw source corpora as read-only.
- Do not write generated outputs into a production database.
- Do not commit generated data, model caches, logs, vector indexes, or downloaded corpus assets.
- Keep the existing 56,579-card bank and its index frozen. Pipeline maintenance must not rebuild, rewrite, append to, or replace that bank.
- Keep existing experiment records associated with their actual bank snapshots and model settings. Record new pipeline runs separately from the frozen bank's construction history.
- Maintain the curated experiment summary in `docs/RESULTS.md` as documentation; raw evaluation outputs remain outside version control.
- Keep every stage resumable by writing stage outputs under a run directory.
- Cache model responses and retrieval results during construction.
- Validate JSON/JSONL interfaces before consuming downstream outputs.

Out of scope for this repository:

- Dataset/evaluation-specific construction.
- Multilingual data generation.
- Standalone assessment/benchmark generation; intermediate corpus-grounded application records are construction artifacts.
- Evaluation scripts and raw outputs.
