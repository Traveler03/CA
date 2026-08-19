from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.io import read_jsonl, write_jsonl
from src.smoke_test.stage_utils import stable_id


DEFAULT_BANK_NAME = "concept_card_qwen3_1024"
DEFAULT_EMBEDDING_MODEL = Path("data/external/models/Qwen3-Embedding-4B")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _norm_key(value: Any) -> str:
    text = str(value or "").lower().strip()
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def _merge_unique_list(*values: Any) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item in (None, ""):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _merge_text_list(*values: Any, limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_list(value):
            key = _norm_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def dedupe_cards_by_subject_concept(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for card in cards:
        concept = card.get("concept") or card.get("canonical_name")
        key = (_norm_key(card.get("subject")), _norm_key(concept))
        if not key[0] or not key[1]:
            deduped.append(dict(card))
            continue
        existing = by_key.get(key)
        if existing is None:
            copied = dict(card)
            copied["_merged_card_count"] = 1
            by_key[key] = copied
            deduped.append(copied)
            continue

        existing["_merged_card_count"] = int(existing.get("_merged_card_count") or 1) + 1
        existing["source_sample_ids"] = _merge_unique_list(existing.get("source_sample_ids"), card.get("source_sample_ids"))
        existing["source_ids"] = _merge_unique_list(existing.get("source_ids"), card.get("source_ids"))
        existing["evidence_claims"] = _merge_unique_list(existing.get("evidence_claims"), card.get("evidence_claims"))
        existing["trigger"] = _merge_text_list(existing.get("trigger"), card.get("trigger"), limit=8)
        existing["decision"] = _merge_text_list(existing.get("decision"), card.get("decision"), limit=10)
        existing["pitfall"] = _merge_text_list(existing.get("pitfall"), card.get("pitfall"), limit=8)
        existing["trigger_conditions"] = _merge_text_list(
            existing.get("trigger_conditions"),
            card.get("trigger_conditions"),
            limit=8,
        )
        existing["decision_procedure"] = _merge_text_list(
            existing.get("decision_procedure"),
            card.get("decision_procedure"),
            limit=10,
        )
        existing["failure_boundaries"] = _merge_text_list(
            existing.get("failure_boundaries"),
            card.get("failure_boundaries"),
            limit=8,
        )
        existing["verification_rules"] = _merge_text_list(
            existing.get("verification_rules"),
            card.get("verification_rules"),
            limit=8,
        )
        existing["benchmark_content_accessed"] = bool(existing.get("benchmark_content_accessed") or card.get("benchmark_content_accessed"))
        existing["uses_gold_answer"] = bool(existing.get("uses_gold_answer") or card.get("uses_gold_answer"))
        existing["can_be_used_for_clean_eval"] = bool(
            existing.get("can_be_used_for_clean_eval", True)
            and card.get("can_be_used_for_clean_eval", True)
            and not existing.get("benchmark_content_accessed")
            and not existing.get("uses_gold_answer")
        )
    return deduped


def _source_label(card: dict[str, Any], source_manifest: dict[str, Any]) -> str:
    if card.get("source_dataset") == "global_mmlu":
        return "Global-MMLU"
    if card.get("source_dataset") == "mmlu_prox":
        return "MMLU-ProX"
    source_snapshot = str(source_manifest.get("source_snapshot") or "")
    construction_mode = str(source_manifest.get("construction_mode") or "")
    if "wikipedia" in source_snapshot.lower() or "wikipag" in construction_mode.lower():
        return "Wikipag"
    if card.get("benchmark_content_accessed") or card.get("uses_gold_answer"):
        return "Benchmark"
    return str(source_manifest.get("source_dataset") or "Wikipag")


def _retrieval_unit(card: dict[str, Any], source_manifest: dict[str, Any]) -> str:
    label = _source_label(card, source_manifest)
    if card.get("uses_gold_answer") or card.get("benchmark_content_accessed"):
        return f"one card per {label} English oracle question-derived concept"
    return f"one card per {label} English corpus-derived usage concept"


def _provenance(card: dict[str, Any], source_manifest: dict[str, Any]) -> str:
    label = _source_label(card, source_manifest)
    if card.get("uses_gold_answer") or card.get("benchmark_content_accessed"):
        return f"{label} English oracle card; diagnostic-only"
    return f"{label} English clean card; corpus-derived"


def normalize_card_body(card: dict[str, Any]) -> dict[str, Any]:
    description = _first_text(
        card.get("description"),
        card.get("concept_boundary"),
        card.get("usage_signature"),
        card.get("usage_pattern"),
    )
    trigger = _as_list(card.get("trigger")) or _as_list(card.get("trigger_conditions"))
    decision = _as_list(card.get("decision")) or _as_list(card.get("decision_procedure")) or _as_list(
        card.get("verification_rules")
    )
    pitfall = _as_list(card.get("pitfall")) or _as_list(card.get("failure_boundaries"))

    if not trigger:
        usage_pattern = _first_text(card.get("usage_pattern"), card.get("usage_signature"))
        if usage_pattern:
            trigger = [usage_pattern]
    if not decision:
        decision = _as_list(card.get("verification_rules"))
    if not pitfall:
        pitfall = ["Do not use this card when the question context does not match the stated concept."]

    return {
        "description": description,
        "trigger": trigger[:8],
        "decision": decision[:10],
        "pitfall": pitfall[:8],
    }


def runtime_description(card: dict[str, Any], source_manifest: dict[str, Any]) -> str:
    body = normalize_card_body(card)
    lines = [
        "Concept-level CA-Mem card.",
        f"Retrieval unit: {_retrieval_unit(card, source_manifest)}.",
        f"Subject: {card.get('subject') or ''}.",
        f"Concept: {card.get('concept') or card.get('canonical_name') or ''}.",
        f"Concept ID: {card.get('concept_id') or ''}.",
        f"Description: {body['description']}",
        "Trigger:",
        *[f"- {item}" for item in body["trigger"]],
        "Decision:",
        *[f"- {item}" for item in body["decision"]],
        "Pitfall:",
        *[f"- {item}" for item in body["pitfall"]],
        f"Provenance: {_provenance(card, source_manifest)}.",
    ]
    return "\n".join(lines).strip()


def runtime_row(card: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, Any]:
    usage_id = str(card.get("usage_id") or stable_id("usage", card.get("concept_id"), card.get("concept")))
    concept_id = str(card.get("concept_id") or stable_id("concept", card.get("subject"), card.get("concept")))
    benchmark_content_accessed = bool(card.get("benchmark_content_accessed") or source_manifest.get("benchmark_content_accessed"))
    uses_gold_answer = bool(card.get("uses_gold_answer") or source_manifest.get("uses_gold_answer"))
    clean_eval = not benchmark_content_accessed and not uses_gold_answer
    return {
        "memory_id": stable_id("mem_concept", usage_id, concept_id),
        "usage_id": usage_id,
        "concept_id": concept_id,
        "subject": card.get("subject"),
        "concept": card.get("concept") or card.get("canonical_name"),
        "description": runtime_description(card, source_manifest),
        "source_sample_ids": card.get("source_sample_ids") or [],
        "bank_component": card.get("bank_component") or ("wikipag_clean" if clean_eval else "benchmark_oracle"),
        "benchmark_content_accessed": benchmark_content_accessed,
        "uses_gold_answer": uses_gold_answer,
        "can_be_used_for_clean_eval": bool(card.get("can_be_used_for_clean_eval", clean_eval)),
    }


def load_manifest(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "bank_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def embed_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    model_dir: Path,
    device: str,
    truncate_dim: int,
    batch_size: int,
) -> dict[str, Any] | None:
    if not rows:
        return None
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device=device)
    texts = [str(row.get("description") or "") for row in rows]
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        kwargs = {"normalize_embeddings": True, "convert_to_numpy": True}
        try:
            vectors = model.encode(batch, prompt_name="document", truncate_dim=truncate_dim, **kwargs)
        except TypeError:
            vectors = model.encode(batch, **kwargs)
            if vectors.shape[1] > truncate_dim:
                vectors = vectors[:, :truncate_dim]
                vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        chunks.append(np.asarray(vectors, dtype=np.float32))
    matrix = np.concatenate(chunks, axis=0)
    index_path = output_dir / "build_index.npy"
    np.save(index_path, matrix)
    return {"embedding_index": str(index_path), "embedding_shape": list(matrix.shape)}


def write_runtime_bank(
    *,
    input_dir: Path,
    output_dir: Path | None = None,
    usage_cards_path: Path | None = None,
    bank_name: str = DEFAULT_BANK_NAME,
    build_embedding_index: bool = True,
    embedding_model_dir: Path = DEFAULT_EMBEDDING_MODEL,
    embedding_device: str = "cuda",
    embedding_truncate_dim: int = 1024,
    embedding_batch_size: int = 64,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    source_manifest = load_manifest(input_dir)
    usage_cards_path = usage_cards_path or input_dir / "usage_cards.jsonl"
    if not usage_cards_path.exists():
        raise FileNotFoundError(f"missing usage cards: {usage_cards_path}")
    output_dir = output_dir or input_dir / "banks" / bank_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = [row for row in read_jsonl(usage_cards_path) if str(row.get("status") or "active") == "active"]
    raw_card_count = len(cards)
    cards = dedupe_cards_by_subject_concept(cards)
    rows = [runtime_row(card, source_manifest) for card in cards]
    write_jsonl(output_dir / "bank.jsonl", rows)

    meta: dict[str, Any] = {
        "bank_dir": str(output_dir),
        "bank_jsonl": str(output_dir / "bank.jsonl"),
        "card_count": len(rows),
        "dedupe_key": "subject+normalized_concept",
        "deduped_card_count": len(rows),
        "raw_active_card_count": raw_card_count,
        "removed_duplicate_subject_concept_count": raw_card_count - len(rows),
        "source_usage_cards": str(usage_cards_path),
        "source_bank_dir": str(input_dir),
        "bank_name": bank_name,
        "schema": "concept_card_runtime_v1",
        "build_embedding_index": bool(build_embedding_index),
        "embedding_index": None,
    }
    if build_embedding_index:
        embedding_meta = embed_rows(
            rows,
            output_dir=output_dir,
            model_dir=embedding_model_dir,
            device=embedding_device,
            truncate_dim=embedding_truncate_dim,
            batch_size=embedding_batch_size,
        )
        if embedding_meta:
            meta.update(embedding_meta)
    (output_dir / "runtime_bank_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export CA concept-card runtime bank from Usage Bank cards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--usage-cards-path", type=Path)
    parser.add_argument("--bank-name", default=DEFAULT_BANK_NAME)
    parser.add_argument("--build-embedding-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model-dir", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-truncate-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    meta = write_runtime_bank(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        usage_cards_path=args.usage_cards_path,
        bank_name=args.bank_name,
        build_embedding_index=bool(args.build_embedding_index),
        embedding_model_dir=args.embedding_model_dir,
        embedding_device=args.embedding_device,
        embedding_truncate_dim=int(args.embedding_truncate_dim),
        embedding_batch_size=int(args.embedding_batch_size),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
