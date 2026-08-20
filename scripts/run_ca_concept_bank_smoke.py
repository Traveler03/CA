from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_multilingual_rag_baselines import JsonCache, LocalVLLMModel, parse_answer_label_from_text
from src.baselines.rag_methods import option_labels, parse_json_object
from src.utils.jsonl import read_jsonl, write_jsonl


DEFAULT_BANK_DIR = Path(
    "/tmp/ca_deleted_old_methods_20260807_125106/"
    "artifacts/ca_mem/final_combined_oracle_plus_clean/banks/concept_card_qwen3_1024"
)
DEFAULT_INPUT = Path("runs/smoke_001/coral_wikipag_qwen3_8b_local_smoke_paired_v1/input.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test concept-card CA answering with query rewrite.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("data/external/models/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--local-model-dir", type=Path, default=Path("/tmp/qwen_models/Qwen3-8B"))
    parser.add_argument("--local-max-model-len", type=int, default=32768)
    parser.add_argument("--local-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--local-batch-timeout-ms", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--max-concurrent-model-requests", type=int, default=4)
    parser.add_argument("--max-model-calls", type=int, default=800)
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-strategy",
        choices=["embedding", "random_global", "random_same_subject"],
        default="embedding",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-card-chars", type=int, default=1200)
    parser.add_argument("--rerank-card-chars", type=int, default=700)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--query-max-tokens", type=int, default=256)
    parser.add_argument("--rerank-max-tokens", type=int, default=1536)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--quiet-items", action="store_true")
    parser.add_argument("--no-rerank-check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-subject-filter", action="store_true")
    return parser.parse_args()


def load_bank(bank_dir: Path, *, require_index: bool = True) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    bank_path = bank_dir / "bank.jsonl"
    index_path = bank_dir / "build_index.npy"
    if not bank_path.exists():
        raise FileNotFoundError(bank_path)
    if require_index and not index_path.exists():
        raise FileNotFoundError(index_path)
    cards = list(read_jsonl(bank_path))
    index = np.load(index_path, mmap_mode="r") if index_path.exists() else None
    if index is not None and index.shape[0] != len(cards):
        raise ValueError(f"index/card count mismatch: {index.shape[0]} != {len(cards)}")
    return cards, index


def format_options(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{label}. {text}" for label, text in options.items())
    if isinstance(options, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{labels[idx]}. {text}" for idx, text in enumerate(options))
    return str(options or "")


def row_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("_dataset") or row.get("dataset") or "global_mmlu")
    language = str(row.get("language") or row.get("_language") or "")
    sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or row.get("question"))
    return f"{dataset}|{language}|{sample_id}"


def concept_query_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You rewrite multilingual multiple-choice questions into compact English concept retrieval queries. "
                "Do not solve the question. Do not mention the gold answer. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language: {row.get('language') or row.get('_language') or ''}\n"
                f"Subject: {row.get('subject') or ''}\n"
                f"Question:\n{row.get('question') or ''}\n\n"
                f"Options:\n{format_options(row.get('options') or {})}\n\n"
                "Return JSON with keys:\n"
                "- concept_query: an English search query focused on the concepts/rules needed to answer.\n"
                "- target_concepts: 1 to 5 short English concept names.\n"
                'Example: {"concept_query":"abstract algebra inverse of product law abelian group criterion","target_concepts":["inverse of product","abelian group"]}'
            ),
        },
    ]


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def rerank_check_messages(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    query_payload: dict[str, Any],
    *,
    final_top_k: int,
    max_card_chars: int,
) -> list[dict[str, str]]:
    candidate_blocks = []
    for item in candidates:
        card = item["card"]
        text = truncate_text(str(card.get("description") or ""), max_card_chars)
        candidate_blocks.append(
            f"[Candidate {item['rank']} | retrieval_score={item['score']:.4f}]\n"
            f"Subject: {card.get('subject') or ''}\n"
            f"Concept: {card.get('concept') or ''}\n"
            f"Card:\n{text}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You rerank and check concept cards for a multilingual multiple-choice question. "
                "Do not solve the question. Do not infer the gold answer. "
                "Use only the candidate card text when writing checked_card. "
                "Remove irrelevant or misleading card content. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language: {row.get('language') or row.get('_language') or ''}\n"
                f"Subject: {row.get('subject') or ''}\n"
                f"Concept retrieval query: {query_payload.get('concept_query') or ''}\n\n"
                f"Question:\n{row.get('question') or ''}\n\n"
                f"Options:\n{format_options(row.get('options') or {})}\n\n"
                f"Candidate concept cards:\n\n{chr(10).join(candidate_blocks)}\n\n"
                f"Select up to {final_top_k} cards that are actually useful for solving this question. "
                "Prefer fewer cards if many are irrelevant. For each selected card, write a compact checked_card in English. "
                "The checked_card should keep only supported trigger/rule/pitfall information from that candidate card.\n\n"
                "Return JSON with this exact shape:\n"
                '{\n'
                '  "selected": [\n'
                '    {"candidate_rank": 1, "relevance": 5, "checked_card": "Concept: ...\\nUse when: ...\\nRule: ...\\nPitfall: ..."}\n'
                "  ]\n"
                "}\n"
                "candidate_rank must refer to one of the candidate numbers above. relevance is 1 to 5."
            ),
        },
    ]


def checked_cards_from_rerank(
    candidates: list[dict[str, Any]],
    rerank_payload: dict[str, Any],
    *,
    final_top_k: int,
) -> list[dict[str, Any]]:
    by_rank = {int(item["rank"]): item for item in candidates}
    selected_rows = rerank_payload.get("selected")
    if not isinstance(selected_rows, list):
        selected_rows = rerank_payload.get("selected_cards")
    if not isinstance(selected_rows, list):
        selected_rows = []
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in selected_rows:
        if not isinstance(row, dict):
            continue
        raw_rank = row.get("candidate_rank", row.get("rank"))
        try:
            candidate_rank = int(raw_rank)
        except Exception:
            continue
        if candidate_rank in seen or candidate_rank not in by_rank:
            continue
        seen.add(candidate_rank)
        item = dict(by_rank[candidate_rank])
        checked_card = str(row.get("checked_card") or "").strip()
        if checked_card:
            item["checked_card"] = checked_card
        item["rerank_relevance"] = row.get("relevance")
        selected.append(item)
        if len(selected) >= final_top_k:
            break
    if selected:
        return selected
    return candidates[:final_top_k]


def answer_messages(row: dict[str, Any], cards: list[dict[str, Any]], query_payload: dict[str, Any], *, max_card_chars: int) -> list[dict[str, str]]:
    labels = ", ".join(option_labels(row.get("options") or {}))
    card_blocks = []
    for idx, item in enumerate(cards, start=1):
        card = item["card"]
        text = str(item.get("checked_card") or card.get("description") or "")
        text = truncate_text(text, max_card_chars)
        rerank_part = ""
        if item.get("rerank_relevance") is not None:
            rerank_part = f" | rerank_relevance={item.get('rerank_relevance')}"
        card_blocks.append(
            f"[Card {idx} | retrieval_score={item['score']:.4f}{rerank_part}]\n"
            f"Subject: {card.get('subject') or ''}\n"
            f"Concept: {card.get('concept') or ''}\n"
            f"{text}"
        )
    cards_text = "\n\n".join(card_blocks) if card_blocks else "(No concept cards retrieved.)"
    return [
        {
            "role": "system",
            "content": (
                "You answer multilingual multiple-choice questions. Use the concept cards only when they are relevant. "
                "If a card conflicts with the question/options, trust the question/options. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language: {row.get('language') or row.get('_language') or ''}\n"
                f"Subject: {row.get('subject') or ''}\n"
                f"Concept retrieval query: {query_payload.get('concept_query') or ''}\n\n"
                f"Question:\n{row.get('question') or ''}\n\n"
                f"Options:\n{format_options(row.get('options') or {})}\n\n"
                f"Retrieved concept cards:\n{cards_text}\n\n"
                f'Return JSON only: {{"answer":"A"}}. The answer must be one of: {labels}.'
            ),
        },
    ]


class ConceptBankRetriever:
    def __init__(self, *, bank_dir: Path, model_dir: Path, device: str, load_embedding_model: bool = True) -> None:
        self.bank_dir = bank_dir
        self.cards, self.index = load_bank(bank_dir, require_index=load_embedding_model)
        self.dim = int(self.index.shape[1]) if self.index is not None else 0
        self.model = None
        if load_embedding_model:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(str(model_dir), device=device)
        self.by_subject: dict[str, np.ndarray] = defaultdict(list)  # type: ignore[assignment]
        subject_lists: dict[str, list[int]] = defaultdict(list)
        for idx, card in enumerate(self.cards):
            subject_lists[str(card.get("subject") or "")].append(idx)
        self.by_subject = {subject: np.asarray(indices, dtype=np.int64) for subject, indices in subject_lists.items()}

    def encode(self, text: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("embedding model is not loaded")
        kwargs = {"prompt_name": "query", "normalize_embeddings": True, "convert_to_numpy": True}
        try:
            vector = self.model.encode([text], truncate_dim=self.dim, **kwargs)
        except TypeError:
            vector = self.model.encode([text], **kwargs)
            if vector.shape[1] > self.dim:
                vector = vector[:, : self.dim]
                vector = vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (1, self.dim):
            raise ValueError(f"query embedding shape {vector.shape} != (1, {self.dim})")
        return vector[0]

    def _hit_rows(self, selected: list[tuple[int, float]]) -> list[dict[str, Any]]:
        return [
            {
                "rank": rank,
                "score": score,
                "memory_id": self.cards[idx].get("memory_id"),
                "subject": self.cards[idx].get("subject"),
                "concept": self.cards[idx].get("concept"),
                "card": self.cards[idx],
            }
            for rank, (idx, score) in enumerate(selected, start=1)
        ]

    def _random_rng(self, *, strategy: str, subject: str, random_seed: int, random_key: str) -> np.random.Generator:
        payload = json.dumps(
            {
                "strategy": strategy,
                "subject": subject,
                "random_seed": int(random_seed),
                "random_key": random_key,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little", signed=False)
        return np.random.default_rng(seed)

    def retrieve_random(
        self,
        *,
        subject: str,
        top_k: int,
        strategy: str,
        random_seed: int,
        random_key: str,
    ) -> list[dict[str, Any]]:
        if strategy == "random_same_subject" and subject in self.by_subject:
            pool = self.by_subject[subject]
        else:
            pool = np.arange(len(self.cards), dtype=np.int64)
        count = min(max(int(top_k), 0), len(pool))
        if count <= 0:
            return []
        rng = self._random_rng(strategy=strategy, subject=subject, random_seed=random_seed, random_key=random_key)
        chosen = rng.choice(pool, size=count, replace=False)
        selected = [(int(idx), 0.0) for idx in chosen.tolist()]
        return self._hit_rows(selected)

    def retrieve(self, *, query: str, subject: str, top_k: int, subject_filter: bool) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("embedding index is not loaded")
        vector = self.encode(query)
        if subject_filter and subject in self.by_subject:
            indices = self.by_subject[subject]
            matrix = np.asarray(self.index[indices], dtype=np.float32)
            scores = matrix @ vector
            local_top = np.argpartition(-scores, kth=min(top_k, len(scores) - 1))[:top_k]
            ordered = local_top[np.argsort(-scores[local_top])]
            selected = [(int(indices[i]), float(scores[i])) for i in ordered]
        else:
            scores = np.asarray(self.index @ vector, dtype=np.float32)
            top = np.argpartition(-scores, kth=min(top_k, len(scores) - 1))[:top_k]
            ordered = top[np.argsort(-scores[top])]
            selected = [(int(i), float(scores[i])) for i in ordered]
        return self._hit_rows(selected)


async def run_one(
    row: dict[str, Any],
    *,
    model: LocalVLLMModel,
    retriever: ConceptBankRetriever,
    output_dir: Path,
    top_k: int,
    candidate_top_k: int,
    max_card_chars: int,
    rerank_card_chars: int,
    answer_max_tokens: int,
    query_max_tokens: int,
    rerank_max_tokens: int,
    rerank_check: bool,
    subject_filter: bool,
    retrieval_strategy: str,
    random_seed: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.perf_counter()
    key = row_key(row)
    async with semaphore:
        try:
            query_payload = await model.json(
                concept_query_messages(row),
                namespace=f"{key}.concept_query",
                max_tokens=query_max_tokens,
                temperature=0.0,
            )
        except ValueError as exc:
            query_payload = {
                "concept_query": "",
                "target_concepts": [],
                "_query_rewrite_error": f"{type(exc).__name__}: {exc}",
            }
        concept_query = str(query_payload.get("concept_query") or "").strip()
        target_concepts = query_payload.get("target_concepts")
        if not concept_query:
            concept_query = f"{row.get('subject') or ''} {row.get('question') or ''} {format_options(row.get('options') or {})}"
        if isinstance(target_concepts, list):
            concept_query = concept_query + " " + " ".join(str(item) for item in target_concepts[:5])
        retrieval_query = f"Subject: {row.get('subject') or ''}\nConcept query: {concept_query}"
        candidate_count = max(top_k, candidate_top_k)
        subject = str(row.get("subject") or "")
        if retrieval_strategy == "embedding":
            candidate_hits = retriever.retrieve(
                query=retrieval_query,
                subject=subject,
                top_k=candidate_count,
                subject_filter=subject_filter,
            )
        else:
            candidate_hits = retriever.retrieve_random(
                subject=subject,
                top_k=candidate_count,
                strategy=retrieval_strategy,
                random_seed=random_seed,
                random_key=key,
            )
        rerank_payload: dict[str, Any] = {"selected": [], "_rerank_check_disabled": True}
        if rerank_check:
            try:
                rerank_payload = await model.json(
                    rerank_check_messages(
                        row,
                        candidate_hits,
                        query_payload,
                        final_top_k=top_k,
                        max_card_chars=rerank_card_chars,
                    ),
                    namespace=f"{key}.rerank_check",
                    max_tokens=rerank_max_tokens,
                    temperature=0.0,
                )
            except ValueError as exc:
                rerank_payload = {
                    "selected": [],
                    "_rerank_check_error": f"{type(exc).__name__}: {exc}",
                }
        hits = checked_cards_from_rerank(candidate_hits, rerank_payload, final_top_k=top_k) if rerank_check else candidate_hits[:top_k]
        answer_text = await model.text(
            answer_messages(row, hits, query_payload, max_card_chars=max_card_chars),
            namespace=f"{key}.answer",
            max_tokens=answer_max_tokens,
            temperature=0.0,
        )
    pred, valid = parse_answer_label_from_text(answer_text, row)
    gold = str(row.get("answer") or "").strip().upper()
    return {
        "eval_key": key,
        "method": (
            f"ca_concept_bank_{retrieval_strategy}_query_rewrite_rerank_check"
            if rerank_check
            else f"ca_concept_bank_{retrieval_strategy}_query_rewrite"
        ),
        "dataset": row.get("_dataset") or row.get("dataset") or "global_mmlu",
        "language": row.get("language") or row.get("_language"),
        "subject": row.get("subject"),
        "sample_id": row.get("sample_id") or row.get("question_id") or row.get("id"),
        "answer": gold,
        "prediction": pred,
        "valid": valid,
        "correct": bool(valid and pred == gold),
        "concept_query_payload": query_payload,
        "rerank_payload": rerank_payload,
        "retrieval_query": retrieval_query,
        "retrieval_strategy": retrieval_strategy,
        "random_seed": random_seed if retrieval_strategy != "embedding" else None,
        "candidate_cards": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "memory_id": hit["memory_id"],
                "subject": hit["subject"],
                "concept": hit["concept"],
                "source_sample_ids": hit["card"].get("source_sample_ids") or [],
                "description_preview": str(hit["card"].get("description") or "")[:500],
            }
            for hit in candidate_hits
        ],
        "retrieved_cards": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "rerank_relevance": hit.get("rerank_relevance"),
                "memory_id": hit["memory_id"],
                "subject": hit["subject"],
                "concept": hit["concept"],
                "source_sample_ids": hit["card"].get("source_sample_ids") or [],
                "checked_card_preview": str(hit.get("checked_card") or "")[:500],
                "description_preview": str(hit["card"].get("description") or "")[:500],
            }
            for hit in hits
        ],
        "raw_answer_text": answer_text,
        "latency_s": time.perf_counter() - started,
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]], model_calls: int, errors: int) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    valid = sum(1 for row in rows if row.get("valid"))
    query_rewrite_fallbacks = sum(1 for row in rows if "_query_rewrite_error" in (row.get("concept_query_payload") or {}))
    rerank_check_fallbacks = sum(1 for row in rows if "_rerank_check_error" in (row.get("rerank_payload") or {}))
    final_card_counts = [len(row.get("retrieved_cards") or []) for row in rows]
    by_language = []
    for language in sorted({str(row.get("language") or "") for row in rows}):
        items = [row for row in rows if str(row.get("language") or "") == language]
        by_language.append(
            {
                "language": language,
                "n": len(items),
                "correct": sum(1 for row in items if row.get("correct")),
                "accuracy": sum(1 for row in items if row.get("correct")) / len(items) if items else 0.0,
            }
        )
    by_subject = []
    for subject in sorted({str(row.get("subject") or "") for row in rows}):
        items = [row for row in rows if str(row.get("subject") or "") == subject]
        by_subject.append(
            {
                "subject": subject,
                "n": len(items),
                "correct": sum(1 for row in items if row.get("correct")),
                "accuracy": sum(1 for row in items if row.get("correct")) / len(items) if items else 0.0,
            }
        )
    summary = {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "valid": valid,
        "parse_rate": valid / n if n else 0.0,
        "errors": errors,
        "model_calls_observed": model_calls,
        "query_rewrite_fallbacks": query_rewrite_fallbacks,
        "rerank_check_fallbacks": rerank_check_fallbacks,
        "avg_final_card_count": sum(final_card_counts) / len(final_card_counts) if final_card_counts else 0.0,
        "by_language": by_language,
        "by_subject": by_subject,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [
        "# CA concept-bank query-rewrite smoke",
        "",
        f"- rows: {n}",
        f"- correct: {correct}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- parse_rate: {summary['parse_rate']:.4f}",
        f"- errors: {errors}",
        f"- model_calls_observed: {model_calls}",
        f"- query_rewrite_fallbacks: {query_rewrite_fallbacks}",
        f"- rerank_check_fallbacks: {rerank_check_fallbacks}",
        f"- avg_final_card_count: {summary['avg_final_card_count']:.2f}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


async def amain() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(args.input_jsonl))[: int(args.max_rows)]
    rerank_check = not args.no_rerank_check
    estimated_model_calls = len(rows) * (3 if rerank_check else 2)
    plan = {
        "rows": len(rows),
        "estimated_model_calls": estimated_model_calls,
        "max_model_calls": args.max_model_calls,
        "input_jsonl": str(args.input_jsonl),
        "bank_dir": str(args.bank_dir),
        "cache_dir": str(args.cache_dir or args.output_dir / "cache" / "local_vllm"),
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
        "retrieval_strategy": args.retrieval_strategy,
        "random_seed": args.random_seed if args.retrieval_strategy != "embedding" else None,
        "rerank_check": rerank_check,
        "rerank_card_chars": args.rerank_card_chars,
        "subject_filter": not args.no_subject_filter,
        "contains_oracle": True,
        "diagnostic_only": True,
    }
    print(json.dumps({"event": "plan", **plan}, ensure_ascii=False), flush=True)
    write_jsonl(args.output_dir / "plan.jsonl", [plan])
    if estimated_model_calls > int(args.max_model_calls):
        raise RuntimeError(f"refusing run: estimated model calls {estimated_model_calls} > max {args.max_model_calls}")

    prediction_path = args.output_dir / "predictions.jsonl"
    error_path = args.output_dir / "errors.jsonl"
    if not args.resume:
        for path in [prediction_path, error_path]:
            if path.exists():
                path.unlink()
    existing = {str(row.get("eval_key")): row for row in read_jsonl(prediction_path)} if args.resume and prediction_path.exists() else {}

    retriever = ConceptBankRetriever(
        bank_dir=args.bank_dir,
        model_dir=args.embedding_model_dir,
        device=args.embedding_device,
        load_embedding_model=args.retrieval_strategy == "embedding",
    )
    model = LocalVLLMModel(
        model_dir=args.local_model_dir,
        max_model_len=args.local_max_model_len,
        gpu_memory_utilization=args.local_gpu_memory_utilization,
        local_batch_size=args.local_batch_size,
        local_batch_timeout_ms=args.local_batch_timeout_ms,
        cache_dir=args.cache_dir or args.output_dir / "cache" / "local_vllm",
    )
    semaphore = asyncio.Semaphore(int(args.max_concurrent_model_requests))
    write_lock = asyncio.Lock()
    results = list(existing.values())
    errors = 0
    completed = len(results)
    started_at = time.perf_counter()
    try:
        pending_rows = [row for row in rows if row_key(row) not in existing]
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        for row in pending_rows:
            queue.put_nowait(row)
        worker_count = max(1, int(args.max_concurrent_model_requests))
        for _ in range(worker_count):
            queue.put_nowait(None)

        async def worker(worker_id: int) -> None:
            nonlocal errors, completed
            while True:
                row = await queue.get()
                if row is None:
                    queue.task_done()
                    return
                key = row_key(row)
                try:
                    result = await run_one(
                        row,
                        model=model,
                        retriever=retriever,
                        output_dir=args.output_dir,
                        top_k=int(args.top_k),
                        candidate_top_k=int(args.candidate_top_k),
                        max_card_chars=int(args.max_card_chars),
                        rerank_card_chars=int(args.rerank_card_chars),
                        answer_max_tokens=int(args.answer_max_tokens),
                        query_max_tokens=int(args.query_max_tokens),
                        rerank_max_tokens=int(args.rerank_max_tokens),
                        rerank_check=rerank_check,
                        subject_filter=not args.no_subject_filter,
                        retrieval_strategy=str(args.retrieval_strategy),
                        random_seed=int(args.random_seed),
                        semaphore=semaphore,
                    )
                    async with write_lock:
                        with prediction_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                        results.append(result)
                        completed += 1
                        done_now = completed
                        elapsed = max(time.perf_counter() - started_at, 1e-9)
                        rate = done_now / elapsed
                        remaining = max(len(rows) - done_now, 0)
                        eta_s = remaining / max(rate, 1e-9)
                    if done_now % max(1, int(args.progress_every)) == 0 or done_now == len(rows):
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "done": done_now,
                                    "total": len(rows),
                                    "rate_rows_per_s": rate,
                                    "eta_s": eta_s,
                                    "model_calls_observed": model.calls,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if not args.quiet_items:
                        print(json.dumps({"event": "item_done", "eval_key": key, "correct": result["correct"], "error": None}, ensure_ascii=False), flush=True)
                except Exception as exc:
                    err = {
                        "eval_key": key,
                        "sample_id": row.get("sample_id") or row.get("question_id") or row.get("id"),
                        "language": row.get("language") or row.get("_language"),
                        "subject": row.get("subject"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "correct": False,
                        "valid": False,
                    }
                    async with write_lock:
                        errors += 1
                        with error_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(err, ensure_ascii=False, sort_keys=True) + "\n")
                        completed += 1
                        done_now = completed
                    print(json.dumps({"event": "item_done", "eval_key": key, "correct": False, "error": err["error"]}, ensure_ascii=False), flush=True)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(worker_count)]
        await queue.join()
        await asyncio.gather(*workers)
        # Keep old sequential logic shape out of the hot path. Results are appended by workers above.
        for row in []:
            key = row_key(row)
            if key in existing:
                continue
            try:
                result = await run_one(
                    row,
                    model=model,
                    retriever=retriever,
                    output_dir=args.output_dir,
                    top_k=int(args.top_k),
                    candidate_top_k=int(args.candidate_top_k),
                    max_card_chars=int(args.max_card_chars),
                    rerank_card_chars=int(args.rerank_card_chars),
                    answer_max_tokens=int(args.answer_max_tokens),
                    query_max_tokens=int(args.query_max_tokens),
                    rerank_max_tokens=int(args.rerank_max_tokens),
                    rerank_check=rerank_check,
                    subject_filter=not args.no_subject_filter,
                    retrieval_strategy=str(args.retrieval_strategy),
                    random_seed=int(args.random_seed),
                    semaphore=semaphore,
                )
                with prediction_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                results.append(result)
                print(json.dumps({"event": "item_done", "eval_key": key, "correct": result["correct"], "error": None}, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors += 1
                err = {
                    "eval_key": key,
                    "sample_id": row.get("sample_id") or row.get("question_id") or row.get("id"),
                    "language": row.get("language") or row.get("_language"),
                    "subject": row.get("subject"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "correct": False,
                    "valid": False,
                }
                with error_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(err, ensure_ascii=False, sort_keys=True) + "\n")
                print(json.dumps({"event": "item_done", "eval_key": key, "correct": False, "error": err["error"]}, ensure_ascii=False), flush=True)
        summary = write_summary(args.output_dir, results, model.calls, errors)
        print(json.dumps({"event": "done", "summary": summary}, ensure_ascii=False), flush=True)
    finally:
        await model.aclose()
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
