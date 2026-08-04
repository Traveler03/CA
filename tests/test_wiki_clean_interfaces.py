from __future__ import annotations

from pathlib import Path

from scripts.query_runtime_cards import query_cards
from scripts.run_subject_concept_smoke import (
    ConceptRegistryRow,
    RuntimeCardRow,
    build_runtime_card_index,
    is_procedural_rule,
    normalize_concept_name,
)
from src.ca_mem.embedding import HashingTextEmbedder
from src.ca_mem.schemas import MemoryNode, Provenance, UsageSummary, bank_hash
from src.smoke_test.io import read_jsonl, write_jsonl
from src.smoke_test.stage_utils import stable_id


def test_normalize_concept_name_is_stable() -> None:
    assert normalize_concept_name("Opportunity-Costs") == "opportunity cost"
    assert stable_id("concept", "subject", "opportunity cost") == stable_id("concept", "subject", "opportunity cost")


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row = ConceptRegistryRow(
        concept_id="ECON-OPPORTUNITY-COST",
        subject="high_school_microeconomics",
        canonical_name="opportunity cost",
        normalized_name="opportunity cost",
        definition="Value of the best forgone alternative.",
        concept_type="economic_principle",
        status="ACTIVE",
        confidence=0.9,
    )
    assert write_jsonl(path, [row]) == 1
    loaded = read_jsonl(path)
    assert loaded[0]["concept_id"] == "ECON-OPPORTUNITY-COST"


def test_minimal_ca_mem_node_hash_and_embedding() -> None:
    node = MemoryNode.create(
        memory_id="mem_1",
        subject="high_school_microeconomics",
        concept="opportunity cost",
        description="Concept-level card.",
        usage_summary=UsageSummary(trigger_invariants=["choice among alternatives"]),
        provenance=Provenance(source_sample_ids=["source_1"], candidate_ids=["cand_1"]),
        order=1,
    )
    assert node.content_hash
    assert bank_hash([node])
    matrix = HashingTextEmbedder(dim=32).embed(["opportunity cost"])
    assert matrix.shape == (1, 32)


def test_runtime_card_index_query(tmp_path: Path) -> None:
    card = RuntimeCardRow(
        card_id="card_1",
        subject="high_school_microeconomics",
        concept_id="ECON-PRICE-ELASTICITY-DEMAND",
        concept="price elasticity of demand",
        definition="Measures how quantity demanded responds to price changes.",
        trigger=["Use when a task asks how quantity demanded responds to a price change."],
        rule=["Elasticity is percentage change in quantity demanded divided by percentage change in price."],
        pitfall=["Do not confuse elasticity with slope."],
        source_ids=["source_1"],
    )
    rows, meta = build_runtime_card_index([card], tmp_path)
    write_jsonl(tmp_path / "runtime_card_index.jsonl", rows)
    assert meta["count"] == 1

    result = query_cards(tmp_path, "price change quantity demanded elasticity", top_k=1)
    assert result["results"][0]["concept"] == "price elasticity of demand"


def test_procedural_rule_detector_handles_proof_steps() -> None:
    assert is_procedural_rule("Introduce the case split P or not-P for any proposition P.")
    assert is_procedural_rule(
        "If you assume not-not-P, combine that with excluded middle: if P, conclude P; if not-P, derive a contradiction."
    )
