from __future__ import annotations

from pathlib import Path

from scripts.run_subject_concept_smoke import ConceptRegistryRow, normalize_concept_name
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
        subject_scopes=["high_school_microeconomics"],
        canonical_name="opportunity cost",
        normalized_name="opportunity cost",
        definition="Value of the best forgone alternative.",
        concept_type="economic_principle",
        status="ACTIVE",
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
