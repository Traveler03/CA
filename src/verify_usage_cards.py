from __future__ import annotations

from dataclasses import replace

from src.smoke_test.schemas import EvidenceClaim, RetrievalResult, UsageCard


def verify_usage_cards(cards: list[UsageCard], retrieval_results: list[RetrievalResult]) -> tuple[list[UsageCard], list[EvidenceClaim]]:
    source_text: dict[str, str] = {}
    for result in retrieval_results:
        for passage in result.passages:
            source_text[passage.source_id] = passage.text

    verified_cards: list[UsageCard] = []
    atomic_claims: list[EvidenceClaim] = []
    for card in cards:
        verified_claims: list[EvidenceClaim] = []
        for claim in card.evidence_claims:
            text = source_text.get(claim.source_id, "")
            if claim.evidence_span and claim.evidence_span in text:
                status = "SUPPORTED"
            elif claim.evidence_span and text:
                status = "PARTIAL"
            else:
                status = "UNSUPPORTED"
            updated = claim.model_copy(update={"status": status})
            verified_claims.append(updated)
            atomic_claims.append(updated)
        critical = [claim for claim in verified_claims if claim.critical]
        if critical and all(claim.status == "SUPPORTED" for claim in critical):
            card_status = "active"
        elif any(claim.status in {"SUPPORTED", "PARTIAL"} for claim in verified_claims):
            card_status = "provisional"
        else:
            card_status = "rejected"
        verified_cards.append(card.model_copy(update={"status": card_status, "evidence_claims": verified_claims}))
    return verified_cards, atomic_claims
