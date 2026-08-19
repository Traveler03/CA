from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.build_project_registry import build_project_registry
from src.build_usage_jobs import build_concepts, build_usage_jobs
from src.extract_project_signatures import extract_project_signatures
from src.extract_usage_cards import extract_usage_cards
from src.replay_seed_projects import replay_seed_projects
from src.retrieve_wikipag import retrieve_wikipag
from src.smoke_test.config import SmokeTestConfig, dump_config, load_config
from src.smoke_test.io import append_error, load_models, write_jsonl
from src.smoke_test.model_client import SmokeModelClient
from src.smoke_test.schemas import (
    ConceptRecord,
    EvidenceClaim,
    ProjectRecord,
    ProjectSignature,
    ReplayResult,
    RetrievalResult,
    UsageCard,
    UsageJob,
)
from src.verify_usage_cards import verify_usage_cards


def estimate_model_calls(project_count: int, usage_job_count: int) -> dict[str, int]:
    signature_calls = project_count
    card_calls = usage_job_count
    replay_calls = project_count * 2
    total = signature_calls + card_calls + replay_calls
    return {
        "signature_calls": signature_calls,
        "usage_card_calls": card_calls,
        "seed_replay_calls": replay_calls,
        "estimated_total": total,
    }


def split_cards(cards: list[UsageCard]) -> tuple[list[UsageCard], list[UsageCard], list[UsageCard]]:
    active = [card for card in cards if card.status == "active"]
    provisional = [card for card in cards if card.status == "provisional"]
    rejected = [card for card in cards if card.status == "rejected"]
    return active, provisional, rejected


def compute_metrics(
    *,
    projects: list[ProjectRecord],
    usage_jobs_count: int,
    retrieval_nonempty_count: int,
    cards: list[UsageCard],
    claims: list[EvidenceClaim],
    replay_correct_without: int,
    replay_correct_with: int,
    dry_run: bool,
    estimated_calls: dict[str, int],
    actual_model_calls: int | str,
    model_usage: dict[str, int] | None = None,
    error_count: int = 0,
) -> dict[str, object]:
    active, provisional, rejected = split_cards(cards)
    claim_counts: dict[str, int] = {}
    for claim in claims:
        claim_counts[claim.status] = claim_counts.get(claim.status, 0) + 1
    project_count = len(projects)
    return {
        "dry_run": dry_run,
        "project_count": project_count,
        "usage_job_count": usage_jobs_count,
        "retrieval_nonempty_count": retrieval_nonempty_count,
        "retrieval_nonempty_rate": retrieval_nonempty_count / usage_jobs_count if usage_jobs_count else 0,
        "usage_cards_total": len(cards),
        "usage_cards_active": len(active),
        "usage_cards_provisional": len(provisional),
        "usage_cards_rejected": len(rejected),
        "claim_status_counts": claim_counts,
        "replay_without_card_accuracy": replay_correct_without / project_count if project_count else 0,
        "replay_with_card_accuracy": replay_correct_with / project_count if project_count else 0,
        "estimated_model_calls_if_real_run": estimated_calls,
        "actual_model_calls": actual_model_calls,
        "model_usage": model_usage or {},
        "error_count": error_count,
    }


def write_report(
    path: Path,
    metrics: dict[str, object],
    *,
    config: SmokeTestConfig,
    output_dir: Path,
    artifacts: list[str],
    canary: bool,
    stage_counts: dict[str, int],
    representative_traces: list[dict[str, object]] | None = None,
) -> None:
    title = "smoke_001 canary report" if canary else "smoke_001 dry-run report"
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        f"- run: `{config.run.name}`",
        f"- model: `{config.model.name}`",
        f"- canary: `{canary}`",
        f"- dry_run: `{metrics['dry_run']}`",
        f"- output_dir: `{output_dir}`",
        f"- subjects: `{', '.join(config.dataset.subjects)}`",
        "",
        "## Stage counts",
        "",
    ]
    lines.extend(f"- {name}: `{count}`" for name, count in stage_counts.items())
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- projects: `{metrics['project_count']}`",
            f"- usage_jobs: `{metrics['usage_job_count']}`",
            f"- retrieval_nonempty_rate: `{metrics['retrieval_nonempty_rate']}`",
            f"- cards active/provisional/rejected: `{metrics['usage_cards_active']}` / `{metrics['usage_cards_provisional']}` / `{metrics['usage_cards_rejected']}`",
            f"- claim status counts: `{metrics['claim_status_counts']}`",
            f"- replay accuracy without/with card: `{metrics['replay_without_card_accuracy']}` / `{metrics['replay_with_card_accuracy']}`",
            f"- actual model calls: `{metrics['actual_model_calls']}`",
            f"- estimated calls if real run: `{metrics['estimated_model_calls_if_real_run']}`",
            f"- error_count: `{metrics['error_count']}`",
            f"- model_usage: `{metrics['model_usage']}`",
            "",
            "## Canary gate",
            "",
        ]
    )
    if canary:
        error_rate = int(metrics["error_count"]) / max(int(metrics["project_count"]), 1)
        should_continue = (
            int(metrics["project_count"]) == config.canary.project_count
            and error_rate <= config.canary.stop_on_error_rate
            and float(metrics["retrieval_nonempty_rate"]) >= 0.90
            and int(metrics["actual_model_calls"]) <= config.model.max_total_calls
        )
        lines.extend(
            [
                f"- project_count_ok: `{int(metrics['project_count']) == config.canary.project_count}`",
                f"- error_rate: `{error_rate}`",
                f"- stop_on_error_rate: `{config.canary.stop_on_error_rate}`",
                f"- retrieval_gate_90pct: `{float(metrics['retrieval_nonempty_rate']) >= 0.90}`",
                f"- model_budget_ok: `{int(metrics['actual_model_calls']) <= config.model.max_total_calls}`",
                f"- recommend_remaining_80: `{should_continue}`",
            ]
        )
    else:
        lines.append("- Real canary execution must be launched explicitly and remain capped at 20 projects first.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend(f"- `{artifact}`" for artifact in artifacts)
    if representative_traces:
        lines.extend(["", "## Representative traces", ""])
        for idx, trace in enumerate(representative_traces[:10], start=1):
            lines.append(f"{idx}. `{trace}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    if metrics["dry_run"]:
        lines.extend(
            [
                "- This run validates the local pipeline shape without calling GPT-5.5.",
                "- Real canary execution must be launched explicitly and remain capped at 20 projects first.",
            ]
        )
    else:
        lines.extend(
            [
                f"- This canary used {config.model.name} only inside the 20-project cap.",
                "- It does not automatically continue to the remaining 80 projects.",
            ]
        )
    lines.append("- The Wikipag/Wikipedia index is used read-only through the local service.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_or_load(path: Path, model: type, *, resume: bool, build) -> list:
    if resume and path.exists() and path.stat().st_size > 0:
        return load_models(path, model)
    rows = build()
    write_jsonl(path, rows)
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def representative_traces(
    *,
    projects: list[ProjectRecord],
    jobs: list[UsageJob],
    retrieval_results: list[RetrievalResult],
    cards: list[UsageCard],
    replay_results: list[ReplayResult],
) -> list[dict[str, object]]:
    jobs_by_project: dict[str, list[UsageJob]] = {}
    for job in jobs:
        for project_id in job.seed_project_ids:
            jobs_by_project.setdefault(project_id, []).append(job)
    retrieval_by_job = {row.usage_job_id: row for row in retrieval_results}
    cards_by_job: dict[str, list[UsageCard]] = {}
    for card in cards:
        cards_by_job.setdefault(card.usage_job_id, []).append(card)
    replay_by_project = {row.project_id: row for row in replay_results}
    traces: list[dict[str, object]] = []
    for project in projects[:10]:
        project_jobs = jobs_by_project.get(project.project_id, [])
        project_cards = [card for job in project_jobs for card in cards_by_job.get(job.usage_job_id, [])]
        trace = {
            "project_id": project.project_id,
            "subject": project.subject,
            "usage_job_ids": [job.usage_job_id for job in project_jobs[:5]],
            "retrieved_passage_counts": {
                job.usage_job_id: len(retrieval_by_job.get(job.usage_job_id, RetrievalResult(usage_job_id=job.usage_job_id, passages=[])).passages)
                for job in project_jobs[:5]
            },
            "usage_card_ids": [card.usage_card_id for card in project_cards[:5]],
            "active_card_count": sum(1 for card in project_cards if card.status == "active"),
            "replay": replay_by_project.get(project.project_id).model_dump(mode="json") if replay_by_project.get(project.project_id) else None,
        }
        traces.append(trace)
    return traces


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.model:
        config = config.model_copy(update={"model": config.model.model_copy(update={"name": args.model})})
    dry_run = bool(args.dry_run or config.run.dry_run)
    if not dry_run and not args.allow_model_calls:
        print("Refusing to run non-dry-run smoke test without --allow-model-calls.", file=sys.stderr)
        return 2
    if not dry_run and not args.canary:
        print("Refusing to run model-backed smoke test without --canary. Run the first 20 projects only.", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit > 100:
        raise ValueError("--limit cannot exceed 100")
    subjects = args.subjects.split(",") if args.subjects else None
    if subjects and len(subjects) > 5:
        raise ValueError("--subjects cannot contain more than 5 subjects")

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.run.output_dir) / "canary" if args.canary else Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    if not args.resume:
        errors_path.write_text("", encoding="utf-8")
    elif not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")

    resolved_config = dump_config(config)
    resolved_config["run"]["dry_run"] = dry_run
    resolved_config["run"]["output_dir"] = str(output_dir)
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")

    print("resolved_config:")
    print(yaml.safe_dump(resolved_config, sort_keys=False))
    print(f"input_dataset: {config.dataset.path}")
    print(f"wikipag_service: {config.retrieval.service_url}")
    print(f"production_registry_write: {config.write.production_registry}")
    print(f"production_vector_index_write: {config.write.production_vector_index}")
    print(f"output_dir: {output_dir}")

    model_client: SmokeModelClient | None = None
    if not dry_run:
        model_client = SmokeModelClient.from_config(
            config.model,
            cache_dir=output_dir / "cache" / "model",
            raw_dir=output_dir / "logs" / "model_raw",
            timeout_s=config.retrieval.timeout_s,
        )

    try:
        registry_limit = None if args.canary else args.limit
        projects = stage_or_load(
            output_dir / "projects.jsonl",
            ProjectRecord,
            resume=args.resume,
            build=lambda: build_project_registry(config, limit=registry_limit, subjects=subjects),
        )
        if args.canary:
            projects = projects[: config.canary.project_count]
            write_jsonl(output_dir / "projects.jsonl", projects)
        elif args.limit is not None:
            projects = projects[: args.limit]
            write_jsonl(output_dir / "projects.jsonl", projects)
    except Exception as exc:
        append_error(errors_path, stage="project_registry", item_id="global", error=str(exc))
        raise

    signatures = stage_or_load(
        output_dir / "project_signatures.jsonl",
        ProjectSignature,
        resume=args.resume,
        build=lambda: extract_project_signatures(projects, config, dry_run=dry_run, client=model_client, errors_path=errors_path),
    )
    concepts = stage_or_load(
        output_dir / "concepts.jsonl",
        ConceptRecord,
        resume=args.resume,
        build=lambda: build_concepts(signatures),
    )
    usage_jobs = stage_or_load(
        output_dir / "usage_jobs.jsonl",
        UsageJob,
        resume=args.resume,
        build=lambda: build_usage_jobs(signatures),
    )

    estimated_calls = estimate_model_calls(len(projects), len(usage_jobs))
    print(f"planned_project_count: {len(projects)}")
    print(f"planned_usage_job_count: {len(usage_jobs)}")
    print(f"estimated_model_calls_if_real_run: {json.dumps(estimated_calls, sort_keys=True)}")
    if estimated_calls["estimated_total"] > config.model.max_total_calls:
        raise RuntimeError(f"estimated model calls exceed budget: {estimated_calls}")

    try:
        retrieval_results = stage_or_load(
            output_dir / "retrieval_results.jsonl",
            RetrievalResult,
            resume=args.resume,
            build=lambda: retrieve_wikipag(usage_jobs, config, cache_dir=output_dir / "cache" / "retrieval"),
        )
    except Exception as exc:
        append_error(errors_path, stage="wikipag_retrieval", item_id="global", error=str(exc))
        raise

    cards_initial = extract_usage_cards(usage_jobs, retrieval_results, config, dry_run=dry_run, client=model_client, errors_path=errors_path)
    verified_cards, claims = verify_usage_cards(cards_initial, retrieval_results)
    active, provisional, rejected = split_cards(verified_cards)
    write_jsonl(output_dir / "usage_cards_provisional.jsonl", provisional)
    write_jsonl(output_dir / "usage_cards_active.jsonl", active)
    write_jsonl(output_dir / "usage_cards_rejected.jsonl", rejected)
    write_jsonl(output_dir / "atomic_claims.jsonl", claims)

    replay_results = replay_seed_projects(projects, usage_jobs, verified_cards, dry_run=dry_run, client=model_client, errors_path=errors_path)
    write_jsonl(output_dir / "replay_results.jsonl", replay_results)

    actual_model_calls: int | str = 0 if dry_run else model_client.network_call_count if model_client else "not_recorded"
    model_usage = {} if dry_run or model_client is None else model_client.usage_summary()
    if isinstance(actual_model_calls, int) and actual_model_calls > config.model.max_total_calls:
        raise RuntimeError(f"actual model calls exceed budget: {actual_model_calls}")

    metrics = compute_metrics(
        projects=projects,
        usage_jobs_count=len(usage_jobs),
        retrieval_nonempty_count=sum(1 for result in retrieval_results if result.passages),
        cards=verified_cards,
        claims=claims,
        replay_correct_without=sum(1 for row in replay_results if row.without_card_correct),
        replay_correct_with=sum(1 for row in replay_results if row.with_card_correct),
        dry_run=dry_run,
        estimated_calls=estimated_calls,
        actual_model_calls=actual_model_calls,
        model_usage=model_usage,
        error_count=count_jsonl(errors_path),
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = [
        "config.resolved.yaml",
        "projects.jsonl",
        "project_signatures.jsonl",
        "concepts.jsonl",
        "usage_jobs.jsonl",
        "retrieval_results.jsonl",
        "usage_cards_provisional.jsonl",
        "usage_cards_active.jsonl",
        "usage_cards_rejected.jsonl",
        "atomic_claims.jsonl",
        "replay_results.jsonl",
        "errors.jsonl",
        "metrics.json",
        "report.md",
    ]
    stage_counts = {
        "projects": len(projects),
        "project_signatures": len(signatures),
        "concepts": len(concepts),
        "usage_jobs": len(usage_jobs),
        "retrieval_results": len(retrieval_results),
        "usage_cards_active": len(active),
        "usage_cards_provisional": len(provisional),
        "usage_cards_rejected": len(rejected),
        "atomic_claims": len(claims),
        "replay_results": len(replay_results),
        "errors": count_jsonl(errors_path),
    }
    write_report(
        output_dir / "report.md",
        metrics,
        config=config,
        output_dir=output_dir,
        artifacts=artifacts,
        canary=args.canary,
        stage_counts=stage_counts,
        representative_traces=representative_traces(
            projects=projects,
            jobs=usage_jobs,
            retrieval_results=retrieval_results,
            cards=verified_cards,
            replay_results=replay_results,
        ),
    )
    if model_client is not None:
        import asyncio

        asyncio.run(model_client.aclose())
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Concept-Usage Bank smoke_001 pipeline.")
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--model", help="Override the configured smoke-test chat model, e.g. gpt-5.4-mini.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subjects")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
