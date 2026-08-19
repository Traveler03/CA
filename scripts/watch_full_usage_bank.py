from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("runs/full_20260730")


COMPONENTS = [
    {
        "name": "oracle",
        "output_dir": ROOT / "global_mmlu_oracle_cap1000_gpt54",
        "match": "runs/full_20260730/global_mmlu_oracle_cap1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_global_mmlu_oracle_usage_bank.py",
            "--subjects",
            "all",
            "--max-rows",
            "0",
            "--max-rows-per-subject",
            "1000",
            "--output-dir",
            "runs/full_20260730/global_mmlu_oracle_cap1000_gpt54",
            "--bank-version",
            "global-mmlu-oracle-cap1000-gpt54-v1",
            "--model",
            "gpt-5.4",
            "--max-completion-tokens",
            "1600",
            "--concurrency",
            "24",
            "--batch-size",
            "512",
            "--resume",
            "--model-timeout-s",
            "180",
        ],
    },
    {
        "name": "mmlu_prox_oracle",
        "output_dir": ROOT / "mmlu_prox_non_global_oracle_cap1000_gpt54",
        "match": "runs/full_20260730/mmlu_prox_non_global_oracle_cap1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_global_mmlu_oracle_usage_bank.py",
            "--input-jsonl",
            "data/processed/mmlu_prox/en.jsonl",
            "--source-dataset",
            "mmlu_prox",
            "--dataset-label",
            "MMLU-ProX",
            "--bank-component",
            "mmlu_prox_non_global_oracle",
            "--row-key-prefix",
            "mmlu_prox",
            "--namespace-salt",
            "non_global",
            "--subjects",
            "all",
            "--source-splits",
            "test",
            "--exclude-global-mmlu-overlap",
            "--max-rows",
            "0",
            "--max-rows-per-subject",
            "1000",
            "--output-dir",
            "runs/full_20260730/mmlu_prox_non_global_oracle_cap1000_gpt54",
            "--bank-version",
            "mmlu-prox-non-global-oracle-cap1000-gpt54-v1",
            "--model",
            "gpt-5.4",
            "--max-completion-tokens",
            "1600",
            "--concurrency",
            "16",
            "--batch-size",
            "512",
            "--resume",
            "--model-timeout-s",
            "180",
        ],
    },
    {
        "name": "clean_shard0",
        "output_dir": ROOT / "wikipag_clean_all_subjects_target1000_gpt54",
        "match": "runs/full_20260730/wikipag_clean_all_subjects_target1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_usage_bank_batch.py",
            "--output-dir",
            "runs/full_20260730/wikipag_clean_all_subjects_target1000_gpt54",
            "--subjects",
            "abstract_algebra",
            "college_chemistry",
            "conceptual_physics",
            "high_school_biology",
            "high_school_macroeconomics",
            "high_school_us_history",
            "logical_fallacies",
            "moral_disputes",
            "professional_law",
            "us_foreign_policy",
            "--bank-version",
            "wikipag-clean-shard0-target1000-gpt54-v2",
        ],
    },
    {
        "name": "clean_shard1",
        "output_dir": ROOT / "wikipag_clean_shard1_target1000_gpt54",
        "match": "runs/full_20260730/wikipag_clean_shard1_target1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_usage_bank_batch.py",
            "--output-dir",
            "runs/full_20260730/wikipag_clean_shard1_target1000_gpt54",
            "--subjects",
            "anatomy",
            "college_computer_science",
            "econometrics",
            "high_school_chemistry",
            "high_school_mathematics",
            "high_school_world_history",
            "machine_learning",
            "moral_scenarios",
            "professional_medicine",
            "virology",
            "--bank-version",
            "wikipag-clean-shard1-target1000-gpt54-v2",
        ],
    },
    {
        "name": "clean_shard2",
        "output_dir": ROOT / "wikipag_clean_shard2_target1000_gpt54",
        "match": "runs/full_20260730/wikipag_clean_shard2_target1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_usage_bank_batch.py",
            "--output-dir",
            "runs/full_20260730/wikipag_clean_shard2_target1000_gpt54",
            "--subjects",
            "astronomy",
            "college_mathematics",
            "electrical_engineering",
            "high_school_computer_science",
            "high_school_microeconomics",
            "human_aging",
            "management",
            "nutrition",
            "professional_psychology",
            "world_religions",
            "--bank-version",
            "wikipag-clean-shard2-target1000-gpt54-v2",
        ],
    },
    {
        "name": "clean_shard3",
        "output_dir": ROOT / "wikipag_clean_shard3_target1000_gpt54",
        "match": "runs/full_20260730/wikipag_clean_shard3_target1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_usage_bank_batch.py",
            "--output-dir",
            "runs/full_20260730/wikipag_clean_shard3_target1000_gpt54",
            "--subjects",
            "business_ethics",
            "college_medicine",
            "elementary_mathematics",
            "high_school_european_history",
            "high_school_physics",
            "human_sexuality",
            "marketing",
            "philosophy",
            "public_relations",
            "--bank-version",
            "wikipag-clean-shard3-target1000-gpt54-v2",
        ],
    },
    {
        "name": "clean_shard4",
        "output_dir": ROOT / "wikipag_clean_shard4_target1000_gpt54",
        "match": "runs/full_20260730/wikipag_clean_shard4_target1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_usage_bank_batch.py",
            "--output-dir",
            "runs/full_20260730/wikipag_clean_shard4_target1000_gpt54",
            "--subjects",
            "clinical_knowledge",
            "college_physics",
            "formal_logic",
            "high_school_geography",
            "high_school_psychology",
            "international_law",
            "medical_genetics",
            "prehistory",
            "security_studies",
            "--bank-version",
            "wikipag-clean-shard4-target1000-gpt54-v2",
        ],
    },
    {
        "name": "clean_shard5",
        "output_dir": ROOT / "wikipag_clean_shard5_target1000_gpt54",
        "match": "runs/full_20260730/wikipag_clean_shard5_target1000_gpt54",
        "cmd": [
            "python",
            "scripts/run_usage_bank_batch.py",
            "--output-dir",
            "runs/full_20260730/wikipag_clean_shard5_target1000_gpt54",
            "--subjects",
            "college_biology",
            "computer_security",
            "global_facts",
            "high_school_government_and_politics",
            "high_school_statistics",
            "jurisprudence",
            "miscellaneous",
            "professional_accounting",
            "sociology",
            "--bank-version",
            "wikipag-clean-shard5-target1000-gpt54-v2",
        ],
    },
]


CLEAN_COMMON_ARGS = [
    "--target-active-concepts",
    "1000",
    "--max-articles",
    "1000",
    "--max-passages",
    "600",
    "--max-seed-queries",
    "32",
    "--top-k-per-query",
    "100",
    "--max-extraction-passages",
    "600",
    "--max-grounding-candidates",
    "2000",
    "--grounding-top-k",
    "5",
    "--max-pair-judge-pairs",
    "1000",
    "--skip-llm-pair-judge",
    "--max-usage-concepts",
    "1000",
    "--max-usage-jobs",
    "1000",
    "--usage-retrieval-top-k",
    "20",
    "--final-materials-per-usage-job",
    "4",
    "--model",
    "gpt-5.4",
    "--max-completion-tokens",
    "4096",
    "--concurrency",
    "8",
    "--max-retries",
    "2",
    "--model-timeout-s",
    "180",
    "--retrieval-timeout-s",
    "180",
    "--embedding-timeout-s",
    "180",
    "--no-write-runtime-bank",
    "--resume",
]


SERVICE = {
    "name": "wikipag_service",
    "match": "scripts/wiki_faiss/serve_sherlock_wiki.py --host 127.0.0.1 --port 8897",
    "cmd": [
        "python",
        "scripts/wiki_faiss/serve_sherlock_wiki.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8897",
        "--index",
        "hnsw_sq.faiss",
        "--device",
        "cuda:1",
    ],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_complete(output_dir: Path) -> bool:
    return (output_dir / "summary.json").exists() and (output_dir / "bank_manifest.json").exists()


def ps_cmds() -> list[str]:
    out = subprocess.run(["ps", "-eo", "pid=,cmd=", "-ww"], text=True, capture_output=True, check=True)
    return out.stdout.splitlines()


def is_running(match: str, commands: list[str]) -> bool:
    return any(match in line and "watch_full_usage_bank.py" not in line for line in commands)


def service_health(url: str = "http://127.0.0.1:8897/health", timeout_s: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": bool(payload.get("ok")), "payload": payload}


def start_process(name: str, cmd: list[str], log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.restart.log"
    handle = log_path.open("ab")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    return int(proc.pid)


def replace_root(value: str, root: Path) -> str:
    return value.replace(str(ROOT), str(root), 1) if str(ROOT) in value else value


def selected_components(args: argparse.Namespace) -> list[dict[str, Any]]:
    components = COMPONENTS
    if args.clean_only:
        components = [component for component in components if component["name"].startswith("clean_")]
    return components


def component_cmd(component: dict[str, Any], root: Path) -> list[str]:
    cmd = [replace_root(str(item), root) for item in component["cmd"]]
    if component["name"].startswith("clean_"):
        return [*cmd, *CLEAN_COMMON_ARGS]
    return cmd


def component_match(component: dict[str, Any], root: Path) -> str:
    return replace_root(str(component["match"]), root)


def final_done(root: Path, *, clean_only: bool) -> bool:
    if clean_only:
        audit_path = root / "wikipag_clean_57subjects_target1000_gpt54_combined" / "audit.json"
    else:
        audit_path = root / "combined_oracle_plus_clean_57subjects_target1000_gpt54" / "audit.json"
    if not audit_path.exists():
        return False
    try:
        return bool(json.loads(audit_path.read_text(encoding="utf-8")).get("ok"))
    except Exception:
        return False


def check_once(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    commands = ps_cmds()
    events: list[dict[str, Any]] = []

    final_is_done = final_done(root, clean_only=args.clean_only)
    health = service_health()
    service_running = is_running(SERVICE["match"], commands)
    if not args.no_service_restart and not final_is_done and not service_running:
        pid = start_process(SERVICE["name"], SERVICE["cmd"], root / "watchdog_logs")
        events.append({"event": "restart", "component": SERVICE["name"], "pid": pid})
        return {
            "ok": True,
            "ts": now(),
            "clean_only": bool(args.clean_only),
            "final_done": final_is_done,
            "service_health": {"ok": False, "status": "started_now"},
            "events": events,
        }
    if not args.no_service_restart and not final_is_done and not health.get("ok"):
        events.append({"event": "waiting", "component": SERVICE["name"], "reason": "service_not_healthy", "health": health})
        return {
            "ok": True,
            "ts": now(),
            "clean_only": bool(args.clean_only),
            "final_done": final_is_done,
            "service_health": health,
            "events": events,
        }

    for component in selected_components(args):
        output_dir = Path(str(component["output_dir"]).replace(str(ROOT), str(root), 1))
        complete = is_complete(output_dir)
        running = is_running(component_match(component, root), commands)
        if complete:
            events.append({"event": "complete", "component": component["name"], "output_dir": str(output_dir)})
            continue
        if not running:
            cmd = component_cmd(component, root)
            pid = start_process(component["name"], cmd, root / "watchdog_logs")
            events.append({"event": "restart", "component": component["name"], "pid": pid, "output_dir": str(output_dir)})
        else:
            events.append({"event": "running", "component": component["name"], "output_dir": str(output_dir)})

    return {"ok": True, "ts": now(), "clean_only": bool(args.clean_only), "final_done": final_is_done, "service_health": health, "events": events}


def run(args: argparse.Namespace) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    while True:
        last = check_once(args)
        print(json.dumps(last, ensure_ascii=False, sort_keys=True), flush=True)
        if args.once or last.get("final_done"):
            return last
        time.sleep(args.poll_interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch full Usage Bank run and resume missing incomplete components.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--poll-interval-s", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-service-restart", action="store_true")
    parser.add_argument("--clean-only", action="store_true", help="Start only clean Wikipag shards; do not start benchmark oracle generation.")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
