from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path("artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16")
INPUT_DIR = Path("artifacts/model_eval_20260821/inputs/full_low_resource_47815")
LLAMA_DIR = Path("data/external/models/Llama-3.1-8B-Instruct")
LLAMA_TOTAL_BYTES_FALLBACK = 16_060_522_496
LLAMA_PID_FILE = Path("artifacts/model_eval_20260821/download_logs/llama_modelscope_download.pid")
FULL_QUEUE_PID_FILE = Path("artifacts/model_eval_20260821/run_control/gpu0_full_queue.pid")
TWO_MODEL_METRICS = Path("artifacts/model_eval_20260821/two_model_summary/metrics.json")


def count_input_rows() -> int:
    total = 0
    for path in sorted(INPUT_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def count_log_items(path: Path) -> dict[str, int]:
    out = {"items": 0, "correct": 0, "wrong": 0, "errors": 0}
    if not path.exists():
        return out
    with path.open(errors="ignore") as f:
        for line in f:
            if '"event": "item_done"' not in line or "coral_wikipag|" not in line:
                continue
            out["items"] += 1
            if '"error": null' not in line:
                out["errors"] += 1
            elif '"correct": true' in line:
                out["correct"] += 1
            elif '"correct": false' in line:
                out["wrong"] += 1
    return out


def add_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {key: int(a.get(key, 0)) + int(b.get(key, 0)) for key in {"items", "correct", "wrong", "errors"}}


def llama_progress() -> dict[str, object]:
    expected_bytes = LLAMA_TOTAL_BYTES_FALLBACK
    index_path = LLAMA_DIR / "model.safetensors.index.json"
    if index_path.exists():
        try:
            metadata = json.loads(index_path.read_text(encoding="utf-8")).get("metadata") or {}
            expected_bytes = int(metadata.get("total_size") or expected_bytes)
        except Exception:
            expected_bytes = LLAMA_TOTAL_BYTES_FALLBACK
    files = []
    bytes_now = 0
    for path in sorted(LLAMA_DIR.glob("model-*.safetensors")):
        size = path.stat().st_size
        bytes_now += size
        files.append({"file": path.name, "bytes": size, "complete_file": True})
    for path in sorted((LLAMA_DIR / "._____temp").glob("model-*.safetensors")):
        size = path.stat().st_size
        bytes_now += size
        files.append({"file": f"._____temp/{path.name}", "bytes": size, "complete_file": False})
    pid: int | None = None
    pid_alive = False
    if LLAMA_PID_FILE.exists():
        try:
            pid = int(LLAMA_PID_FILE.read_text(encoding="utf-8").strip())
            pid_alive = Path(f"/proc/{pid}").exists()
        except Exception:
            pid = None
    return {
        "dir": str(LLAMA_DIR),
        "download_pid": pid,
        "download_pid_alive": pid_alive,
        "bytes": bytes_now,
        "gb": bytes_now / 1_000_000_000,
        "expected_gb": expected_bytes / 1_000_000_000,
        "percent": bytes_now / expected_bytes if expected_bytes else 0.0,
        "complete_safetensor_files": sum(1 for item in files if item["complete_file"]),
        "temp_safetensor_files": sum(1 for item in files if not item["complete_file"]),
        "looks_complete": len(list(LLAMA_DIR.glob("model-*.safetensors"))) == 4 and bytes_now >= expected_bytes * 0.98,
        "files": files,
    }


def pid_status(pid_file: Path) -> dict[str, object]:
    pid: int | None = None
    alive = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            alive = Path(f"/proc/{pid}").exists()
        except Exception:
            pid = None
    return {"pid_file": str(pid_file), "pid": pid, "alive": alive}


def latest_two_model_coverage() -> list[dict[str, object]]:
    if not TWO_MODEL_METRICS.exists():
        return []
    try:
        payload = json.loads(TWO_MODEL_METRICS.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for row in payload.get("rows") or []:
        rows.append(
            {
                "model": row.get("model"),
                "method": row.get("method"),
                "coverage": row.get("coverage"),
                "missing": row.get("missing"),
                "overall_accuracy": row.get("overall"),
                "global_mmlu_accuracy": row.get("global_mmlu"),
                "mmlu_prox_accuracy": row.get("mmlu_prox"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print current model-eval and Llama download status.")
    parser.add_argument("--watch-seconds", type=int, default=0, help="If positive, print status, sleep, then print delta status.")
    return parser.parse_args()


def collect_status() -> dict[str, object]:
    target = count_input_rows()
    logs = {
        "32": [ROOT / "baselines_shard0" / "run_coral_resume_32.log", ROOT / "baselines_shard1" / "run_coral_resume_32.log"],
        "64": [ROOT / "baselines_shard0" / "run_coral_resume_64.log", ROOT / "baselines_shard1" / "run_coral_resume_64.log"],
        "64_patch": [
            ROOT / "baselines_shard0" / "run_coral_resume_64_patch.log",
            ROOT / "baselines_shard1" / "run_coral_resume_64_patch.log",
        ],
        "gpu0_sequence": [ROOT / "run_control" / "ministral_coral_gpu0_sequence.nohup.log"],
    }
    coral_by_log = {}
    coral_total = {"items": 0, "correct": 0, "wrong": 0, "errors": 0}
    for label, paths in logs.items():
        merged = {"items": 0, "correct": 0, "wrong": 0, "errors": 0}
        for path in paths:
            merged = add_counts(merged, count_log_items(path))
        coral_by_log[label] = merged
        coral_total = add_counts(coral_total, merged)
    return {
        "timestamp": time.time(),
        "target_rows": target,
        "ministral_coral_log_progress": {
            **coral_total,
            "remaining_approx": max(target - coral_total["items"], 0),
            "coverage_approx": coral_total["items"] / target if target else 0.0,
            "accuracy_on_logged_non_errors": (
                coral_total["correct"] / max(coral_total["correct"] + coral_total["wrong"], 1)
            ),
            "by_log": coral_by_log,
        },
        "llama_download": llama_progress(),
        "gpu0_full_queue": pid_status(FULL_QUEUE_PID_FILE),
        "latest_two_model_coverage": latest_two_model_coverage(),
    }


def main() -> int:
    args = parse_args()
    first = collect_status()
    if args.watch_seconds <= 0:
        print(json.dumps(first, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    time.sleep(int(args.watch_seconds))
    second = collect_status()
    elapsed = max(float(second["timestamp"]) - float(first["timestamp"]), 1e-9)
    first_coral = first["ministral_coral_log_progress"]
    second_coral = second["ministral_coral_log_progress"]
    first_llama = first["llama_download"]
    second_llama = second["llama_download"]
    delta = {
        "elapsed_s": elapsed,
        "ministral_coral_items_delta": int(second_coral["items"]) - int(first_coral["items"]),
        "ministral_coral_items_per_min": (int(second_coral["items"]) - int(first_coral["items"])) / elapsed * 60.0,
        "llama_bytes_delta": int(second_llama["bytes"]) - int(first_llama["bytes"]),
        "llama_mb_per_s": (int(second_llama["bytes"]) - int(first_llama["bytes"])) / elapsed / 1_000_000.0,
    }
    remaining_items = int(second_coral["remaining_approx"])
    if delta["ministral_coral_items_per_min"] > 0:
        delta["ministral_coral_eta_h"] = remaining_items / delta["ministral_coral_items_per_min"] / 60.0
    remaining_bytes = max(int(float(second_llama.get("expected_gb", 0.0)) * 1_000_000_000) - int(second_llama["bytes"]), 0)
    if delta["llama_mb_per_s"] > 0:
        delta["llama_download_eta_h"] = remaining_bytes / (delta["llama_mb_per_s"] * 1_000_000.0) / 3600.0
    print(json.dumps({"first": first, "second": second, "delta": delta}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
