#!/usr/bin/env bash
set -euo pipefail

cd /home/work/migoo_ai_public/linxuan/CA

RUN_CONTROL="artifacts/model_eval_20260821/run_control"
MINISTRAL_RC="artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16/run_control"
INPUT_DIR="artifacts/model_eval_20260821/inputs/full_low_resource_47815"
MINISTRAL_OUT="artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16"
LLAMA_OUT="artifacts/model_eval_20260821/llama3_1_8b_instruct"
LLAMA_DIR="data/external/models/Llama-3.1-8B-Instruct"
LLAMA_PID_FILE="artifacts/model_eval_20260821/download_logs/llama_modelscope_download.pid"
mkdir -p "$RUN_CONTROL"

log() {
  echo "[$(date -Is)] $*" >&2
}

is_alive() {
  local pid="$1"
  [[ -n "$pid" && -d "/proc/$pid" ]]
}

wait_for_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [[ ! -s "$pid_file" ]]; then
    return 0
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$pid_file")"
  if ! is_alive "$pid"; then
    return 0
  fi
  log "waiting for $label pid=$pid"
  while is_alive "$pid"; do
    sleep 60
  done
  log "$label pid=$pid finished"
}

llama_ready() {
  [[ -s "$LLAMA_DIR/config.json" ]] || return 1
  [[ -s "$LLAMA_DIR/model.safetensors.index.json" ]] || return 1
  [[ -s "$LLAMA_DIR/tokenizer.json" ]] || return 1
  [[ -s "$LLAMA_DIR/tokenizer_config.json" ]] || return 1
  [[ -s "$LLAMA_DIR/model-00001-of-00004.safetensors" ]] || return 1
  [[ -s "$LLAMA_DIR/model-00002-of-00004.safetensors" ]] || return 1
  [[ -s "$LLAMA_DIR/model-00003-of-00004.safetensors" ]] || return 1
  [[ -s "$LLAMA_DIR/model-00004-of-00004.safetensors" ]] || return 1
}

wait_for_llama() {
  if llama_ready; then
    log "Llama weights already complete"
    return 0
  fi
  log "waiting for Llama weights in $LLAMA_DIR"
  while ! llama_ready; do
    if [[ -s "$LLAMA_PID_FILE" ]]; then
      local pid
      pid="$(tr -d '[:space:]' < "$LLAMA_PID_FILE")"
      if ! is_alive "$pid"; then
        log "Llama download pid=$pid is not alive and weights are incomplete"
        return 1
      fi
    fi
    sleep 60
  done
  log "Llama weights complete"
}

health_ok() {
  python - "$@" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

start_english_service() {
  if health_ok "http://127.0.0.1:8897/health"; then
    log "English FAISS service already healthy on 8897"
    echo ""
    return 0
  fi
  log "starting English FAISS service on GPU0 port 8897"
  scripts/launch_model_eval_20260821.sh english-8897 \
    > "$RUN_CONTROL/english_8897_gpu0.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_CONTROL/english_8897_gpu0.pid"
  for _ in $(seq 1 120); do
    if health_ok "http://127.0.0.1:8897/health"; then
      log "English FAISS service healthy pid=$pid"
      echo "$pid"
      return 0
    fi
    if ! is_alive "$pid"; then
      log "English FAISS service exited before becoming healthy"
      return 1
    fi
    sleep 5
  done
  log "English FAISS service did not become healthy in time"
  return 1
}

stop_pid() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && is_alive "$pid"; then
    log "stopping pid=$pid"
    kill "$pid" || true
    sleep 5
    if is_alive "$pid"; then
      kill -TERM "$pid" || true
    fi
  fi
}

run_stage() {
  local label="$1"
  shift
  local stage_log="$RUN_CONTROL/${label}.log"
  log "start $label: $*"
  "$@" > "$stage_log" 2>&1
  log "done $label"
}

refresh_report() {
  local report_log="$RUN_CONTROL/refresh_two_model_summary.log"
  log "refreshing two-model summary"
  python scripts/summarize_two_model_eval_20260821.py >> "$report_log" 2>&1
  log "two-model summary refreshed"
}

coverage_json() {
  local model_root="$1"
  local method="$2"
  python - "$INPUT_DIR" "$model_root" "$method" <<'PY'
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
model_root = Path(sys.argv[2])
method_label = sys.argv[3]

def iter_jsonl(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def split_eval_key(value):
    parts = str(value or "").split("|")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], "|".join(parts[3:])
    if len(parts) == 3:
        return None, parts[0], parts[1], parts[2]
    return None, None, None, None

def target_key(row):
    _, key_dataset, key_language, key_sample = split_eval_key(row.get("eval_key"))
    dataset = str(row.get("_dataset") or row.get("dataset") or key_dataset or "").strip()
    language = str(row.get("language") or row.get("_language") or key_language or "").strip()
    sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or key_sample or "").strip()
    if not dataset or not language or not sample_id:
        return None
    return dataset, language, sample_id

def matches(row):
    method = str(row.get("method") or split_eval_key(row.get("eval_key"))[0] or "")
    if method_label == "zero_shot":
        return method == "zero_shot"
    if method_label == "tCRAG":
        return method in {"trag", "tcrag", "tCRAG"}
    if method_label == "CORAL-Wikipag":
        return method == "coral_wikipag"
    if method_label == "CA":
        return method.startswith("ca_concept_bank")
    raise SystemExit(f"unknown method: {method_label}")

targets = {}
for path in sorted(input_dir.glob("*.jsonl")):
    for row in iter_jsonl(path):
        key = target_key(row)
        if key:
            targets[key] = row

if method_label == "CA":
    paths = [model_root / "ca_shard0" / "predictions.jsonl", model_root / "ca_shard1" / "predictions.jsonl"]
else:
    paths = [model_root / "baselines_shard0" / "predictions.jsonl", model_root / "baselines_shard1" / "predictions.jsonl"]

present = set()
errors = 0
for path in paths:
    for row in iter_jsonl(path):
        if not matches(row):
            continue
        if row.get("error"):
            errors += 1
            continue
        key = target_key(row)
        if key in targets:
            present.add(key)

payload = {
    "model_root": str(model_root),
    "method": method_label,
    "target": len(targets),
    "present": len(present),
    "missing": max(len(targets) - len(present), 0),
    "coverage": len(present) / len(targets) if targets else 0.0,
    "error_rows": errors,
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if payload["missing"] == 0 else 1)
PY
}

method_complete() {
  local model_root="$1"
  local method="$2"
  local status
  if status="$(coverage_json "$model_root" "$method")"; then
    log "coverage complete: $status"
    return 0
  fi
  log "coverage incomplete: $status"
  return 1
}

record_method_status() {
  local model_root="$1"
  local method="$2"
  if ! method_complete "$model_root" "$method"; then
    log "WARNING: $method is still incomplete under $model_root; continuing queue so other methods can run"
  fi
}

log "GPU0 full queue started"

wait_for_pid_file "$MINISTRAL_RC/ministral_coral_gpu0_sequence.pid" "existing Ministral CORAL GPU0 sequence"

if ! method_complete "$MINISTRAL_OUT" "CORAL-Wikipag"; then
  run_stage ministral_coral_repair_shard0 scripts/launch_model_eval_20260821.sh ministral-coral-shard0
  run_stage ministral_coral_repair_shard1 scripts/launch_model_eval_20260821.sh ministral-coral-shard1
  record_method_status "$MINISTRAL_OUT" "CORAL-Wikipag"
fi
refresh_report

run_stage ministral_ca_shard0 scripts/launch_model_eval_20260821.sh ministral-ca-shard0
run_stage ministral_ca_shard1 scripts/launch_model_eval_20260821.sh ministral-ca-shard1
record_method_status "$MINISTRAL_OUT" "CA"
refresh_report

wait_for_llama

english_pid="$(start_english_service)"
trap 'stop_pid "${english_pid:-}"' EXIT
run_stage llama_base_shard0 scripts/launch_model_eval_20260821.sh llama-base-shard0
run_stage llama_base_shard1 scripts/launch_model_eval_20260821.sh llama-base-shard1
record_method_status "$LLAMA_OUT" "zero_shot"
record_method_status "$LLAMA_OUT" "tCRAG"
refresh_report
stop_pid "${english_pid:-}"
english_pid=""
trap - EXIT

run_stage llama_coral_shard0 scripts/launch_model_eval_20260821.sh llama-coral-shard0
run_stage llama_coral_shard1 scripts/launch_model_eval_20260821.sh llama-coral-shard1
record_method_status "$LLAMA_OUT" "CORAL-Wikipag"
refresh_report

run_stage llama_ca_shard0 scripts/launch_model_eval_20260821.sh llama-ca-shard0
run_stage llama_ca_shard1 scripts/launch_model_eval_20260821.sh llama-ca-shard1
record_method_status "$LLAMA_OUT" "CA"
refresh_report

log "GPU0 full queue complete"
