#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_ROOT="artifacts/model_eval_20260821/inputs/full_low_resource_47815"
MULTILINGUAL_ROOT="data/external/wikipedia-multilingual-20231101-qwen3-embedding-4b/full_20231101"
EMBEDDING_MODEL="data/external/models/Qwen3-Embedding-4B"
MINISTRAL_MODEL="data/external/models/Ministral-3-8B-Instruct-2512-BF16"
LLAMA_MODEL="data/external/models/Llama-3.1-8B-Instruct"
CARD_BANK="artifacts/ca_mem/full_oracle_plus_wikipag_clean_57subjects_target1000_gpt54_one_card_v1/combined_oracle_plus_clean_57subjects_target1000_gpt54/banks/concept_card_qwen3_1024"

MINISTRAL_OUT="artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16"
LLAMA_OUT="artifacts/model_eval_20260821/llama3_1_8b_instruct"

common_rag_args=(
  --config configs/global_mmlu_low_resource_completion.yaml
  --retrieval-mode faithful
  --multilingual-root "$MULTILINGUAL_ROOT"
  --candidate-top-k 50
  --top-k 50
  --final-top-k 5
  --generation-top-k 5
  --live-model
  --local-vllm
  --local-max-model-len 16384
  --local-gpu-memory-utilization 0.72
  --local-batch-size 64
  --local-batch-timeout-ms 25
  --batch-max-tokens 8192
  --answer-max-tokens 32
  --drag-icl-answer-max-tokens 32
  --max-concurrent-model-requests 64
  --allow-full-run
  --resume
)

coral_args=(
  --methods coral_wikipag
  --coral-retrieval-source wikipag
  --coral-language-pool bn sw te ne hi
  --coral-max-corpora 1
  --coral-top-k-per-corpus 8
  --coral-final-top-k 5
  --coral-max-rounds 1
  --coral-batch-critic
  --coral-translate-query
  --coral-dual-query-retrieval
  --coral-adaptive-direct-gate
  --max-model-calls 250000
)

base_args=(
  --methods zero_shot trag
  --max-model-calls 150000
)

ca_common_args=(
  --bank-dir "$CARD_BANK"
  --embedding-model-dir "$EMBEDDING_MODEL"
  --embedding-device cuda
  --local-max-model-len 16384
  --local-gpu-memory-utilization 0.72
  --local-batch-size 64
  --local-batch-timeout-ms 25
  --max-concurrent-model-requests 64
  --max-model-calls 250000
  --candidate-top-k 20
  --top-k 5
  --answer-max-tokens 32
  --progress-every 200
  --quiet-items
  --resume
)

run_english_service() {
  local gpu="$1"
  local port="$2"
  exec env CUDA_VISIBLE_DEVICES="$gpu" python scripts/wiki_faiss/serve_sherlock_wiki.py \
    --host 127.0.0.1 \
    --port "$port" \
    --index hnsw_sq.faiss \
    --device cuda
}

run_llama_base() {
  local shard="$1"
  local gpu="$2"
  local port="$3"
  exec env CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_multilingual_rag_baselines.py \
    "${common_rag_args[@]}" \
    "${base_args[@]}" \
    --input-jsonl "$INPUT_ROOT/shard${shard}.jsonl" \
    --output-dir "$LLAMA_OUT/baselines_shard${shard}" \
    --english-service-url "http://127.0.0.1:${port}" \
    --local-model-dir "$LLAMA_MODEL" \
    --local-model-name Llama-3.1-8B-Instruct
}

run_llama_coral() {
  local shard="$1"
  local gpu="$2"
  exec env CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_multilingual_rag_baselines.py \
    "${common_rag_args[@]}" \
    "${coral_args[@]}" \
    --input-jsonl "$INPUT_ROOT/shard${shard}.jsonl" \
    --output-dir "$LLAMA_OUT/baselines_shard${shard}" \
    --local-model-dir "$LLAMA_MODEL" \
    --local-model-name Llama-3.1-8B-Instruct
}

run_ministral_coral() {
  local shard="$1"
  local gpu="$2"
  exec env CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_multilingual_rag_baselines.py \
    "${common_rag_args[@]}" \
    "${coral_args[@]}" \
    --input-jsonl "$INPUT_ROOT/shard${shard}.jsonl" \
    --output-dir "$MINISTRAL_OUT/baselines_shard${shard}" \
    --local-model-dir "$MINISTRAL_MODEL" \
    --local-model-name Ministral-3-8B-Instruct-2512-BF16
}

run_ca() {
  local model_dir="$1"
  local output_root="$2"
  local model_name="$3"
  local shard="$4"
  local gpu="$5"
  local rows="$6"
  exec env CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_ca_concept_bank_smoke.py \
    --input-jsonl "$INPUT_ROOT/shard${shard}.jsonl" \
    --output-dir "$output_root/ca_shard${shard}" \
    --local-model-dir "$model_dir" \
    "${ca_common_args[@]}" \
    --max-rows "$rows"
}

case "${1:-}" in
  english-8897)
    run_english_service 0 8897
    ;;
  english-8898)
    run_english_service 0 8898
    ;;
  llama-base-shard0)
    run_llama_base 0 0 8897
    ;;
  llama-base-shard1)
    run_llama_base 1 0 8897
    ;;
  llama-coral-shard0)
    run_llama_coral 0 0
    ;;
  llama-coral-shard1)
    run_llama_coral 1 0
    ;;
  ministral-coral-shard0)
    run_ministral_coral 0 0
    ;;
  ministral-coral-shard1)
    run_ministral_coral 1 0
    ;;
  ministral-ca-shard0)
    run_ca "$MINISTRAL_MODEL" "$MINISTRAL_OUT" Ministral-3-8B-Instruct-2512-BF16 0 0 23908
    ;;
  ministral-ca-shard1)
    run_ca "$MINISTRAL_MODEL" "$MINISTRAL_OUT" Ministral-3-8B-Instruct-2512-BF16 1 0 23907
    ;;
  llama-ca-shard0)
    run_ca "$LLAMA_MODEL" "$LLAMA_OUT" Llama-3.1-8B-Instruct 0 0 23908
    ;;
  llama-ca-shard1)
    run_ca "$LLAMA_MODEL" "$LLAMA_OUT" Llama-3.1-8B-Instruct 1 0 23907
    ;;
  *)
    cat <<'USAGE'
Usage: scripts/launch_model_eval_20260821.sh <command>

Commands:
  english-8897          Start Sherlock English FAISS service on GPU 0 / port 8897.
  english-8898          Start Sherlock English FAISS service on GPU 0 / port 8898.
  llama-base-shard0     Run Llama zero-shot + tCRAG on shard0. Requires english-8897.
  llama-base-shard1     Run Llama zero-shot + tCRAG on shard1 using GPU 0. Requires english-8897.
  llama-coral-shard0    Run Llama CORAL-Wikipag on shard0.
  llama-coral-shard1    Run Llama CORAL-Wikipag on shard1.
  ministral-coral-shard0 Run Ministral CORAL-Wikipag on shard0 using GPU 0.
  ministral-coral-shard1 Run Ministral CORAL-Wikipag on shard1 using GPU 0.
  ministral-ca-shard0   Run Ministral CA on shard0.
  ministral-ca-shard1   Run Ministral CA on shard1.
  llama-ca-shard0       Run Llama CA on shard0.
  llama-ca-shard1       Run Llama CA on shard1.
USAGE
    exit 2
    ;;
esac
