#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

RUN_TAG="${RUN_TAG:-Phase1_validate_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-logs/RecoveryGen/Phase1_validation}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${BASE_DIR}/server_logs/${RUN_TAG}}"
SERVER_SIG_FILE="${SERVER_SIG_FILE:-${BASE_DIR}/current_server_sig.json}"

mkdir -p "${SERVER_LOG_DIR}" "$(dirname "${SERVER_SIG_FILE}")"

# Old phase1 validation defaults (override by env if needed).
export VLLM_RECOVERY_OBS="${VLLM_RECOVERY_OBS:-1}"
export VLLM_RECOVERY_PHASE="${VLLM_RECOVERY_PHASE:-1}"
export VLLM_RECOVERY_V2="${VLLM_RECOVERY_V2:-0}"
export VLLM_RECOVERY_BUDGET="${VLLM_RECOVERY_BUDGET:-16}"
export VLLM_RECOVERY_TS_PERIOD_MS="${VLLM_RECOVERY_TS_PERIOD_MS:-50}"
export VLLM_RECOVERY_LOG_DIR="${VLLM_RECOVERY_LOG_DIR:-${SERVER_LOG_DIR}/recovery}"

export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export SERVED="${SERVED:-Qwen2.5-7B-Instruct}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export READY_HOST="${READY_HOST:-127.0.0.1}"

export MEM="${MEM:-0.75}"
export MAXLEN="${MAXLEN:-15000}"
export MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export PMODE="${PMODE:-recompute}"
export SWAP_SPACE_GB="${SWAP_SPACE_GB:-0}"
export NUM_SCHED_STEPS="${NUM_SCHED_STEPS:-1}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export CFG_TAG="${CFG_TAG:-phase1_obs_m2}"

echo "[INFO] Phase1 validation server config:"
echo "  RUN_TAG=${RUN_TAG}"
echo "  BASE_DIR=${BASE_DIR}"
echo "  SERVER_LOG_DIR=${SERVER_LOG_DIR}"
echo "  SERVER_SIG_FILE=${SERVER_SIG_FILE}"
echo "  VLLM_RECOVERY_LOG_DIR=${VLLM_RECOVERY_LOG_DIR}"
echo "  VLLM_RECOVERY_PHASE=${VLLM_RECOVERY_PHASE}"
echo "  VLLM_RECOVERY_BUDGET=${VLLM_RECOVERY_BUDGET}"
echo "  MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS}"
echo "  PMODE=${PMODE}"
echo "  ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL}"

exec env \
  BASE_DIR="${BASE_DIR}" \
  SERVER_LOG_DIR="${SERVER_LOG_DIR}" \
  SERVER_SIG_FILE="${SERVER_SIG_FILE}" \
  MODEL="${MODEL}" \
  SERVED="${SERVED}" \
  HOST="${HOST}" \
  PORT="${PORT}" \
  READY_HOST="${READY_HOST}" \
  MEM="${MEM}" \
  MAXLEN="${MAXLEN}" \
  MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PMODE="${PMODE}" \
  SWAP_SPACE_GB="${SWAP_SPACE_GB}" \
  NUM_SCHED_STEPS="${NUM_SCHED_STEPS}" \
  ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}" \
  CFG_TAG="${CFG_TAG}" \
  bash "${SCRIPT_DIR}/baseline_chunked_server.sh"
