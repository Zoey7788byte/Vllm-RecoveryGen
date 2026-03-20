#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd /home/ad/zteng/6022/vllm

# ---- HF offline guard (avoid optional template path 404 on Hub) ----
HF_FORCE_ONLINE="${HF_FORCE_ONLINE:-0}"
if [[ "${HF_FORCE_ONLINE}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_ENABLE_HF_TRANSFER=0
  echo "[INFO] HF offline enabled."
else
  unset HF_HUB_OFFLINE || true
  unset TRANSFORMERS_OFFLINE || true
  echo "[INFO] HF online mode: HF_FORCE_ONLINE=1"
fi

# ============================================================
# Server for sweep trace replay (BurstGPT trace)
# - Keep server MAXLEN fixed
# - Client will filter trace Total tokens <= MAX_TOTAL_TOKENS
# ============================================================

BASE_DIR="${BASE_DIR:-logs/RecoveryGen/Mot2_tail_instability}"
mkdir -p "${BASE_DIR}/server_logs" "${BASE_DIR}/summary"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED="${SERVED:-Qwen2.5-7B-Instruct}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

MEM="${MEM:-0.85}"
MAXLEN="${MAXLEN:-15000}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16384}"  # MUST be >= MAXLEN
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

PMODE="${PMODE:-recompute}"
SWAP_SPACE_GB="${SWAP_SPACE_GB:-0}"
CFG_TAG="${CFG_TAG:-swap${SWAP_SPACE_GB}}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"  # 15min
READY_POLL_S="${READY_POLL_S:-1}"          # poll interval
READY_PRINT_EVERY_S="${READY_PRINT_EVERY_S:-10}"  # status print cadence

# -----------------------------
# guards
# -----------------------------
if awk "BEGIN{exit !(${MEM} < 0.70)}"; then
  echo "[WARN] MEM=${MEM} < 0.70 is not allowed on this machine. Auto bump to 0.70"
  MEM="0.70"
fi

if [[ "${MAX_BATCH_TOKENS}" -lt "${MAXLEN}" ]]; then
  echo "[WARN] MAX_BATCH_TOKENS(${MAX_BATCH_TOKENS}) < MAXLEN(${MAXLEN}), auto bump to ${MAXLEN}"
  MAX_BATCH_TOKENS="${MAXLEN}"
fi

# -----------------------------
# GPU selection: respect explicit CUDA_VISIBLE_DEVICES
# -----------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] Respect user CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="1"
  echo "[INFO] CUDA_VISIBLE_DEVICES not set; default to ${CUDA_VISIBLE_DEVICES}"
fi

# --- fix: avoid picking /usr/local/cuda/lib64/libnvJitLink.so.12 ---
unset CUDA_HOME CUDA_PATH

# 如果你的 LD_LIBRARY_PATH 里含 /usr/local/cuda/lib64，把它移除
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda/lib64$' | paste -sd: -)"
fi

# 把 conda 的 lib 放到最前（保证优先级）
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# （可选）也把 torch 自带库路径放前面，进一步稳
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"

echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# -----------------------------
# sanity checks (torch + cuda)
# -----------------------------
python - <<'PY'
import os, sys
try:
    import torch
    print("[OK] torch import:", torch.__version__, "cuda:", torch.version.cuda)
    print("[OK] cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
    if torch.cuda.device_count() <= 0:
        print("[FATAL] torch sees 0 CUDA devices. Check CUDA_VISIBLE_DEVICES and driver.", file=sys.stderr)
        sys.exit(2)
except Exception as e:
    print("[FATAL] torch import/cuda failed:", repr(e), file=sys.stderr)
    sys.exit(2)
PY

TAG="server_sweep_trace_cfg=${CFG_TAG}_model=${SERVED}_mem=${MEM}_maxlen=${MAXLEN}_mbt=${MAX_BATCH_TOKENS}_mseq=${MAX_NUM_SEQS}_pmode=${PMODE}_swapgb=${SWAP_SPACE_GB}_cvis=${CUDA_VISIBLE_DEVICES}"
LOG_FILE="${BASE_DIR}/server_logs/${TAG}.log"
PID_FILE="${BASE_DIR}/server_logs/${TAG}.pid"
CFG_FILE="${BASE_DIR}/summary/${TAG}.server_config.json"

# -----------------------------
# port check (non-fatal if lsof missing)
# -----------------------------
if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[ERROR] Port ${PORT} is already in use. Stop the existing server first."
    exit 1
  fi
else
  echo "[WARN] lsof not found; skip port-in-use check."
fi

# -----------------------------
# write config snapshot
# -----------------------------
{
  echo "{"
  echo "  \"step\": \"server_sweep_trace\","
  echo "  \"cfg_tag\": \"${CFG_TAG}\","
  echo "  \"model\": \"${MODEL}\","
  echo "  \"served_model_name\": \"${SERVED}\","
  echo "  \"host\": \"${HOST}\","
  echo "  \"port\": ${PORT},"
  echo "  \"gpu_memory_utilization\": ${MEM},"
  echo "  \"max_model_len\": ${MAXLEN},"
  echo "  \"max_num_batched_tokens\": ${MAX_BATCH_TOKENS},"
  echo "  \"max_num_seqs\": ${MAX_NUM_SEQS},"
  echo "  \"preemption_mode\": \"${PMODE}\","
  echo "  \"swap_space_gb_assumed\": ${SWAP_SPACE_GB},"
  echo "  \"cuda_visible_devices\": \"${CUDA_VISIBLE_DEVICES}\","
  echo "  \"hf_home\": \"${HF_HOME}\","
  echo "  \"ready_timeout_s\": ${READY_TIMEOUT_S},"
  echo "  \"client_max_total_tokens_assumed\": $((MAXLEN-500)),"
  echo "  \"client_trace_window_duration_s_assumed\": 600,"
  echo "  \"time\": \"$(date -Is)\""
  echo "}"
} > "${CFG_FILE}"

echo "[INFO] Starting vLLM server..."
echo "[INFO] TAG:  ${TAG}"
echo "[INFO] LOG:  ${LOG_FILE}"
echo "[INFO] CFG:  ${CFG_FILE}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host "${HOST}" --port "${PORT}" \
  --device cuda \
  --dtype half \
  --gpu-memory-utilization "${MEM}" \
  --max-model-len "${MAXLEN}" \
  --max-num-batched-tokens "${MAX_BATCH_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --preemption-mode "${PMODE}" \
  --swap-space "${SWAP_SPACE_GB}" \
  --disable-log-requests \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[INFO] PID: $(cat "${PID_FILE}")"

# -----------------------------
# readiness check (quiet): probe /v1/completions
# - Do NOT print curl connection errors
# - Print status every READY_PRINT_EVERY_S seconds
# - Exit early if server process dies
# -----------------------------
if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl not found; cannot do readiness check."
  exit 1
fi

echo "[INFO] Waiting for engine readiness via /v1/completions (timeout ${READY_TIMEOUT_S}s) ..."

READY_TMP="${BASE_DIR}/server_logs/${TAG}.ready_probe.json"
pid="$(cat "${PID_FILE}")"
last_print=0

for ((t=1; t<=READY_TIMEOUT_S; t+=READY_POLL_S)); do
  # If process died, stop waiting immediately
  if ! ps -p "${pid}" >/dev/null 2>&1; then
    echo "[ERROR] server process (pid=${pid}) exited early. Check ${LOG_FILE}"
    exit 1
  fi

  # http_code 200 means engine is ready.
  # http_code 000 means connection failure (port not up yet) or other transport error.
  code="$(curl -sS --connect-timeout 1 --max-time 2 \
    -o "${READY_TMP}" -w "%{http_code}" \
    "http://${HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" \
    2>/dev/null || echo "000")"

  if [[ "${code}" == "200" ]]; then
    echo "[INFO] Server is up and engine is ready."
    exit 0
  fi

  # Print compact status periodically (no curl stderr spam)
  if (( t - last_print >= READY_PRINT_EVERY_S )); then
    last_print=$t
    if [[ "${code}" == "000" ]]; then
      echo "[INFO] waiting... t=${t}s http=000 (port not ready yet)"
    else
      echo "[INFO] waiting... t=${t}s http=${code}"
      head -c 200 "${READY_TMP}" 2>/dev/null || true
      echo
    fi
  fi

  sleep "${READY_POLL_S}"
done

echo "[ERROR] Engine not ready in ${READY_TIMEOUT_S}s. Check ${LOG_FILE}"
echo "[INFO] Use this model name in client payload: ${SERVED}"

exit 1
