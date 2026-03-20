#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd /home/ad/zteng/6022/vllm

# ---- avoid proxy interference ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true

BASE_DIR="${BASE_DIR:-logs/RecoveryGen/Phase2_validation}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${BASE_DIR}/server_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${SERVER_LOG_DIR}"

# Recovery observability + Phase2 knobs
export VLLM_RECOVERY_OBS="${VLLM_RECOVERY_OBS:-1}"
export VLLM_RECOVERY_PHASE="${VLLM_RECOVERY_PHASE:-2}"
# P2 enable switch: budget=1 enables BudgetController + TaskSelector semantics.
export VLLM_RECOVERY_BUDGET="${VLLM_RECOVERY_BUDGET:-1}"
export VLLM_RECOVERY_TS_PERIOD_MS="${VLLM_RECOVERY_TS_PERIOD_MS:-50}"
export VLLM_RECOVERY_SWAP_MS_PER_BLOCK="${VLLM_RECOVERY_SWAP_MS_PER_BLOCK:-0.2}"
export VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK="${VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK:-0}"
export REC_T_CYC_MS="${REC_T_CYC_MS:-80}"
export REC_BUDGET_INIT_MS="${REC_BUDGET_INIT_MS:-5}"
export REC_BUDGET_MIN_MS="${REC_BUDGET_MIN_MS:-1}"
export REC_DPLUS_MS="${REC_DPLUS_MS:-3}"
export REC_DMINUS_MS="${REC_DMINUS_MS:-8}"
export REC_STALL_MS="${REC_STALL_MS:-300}"
export REC_PREFETCH_MAX_BLOCKS="${REC_PREFETCH_MAX_BLOCKS:-2048}"
export REC_MODE_STABLE_WINDOW_MS="${REC_MODE_STABLE_WINDOW_MS:-120}"
export REC_MWS_PREFIX_TOKENS="${REC_MWS_PREFIX_TOKENS:-256}"
export REC_MWS_RECENT_TOKENS="${REC_MWS_RECENT_TOKENS:-1024}"
export REC_MWS_ADMIT_RHO="${REC_MWS_ADMIT_RHO:-0.25}"
export REC_FALLBACK_STALL_MS="${REC_FALLBACK_STALL_MS:-600}"
export REC_FALLBACK_RESIDENCE_MS="${REC_FALLBACK_RESIDENCE_MS:-300}"
export REC_FALLBACK_PREEMPT_THRESHOLD="${REC_FALLBACK_PREEMPT_THRESHOLD:-3}"
export REC_FALLBACK_PROTECT_PRIORITY_GTE="${REC_FALLBACK_PROTECT_PRIORITY_GTE:--1}"
export REC_FALLBACK_PAUSE_DECODE="${REC_FALLBACK_PAUSE_DECODE:-0}"
export REC_FALLBACK_DECODE_INTERVAL_CYCLES="${REC_FALLBACK_DECODE_INTERVAL_CYCLES:-1}"
export REC_FALLBACK_BUDGET_BOOST_MS="${REC_FALLBACK_BUDGET_BOOST_MS:-0.5}"
export VLLM_RECOVERY_LOG_DIR="${VLLM_RECOVERY_LOG_DIR:-${SERVER_LOG_DIR}/recovery}"
mkdir -p "${VLLM_RECOVERY_LOG_DIR}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED="${SERVED:-Qwen2.5-7B-Instruct}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
READY_HOST="${READY_HOST:-127.0.0.1}"

MEM="${MEM:-0.75}"
MAXLEN="${MAXLEN:-15000}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

PMODE="${PMODE:-swap}"               # recompute | swap
SWAP_SPACE_GB="${SWAP_SPACE_GB:-8}"  # >0 for swap path
NUM_SCHED_STEPS="${NUM_SCHED_STEPS:-1}"

ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
CFG_TAG="${CFG_TAG:-phase2_m5_eval}"

# -----------------------------
# HF offline guard
# -----------------------------
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

case "${ENABLE_CHUNKED_PREFILL}" in
  1|true|TRUE|yes|YES) ENABLE_CHUNKED_PREFILL="1" ;;
  0|false|FALSE|no|NO|"") ENABLE_CHUNKED_PREFILL="0" ;;
  *) echo "[WARN] ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL} not recognized, treat as 0"; ENABLE_CHUNKED_PREFILL="0" ;;
esac

if ! [[ "${NUM_SCHED_STEPS}" =~ ^[0-9]+$ ]] || [[ "${NUM_SCHED_STEPS}" -lt 1 ]]; then
  echo "[WARN] NUM_SCHED_STEPS=${NUM_SCHED_STEPS} invalid, set to 1"
  NUM_SCHED_STEPS="1"
fi

if [[ "${ENABLE_CHUNKED_PREFILL}" == "0" ]]; then
  if [[ "${MAX_BATCH_TOKENS}" -lt "${MAXLEN}" ]]; then
    echo "[WARN] chunked_prefill disabled: MAX_BATCH_TOKENS(${MAX_BATCH_TOKENS}) < MAXLEN(${MAXLEN}), bump to ${MAXLEN}"
    MAX_BATCH_TOKENS="${MAXLEN}"
  fi
fi

# -----------------------------
# GPU selection
# -----------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] Respect user CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="2"
  echo "[INFO] CUDA_VISIBLE_DEVICES not set; default to ${CUDA_VISIBLE_DEVICES}"
fi

# ---- CUDA lib conflict guard ----
unset CUDA_HOME CUDA_PATH
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda/lib64$' | paste -sd: -)"
fi
CUDA_SYS_LIB="/usr/local/cuda/targets/x86_64-linux/lib"
CONDA_NVJITLINK_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="${CONDA_NVJITLINK_LIB}:${CUDA_SYS_LIB}:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

python - <<'PY'
import sys
try:
    import torch
    print("[OK] torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("[OK] cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
    if torch.cuda.device_count() <= 0:
        print("[FATAL] torch sees 0 CUDA devices.", file=sys.stderr)
        sys.exit(2)
except Exception as e:
    print("[FATAL] torch import/cuda failed:", repr(e), file=sys.stderr)
    sys.exit(2)
PY

# -----------------------------
# port check (avoid binding to old server)
# -----------------------------
if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[ERROR] Port ${PORT} already in use. Stop existing server first."
    exit 1
  fi
fi

TAG="phase2_cfg=${CFG_TAG}_served=${SERVED}_mem=${MEM}_maxlen=${MAXLEN}_mbt=${MAX_BATCH_TOKENS}_mseq=${MAX_NUM_SEQS}_pmode=${PMODE}_swap=${SWAP_SPACE_GB}_chunk=${ENABLE_CHUNKED_PREFILL}_ss=${NUM_SCHED_STEPS}_rbudget=${VLLM_RECOVERY_BUDGET}_tcyc=${REC_T_CYC_MS}_binit=${REC_BUDGET_INIT_MS}_bmin=${REC_BUDGET_MIN_MS}_dplus=${REC_DPLUS_MS}_dminus=${REC_DMINUS_MS}_stall=${REC_STALL_MS}_pfmax=${REC_PREFETCH_MAX_BLOCKS}_cvis=${CUDA_VISIBLE_DEVICES}_port=${PORT}"
LOG_FILE="${SERVER_LOG_DIR}/server.log"
PID_FILE="${SERVER_LOG_DIR}/server.pid"
CFG_FILE="${SERVER_LOG_DIR}/server_config.json"
SERVER_SIG_FILE="${SERVER_SIG_FILE:-${BASE_DIR}/current_server_sig.json}"
SERVER_SIG_STR="pmode=${PMODE}_chunk=${ENABLE_CHUNKED_PREFILL}_mbt=${MAX_BATCH_TOKENS}_mseq=${MAX_NUM_SEQS}_ss=${NUM_SCHED_STEPS}_mem=${MEM}_maxlen=${MAXLEN}_swap=${SWAP_SPACE_GB}_rbudget=${VLLM_RECOVERY_BUDGET}_tcyc=${REC_T_CYC_MS}_binit=${REC_BUDGET_INIT_MS}_bmin=${REC_BUDGET_MIN_MS}_dplus=${REC_DPLUS_MS}_dminus=${REC_DMINUS_MS}_stall=${REC_STALL_MS}_pfmax=${REC_PREFETCH_MAX_BLOCKS}_msw=${REC_MODE_STABLE_WINDOW_MS}_mwsA=${REC_MWS_PREFIX_TOKENS}_mwsW=${REC_MWS_RECENT_TOKENS}_mwsR=${REC_MWS_ADMIT_RHO}_fbstall=${REC_FALLBACK_STALL_MS}_fbr=${REC_FALLBACK_RESIDENCE_MS}_fbpre=${REC_FALLBACK_PREEMPT_THRESHOLD}_fbprot=${REC_FALLBACK_PROTECT_PRIORITY_GTE}_fbpause=${REC_FALLBACK_PAUSE_DECODE}_fbint=${REC_FALLBACK_DECODE_INTERVAL_CYCLES}_fbboost=${REC_FALLBACK_BUDGET_BOOST_MS}_cvis=${CUDA_VISIBLE_DEVICES}_port=${PORT}"

cat > "${CFG_FILE}" <<JSON
{
  "tag": "${TAG}",
  "cfg_tag": "${CFG_TAG}",
  "model_repo": "${MODEL}",
  "served_model_name": "${SERVED}",
  "host": "${HOST}",
  "port": ${PORT},
  "ready_host": "${READY_HOST}",
  "gpu_memory_utilization": ${MEM},
  "max_model_len": ${MAXLEN},
  "max_num_batched_tokens": ${MAX_BATCH_TOKENS},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "preemption_mode": "${PMODE}",
  "swap_space_gb": ${SWAP_SPACE_GB},
  "enable_chunked_prefill": ${ENABLE_CHUNKED_PREFILL},
  "num_scheduler_steps": ${NUM_SCHED_STEPS},
  "recovery_budget": ${VLLM_RECOVERY_BUDGET},
  "recovery_swap_ms_per_block": ${VLLM_RECOVERY_SWAP_MS_PER_BLOCK},
  "recovery_swap_bytes_per_block": ${VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK},
  "rec_t_cyc_ms": ${REC_T_CYC_MS},
  "rec_budget_init_ms": ${REC_BUDGET_INIT_MS},
  "rec_budget_min_ms": ${REC_BUDGET_MIN_MS},
  "rec_dplus_ms": ${REC_DPLUS_MS},
  "rec_dminus_ms": ${REC_DMINUS_MS},
  "rec_stall_ms": ${REC_STALL_MS},
  "rec_prefetch_max_blocks": ${REC_PREFETCH_MAX_BLOCKS},
  "rec_mode_stable_window_ms": ${REC_MODE_STABLE_WINDOW_MS},
  "rec_mws_prefix_tokens": ${REC_MWS_PREFIX_TOKENS},
  "rec_mws_recent_tokens": ${REC_MWS_RECENT_TOKENS},
  "rec_mws_admit_rho": ${REC_MWS_ADMIT_RHO},
  "rec_fallback_stall_ms": ${REC_FALLBACK_STALL_MS},
  "rec_fallback_residence_ms": ${REC_FALLBACK_RESIDENCE_MS},
  "rec_fallback_preempt_threshold": ${REC_FALLBACK_PREEMPT_THRESHOLD},
  "rec_fallback_protect_priority_gte": ${REC_FALLBACK_PROTECT_PRIORITY_GTE},
  "rec_fallback_pause_decode": ${REC_FALLBACK_PAUSE_DECODE},
  "rec_fallback_decode_interval_cycles": ${REC_FALLBACK_DECODE_INTERVAL_CYCLES},
  "rec_fallback_budget_boost_ms": ${REC_FALLBACK_BUDGET_BOOST_MS},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "time": "$(date -Is)"
}
JSON

CMD=(python -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name "${SERVED}"
  --host "${HOST}" --port "${PORT}"
  --device cuda
  --dtype half
  --gpu-memory-utilization "${MEM}"
  --max-model-len "${MAXLEN}"
  --max-num-batched-tokens "${MAX_BATCH_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --preemption-mode "${PMODE}"
  --swap-space "${SWAP_SPACE_GB}"
  --num-scheduler-steps "${NUM_SCHED_STEPS}"
  --disable-log-requests
  --trust-remote-code
)
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  CMD+=(--enable-chunked-prefill)
fi

{
  echo "[INFO] TAG=${TAG}"
  echo "[INFO] LOG_FILE=${LOG_FILE}"
  echo "[INFO] PID_FILE=${PID_FILE}"
  echo "[INFO] CFG_FILE=${CFG_FILE}"
  echo "[CMD] ${CMD[*]}"
} | tee -a "${LOG_FILE}"

nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[INFO] PID=$(cat "${PID_FILE}")"
echo "[INFO] tail -f ${LOG_FILE}"

echo "[INFO] Waiting readiness: GET /v1/models then POST /v1/completions (model=${SERVED}) ..."
pid="$(cat "${PID_FILE}")"
for i in $(seq 1 600); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[ERROR] Server process ${pid} exited before readiness. Check: ${LOG_FILE}"
    exit 1
  fi

  mcode="$(curl -sS --connect-timeout 1 --max-time 2 -o /tmp/vllm_models.json -w "%{http_code}" \
    "http://${READY_HOST}:${PORT}/v1/models" 2>/dev/null || echo "000")"
  ccode="$(curl -sS --connect-timeout 1 --max-time 2 -o /tmp/vllm_ready.json -w "%{http_code}" \
    "http://${READY_HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" \
    2>/dev/null || echo "000")"
  if [[ "${mcode}" == "200" && "${ccode}" == "200" ]]; then
    mkdir -p "$(dirname "${SERVER_SIG_FILE}")"
    {
      echo "{"
      echo "  \"server_sig_str\": \"${SERVER_SIG_STR}\","
      echo "  \"cfg_tag\": \"${CFG_TAG}\","
      echo "  \"pmode\": \"${PMODE}\","
      echo "  \"chunked\": ${ENABLE_CHUNKED_PREFILL},"
      echo "  \"mbt\": ${MAX_BATCH_TOKENS},"
      echo "  \"mseq\": ${MAX_NUM_SEQS},"
      echo "  \"ss\": ${NUM_SCHED_STEPS},"
      echo "  \"mem\": ${MEM},"
      echo "  \"maxlen\": ${MAXLEN},"
      echo "  \"swap_gb\": ${SWAP_SPACE_GB},"
      echo "  \"recovery_budget\": ${VLLM_RECOVERY_BUDGET},"
      echo "  \"recovery_swap_ms_per_block\": ${VLLM_RECOVERY_SWAP_MS_PER_BLOCK},"
      echo "  \"recovery_swap_bytes_per_block\": ${VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK},"
      echo "  \"rec_t_cyc_ms\": ${REC_T_CYC_MS},"
      echo "  \"rec_budget_init_ms\": ${REC_BUDGET_INIT_MS},"
      echo "  \"rec_budget_min_ms\": ${REC_BUDGET_MIN_MS},"
      echo "  \"rec_dplus_ms\": ${REC_DPLUS_MS},"
      echo "  \"rec_dminus_ms\": ${REC_DMINUS_MS},"
      echo "  \"rec_stall_ms\": ${REC_STALL_MS},"
      echo "  \"rec_prefetch_max_blocks\": ${REC_PREFETCH_MAX_BLOCKS},"
      echo "  \"rec_mode_stable_window_ms\": ${REC_MODE_STABLE_WINDOW_MS},"
      echo "  \"rec_mws_prefix_tokens\": ${REC_MWS_PREFIX_TOKENS},"
      echo "  \"rec_mws_recent_tokens\": ${REC_MWS_RECENT_TOKENS},"
      echo "  \"rec_mws_admit_rho\": ${REC_MWS_ADMIT_RHO},"
      echo "  \"rec_fallback_stall_ms\": ${REC_FALLBACK_STALL_MS},"
      echo "  \"rec_fallback_residence_ms\": ${REC_FALLBACK_RESIDENCE_MS},"
      echo "  \"rec_fallback_preempt_threshold\": ${REC_FALLBACK_PREEMPT_THRESHOLD},"
      echo "  \"rec_fallback_protect_priority_gte\": ${REC_FALLBACK_PROTECT_PRIORITY_GTE},"
      echo "  \"rec_fallback_pause_decode\": ${REC_FALLBACK_PAUSE_DECODE},"
      echo "  \"rec_fallback_decode_interval_cycles\": ${REC_FALLBACK_DECODE_INTERVAL_CYCLES},"
      echo "  \"rec_fallback_budget_boost_ms\": ${REC_FALLBACK_BUDGET_BOOST_MS},"
      echo "  \"cuda_visible_devices\": \"${CUDA_VISIBLE_DEVICES}\","
      echo "  \"host\": \"${HOST}\","
      echo "  \"phase2_log_dir\": \"${SERVER_LOG_DIR}\","
      echo "  \"port\": ${PORT},"
      echo "  \"pid\": ${pid},"
      echo "  \"time\": \"$(date -Is)\""
      echo "}"
    } > "${SERVER_SIG_FILE}"
    echo "[INFO] wrote server signature: ${SERVER_SIG_FILE}"
    echo "[INFO] server_sig_str=${SERVER_SIG_STR}"
    echo "[INFO] Server ready."
    exit 0
  fi
  if (( i % 10 == 0 )); then
    echo "[INFO] still waiting... t=${i}s models_http=${mcode} comp_http=${ccode}"
  fi
  sleep 1
done

echo "[ERROR] Server not ready in 600s, check: ${LOG_FILE}"
exit 1
