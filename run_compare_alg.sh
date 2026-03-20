#!/usr/bin/env bash
set -euo pipefail

kill_pid_list() {
  local signal="$1"
  shift
  local pid
  for pid in "$@"; do
    [[ -z "${pid}" ]] && continue
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    [[ "${pid}" -eq $$ ]] && continue
    kill "-${signal}" "${pid}" 2>/dev/null || true
  done
}

collect_residual_vllm_pids() {
  local pid cmdline
  local -a pids=()
  local -a patterns=(
    "vllm.entrypoints.openai.api_server"
    "vllm/engine/multiprocessing/engine.py"
    "MQLLMEngine"
    "Qwen/Qwen2.5-7B-Instruct"
  )

  for pattern in "${patterns[@]}"; do
    while read -r pid; do
      [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
      if [[ "${cmdline}" == *"python"* ]] && [[ "${cmdline}" == *"/home/ad/zteng/6022/vllm"* || "${cmdline}" == *"vllm"* ]]; then
        pids+=("${pid}")
      fi
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -u)
  fi

  if [[ "${#pids[@]}" -gt 0 ]]; then
    printf '%s\n' "${pids[@]}" | awk 'NF && !seen[$1]++'
  fi
}

kill_server_8000() {
  local pid pid2
  local -a residual_pids=()

  pid="$(lsof -tiTCP:8000 -sTCP:LISTEN || true)"
  if [[ -n "${pid}" ]]; then
    echo "[INFO] Killing server on :8000 PID=${pid}"
    kill_pid_list TERM ${pid}
    sleep 2
    pid2="$(lsof -tiTCP:8000 -sTCP:LISTEN || true)"
    if [[ -n "${pid2}" ]]; then
      echo "[INFO] Force killing PID=${pid2}"
      kill_pid_list KILL ${pid2}
    fi
  else
    echo "[INFO] No server listening on :8000"
  fi

  while read -r pid; do
    [[ -n "${pid}" ]] && residual_pids+=("${pid}")
  done < <(collect_residual_vllm_pids)

  if [[ "${#residual_pids[@]}" -gt 0 ]]; then
    echo "[INFO] Cleaning residual vLLM/GPU processes: ${residual_pids[*]}"
    kill_pid_list TERM "${residual_pids[@]}"
    sleep 3

    residual_pids=()
    while read -r pid; do
      [[ -n "${pid}" ]] && residual_pids+=("${pid}")
    done < <(collect_residual_vllm_pids)

    if [[ "${#residual_pids[@]}" -gt 0 ]]; then
      echo "[INFO] Force killing residual processes: ${residual_pids[*]}"
      kill_pid_list KILL "${residual_pids[@]}"
      sleep 2
    fi
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"
SCRIPT_PATH="${REPO_ROOT}/run_compare_alg.sh"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066

trap kill_server_8000 EXIT

# ---- shared server params ----
CUDA_VISIBLE_DEVICES_COMMON=2
VLLM_RECOVERY_OBS_COMMON=1
VLLM_RECOVERY_PHASE_COMMON=0
VLLM_RECOVERY_BUDGET_COMMON=0
# Leave empty to avoid forcing gpu-memory-utilization from this wrapper.
MEM_COMMON="${MEM_COMMON:-}"
MAXLEN_COMMON=15000
MAX_BATCH_TOKENS_COMMON=16384
MAX_NUM_SEQS_COMMON=16
PMODE_COMMON=swap
SWAP_SPACE_GB_COMMON=8
NUM_SCHED_STEPS_COMMON=1

# ---- shared client params ----
SEGMENT_TIMEOUT_S_COMMON=2400
SEGMENT_TIMEOUT_KILL_S_COMMON=30
MAX_OUTSTANDING_COMMON=64
CLIENT_REQ_TIMEOUT_S_COMMON=180
CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S_COMMON=10
CLIENT_REQ_SOCK_READ_TIMEOUT_S_COMMON=45
SEED_REQSET_COMMON=0
SEED_ARRIVAL_BASE_COMMON=0
FIXED_ARRIVAL_SEED_COMMON=1
GATE_ENABLE_COMMON=0
PHASE_SLICE_S_COMMON=0
SINGLE_LAMBDA_COMMON="${SINGLE_LAMBDA_COMMON:-0.85}"
LAMBDA_LOW_COMMON="${LAMBDA_LOW_COMMON:-${SINGLE_LAMBDA_COMMON}}"
LAMBDA_HIGH_COMMON="${LAMBDA_HIGH_COMMON:-${SINGLE_LAMBDA_COMMON}}"
LOW_S_COMMON="${LOW_S_COMMON:-180}"
HIGH_S_COMMON="${HIGH_S_COMMON:-180}"
NUM_CYCLES_COMMON="${NUM_CYCLES_COMMON:-8}"
TRACE_PATH_COMMON="${REPO_ROOT}/traces/BurstGPT_without_fails_1.csv"
START_TS_COMMON=2032575.0
TRACE_WIN_S_COMMON=300
MAX_TOTAL_TOKENS_COMMON=14500
BATCH_LAMBDAS="${BATCH_LAMBDAS:-}"
BATCH_COMPARE_GAP_METRIC="${BATCH_COMPARE_GAP_METRIC:-stall_max}"
BATCH_RUN_CHILD="${BATCH_RUN_CHILD:-0}"
DEFER_COMPARE_PLOT="${DEFER_COMPARE_PLOT:-0}"
RECOVERY_ROOT_COMMON="${RECOVERY_ROOT_COMMON:-logs/RecoveryGen}"
BATCH_PLOT_ROOT="${BATCH_PLOT_ROOT:-${REPO_ROOT}/${RECOVERY_ROOT_COMMON}/compare_lambda_plots_batch}"

LAM_LOW_TAG="$(echo "${LAMBDA_LOW_COMMON}" | tr '.' 'p')"
LAM_HIGH_TAG="$(echo "${LAMBDA_HIGH_COMMON}" | tr '.' 'p')"
if [[ "${LAMBDA_LOW_COMMON}" == "${LAMBDA_HIGH_COMMON}" ]]; then
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-lam${LAM_LOW_TAG}_single}"
else
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-lam${LAM_LOW_TAG}_to_${LAM_HIGH_TAG}}"
fi
EXPERIMENT_TAG="$(echo "${EXPERIMENT_TAG}" | sed 's/[^A-Za-z0-9._-]/_/g')"
echo "[INFO] script_path=${SCRIPT_PATH}"
echo "[INFO] EXPERIMENT_TAG=${EXPERIMENT_TAG}, lambda_low=${LAMBDA_LOW_COMMON}, lambda_high=${LAMBDA_HIGH_COMMON}"
if [[ "${LAMBDA_LOW_COMMON}" == "${LAMBDA_HIGH_COMMON}" ]]; then
  echo "[INFO] lambda mode: single (lambda=${LAMBDA_LOW_COMMON})"
else
  echo "[INFO] lambda mode: low/high oscillation (low=${LAMBDA_LOW_COMMON}, high=${LAMBDA_HIGH_COMMON})"
fi
if [[ -n "${MEM_COMMON}" ]]; then
  echo "[INFO] MEM_COMMON=${MEM_COMMON} (wrapper sets MEM for server)"
else
  echo "[INFO] MEM_COMMON is empty (wrapper does NOT set MEM for server)"
fi
if [[ -n "${BATCH_LAMBDAS}" ]]; then
  echo "[INFO] BATCH_LAMBDAS=${BATCH_LAMBDAS}"
fi
echo "[INFO] RECOVERY_ROOT_COMMON=${RECOVERY_ROOT_COMMON}"

declare -A CASE_SUMMARY
declare -A CASE_SERVER_LOG

canonicalize_case_name() {
  local raw key
  raw="${1:-}"
  key="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${key}" in
    all)
      echo "all"
      ;;
    baseline|baseline_chunked|vllm_chunked)
      echo "baseline_chunked"
      ;;
    recovergen|recovergen_chunked)
      echo "recovergen_chunked"
      ;;
    baseline_nochunk|baseline_no_chunked|vllm_nochunk|nochunk)
      echo "baseline_nochunk"
      ;;
    recovergen_nochunk|recovergen_no_chunked|recovergen_no_chunk)
      echo "recovergen_nochunk"
      ;;
    *)
      echo "__invalid__:${raw}"
      ;;
  esac
}

if [[ "$#" -gt 1 ]]; then
  echo "[ERROR] too many arguments."
  echo "[INFO] usage:"
  echo "  bash run_compare_alg.sh                    # run all cases"
  echo "  bash run_compare_alg.sh recovergen_nochunk # run one case"
  echo "  bash run_compare_alg.sh a,b                # run multiple cases"
  exit 2
fi

TARGET_CASES_RAW="${1:-all}"
case "${TARGET_CASES_RAW}" in
  --all)
    TARGET_CASES_RAW="all"
    ;;
  --only-case=*)
    TARGET_CASES_RAW="${TARGET_CASES_RAW#--only-case=}"
    ;;
esac
TARGET_CASES_RAW="${TARGET_CASES_RAW// /}"
RUN_ALL_CASES=0
declare -A TARGET_CASES=()

IFS=',' read -r -a _target_tokens <<< "${TARGET_CASES_RAW}"
for _token in "${_target_tokens[@]}"; do
  [[ -z "${_token}" ]] && continue
  _canon="$(canonicalize_case_name "${_token}")"
  if [[ "${_canon}" == "all" ]]; then
    RUN_ALL_CASES=1
    break
  fi
  if [[ "${_canon}" == __invalid__:* ]]; then
    echo "[ERROR] invalid case: ${_token}"
    echo "[INFO] valid: all, baseline_chunked, recovergen_chunked, baseline_nochunk, recovergen_nochunk"
    exit 2
  fi
  TARGET_CASES["${_canon}"]=1
done

if [[ "${RUN_ALL_CASES}" -eq 0 && "${#TARGET_CASES[@]}" -eq 0 ]]; then
  RUN_ALL_CASES=1
fi

should_run_case() {
  local case_key="$1"
  if [[ "${RUN_ALL_CASES}" -eq 1 ]]; then
    return 0
  fi
  [[ -n "${TARGET_CASES[${case_key}]:-}" ]]
}

run_batch_lambdas() {
  local lambdas_raw lam lam_tag child_tag child_status
  local plot_ts plot_dir plot_json
  lambdas_raw="${BATCH_LAMBDAS//,/ }"
  plot_ts="$(date +%Y%m%d_%H%M%S)"
  plot_dir="${BATCH_PLOT_ROOT}/batch_${plot_ts}"
  plot_json="${plot_dir}/report.json"

  mkdir -p "${plot_dir}"
  echo "[INFO] batch mode: lambdas=${BATCH_LAMBDAS}"
  echo "[INFO] batch plot dir=${plot_dir}"

  for lam in ${lambdas_raw}; do
    [[ -z "${lam}" ]] && continue
    lam_tag="$(echo "${lam}" | tr '.' 'p')"
    child_tag="lam${lam_tag}_batch_$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] ===== Batch lambda ${lam} ====="
    set +e
    env       BATCH_RUN_CHILD=1       DEFER_COMPARE_PLOT=1       SINGLE_LAMBDA_COMMON="${lam}"       EXPERIMENT_TAG="${child_tag}"       MEM_COMMON="${MEM_COMMON}"       BATCH_COMPARE_GAP_METRIC="${BATCH_COMPARE_GAP_METRIC}"       bash "${SCRIPT_PATH}" "${TARGET_CASES_RAW}"
    child_status=$?
    set -e
    if [[ "${child_status}" -ne 0 ]]; then
      echo "[ERROR] batch child failed for lambda=${lam} exit=${child_status}"
    else
      echo "[DONE] lambda=${lam}"
    fi
  done

  python3     "${REPO_ROOT}/scripts/plot_compare_alg_lambda_results.py"     --target-lambdas "${BATCH_LAMBDAS}"     --gap-metric "${BATCH_COMPARE_GAP_METRIC}"     --plot-dir "${plot_dir}"     --json-out "${plot_json}"
}

if [[ -n "${BATCH_LAMBDAS}" && "${BATCH_RUN_CHILD}" != "1" ]]; then
  run_batch_lambdas
  exit 0
fi

phase3_extra_server_env=(
  "REC_T_CYC_MS=80"
  "REC_BUDGET_INIT_MS=5"
  "REC_BUDGET_MIN_MS=1"
  "REC_DPLUS_MS=3"
  "REC_DMINUS_MS=8"
  "REC_STALL_MS=300"
  "REC_PREFETCH_MAX_BLOCKS=2048"
  "REC_MODE_STABLE_WINDOW_MS=120"
  "REC_MWS_PREFIX_TOKENS=256"
  "REC_MWS_RECENT_TOKENS=1024"
  "REC_MWS_ADMIT_RHO=0.25"
  "REC_FALLBACK_STALL_MS=600"
  "REC_FALLBACK_RESIDENCE_MS=300"
  "REC_FALLBACK_PREEMPT_THRESHOLD=3"
  "REC_FALLBACK_PROTECT_PRIORITY_GTE=-1"
  "REC_FALLBACK_PAUSE_DECODE=1"
  "REC_FALLBACK_DECODE_INTERVAL_CYCLES=1"
  "REC_FALLBACK_BUDGET_BOOST_MS=0.5"
)

run_case() {
  local case_key="$1"
  local base_dir="$2"
  local run_tag_prefix="$3"
  local server_script="$4"
  local client_script="$5"
  local enable_chunked="$6"
  local cfg_tag="$7"
  local recovery_phase="$8"
  local recovery_budget="$9"
  local use_phase3_extra="${10}"

  local run_ts run_tag server_log_dir client_out_base server_sig_file summary_path
  run_ts="$(date +%Y%m%d_%H%M%S)"
  run_tag="${run_tag_prefix}_${EXPERIMENT_TAG}_${run_ts}"
  server_log_dir="${base_dir}/server_logs/${run_ts}"
  client_out_base="${base_dir}/client_logs/${run_tag}"
  server_sig_file="${base_dir}/current_server_sig.json"
  summary_path="${REPO_ROOT}/${client_out_base}/summary.csv"

  mkdir -p "${server_log_dir}" "${client_out_base}"
  echo "[INFO] ===== Running ${case_key} ====="
  echo "[INFO] server_log_dir=${server_log_dir}"
  echo "[INFO] client_out_base=${client_out_base}"

  local -a server_env=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_COMMON}"
    "BASE_DIR=${base_dir}"
    "SERVER_LOG_DIR=${server_log_dir}"
    "SERVER_SIG_FILE=${server_sig_file}"
    "VLLM_RECOVERY_OBS=${VLLM_RECOVERY_OBS_COMMON}"
    "VLLM_RECOVERY_PHASE=${recovery_phase}"
    "VLLM_RECOVERY_BUDGET=${recovery_budget}"
    "MAXLEN=${MAXLEN_COMMON}"
    "MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS_COMMON}"
    "MAX_NUM_SEQS=${MAX_NUM_SEQS_COMMON}"
    "PMODE=${PMODE_COMMON}"
    "SWAP_SPACE_GB=${SWAP_SPACE_GB_COMMON}"
    "ENABLE_CHUNKED_PREFILL=${enable_chunked}"
    "NUM_SCHED_STEPS=${NUM_SCHED_STEPS_COMMON}"
    "CFG_TAG=${cfg_tag}"
  )
  if [[ -n "${MEM_COMMON}" ]]; then
    server_env+=("MEM=${MEM_COMMON}")
  fi
  if [[ "${use_phase3_extra}" == "1" ]]; then
    server_env+=("${phase3_extra_server_env[@]}")
  fi
  env "${server_env[@]}" bash "${REPO_ROOT}/${server_script}"

  env \
    "NOHUP_LAUNCHED=1" \
    "RUN_TAG=${run_tag}" \
    "OUT_BASE=${client_out_base}" \
    "SERVER_SIG_FILE=${server_sig_file}" \
    "SEGMENT_TIMEOUT_S=${SEGMENT_TIMEOUT_S_COMMON}" \
    "SEGMENT_TIMEOUT_KILL_S=${SEGMENT_TIMEOUT_KILL_S_COMMON}" \
    "MAX_OUTSTANDING=${MAX_OUTSTANDING_COMMON}" \
    "CLIENT_REQ_TIMEOUT_S=${CLIENT_REQ_TIMEOUT_S_COMMON}" \
    "CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S=${CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S_COMMON}" \
    "CLIENT_REQ_SOCK_READ_TIMEOUT_S=${CLIENT_REQ_SOCK_READ_TIMEOUT_S_COMMON}" \
    "SEED_REQSET=${SEED_REQSET_COMMON}" \
    "SEED_ARRIVAL_BASE=${SEED_ARRIVAL_BASE_COMMON}" \
    "FIXED_ARRIVAL_SEED=${FIXED_ARRIVAL_SEED_COMMON}" \
    "GATE_ENABLE=${GATE_ENABLE_COMMON}" \
    "PHASE_SLICE_S=${PHASE_SLICE_S_COMMON}" \
    "LAMBDA_LOW=${LAMBDA_LOW_COMMON}" \
    "LAMBDA_HIGH=${LAMBDA_HIGH_COMMON}" \
    "LOW_S=${LOW_S_COMMON}" \
    "HIGH_S=${HIGH_S_COMMON}" \
    "NUM_CYCLES=${NUM_CYCLES_COMMON}" \
    "TRACE_PATH=${TRACE_PATH_COMMON}" \
    "START_TS=${START_TS_COMMON}" \
    "TRACE_WIN_S=${TRACE_WIN_S_COMMON}" \
    "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS_COMMON}" \
    bash "${REPO_ROOT}/${client_script}"

  if [[ ! -s "${summary_path}" ]]; then
    echo "[ERROR] summary file not found or empty: ${summary_path}"
    exit 2
  fi

  CASE_SUMMARY["${case_key}"]="${summary_path}"
  CASE_SERVER_LOG["${case_key}"]="${REPO_ROOT}/${server_log_dir}"

  kill_server_8000
  sleep 2
}

kill_server_8000

# 1) vllm + chunked prefill
if should_run_case "baseline_chunked"; then
  run_case \
    "baseline_chunked" \
    "${RECOVERY_ROOT_COMMON}/Baseline" \
    "baseline" \
    "scripts/baseline_chunked_server.sh" \
    "scripts/baseline_chunked_client.sh" \
    "1" \
    "recovergen_baseline" \
    "${VLLM_RECOVERY_PHASE_COMMON}" \
    "${VLLM_RECOVERY_BUDGET_COMMON}" \
    "0"
fi

# 2) RecoverGen + chunked prefill
if should_run_case "recovergen_chunked"; then
  run_case \
    "recovergen_chunked" \
    "${RECOVERY_ROOT_COMMON}/RecoverGen" \
    "recovergen" \
    "scripts/phase3_recovery_server.sh" \
    "scripts/phase3_recovery_client.sh" \
    "1" \
    "recovergen_phase3_compare" \
    "3" \
    "1" \
    "1"
fi

# 3) vllm + no_chunked prefill
if should_run_case "baseline_nochunk"; then
  run_case \
    "baseline_nochunk" \
    "${RECOVERY_ROOT_COMMON}/Baseline_no_chunked" \
    "baseline_nochunk" \
    "scripts/baseline_chunked_server.sh" \
    "scripts/baseline_chunked_client.sh" \
    "0" \
    "recovergen_baseline_nochunk" \
    "${VLLM_RECOVERY_PHASE_COMMON}" \
    "${VLLM_RECOVERY_BUDGET_COMMON}" \
    "0"
fi

# 4) RecoverGen + no_chunked prefill
if should_run_case "recovergen_nochunk"; then
  run_case \
    "recovergen_nochunk" \
    "${RECOVERY_ROOT_COMMON}/RecoverGen_no_chunked" \
    "recovergen_nochunk" \
    "scripts/phase3_recovery_server.sh" \
    "scripts/phase3_recovery_client.sh" \
    "0" \
    "recovergen_phase3_nochunk" \
    "3" \
    "1" \
    "1"
fi

if [[ "${DEFER_COMPARE_PLOT}" == "1" ]]; then
  echo "[INFO] defer compare plot: skip per-run plotting"
  echo "[INFO] finished selected run(s):"
  for case_key in baseline_chunked recovergen_chunked baseline_nochunk recovergen_nochunk; do
    if [[ -n "${CASE_SUMMARY[${case_key}]:-}" ]]; then
      echo "  - ${case_key}: ${CASE_SUMMARY[${case_key}]}"
    fi
  done
elif [[ -n "${CASE_SUMMARY[baseline_chunked]:-}" &&         -n "${CASE_SUMMARY[recovergen_chunked]:-}" &&         -n "${CASE_SUMMARY[baseline_nochunk]:-}" ]]; then
  COMPARE_OUT_DIR="${REPO_ROOT}/${RECOVERY_ROOT_COMMON}/RecoverGen/compare_reports"
  COMPARE_TS="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${COMPARE_OUT_DIR}"
  COMPARE_SCRIPT="${REPO_ROOT}/scripts/plot_compare_run_compare_alg_results.py"

  if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
    echo "[WARN] compare script missing: ${COMPARE_SCRIPT}"
    echo "[INFO] finished selected run(s):"
    for case_key in baseline_chunked recovergen_chunked baseline_nochunk recovergen_nochunk; do
      if [[ -n "${CASE_SUMMARY[${case_key}]:-}" ]]; then
        echo "  - ${case_key}: ${CASE_SUMMARY[${case_key}]}"
      fi
    done
    exit 0
  fi

  compare_cmd=(
    python3
    "${COMPARE_SCRIPT}"
    --baseline-summary "${CASE_SUMMARY[baseline_chunked]}"
    --recovergen-summary "${CASE_SUMMARY[recovergen_chunked]}"
    --nochunk-summary "${CASE_SUMMARY[baseline_nochunk]}"
    --baseline-server-log "${CASE_SERVER_LOG[baseline_chunked]}"
    --recovergen-server-log "${CASE_SERVER_LOG[recovergen_chunked]}"
    --nochunk-server-log "${CASE_SERVER_LOG[baseline_nochunk]}"
    --plot-dir "${COMPARE_OUT_DIR}/compare_${EXPERIMENT_TAG}_${COMPARE_TS}_plots"
    --json-out "${COMPARE_OUT_DIR}/compare_${EXPERIMENT_TAG}_${COMPARE_TS}.json"
  )
  if [[ -n "${CASE_SUMMARY[recovergen_nochunk]:-}" ]]; then
    compare_cmd+=(
      --recovergen-nochunk-summary "${CASE_SUMMARY[recovergen_nochunk]}"
      --recovergen-nochunk-server-log "${CASE_SERVER_LOG[recovergen_nochunk]}"
    )
  fi
  "${compare_cmd[@]}" | tee "${COMPARE_OUT_DIR}/compare_${EXPERIMENT_TAG}_${COMPARE_TS}.txt"
else
  echo "[INFO] Skip compare: need baseline_chunked + recovergen_chunked + baseline_nochunk."
  echo "[INFO] finished selected run(s):"
  for case_key in baseline_chunked recovergen_chunked baseline_nochunk recovergen_nochunk; do
    if [[ -n "${CASE_SUMMARY[${case_key}]:-}" ]]; then
      echo "  - ${case_key}: ${CASE_SUMMARY[${case_key}]}"
    fi
  done
fi
