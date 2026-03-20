#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${PROJECT_ROOT}"

# =========================
# nohup wrapper
# =========================
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-logs/RecoveryGen/Baseline_chunked/client_logs/${RUN_TAG}}"
mkdir -p "${OUT_BASE}"

NOHUP_OUT="${NOHUP_OUT:-${OUT_BASE}/nohup.out}"
NOHUP_PID="${NOHUP_PID:-${OUT_BASE}/nohup.pid}"

if [[ "${NOHUP_LAUNCHED:-0}" != "1" ]]; then
  nohup env NOHUP_LAUNCHED=1 RUN_TAG="${RUN_TAG}" OUT_BASE="${OUT_BASE}" bash "$0" \
    > "${NOHUP_OUT}" 2>&1 &
  echo $! > "${NOHUP_PID}"
  echo "[OK] launched nohup: pid=$(cat "${NOHUP_PID}") log=${NOHUP_OUT} out_base=${OUT_BASE}"
  exit 0
fi

# =========================
# env + conda
# =========================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066
export PYTHONNOUSERSITE=1

# ---- avoid proxy interference ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true

# ---- HF offline guard (CRITICAL for tokenizer) ----
HF_FORCE_ONLINE="${HF_FORCE_ONLINE:-0}"
if [[ "${HF_FORCE_ONLINE}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_ENABLE_HF_TRANSFER=0
  echo "[INFO] HF offline enabled for client tokenizer."
else
  unset HF_HUB_OFFLINE || true
  unset TRANSFORMERS_OFFLINE || true
  echo "[INFO] HF online mode for client: HF_FORCE_ONLINE=1"
fi

# ---- CUDA lib conflict guard (match server) ----
unset CUDA_HOME CUDA_PATH
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda/lib64$' | paste -sd: -)"
fi
CUDA_SYS_LIB="/usr/local/cuda/targets/x86_64-linux/lib"
CONDA_NVJITLINK_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="${CONDA_NVJITLINK_LIB}:${CUDA_SYS_LIB}:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# =========================
# endpoints
# =========================
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1/completions}"
MODELS_URL="${MODELS_URL:-http://${HOST}:${PORT}/v1/models}"
METRICS_URL="${METRICS_URL:-http://${HOST}:${PORT}/metrics}"
HEALTH_URL="${HEALTH_URL:-http://${HOST}:${PORT}/health}"

MODEL="${MODEL:-__AUTO__}"

SERVER_SIG_FILE="${SERVER_SIG_FILE:-logs/RecoveryGen/Baseline_chunked/current_server_sig.json}"
SERVER_LOG_ROOT="${SERVER_LOG_ROOT:-$(dirname "${SERVER_SIG_FILE}")/server_logs}"

# =========================
# trace replay config
# =========================
TRACE_PATH="${TRACE_PATH:-/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv}"
START_TS="${START_TS:-2032575.0}"
TRACE_WIN_S="${TRACE_WIN_S:-300}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-14500}"

PROMPTS_JSON="${PROMPTS_JSON:-/home/ad/zteng/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

# NOTE: tokenizer name MUST be cached locally when offline
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"

# =========================
# schedule
# =========================
MODE_LOW="${MODE_LOW:-poisson}"
MODE_HIGH="${MODE_HIGH:-${MODE_LOW}}"
LOW_LAMBDA="${LAMBDA_LOW:-${LOW_LAMBDA:-0.60}}"
HIGH_LAMBDA="${LAMBDA_HIGH:-${HIGH_LAMBDA:-0.85}}"
LOW_S="${LOW_S:-60}"
HIGH_S="${HIGH_S:-60}"
NUM_CYCLES="${NUM_CYCLES:-2}"

BURST_B="${BURST_B:-4}"
BURST_WINDOW_S="${BURST_WINDOW_S:-2.0}"
MAX_OUTSTANDING="${MAX_OUTSTANDING:-256}"
# Optional per-segment hard timeout for strict comparability with phase2 client.
SEGMENT_TIMEOUT_S="${SEGMENT_TIMEOUT_S:-0}"
SEGMENT_TIMEOUT_KILL_S="${SEGMENT_TIMEOUT_KILL_S:-30}"

# =========================
# metrics sampling
# =========================
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-0.5}"
METRICS_CONTINUOUS="${METRICS_CONTINUOUS:-0}"

SEED_REQSET="${SEED_REQSET:-0}"
SEED_ARRIVAL_BASE="${SEED_ARRIVAL_BASE:-0}"
FIXED_ARRIVAL_SEED="${FIXED_ARRIVAL_SEED:-0}"

# =========================
# output naming
# =========================
SERVER_SIG_STR="unknown_server"
if [[ -s "${SERVER_SIG_FILE}" ]]; then
  SERVER_SIG_STR="$(python3 - <<PY
import json
p="${SERVER_SIG_FILE}"
try:
    d=json.load(open(p,"r",encoding="utf-8"))
    s=d.get("server_sig_str","unknown_server")
    s=s.replace("/","_").replace(" ","").replace(".","p")
    print(s)
except Exception:
    print("unknown_server")
PY
)"
else
  echo "[WARN] SERVER_SIG_FILE not found: ${SERVER_SIG_FILE} (use unknown_server)"
fi

GATE_SIG="gate0"
GATE_SIG="${GATE_SIG//./p}"

if [[ "${OUT_BASE}" == "logs/RecoveryGen/Baseline_chunked/client_logs/${RUN_TAG}" ]]; then
  SERVER_SIG_SHORT="$(python3 - <<PY
import hashlib
s = """${SERVER_SIG_STR}"""
h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
prefix = s[:96]
print(f"{prefix}_h{h}")
PY
)"
  OUT_BASE="logs/RecoveryGen/Baseline_chunked/client_logs/${RUN_TAG}_${SERVER_SIG_SHORT}_${GATE_SIG}"
  mkdir -p "${OUT_BASE}"
fi

REQSET_PATH="${REQSET_PATH:-${OUT_BASE}/reqset_trace.jsonl}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_BASE}/summary.csv}"
PROGRESS_TXT="${PROGRESS_TXT:-${OUT_BASE}/progress.txt}"
PHASE_FILE="${PHASE_FILE:-${OUT_BASE}/phase.txt}"
PHASE_LOG="${PHASE_LOG:-${OUT_BASE}/phase_changes.csv}"

echo "[INFO] OUT_BASE=${OUT_BASE}"
echo "[INFO] server_sig_str=${SERVER_SIG_STR}"
echo "[INFO] base_url=${BASE_URL}"
echo "[INFO] models_url=${MODELS_URL}"
echo "[INFO] metrics_url=${METRICS_URL}"
echo "[INFO] model_env=${MODEL}"
echo "[INFO] tokenizer=${TOKENIZER} (offline=${HF_HUB_OFFLINE:-0})"

write_progress() {
  local stage="$1"; local extra="${2:-}"
  {
    echo "ts=$(date +%s)"
    echo "stage=${stage}"
    echo "out_base=${OUT_BASE}"
    echo "summary_csv=${SUMMARY_CSV}"
    echo "extra=${extra}"
  } > "${PROGRESS_TXT}"
}

log_phase_change() {
  local cycle_idx="$1" phase_tag="$2" lam="$3" mode="$4"
  local ts
  ts="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
  if [[ ! -s "${PHASE_LOG}" ]]; then
    echo "ts,cycle,phase,lambda_rps,mode" > "${PHASE_LOG}"
  fi
  echo "${ts},${cycle_idx},${phase_tag},${lam},${mode}" >> "${PHASE_LOG}"
}

resolve_model_id() {
  python3 - "$MODELS_URL" "$MODEL" <<'PY'
import sys, time
import requests

url = sys.argv[1]
want = sys.argv[2]
last_err = None
for _ in range(60):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}"
            time.sleep(1); continue
        d = r.json()
        ids = [x.get("id") for x in d.get("data", []) if isinstance(x, dict) and x.get("id")]
        if not ids:
            last_err = "empty id list"
            time.sleep(1); continue
        first = ids[0]
        if want == "__AUTO__":
            print(first); sys.exit(0)
        if want in ids:
            print(want); sys.exit(0)
        print(first); sys.exit(0)
    except Exception as e:
        last_err = repr(e)
        time.sleep(1)
print(f"__ERROR__ {last_err}")
sys.exit(2)
PY
}

wait_ready() {
  local timeout_s="${1:-600}"
  echo "[INFO] waiting server readiness (timeout ${timeout_s}s) ..."
  for ((i=1;i<=timeout_s;i++)); do
    code="$(curl -sS --connect-timeout 1 --max-time 2 \
      -o /tmp/vllm_ready_probe.json -w "%{http_code}" \
      "${BASE_URL}" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" \
      2>/dev/null || echo "000")"
    if [[ "${code}" == "200" ]]; then
      echo "[INFO] server ready."
      return 0
    fi
    if (( i % 10 == 0 )); then
      echo "[INFO] still waiting... t=${i}s http=${code} model=${MODEL}"
      echo "[INFO] models snapshot:"
      curl -sS --connect-timeout 1 --max-time 2 "${MODELS_URL}" | head -c 240 || true
      echo ""
    fi
    sleep 1
  done
  echo "[FATAL] server not ready within ${timeout_s}s" >&2
  return 1
}

start_metrics_sampler() {
  local raw_out="$1" csv_out="$2" interval_s="$3" phase_file="$4" start_ts="$5"
  python3 -u - "$METRICS_URL" "$raw_out" "$csv_out" "$interval_s" "$phase_file" "$start_ts" <<'PY' &
import sys, time, math, requests
url, raw_out, csv_out, interval, phase_file, start_ts = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], float(sys.argv[6])

def read_phase():
    try:
        with open(phase_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

import re

SWAPIN_COUNTER_CANDIDATES = [
    "vllm:recovery_swapin_blocks_total",
    "vllm:swapin_blocks_total",
    "recovery_swapin_blocks_total",
    "swapin_blocks_total",
]
SWAPOUT_COUNTER_CANDIDATES = [
    "vllm:recovery_swapout_blocks_total",
    "vllm:swapout_blocks_total",
    "recovery_swapout_blocks_total",
    "swapout_blocks_total",
]
RECOMPUTE_COUNTER_CANDIDATES = [
    "vllm:recovery_recompute_tokens_total",
    "vllm:recompute_tokens_total",
    "recovery_recompute_tokens_total",
    "recompute_tokens_total",
]
STALL_MS_COUNTER_CANDIDATES = [
    "vllm:recovery_restore_progress_stall_ms_total",
    "vllm:restore_progress_stall_ms_total",
    "recovery_restore_progress_stall_ms_total",
    "restore_progress_stall_ms_total",
]

def parse_metric(text, key):
    # Match metric line with optional labels: key{...} value
    pat = re.compile(
        rf"^{re.escape(key)}(?:\{{.*\}})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return float("nan")
    return float("nan")

def parse_first_metric(text, keys):
    for key in keys:
        val = parse_metric(text, key)
        if math.isfinite(val):
            return val
    return float("nan")

def delta_nonneg(curr, prev):
    if math.isfinite(curr) and math.isfinite(prev):
        return max(0.0, curr - prev)
    return float("nan")

prev_ts = None
prev_pre = None
prev_swapin = None
prev_swapout = None
prev_recompute = None
prev_restore_stall = None

with open(raw_out, "w", encoding="utf-8") as fraw, open(csv_out, "w", encoding="utf-8") as fcsv:
    fcsv.write("ts,t_rel_s,phase,cycle,mode,lambda_rps,preempt_total,preempt_rate_per_s,req_waiting,req_running,gpu_cache_usage_perc,cycle_id,on_preempt_count_delta,on_waiting_len,T_on_ms,S_ms,swapin_blocks,swapout_blocks,recompute_tokens,gpu_kv_blocks_free,mode_cnt_normal,mode_cnt_recovery,mode_cnt_fallback,restore_progress_stall_ms\n")
    fcsv.flush()
    while True:
        ts = time.time()
        phase = read_phase()
        try:
            text = requests.get(url, timeout=2).text
        except Exception as e:
            text = f"#ERROR {repr(e)}"

        fraw.write(f"@@TS {ts:.6f}\n")
        fraw.write(text)
        if not text.endswith("\n"):
            fraw.write("\n")
        fraw.flush()

        pre = parse_metric(text, "vllm:num_preemptions_total")
        waiting = parse_metric(text, "vllm:num_requests_waiting")
        running = parse_metric(text, "vllm:num_requests_running")
        gpu_cache_perc = parse_metric(text, "vllm:gpu_cache_usage_perc")
        swapin_total = parse_first_metric(text, SWAPIN_COUNTER_CANDIDATES)
        swapout_total = parse_first_metric(text, SWAPOUT_COUNTER_CANDIDATES)
        recompute_total = parse_first_metric(text, RECOMPUTE_COUNTER_CANDIDATES)
        restore_stall_total = parse_first_metric(text, STALL_MS_COUNTER_CANDIDATES)

        prev_ts_old = prev_ts
        prev_pre_old = prev_pre
        prev_swapin_old = prev_swapin
        prev_swapout_old = prev_swapout
        prev_recompute_old = prev_recompute
        prev_restore_stall_old = prev_restore_stall
        rate = float("nan")
        if prev_ts_old is not None and math.isfinite(pre):
            dt = max(1e-9, ts - prev_ts_old)
            if prev_pre_old is not None and math.isfinite(prev_pre_old):
                rate = delta_nonneg(pre, prev_pre_old) / dt

        # phase string format: "cycle=X phase=low lam=0.6 mode=poisson"
        cycle = ""
        mode = ""
        lam = ""
        for part in phase.split():
            if part.startswith("cycle="):
                cycle = part.split("=", 1)[1]
            elif part.startswith("phase="):
                phase = part.split("=", 1)[1]
            elif part.startswith("lam="):
                lam = part.split("=", 1)[1]
            elif part.startswith("mode="):
                mode = part.split("=", 1)[1]

        t_rel = ts - start_ts
        interval_ms = interval * 1000.0
        preempt_delta = delta_nonneg(pre, prev_pre_old) if prev_pre_old is not None else float("nan")
        on_waiting_len = waiting
        t_on_ms = interval_ms
        s_ms = 0.0
        swapin_blocks = delta_nonneg(swapin_total, prev_swapin_old) if prev_swapin_old is not None else float("nan")
        swapout_blocks = delta_nonneg(swapout_total, prev_swapout_old) if prev_swapout_old is not None else float("nan")
        recompute_tokens = delta_nonneg(recompute_total, prev_recompute_old) if prev_recompute_old is not None else float("nan")
        restore_stall_ms = delta_nonneg(restore_stall_total, prev_restore_stall_old) if prev_restore_stall_old is not None else float("nan")
        if math.isfinite(gpu_cache_perc):
            if gpu_cache_perc <= 1.0 + 1e-6:
                gpu_kv_blocks_free = max(0.0, 1.0 - gpu_cache_perc)
            else:
                gpu_kv_blocks_free = max(0.0, 100.0 - gpu_cache_perc)
        else:
            gpu_kv_blocks_free = float("nan")
        mode_cnt_normal = 1
        mode_cnt_recovery = 0
        mode_cnt_fallback = 0
        if not math.isfinite(restore_stall_ms):
            # Fallback proxy when recovery counters are unavailable.
            if math.isfinite(preempt_delta) and preempt_delta > 0:
                restore_stall_ms = 0.0
            else:
                restore_stall_ms = interval_ms
        fcsv.write(f"{ts:.6f},{t_rel:.6f},{phase},{cycle},{mode},{lam},{pre},{rate},{waiting},{running},{gpu_cache_perc},{cycle},{preempt_delta},{on_waiting_len},{t_on_ms},{s_ms},{swapin_blocks},{swapout_blocks},{recompute_tokens},{gpu_kv_blocks_free},{mode_cnt_normal},{mode_cnt_recovery},{mode_cnt_fallback},{restore_stall_ms}\n")
        fcsv.flush()
        prev_ts = ts
        prev_pre = pre
        prev_swapin = swapin_total
        prev_swapout = swapout_total
        prev_recompute = recompute_total
        prev_restore_stall = restore_stall_total
        time.sleep(interval)
PY
}

run_one_segment() {
  local out_dir="$1" lam="$2" mode="$3" dur_s="$4" seed_arrival="$5"
  mkdir -p "${out_dir}"
  local client_csv="${out_dir}/client_results.csv"
  local meta_json="${out_dir}/replay_meta.json"
  local seg_csv="${out_dir}/burst_segments.csv"
  local client_log="${out_dir}/client.log"

  set +e
  if [[ "${SEGMENT_TIMEOUT_S}" =~ ^[0-9]+$ ]] && [[ "${SEGMENT_TIMEOUT_S}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after="${SEGMENT_TIMEOUT_KILL_S}" "${SEGMENT_TIMEOUT_S}" \
      python3 -u scripts/controlled_replay_sweep_trace.py \
      "${PROMPTS_JSON}" \
      "${TOKENIZER}" \
      "${REQSET_PATH}" \
      "${SEED_REQSET}" \
      "${BASE_URL}" \
      "${MODEL}" \
      "${mode}" \
      "${lam}" \
      "${dur_s}" \
      "${BURST_B}" \
      "${BURST_WINDOW_S}" \
      "${seed_arrival}" \
      "${MAX_OUTSTANDING}" \
      "${MAX_TOTAL_TOKENS}" \
      "${client_csv}" \
      "${meta_json}" \
      "${seg_csv}" \
      > "${client_log}" 2>&1
    local rc=$?
  else
    if [[ "${SEGMENT_TIMEOUT_S}" =~ ^[0-9]+$ ]] && [[ "${SEGMENT_TIMEOUT_S}" -gt 0 ]]; then
      echo "[WARN] 'timeout' not found; running segment without hard timeout." > "${client_log}"
      python3 -u scripts/controlled_replay_sweep_trace.py \
        "${PROMPTS_JSON}" \
        "${TOKENIZER}" \
        "${REQSET_PATH}" \
        "${SEED_REQSET}" \
        "${BASE_URL}" \
        "${MODEL}" \
        "${mode}" \
        "${lam}" \
        "${dur_s}" \
        "${BURST_B}" \
        "${BURST_WINDOW_S}" \
        "${seed_arrival}" \
        "${MAX_OUTSTANDING}" \
        "${MAX_TOTAL_TOKENS}" \
        "${client_csv}" \
        "${meta_json}" \
        "${seg_csv}" \
        >> "${client_log}" 2>&1
    else
      python3 -u scripts/controlled_replay_sweep_trace.py \
        "${PROMPTS_JSON}" \
        "${TOKENIZER}" \
        "${REQSET_PATH}" \
        "${SEED_REQSET}" \
        "${BASE_URL}" \
        "${MODEL}" \
        "${mode}" \
        "${lam}" \
        "${dur_s}" \
        "${BURST_B}" \
        "${BURST_WINDOW_S}" \
        "${seed_arrival}" \
        "${MAX_OUTSTANDING}" \
        "${MAX_TOTAL_TOKENS}" \
        "${client_csv}" \
        "${meta_json}" \
        "${seg_csv}" \
        > "${client_log}" 2>&1
    fi
    local rc=$?
  fi
  set -e
  echo "${rc}"
}

summarize_one_segment() {
  local out_dir="$1" lam="$2" dur_s="$3" wall="$4"
  local meta_json="${out_dir}/replay_meta.json"
  python3 -u "${SCRIPT_DIR}/summarize_one_run_sweep_trace.py" \
    "${out_dir}" \
    "${lam}" \
    "${dur_s}" \
    "${wall}" \
    "${meta_json}" \
    "${SUMMARY_CSV}"
}

run_phase() {
  local cycle_idx="$1" phase_tag="$2" lam="$3" seed_arrival="$4" mode="$5" dur_s="$6"

  local lam_tag
  lam_tag="$(python3 - <<PY
x=float("${lam}")
print(f"{x:.2f}".replace(".","p"))
PY
)"
  local out_dir="${OUT_BASE}/rc_cycle$(printf "%02d" "${cycle_idx}")_${phase_tag}_lam${lam_tag}_mode${mode}"
  mkdir -p "${out_dir}"

  echo "cycle=${cycle_idx} phase=${phase_tag} lam=${lam} mode=${mode}" > "${PHASE_FILE}"

  local metr_raw="${out_dir}/metrics_raw.txt"
  local metr_csv="${out_dir}/metrics_timeseries.csv"
  local samp_pid=""
  if [[ "${METRICS_CONTINUOUS}" == "0" ]]; then
    start_metrics_sampler "${metr_raw}" "${metr_csv}" "${METRICS_INTERVAL_S}" "${PHASE_FILE}" "${RUN_START_TS}"
    samp_pid=$!
  fi

  local t0 t1 wall rc
  t0="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
  rc="$(run_one_segment "${out_dir}" "${lam}" "${mode}" "${dur_s}" "${seed_arrival}")"
  t1="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
  wall="$(python3 - <<PY
t0=float("${t0}"); t1=float("${t1}"); print(f"{(t1-t0):.6f}")
PY
)"

  if [[ "${METRICS_CONTINUOUS}" == "0" ]]; then
    kill "${samp_pid}" >/dev/null 2>&1 || true
    wait "${samp_pid}" >/dev/null 2>&1 || true
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "[FATAL] replay failed rc=${rc} (skip summarize). See: ${out_dir}/client.log" | tee -a "${OUT_BASE}/fatal.log"
    return "${rc}"
  fi

  summarize_one_segment "${out_dir}" "${lam}" "${dur_s}" "${wall}" | tee -a "${OUT_BASE}/summarize.log"
}

# =========================
# main
# =========================
write_progress "INIT" ""
RUN_START_TS="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"

write_progress "RESOLVE_MODEL" ""
RESOLVED="$(resolve_model_id)"
if [[ "${RESOLVED}" == __ERROR__* ]]; then
  echo "[FATAL] resolve_model_id failed: ${RESOLVED}"
  exit 2
fi
if [[ "${MODEL}" != "${RESOLVED}" ]]; then
  echo "[WARN] MODEL overridden: ${MODEL} -> ${RESOLVED} (use /v1/models id)"
fi
MODEL="${RESOLVED}"
echo "[INFO] model=${MODEL}"

wait_ready 600

write_progress "BUILD_REQSET" ""
if [[ ! -s "${REQSET_PATH}" ]]; then
  python3 -u scripts/build_reqset_sweep_trace.py \
    "${TRACE_PATH}" "${START_TS}" "${TRACE_WIN_S}" "${MAX_TOTAL_TOKENS}" "${REQSET_PATH}" \
    | tee "${OUT_BASE}/build_reqset.log"
else
  echo "[INFO] reuse existing reqset: ${REQSET_PATH}"
fi

if [[ "${METRICS_CONTINUOUS}" == "1" ]]; then
  echo "cycle=0 phase=init lam=0 mode=${MODE_LOW}" > "${PHASE_FILE}"
  start_metrics_sampler "${OUT_BASE}/metrics_raw.txt" "${OUT_BASE}/metrics_timeseries.csv" "${METRICS_INTERVAL_S}" "${PHASE_FILE}" "${RUN_START_TS}"
  CONT_PID=$!
  echo "[INFO] metrics continuous pid=${CONT_PID}"
fi

set +e
for ((c=1; c<=NUM_CYCLES; c++)); do
  low_seed="${SEED_ARRIVAL_BASE}"
  high_seed="${SEED_ARRIVAL_BASE}"
  if [[ "${FIXED_ARRIVAL_SEED}" != "1" ]]; then
    low_seed="$((SEED_ARRIVAL_BASE + c*2))"
    high_seed="$((SEED_ARRIVAL_BASE + c*2 + 1))"
  fi
  log_phase_change "${c}" "low" "${LOW_LAMBDA}" "${MODE_LOW}"
  run_phase "${c}" "low" "${LOW_LAMBDA}" "${low_seed}" "${MODE_LOW}" "${LOW_S}" || break

  log_phase_change "${c}" "high" "${HIGH_LAMBDA}" "${MODE_HIGH}"
  run_phase "${c}" "high" "${HIGH_LAMBDA}" "${high_seed}" "${MODE_HIGH}" "${HIGH_S}" || break
done
rc_all=$?
set -e

if [[ "${METRICS_CONTINUOUS}" == "1" ]]; then
  kill "${CONT_PID}" >/dev/null 2>&1 || true
  wait "${CONT_PID}" >/dev/null 2>&1 || true
fi

if [[ "${rc_all}" -ne 0 ]]; then
  echo "[FATAL] run aborted rc=${rc_all}. See ${OUT_BASE}/fatal.log" >&2
  exit "${rc_all}"
fi

write_progress "DONE_ALL" ""
FINALIZE_SCRIPT="scripts/finalize_recovery_run_artifacts.py"
if [[ -f "${FINALIZE_SCRIPT}" ]]; then
  python3 -u "${FINALIZE_SCRIPT}" \
    --out-base "${OUT_BASE}" \
    --server-sig-file "${SERVER_SIG_FILE}" \
    --server-log-root "${SERVER_LOG_ROOT}"
else
  echo "[WARN] finalize script missing: ${FINALIZE_SCRIPT}, skip finalize artifacts."
fi
echo "[OK] Phase0 run done: ${OUT_BASE}"
echo "[OK] summary: ${SUMMARY_CSV}"
