#!/usr/bin/env bash
set -euo pipefail

cd /home/ad/zteng/6022/vllm

# =========================
# nohup wrapper
# =========================
RUN_TAG="${RUN_TAG:-Mot2_tail_instability_$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-logs/RecoveryGen/Mot2_tail_instability/${RUN_TAG}}"
mkdir -p "${OUT_BASE}"

NOHUP_OUT="${OUT_BASE}/nohup.out"
NOHUP_PID="${OUT_BASE}/nohup.pid"

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

# ---- HF offline guard (client tokenizer) ----
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# =========================
# config
# =========================
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1/completions}"
METRICS_URL="${METRICS_URL:-http://${HOST}:${PORT}/metrics}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"

TRACE_PATH="${TRACE_PATH:-/home/ad/zteng/6022/vllm/traces/BurstGPT_without_fails_1.csv}"
START_TS="${START_TS:-3063938.0}"
TRACE_WIN_S="${TRACE_WIN_S:-600}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-14500}"

PROMPTS_JSON="${PROMPTS_JSON:-/home/ad/zteng/6022/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"

# oscillation schedule
# allow legacy/alias envs: LAMBDA_LOW/LAMBDA_HIGH, LOW_S/HIGH_S, NUM_CYCLES, MODE_LOW/MODE_HIGH
MODE_LOW="${MODE_LOW:-${MODE:-poisson}}"     # poisson | burst
MODE_HIGH="${MODE_HIGH:-${MODE_LOW}}"
LOW_LAMBDA="${LAMBDA_LOW:-${LOW_LAMBDA:-0.2}}"
HIGH_LAMBDA="${LAMBDA_HIGH:-${HIGH_LAMBDA:-0.9}}"
LOW_S="${LOW_S:-${PHASE_S:-120}}"
HIGH_S="${HIGH_S:-${PHASE_S:-120}}"
NUM_CYCLES="${NUM_CYCLES:-${CYCLES:-6}}"

# burst params (only used when MODE=burst)
BURST_B="${BURST_B:-4}"
BURST_WINDOW_S="${BURST_WINDOW_S:-2.0}"

MAX_OUTSTANDING="${MAX_OUTSTANDING:-256}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-0.2}"

SEED_REQSET="${SEED_REQSET:-0}"
SEED_ARRIVAL_BASE="${SEED_ARRIVAL_BASE:-0}"


# optional: continuous metrics sampler (1) vs per-phase (0)
METRICS_CONTINUOUS="${METRICS_CONTINUOUS:-0}"

# compose OUT_BASE if not overridden
if [[ "${OUT_BASE}" == "logs/RecoveryGen/Mot2_tail_instability/${RUN_TAG}" ]]; then
  OUT_BASE="logs/RecoveryGen/Mot2_tail_instability/${RUN_TAG}_low${LOW_LAMBDA}_high${HIGH_LAMBDA}_${MODE_LOW}-${MODE_HIGH}_b${BURST_B}"
  mkdir -p "${OUT_BASE}"
fi

# recompute derived paths after OUT_BASE finalization
REQSET_PATH="${REQSET_PATH:-${OUT_BASE}/reqset_trace.jsonl}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_BASE}/summary.csv}"
PROGRESS_TXT="${PROGRESS_TXT:-${OUT_BASE}/progress.txt}"
PHASE_FILE="${PHASE_FILE:-${OUT_BASE}/phase.txt}"
PHASE_LOG="${PHASE_LOG:-${OUT_BASE}/phase_changes.csv}"

echo "[INFO] OUT_BASE=${OUT_BASE}"
echo "[INFO] trace=${TRACE_PATH} start_ts=${START_TS} trace_win_s=${TRACE_WIN_S} max_total_tokens=${MAX_TOTAL_TOKENS}"
echo "[INFO] mode_low=${MODE_LOW} mode_high=${MODE_HIGH} low=${LOW_LAMBDA} high=${HIGH_LAMBDA} low_s=${LOW_S} high_s=${HIGH_S} cycles=${NUM_CYCLES}"
echo "[INFO] burst_b=${BURST_B} burst_window_s=${BURST_WINDOW_S} max_outstanding=${MAX_OUTSTANDING}"

# =========================
# helpers
# =========================
start_metrics_sampler() {
  local raw_out="$1"
  local csv_out="$2"
  local interval_s="$3"
  local phase_file="$4"
  local start_ts="$5"
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

write_progress() {
  local stage="$1"
  local extra="${2:-}"
  {
    echo "ts=$(date +%s)"
    echo "stage=${stage}"
    echo "out_base=${OUT_BASE}"
    echo "summary_csv=${SUMMARY_CSV}"
    echo "extra=${extra}"
  } > "${PROGRESS_TXT}"
}

log_phase_change() {
  local cycle_idx="$1"
  local phase_tag="$2"
  local lam="$3"
  local mode="$4"
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

wait_ready() {
  local timeout_s="${1:-600}"
  echo "[INFO] waiting server readiness (timeout ${timeout_s}s) ..."
  for ((i=1;i<=timeout_s;i++)); do
    code="$(curl -sS --connect-timeout 1 --max-time 2 \
      -o /tmp/vllm_ready_probe.json -w "%{http_code}" \
      "http://${HOST}:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" \
      2>/dev/null || echo "000")"
    if [[ "${code}" == "200" ]]; then
      echo "[INFO] server ready."
      return 0
    fi
    if (( i % 10 == 0 )); then
      echo "[INFO] still waiting... t=${i}s http=${code}"
    fi
    sleep 1
  done
  echo "[FATAL] server not ready within ${timeout_s}s" >&2
  return 1
}

run_phase() {
  local cycle_idx="$1"
  local phase_tag="$2"
  local lam="$3"
  local seed_arrival="$4"
  local mode="$5"
  local dur_s="$6"

  local out_dir="${OUT_BASE}/cycle_${cycle_idx}_${phase_tag}_lambda_${lam}"
  mkdir -p "${out_dir}"

  local metr_raw="${out_dir}/metrics_raw.txt"
  local metr_csv="${out_dir}/metrics_timeseries.csv"
  local client_csv="${out_dir}/client_results.csv"
  local meta_json="${out_dir}/replay_meta.json"
  local seg_csv="${out_dir}/burst_segments.csv"
  local client_log="${out_dir}/client.log"

  local samp_pid=""
  # update phase marker for both continuous and per-phase sampling
  echo "cycle=${cycle_idx} phase=${phase_tag} lam=${lam} mode=${mode}" > "${PHASE_FILE}"
  if [[ "${METRICS_CONTINUOUS}" == "0" ]]; then
    start_metrics_sampler "${metr_raw}" "${metr_csv}" "${METRICS_INTERVAL_S}" "${PHASE_FILE}" "${RUN_START_TS}"
    samp_pid=$!
  fi

  local t0
  t0="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"

  set +e
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
  set -e

  if [[ "${METRICS_CONTINUOUS}" == "0" ]]; then
    kill "${samp_pid}" >/dev/null 2>&1 || true
    wait "${samp_pid}" >/dev/null 2>&1 || true
  fi

  local t1 wall
  t1="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
  wall="$(python3 - <<PY
t0=float("${t0}"); t1=float("${t1}"); print(f"{(t1-t0):.6f}")
PY
)"

  if [[ "${rc}" -ne 0 ]]; then
    echo "[WARN] replay rc=${rc} cycle=${cycle_idx} phase=${phase_tag} lam=${lam} (still summarizing what exists)"
  fi

  python3 -u scripts/summarize_one_run_sweep_trace.py \
    "${out_dir}" \
    "${lam}" \
    "${dur_s}" \
    "${wall}" \
    "${meta_json}" \
    "${SUMMARY_CSV}" \
    | tee -a "${OUT_BASE}/summarize.log"
}

# =========================
# main
# =========================
write_progress "INIT" ""
RUN_START_TS="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"

wait_ready 600

# build reqset once
write_progress "BUILD_REQSET" "reqset_path=${REQSET_PATH}"
if [[ ! -s "${REQSET_PATH}" ]]; then
  python3 -u scripts/build_reqset_sweep_trace.py \
    "${TRACE_PATH}" "${START_TS}" "${TRACE_WIN_S}" "${MAX_TOTAL_TOKENS}" "${REQSET_PATH}" \
    | tee "${OUT_BASE}/build_reqset.log"
else
  echo "[INFO] reuse existing reqset: ${REQSET_PATH}"
fi

# optional continuous metrics sampler
if [[ "${METRICS_CONTINUOUS}" == "1" ]]; then
  METR_CONT_RAW="${OUT_BASE}/metrics_raw.txt"
  METR_CONT_CSV="${OUT_BASE}/metrics_timeseries.csv"
  PHASE_FILE="${OUT_BASE}/phase.txt"
  echo "cycle=0 phase=init lam=0 mode=${MODE_LOW}" > "${PHASE_FILE}"
  start_metrics_sampler "${METR_CONT_RAW}" "${METR_CONT_CSV}" "${METRICS_INTERVAL_S}" "${PHASE_FILE}" "${RUN_START_TS}"
  CONT_PID=$!
  echo "[INFO] metrics continuous: ${METR_CONT_RAW} / ${METR_CONT_CSV} (pid=${CONT_PID})"
fi

for ((c=1; c<=NUM_CYCLES; c++)); do
  write_progress "CYCLE_LOW" "cycle=${c} lam=${LOW_LAMBDA} mode=${MODE_LOW} dur_s=${LOW_S}"
  log_phase_change "${c}" "low" "${LOW_LAMBDA}" "${MODE_LOW}"
  run_phase "${c}" "low" "${LOW_LAMBDA}" "$((SEED_ARRIVAL_BASE + c*2))" "${MODE_LOW}" "${LOW_S}"

  write_progress "CYCLE_HIGH" "cycle=${c} lam=${HIGH_LAMBDA} mode=${MODE_HIGH} dur_s=${HIGH_S}"
  log_phase_change "${c}" "high" "${HIGH_LAMBDA}" "${MODE_HIGH}"
  run_phase "${c}" "high" "${HIGH_LAMBDA}" "$((SEED_ARRIVAL_BASE + c*2 + 1))" "${MODE_HIGH}" "${HIGH_S}"
done

if [[ "${METRICS_CONTINUOUS}" == "1" ]]; then
  kill "${CONT_PID}" >/dev/null 2>&1 || true
  wait "${CONT_PID}" >/dev/null 2>&1 || true
fi

write_progress "DONE_ALL" ""
echo "[OK] oscillation run done: ${OUT_BASE}"
echo "[OK] summary: ${SUMMARY_CSV}"
