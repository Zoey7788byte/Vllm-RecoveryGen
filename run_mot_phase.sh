# 1) kill any server on port 8000
PID=$(lsof -tiTCP:8000 -sTCP:LISTEN || true)
if [ -n "$PID" ]; then
  echo "[INFO] Killing old server PID=$PID"
  kill "$PID" || true
  sleep 2
  PID2=$(lsof -tiTCP:8000 -sTCP:LISTEN || true)
  if [ -n "$PID2" ]; then
    echo "[INFO] Force killing PID=$PID2"
    kill -9 "$PID2" || true
  fi
else
  echo "[INFO] No server listening on :8000"
fi

#############motivation1,mem_vs_concurrency
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066
cd /home/ad/zteng/6022/vllm

GPU_DEV=2
MOT1_CASE_TAG="${MOT1_CASE_TAG:-rerun_$(date +%Y%m%d_%H%M%S)}"
MOT1_BASE_DIR="logs/RecoveryGen/Mot1_mem_concy"

############################
# MOT1: server (KV dominance / memory pressure)
############################
CUDA_VISIBLE_DEVICES=${GPU_DEV} \
VLLM_RECOVERY_BUDGET=0 VLLM_RECOVERY_PHASE=0 \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_FLAGS_JSON= \
MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 CFG_TAG=mot1_mem_concy \
BASE_DIR="${MOT1_BASE_DIR}" CASE_TAG="${MOT1_CASE_TAG}" \
bash scripts/mot1_mem_concy_server.sh


############################
# MOT1: client (single-point lambda sweep)
############################
RUN_TAG=Mot1_mem_concy_${MOT1_CASE_TAG}_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=${GPU_DEV} \
BASE_DIR="${MOT1_BASE_DIR}" CASE_TAG="${MOT1_CASE_TAG}" \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 \
LAMBDA_LIST="0.60 0.70 0.80 0.90 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90" \
DURATION_S=300 MAX_OUTSTANDING=256 MAX_TOTAL_TOKENS=14500 \
bash scripts/mot1_mem_concy_client.sh


# 3) 画图
cd /home/ad/zteng/6022/vllm

MOT1_SUMMARY="$(ls -dt logs/RecoveryGen/Mot1_mem_concy/*/Mot1_mem_concy_*_lam*/summary.csv 2>/dev/null | head -n 1)"
[[ -n "${MOT1_SUMMARY}" ]] || { echo "[ERROR] no mot1 summary found"; exit 1; }

MOT1_CASE_TAG="$(basename "$(dirname "$(dirname "${MOT1_SUMMARY}")")")"

echo "[INFO] MOT1_CASE_TAG=${MOT1_CASE_TAG}"
echo "[INFO] MOT1_SUMMARY=${MOT1_SUMMARY}"

python scripts/plot_mot1_kv_dominance.py \
  --summary "${MOT1_SUMMARY}" \
  --labels "${MOT1_CASE_TAG}" \
  --out figures/kv_dominance.pdf \
  --out-png figures/kv_dominance.png


  #############motivation2,recovery_induced_tail_instability
# 目标：
# - 在 low/high 负载交替下观测 preempt burst + waiting 峰值（oscillation）
# - 导出 TTFT/TPOT p99/p99.9 对比（tail instability）
# - 产出可直接用于论文图的 recovery_instability.{png,pdf}

# 1) 启 server（建议单独终端）
CUDA_VISIBLE_DEVICES=${GPU_DEV} \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 CFG_TAG=mot2_tail_instability \
bash scripts/mot2_tail_instability_server.sh


# 2) 跑 low/high 交替负载（建议新终端）
RUN_TAG=Mot2_tail_instability_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=${GPU_DEV} \
TRACE_PATH=/home/ad/zteng/6022/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=600 MAX_TOTAL_TOKENS=14500 \
MODE_LOW=poisson MODE_HIGH=poisson \
LOW_LAMBDA=0.60 HIGH_LAMBDA=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
MAX_OUTSTANDING=256 METRICS_INTERVAL_S=0.2 METRICS_CONTINUOUS=1 \
bash scripts/mot2_tail_instability_client.sh

# 3) 提取 proof2 统计 + 产物（图/表）
MOT2_RUN_DIR="$(ls -dt logs/RecoveryGen/Mot2_tail_instability/*_low0.60_high0.85_poisson-poisson_b4 | head -n 1)"
echo "[INFO] MOT2_RUN_DIR=${MOT2_RUN_DIR}"
bash scripts/plot_mot2_extract_from_timeserics.sh "${MOT2_RUN_DIR}"
python scripts/mot2_make_artifacts.py "${MOT2_RUN_DIR}"


################ Phase1 validation (manual commands) ############
#server
RUN_ID=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=2 \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_PHASE=1 VLLM_RECOVERY_BUDGET=16 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
BASE_DIR=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation \
SERVER_LOG_DIR=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/server_logs/${RUN_ID} \
SERVER_SIG_FILE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/current_server_sig.json \
bash /home/ad/zteng/6022/vllm/scripts/Phase1_validation_server.sh

#client
RUN_TAG=phase1_validate_${RUN_ID} \
OUT_BASE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/client_logs/phase1_validate_${RUN_ID} \
HOST=127.0.0.1 PORT=8000 \
TRACE_PATH=/home/ad/zteng/6022/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
MODE_LOW=poisson MODE_HIGH=poisson \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
BURST_B=4 BURST_WINDOW_S=2.0 MAX_OUTSTANDING=64 \
METRICS_INTERVAL_S=0.5 METRICS_CONTINUOUS=0 \
SERVER_SIG_FILE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/current_server_sig.json \
bash /home/ad/zteng/6022/vllm/scripts/Phase1_validation_client.sh


#################baseline与phase1的性能对比：
python3 /home/ad/zteng/6022/vllm/scripts/plot_analyze_phase0_phase1_effectiveness.py \
  --phase0-summary /home/ad/zteng/6022/vllm/logs/RecoveryGen/Baseline/client_logs/baseline_20260228_015032/summary.csv \
  --phase1-summary /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/client_logs/phase1_validate_${RUN_ID}/summary.csv \
  --phase1-recovery-ts /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/server_logs/${RUN_ID}/recovery/recovery_ts.csv \
  --phase1-events /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/server_logs/${RUN_ID}/recovery/recovery_events.jsonl \
  --json-out /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/phase1_effectiveness_report_${RUN_ID}.json \
  --plot-dir /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/plots_${RUN_ID}


################ Phase2 validation (manual commands) ############
# 可比性原则：与 Phase1 保持相同工作负载与系统参数，仅切算法开关
#   Phase1: VLLM_RECOVERY_PHASE=1, VLLM_RECOVERY_BUDGET=16
#   Phase2: VLLM_RECOVERY_PHASE=2, VLLM_RECOVERY_BUDGET=1
PHASE2_RUN_ID=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=2 \
BASE_DIR=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation \
SERVER_LOG_DIR=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/server_logs/${PHASE2_RUN_ID} \
SERVER_SIG_FILE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/current_server_sig.json \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_PHASE=2 VLLM_RECOVERY_BUDGET=1 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
bash /home/ad/zteng/6022/vllm/scripts/phase2_recovery_server.sh

# client
RUN_TAG=phase2_validate_${PHASE2_RUN_ID} \
OUT_BASE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/client_logs/phase2_validate_${PHASE2_RUN_ID} \
HOST=127.0.0.1 PORT=8000 \
TRACE_PATH=/home/ad/zteng/6022/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
MODE_LOW=poisson MODE_HIGH=poisson \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
BURST_B=4 BURST_WINDOW_S=2.0 MAX_OUTSTANDING=64 \
METRICS_INTERVAL_S=0.5 METRICS_CONTINUOUS=0 \
SERVER_SIG_FILE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/current_server_sig.json \
bash /home/ad/zteng/6022/vllm/scripts/phase2_recovery_client.sh

# phase1 vs phase2 effectiveness gates
PHASE1_RUN_ID="$(basename "$(ls -dt /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/server_logs/* | head -n 1)")"
PHASE2_RUN_ID_ANALYZE="$(basename "$(ls -dt /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/server_logs/* | head -n 1)")"

python3 /home/ad/zteng/6022/vllm/scripts/plot_analyze_phase1_phase2_effectiveness.py \
  --phase1-summary /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/client_logs/phase1_validate_${PHASE1_RUN_ID}/summary.csv \
  --phase2-summary /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/client_logs/phase2_validate_${PHASE2_RUN_ID_ANALYZE}/summary.csv \
  --phase1-ts /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/server_logs/${PHASE1_RUN_ID}/recovery/recovery_ts.csv \
  --phase2-ts /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/server_logs/${PHASE2_RUN_ID_ANALYZE}/recovery/recovery_ts.csv \
  --phase1-events /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase1_validation/server_logs/${PHASE1_RUN_ID}/recovery/recovery_events.jsonl \
  --phase2-events /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/server_logs/${PHASE2_RUN_ID_ANALYZE}/recovery/recovery_events.jsonl \
  --strict-m1-m2 \
  --json-out /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/phase2_effectiveness_report_${PHASE2_RUN_ID_ANALYZE}.json \
  --plot-dir /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/plots_${PHASE2_RUN_ID_ANALYZE}


################ Phase3 validation (manual commands) ############
# 可比性原则：与 Phase2 保持相同工作负载与系统参数，仅切算法开关
#   Phase2: VLLM_RECOVERY_PHASE=2, VLLM_RECOVERY_BUDGET=1
#   Phase3: VLLM_RECOVERY_PHASE=3, VLLM_RECOVERY_BUDGET=1
PHASE3_RUN_ID=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=2 \
BASE_DIR=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation \
SERVER_LOG_DIR=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/server_logs/${PHASE3_RUN_ID} \
SERVER_SIG_FILE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/current_server_sig.json \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_PHASE=3 VLLM_RECOVERY_BUDGET=1 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
bash /home/ad/zteng/6022/vllm/scripts/phase3_recovery_server.sh

#######client
RUN_TAG=phase3_validate_${PHASE3_RUN_ID} \
OUT_BASE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/client_logs/phase3_validate_${PHASE3_RUN_ID} \
HOST=127.0.0.1 PORT=8000 \
TRACE_PATH=/home/ad/zteng/6022/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
MODE_LOW=poisson MODE_HIGH=poisson \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
BURST_B=4 BURST_WINDOW_S=2.0 MAX_OUTSTANDING=64 \
METRICS_INTERVAL_S=0.5 METRICS_CONTINUOUS=0 \
SERVER_SIG_FILE=/home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/current_server_sig.json \
bash /home/ad/zteng/6022/vllm/scripts/phase3_recovery_client.sh


# phase2 vs phase3 effectiveness gates
# 2) 对比 Phase2 vs 新 Phase3
PHASE3_RUN_ID="$(basename "$(ls -dt /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/server_logs/* | head -n 1)")"

python3 /home/ad/zteng/6022/vllm/scripts/plot_analyze_phase2_phase3_effectiveness.py \
  --phase2-summary /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/client_logs/phase2_validate_20260301_162718/summary.csv \
  --phase2-ts /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/server_logs/20260301_162718/recovery/recovery_ts.csv \
  --phase2-events /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase2_validation/server_logs/20260301_162718/recovery/recovery_events.jsonl \
  --phase3-summary /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/client_logs/phase3_validate_${PHASE3_RUN_ID}/summary.csv \
  --phase3-ts /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/server_logs/${PHASE3_RUN_ID}/recovery/recovery_ts.csv \
  --phase3-events /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/server_logs/${PHASE3_RUN_ID}/recovery/recovery_events.jsonl \
  --json-out /home/ad/zteng/6022/vllm/logs/RecoveryGen/Phase3_validation/phase3_effectiveness_report_${PHASE3_RUN_ID}.json


#跑run_compare_alg所有算法的对比：
cd /home/ad/zteng/6022/vllm
TS=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="/home/ad/zteng/6022/vllm/logs/run_compare_alg_all_lambda1p2_${TS}.nohup.log"
PID_FILE="/home/ad/zteng/6022/vllm/logs/run_compare_alg_all_lambda1p2_${TS}.pid"

nohup env MEM_COMMON=0.75 bash /home/ad/zteng/6022/vllm/run_compare_alg.sh \
  > "${NOHUP_LOG}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "PID=${PID}"
echo "NOHUP_LOG=${NOHUP_LOG}"

#实时查看
tail -f /home/ad/zteng/6022/vllm/logs/run_compare_alg_all_lambda1p2_*.nohup.log


cd /home/ad/zteng/6022/vllm
TS=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="/home/ad/zteng/6022/vllm/logs/run_compare_alg_batch_${TS}.nohup.log"
PID_FILE="/home/ad/zteng/6022/vllm/logs/run_compare_alg_batch_${TS}.pid"
PLOT_ROOT="/home/ad/zteng/6022/vllm/logs/RecoveryGen/compare_lambda_plots_batch"

nohup env \
  MEM_COMMON=0.75 \
  BATCH_LAMBDAS=0.9,1.2 \
  BATCH_COMPARE_GAP_METRIC=stall_max \
  BATCH_PLOT_ROOT="${PLOT_ROOT}" \
  bash /home/ad/zteng/6022/vllm/run_compare_alg.sh all \
  > "${NOHUP_LOG}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "PID=${PID}"
echo "NOHUP_LOG=${NOHUP_LOG}"



