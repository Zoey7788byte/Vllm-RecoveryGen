
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
cd /home/ad/zteng/vllm

GPU_DEV=2
MOT1_CASE_TAG="${MOT1_CASE_TAG:-rerun_$(date +%Y%m%d_%H%M%S)}"
MOT1_BASE_DIR="logs/RecoveryGen/Mot1_mem_concy"

# 1) 启 server
CUDA_VISIBLE_DEVICES=${GPU_DEV} \
VLLM_RECOVERY_BUDGET=0 VLLM_RECOVERY_PHASE=0 \
VLLM_RECOVERY_OBS=0 VLLM_RECOVERY_FLAGS_JSON= \
MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 CFG_TAG=mot1_mem_concy \
BASE_DIR="${MOT1_BASE_DIR}" CASE_TAG="${MOT1_CASE_TAG}" \
bash scripts/mot1_mem_concy_server.sh

# 2) client: 新窗口 + 单点 lambda 探测
RUN_TAG=Mot1_mem_concy_${MOT1_CASE_TAG}_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=${GPU_DEV} \
BASE_DIR="${MOT1_BASE_DIR}" CASE_TAG="${MOT1_CASE_TAG}" \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 \
LAMBDA_LIST="0.60 0.70 0.80 0.9 1.00 1.10 1.20 1.30 1.40 1.50" \
DURATION_S=180 MAX_OUTSTANDING=64 MAX_TOTAL_TOKENS=6000 \
bash scripts/mot1_mem_concy_client.sh

# 3) 画图
MOT1_SUMMARY="$(ls -dt ${MOT1_BASE_DIR}/${MOT1_CASE_TAG}/Mot1_mem_concy_${MOT1_CASE_TAG}_*_lam*/summary.csv | head -n 1)"
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
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=600 MAX_TOTAL_TOKENS=14500 \
MODE_LOW=poisson MODE_HIGH=poisson \
LOW_LAMBDA=0.60 HIGH_LAMBDA=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
MAX_OUTSTANDING=256 METRICS_INTERVAL_S=0.2 METRICS_CONTINUOUS=1 \
bash scripts/mot2_tail_instability_client.sh

# 3) 提取 proof2 统计 + 产物（图/表）
MOT2_RUN_DIR="$(ls -dt logs/RecoveryGen/Mot2_tail_instability/*_low0.60_high0.85_poisson-poisson_b4 | head -n 1)"
echo "[INFO] MOT2_RUN_DIR=${MOT2_RUN_DIR}"
bash scripts/proofmot2_extract_from_timeserics.sh "${MOT2_RUN_DIR}"
python scripts/mot2_make_artifacts.py "${MOT2_RUN_DIR}"

# 4) 关键产物位置
# - ${MOT2_RUN_DIR}/artifacts_proof2/recovery_instability.pdf
# - ${MOT2_RUN_DIR}/artifacts_proof2/table_tail_pairs.csv
# - ${MOT2_RUN_DIR}/summary.csv


##################3.算法对比######
# 目标：baseline(chunked=1/0) 与 full 全部对比
cd /data/home/ad/zteng/vllm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066

# Vllm(chunked) server
CUDA_VISIBLE_DEVICES=2 \
BASE_DIR=logs/RecoveryGen/Validate/Baseline \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_PHASE=0 VLLM_RECOVERY_BUDGET=0 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
CFG_TAG=recovergen_validate_baseline \
bash scripts/phase0_recovery_server.sh


# Vllm(chunked) client
RUN_TAG=baseline_$(date +%Y%m%d_%H%M%S) \
OUT_BASE=logs/RecoveryGen/Validate/Baseline/client_logs/${RUN_TAG} \
SERVER_SIG_FILE=logs/RecoveryGen/Validate/Baseline/current_server_sig.json \
SEGMENT_TIMEOUT_S=2400 SEGMENT_TIMEOUT_KILL_S=30 \
MAX_OUTSTANDING=64 \
CLIENT_REQ_TIMEOUT_S=180 \
CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S=10 \
CLIENT_REQ_SOCK_READ_TIMEOUT_S=45 \
SEED_REQSET=0 SEED_ARRIVAL_BASE=0 \
FIXED_ARRIVAL_SEED=1 \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/phase0_recovery_client.sh



# Vllm (no chunked prefill)
cd /home/ad/zteng/vllm

BASE_DIR="/home/ad/zteng/vllm/logs/RecoveryGen/Validate/Baseline_no_chunked"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="baseline_nochunk_${RUN_TS}"
SERVER_LOG_DIR="${BASE_DIR}/server_logs/${RUN_TS}"

mkdir -p "${SERVER_LOG_DIR}" "${CLIENT_OUT_BASE}"

# Vllm server
CUDA_VISIBLE_DEVICES=2 \
BASE_DIR="${BASE_DIR}" \
SERVER_LOG_DIR="${SERVER_LOG_DIR}" \
SERVER_SIG_FILE="${BASE_DIR}/current_server_sig.json" \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_PHASE=0 VLLM_RECOVERY_BUDGET=0 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 ENABLE_CHUNKED_PREFILL=0 NUM_SCHED_STEPS=1 \
CFG_TAG=recovergen_validate_baseline_nochunk \
bash scripts/phase0_recovery_server.sh

# Vllm client
RUN_TAG="${RUN_TAG}" \
OUT_BASE="${CLIENT_OUT_BASE}" \
SERVER_SIG_FILE="${BASE_DIR}/current_server_sig.json" \
SEGMENT_TIMEOUT_S=2400 SEGMENT_TIMEOUT_KILL_S=30 \
MAX_OUTSTANDING=64 \
CLIENT_REQ_TIMEOUT_S=180 \
CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S=10 \
CLIENT_REQ_SOCK_READ_TIMEOUT_S=45 \
SEED_REQSET=0 SEED_ARRIVAL_BASE=0 \
FIXED_ARRIVAL_SEED=1 \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/phase0_recovery_client.sh



# RecoverGen server（phase1+phase2+phase3）
CUDA_VISIBLE_DEVICES=2 \
BASE_DIR=logs/RecoveryGen/Validate/Full \
VLLM_RECOVERY_OBS=1 VLLM_RECOVERY_PHASE=3 VLLM_RECOVERY_BUDGET=8 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=8 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
REC_T_CYC_MS=60 REC_BUDGET_INIT_MS=1.6 REC_BUDGET_MIN_MS=1.2 \
REC_DPLUS_MS=0.5 REC_DMINUS_MS=4.0 REC_STALL_MS=120 \
VLLM_RECOVERY_SWAP_MS_PER_BLOCK=0.30 REC_PREFETCH_MAX_BLOCKS=512 \
CFG_TAG=recovergen_validate_full_p123 \
bash scripts/phase3_recovery_server.sh

# RecoverGen client
RUN_TAG=full_$(date +%Y%m%d_%H%M%S) \
OUT_BASE=logs/RecoveryGen/Validate/Full/client_logs/${RUN_TAG} \
SERVER_SIG_FILE=logs/RecoveryGen/Validate/Full/current_server_sig.json \
SEGMENT_TIMEOUT_S=2400 SEGMENT_TIMEOUT_KILL_S=30 \
MAX_OUTSTANDING=64 \
CLIENT_REQ_TIMEOUT_S=180 \
CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S=10 \
CLIENT_REQ_SOCK_READ_TIMEOUT_S=45 \
SEED_REQSET=0 SEED_ARRIVAL_BASE=0 \
FIXED_ARRIVAL_SEED=1 \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/phase2_recovery_client.sh




# 三方同屏对比：Vllm(chunked) / Vllm / RecoverGen
cd /home/ad/zteng/vllm
OUT_JSON_THREE_WAY="/data/home/ad/zteng/vllm/logs/RecoveryGen/Validate/effectiveness_compare_Vllm(chunked)_Vllm_RecoverGen_$(date +%Y%m%d_%H%M%S).json"
bash scripts/compare_three_recoverygen.sh \
  --chunk-on-root /data/home/ad/zteng/vllm/logs/RecoveryGen/Validate/Baseline \
  --chunk-off-root /data/home/ad/zteng/vllm/logs/RecoveryGen/Validate/Baseline_no_chunked \
  --full-root /data/home/ad/zteng/vllm/logs/RecoveryGen/Validate/Full \
  --thresholds "1000,2000,5000" \
  --out "${OUT_JSON_THREE_WAY}"
echo "[INFO] three-way compare out: ${OUT_JSON_THREE_WAY}"
