#!/usr/bin/env bash
set -euo pipefail

# =========================
# proof2_extract.sh
# 从每个 phase 子目录的 metrics_timeseries.csv 提取证明2所需统计量
# 输出：
#   - proof2_phase_summary.csv  (每个 phase 一行)
#   - proof2_cycle_summary.csv  (每个 cycle 一行，low/high 对齐)
#   - metrics_timeseries_all.csv (可选拼接，全局 time series)
# 用法：
#   bash proof2_extract.sh logs/RecoveryGen/Mot2_tail_instability/<RUN_DIR_NAME>
# =========================

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: $0 <mot2_run_dir>"
  echo "Example: $0 logs/RecoveryGen/Mot2_tail_instability/Mot2_tail_instability_20260223_140000_low0.60_high0.85_poisson-poisson_b4"
  exit 1
fi

OUT_PHASE="${RUN_DIR}/proof2_phase_summary.csv"
OUT_CYCLE="${RUN_DIR}/proof2_cycle_summary.csv"
OUT_ALL_TS="${RUN_DIR}/metrics_timeseries_all.csv"
RUN_LEVEL_TS="${RUN_DIR}/metrics_timeseries.csv"
TMP_SPLIT_DIR=""
if [[ -f "${RUN_LEVEL_TS}" ]]; then
  TMP_SPLIT_DIR="$(mktemp -d /tmp/proof2_extract_split.XXXXXX)"
fi

# 你脚本的 metrics_timeseries.csv 列顺序（你贴出来的是这样）：
# ts,t_rel_s,phase,cycle,mode,lambda_rps,preempt_total,preempt_rate_per_s,req_waiting,req_running,gpu_cache_usage_perc
# 如果以后列变了，只要在 awk 里改列号即可。
COL_TS=1
COL_TREL=2
COL_PHASE=3
COL_CYCLE=4
COL_MODE=5
COL_LAM=6
COL_PREEMPT_TOT=7
COL_PREEMPT_RATE=8
COL_WAITING=9
COL_RUNNING=10
COL_GPU=11

echo "[INFO] RUN_DIR=${RUN_DIR}"

# -------------------------
# 1) phase-level 汇总：每个 phase 子目录一行
# -------------------------
echo "cond_dir,cycle,phase,mode,lambda_rps,T_s,rows,preempt_min,preempt_max,preempt_delta,preempt_rate_avg,preempt_rate_max,waiting_avg,waiting_max,running_avg,running_max,gpu_avg,gpu_max,parse_nan_rows" \
  > "${OUT_PHASE}"

shopt -s nullglob
phase_dirs=( "${RUN_DIR}"/cycle_*_*_lambda_* )
if (( ${#phase_dirs[@]} == 0 )); then
  echo "[ERR] No phase dirs found under ${RUN_DIR} (expect cycle_*_*_lambda_*)"
  exit 1
fi

for d in "${phase_dirs[@]}"; do
  f="${d}/metrics_timeseries.csv"
  if [[ ! -f "${f}" ]]; then
    if [[ -f "${RUN_LEVEL_TS}" ]]; then
      bname="$(basename "${d}")"
      if [[ "${bname}" =~ ^cycle_([0-9]+)_(low|high)_lambda_ ]]; then
        cyc="${BASH_REMATCH[1]}"
        ph="${BASH_REMATCH[2]}"
        f="${TMP_SPLIT_DIR}/${bname}.metrics_timeseries.csv"
        head -n 1 "${RUN_LEVEL_TS}" > "${f}"
        awk -F, -v cyc="${cyc}" -v ph="${ph}" '
          NR==1 {next}
          {
            p=tolower($3)
            c=$4
            sub(/^[[:space:]]+/, "", p); sub(/[[:space:]]+$/, "", p)
            sub(/^[[:space:]]+/, "", c); sub(/[[:space:]]+$/, "", c)
            if (p==ph && c==cyc) print $0
          }' "${RUN_LEVEL_TS}" >> "${f}"
        if [[ $(wc -l < "${f}") -le 1 ]]; then
          echo "[WARN] cannot split ${bname} from ${RUN_LEVEL_TS}, skip"
          continue
        fi
      else
        echo "[WARN] missing ${f}, and cannot parse cycle/phase from ${bname}, skip"
        continue
      fi
    else
      echo "[WARN] missing ${f}, skip"
      continue
    fi
  fi

  # 计算 T_s：用 t_rel_s 的 max-min 近似（只用于 sanity check）
  awk -F, -v cond_dir="${d}" \
    -v CTS=${COL_TS} -v CTREL=${COL_TREL} -v CPH=${COL_PHASE} -v CCYC=${COL_CYCLE} -v CMODE=${COL_MODE} -v CLAM=${COL_LAM} \
    -v CPRET=${COL_PREEMPT_TOT} -v CPRER=${COL_PREEMPT_RATE} -v CWAIT=${COL_WAITING} -v CRUN=${COL_RUNNING} -v CGPU=${COL_GPU} \
    '
    function is_nan(x){ return (x=="" || x=="nan" || x=="NaN" || x=="NAN"); }
    NR==1{ next }  # skip header
    {
      rows++

      phase=$CPH
      cycle=$CCYC
      mode=$CMODE
      lam=$CLAM

      # t_rel range
      tr=$CTREL+0
      if(rows==1){ tr_min=tr; tr_max=tr; } else { if(tr<tr_min)tr_min=tr; if(tr>tr_max)tr_max=tr; }

      # preempt_total min/max
      pt=$CPRET
      if(is_nan(pt)){ nan_rows++; next_row_nan=1; } else { pt=pt+0; next_row_nan=0; }

      if(!next_row_nan){
        if(pt_min=="" || pt<pt_min) pt_min=pt
        if(pt_max=="" || pt>pt_max) pt_max=pt
      }

      # preempt_rate avg/max（忽略 nan）
      pr=$CPRER
      if(!is_nan(pr)){
        pr=pr+0
        pr_sum+=pr; pr_n++
        if(pr_max=="" || pr>pr_max) pr_max=pr
      } else {
        nan_rows++
      }

      # waiting avg/max（忽略 nan）
      w=$CWAIT
      if(!is_nan(w)){
        w=w+0
        w_sum+=w; w_n++
        if(w_max=="" || w>w_max) w_max=w
      } else { nan_rows++ }

      # running avg/max（忽略 nan）
      r=$CRUN
      if(!is_nan(r)){
        r=r+0
        r_sum+=r; r_n++
        if(r_max=="" || r>r_max) r_max=r
      } else { nan_rows++ }

      # gpu avg/max（忽略 nan）
      g=$CGPU
      if(!is_nan(g)){
        g=g+0
        g_sum+=g; g_n++
        if(g_max=="" || g>g_max) g_max=g
      } else { nan_rows++ }
    }
    END{
      # 保护：如果 preempt_total 没采到，置空并提示
      if(pt_min=="" || pt_max==""){ pt_min=""; pt_max=""; pt_delta=""; } else { pt_delta=pt_max-pt_min; }

      T_s = (rows>0 ? (tr_max-tr_min) : 0)

      pr_avg = (pr_n>0 ? pr_sum/pr_n : "")
      w_avg  = (w_n>0  ? w_sum/w_n  : "")
      r_avg  = (r_n>0  ? r_sum/r_n  : "")
      g_avg  = (g_n>0  ? g_sum/g_n  : "")

      # 输出
      printf "%s,%s,%s,%s,%.6f,%.3f,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d\n",
        cond_dir, cycle, phase, mode, lam+0, T_s, rows,
        pt_min, pt_max, pt_delta,
        pr_avg, pr_max,
        w_avg, w_max,
        r_avg, r_max,
        g_avg, g_max,
        nan_rows
    }' "${f}" >> "${OUT_PHASE}"
done

echo "[OK] wrote ${OUT_PHASE}"

# -------------------------
# 2) cycle-level 汇总：把同一 cycle 的 low/high 拼到一行，便于看锯齿条件
# -------------------------
# 字段：cycle, low_preempt_delta, high_preempt_delta, low_waiting_max, high_waiting_max, low_preempt_rate_max, high_preempt_rate_max ...
python3 - "${RUN_DIR}" <<'PY'
import csv, os, sys, math
run_dir = sys.argv[1]
phase_csv = os.path.join(run_dir, "proof2_phase_summary.csv")
out_csv   = os.path.join(run_dir, "proof2_cycle_summary.csv")

def fnum(x):
    if x is None: return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan": return None
    try:
        return float(s)
    except:
        return None

rows = []
with open(phase_csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# group by cycle
by_cycle = {}
for row in rows:
    cyc = row["cycle"].strip()
    ph  = row["phase"].strip().lower()  # low/high
    by_cycle.setdefault(cyc, {})[ph] = row

# write
fields = [
    "cycle",
    "low_lambda_rps","high_lambda_rps",
    "low_preempt_delta","high_preempt_delta",
    "low_preempt_rate_avg","high_preempt_rate_avg",
    "low_preempt_rate_max","high_preempt_rate_max",
    "low_waiting_avg","high_waiting_avg",
    "low_waiting_max","high_waiting_max",
    "low_running_avg","high_running_avg",
    "low_running_max","high_running_max",
    "separation_ok"
]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for cyc in sorted(by_cycle.keys(), key=lambda x: int(x) if x.isdigit() else x):
        low = by_cycle[cyc].get("low")
        high= by_cycle[cyc].get("high")

        def get(row, k):
            return row.get(k,"") if row else ""

        low_delta  = fnum(get(low,  "preempt_delta"))
        high_delta = fnum(get(high, "preempt_delta"))
        low_prmax  = fnum(get(low,  "preempt_rate_max"))
        high_prmax = fnum(get(high, "preempt_rate_max"))
        low_wmax   = fnum(get(low,  "waiting_max"))
        high_wmax  = fnum(get(high, "waiting_max"))

        # separation_ok：用于证明2的必要条件之一（不是充分）
        # 规则：low_preempt_delta==0 且 high_preempt_delta>0 且 high_waiting_max >= low_waiting_max
        sep = ""
        if low_delta is not None and high_delta is not None:
            ok = (abs(low_delta) < 1e-9) and (high_delta > 0)
            if low_wmax is not None and high_wmax is not None:
                ok = ok and (high_wmax >= low_wmax)
            sep = "1" if ok else "0"

        w.writerow({
            "cycle": cyc,
            "low_lambda_rps":  get(low,  "lambda_rps"),
            "high_lambda_rps": get(high, "lambda_rps"),
            "low_preempt_delta":  get(low,  "preempt_delta"),
            "high_preempt_delta": get(high, "preempt_delta"),
            "low_preempt_rate_avg":  get(low,  "preempt_rate_avg"),
            "high_preempt_rate_avg": get(high, "preempt_rate_avg"),
            "low_preempt_rate_max":  get(low,  "preempt_rate_max"),
            "high_preempt_rate_max": get(high, "preempt_rate_max"),
            "low_waiting_avg":  get(low,  "waiting_avg"),
            "high_waiting_avg": get(high, "waiting_avg"),
            "low_waiting_max":  get(low,  "waiting_max"),
            "high_waiting_max": get(high, "waiting_max"),
            "low_running_avg":  get(low,  "running_avg"),
            "high_running_avg": get(high, "running_avg"),
            "low_running_max":  get(low,  "running_max"),
            "high_running_max": get(high, "running_max"),
            "separation_ok": sep
        })

print(f"[OK] wrote {out_csv}")
PY

# -------------------------
# 3) （可选）拼接全局 time series，便于画锯齿图
# -------------------------
# 若你想画 “preempt_rate_per_s vs t_rel_s (按 phase 着色)” 之类的图，建议拼起来
first=""
for d in "${phase_dirs[@]}"; do
  f="${d}/metrics_timeseries.csv"
  if [[ -f "${f}" ]]; then
    first="${f}"
    break
  fi
done

if [[ -n "${first}" ]]; then
  head -n 1 "${first}" > "${OUT_ALL_TS}"
  for d in "${phase_dirs[@]}"; do
    f="${d}/metrics_timeseries.csv"
    [[ -f "${f}" ]] || continue
    tail -n +2 "${f}"
  done | sort -t, -k1,1 >> "${OUT_ALL_TS}"
  echo "[OK] wrote ${OUT_ALL_TS}"
elif [[ -f "${RUN_LEVEL_TS}" ]]; then
  cp -f "${RUN_LEVEL_TS}" "${OUT_ALL_TS}"
  echo "[OK] wrote ${OUT_ALL_TS} (from run-level metrics_timeseries.csv)"
else
  echo "[WARN] cannot build ${OUT_ALL_TS} (no metrics_timeseries.csv found)"
fi

if [[ -n "${TMP_SPLIT_DIR}" && -d "${TMP_SPLIT_DIR}" ]]; then
  rm -rf "${TMP_SPLIT_DIR}"
fi

echo "[DONE] Proof2 extraction finished."
echo "  - Phase summary: ${OUT_PHASE}"
echo "  - Cycle summary: ${OUT_CYCLE}"
echo "  - All time series: ${OUT_ALL_TS}"
