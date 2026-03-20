#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import re
import sys

import numpy as np

NUM_RE = r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
TS_PAT = re.compile(rf"^@@TS\s+{NUM_RE}\s*$")
GPU_USED_PAT = re.compile(rf"^@@GPU_MEM_USED_MB\s+{NUM_RE}\s*$")
GPU_TOTAL_PAT = re.compile(rf"^@@GPU_MEM_TOTAL_MB\s+{NUM_RE}\s*$")
METRIC_PAT = re.compile(rf"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{{.*\}})?\s+{NUM_RE}\s*$")

PREEMPT_METRIC = "vllm:num_preemptions_total"
GPU_CACHE_METRIC = "vllm:gpu_cache_usage_perc"
AVG_PROMPT_TP_METRIC = "vllm:avg_prompt_throughput_toks_per_s"
AVG_GEN_TP_METRIC = "vllm:avg_generation_throughput_toks_per_s"
REQ_SWAPPED_METRIC = "vllm:num_requests_swapped"
PROMPT_TOK_METRIC = "vllm:prompt_tokens_total"
GEN_TOK_METRIC = "vllm:generation_tokens_total"

SWAPIN_COUNTER_CANDIDATES = {
    "vllm:recovery_swapin_blocks_total",
    "vllm:swapin_blocks_total",
    "recovery_swapin_blocks_total",
    "swapin_blocks_total",
}
SWAPOUT_COUNTER_CANDIDATES = {
    "vllm:recovery_swapout_blocks_total",
    "vllm:swapout_blocks_total",
    "recovery_swapout_blocks_total",
    "swapout_blocks_total",
}
RECOMPUTE_COUNTER_CANDIDATES = {
    "vllm:recovery_recompute_tokens_total",
    "vllm:recompute_tokens_total",
    "recovery_recompute_tokens_total",
    "recompute_tokens_total",
}
STALL_MS_COUNTER_CANDIDATES = {
    "vllm:recovery_restore_progress_stall_ms_total",
    "vllm:restore_progress_stall_ms_total",
    "recovery_restore_progress_stall_ms_total",
    "restore_progress_stall_ms_total",
}


def percentile_or_nan(arr, p: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(np.array(arr, dtype=float), p))


def max_or_nan(arr) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.max(np.array(arr, dtype=float)))


def delta_or_nan(arr, default_nan: bool = True) -> float:
    if len(arr) < 2:
        return float("nan") if default_nan else 0.0
    return float(arr[-1] - arr[0])


def read_metrics_summary(metrics_raw: str):
    if not os.path.exists(metrics_raw):
        return {
            "preempt_sum_delta": 0.0,
            "gpu_mem_used_mb_p50": float("nan"),
            "gpu_mem_used_mb_p90": float("nan"),
            "gpu_mem_used_mb_p99": float("nan"),
            "gpu_mem_used_mb_max": float("nan"),
            "gpu_mem_total_mb_p50": float("nan"),
            "gpu_mem_util_perc_p50": float("nan"),
            "gpu_mem_util_perc_p90": float("nan"),
            "gpu_mem_util_perc_p99": float("nan"),
            "gpu_mem_util_perc_max": float("nan"),
            "gpu_cache_usage_perc_p50": float("nan"),
            "gpu_cache_usage_perc_p90": float("nan"),
            "gpu_cache_usage_perc_p99": float("nan"),
            "gpu_cache_usage_perc_max": float("nan"),
            "avg_prompt_toks_per_s_p50": float("nan"),
            "avg_prompt_toks_per_s_p90": float("nan"),
            "avg_prompt_toks_per_s_p99": float("nan"),
            "avg_prompt_toks_per_s_max": float("nan"),
            "avg_generation_toks_per_s_p50": float("nan"),
            "avg_generation_toks_per_s_p90": float("nan"),
            "avg_generation_toks_per_s_p99": float("nan"),
            "avg_generation_toks_per_s_max": float("nan"),
            "num_requests_swapped_p50": float("nan"),
            "num_requests_swapped_p90": float("nan"),
            "num_requests_swapped_p99": float("nan"),
            "num_requests_swapped_max": float("nan"),
            "prompt_tokens_delta": float("nan"),
            "generation_tokens_delta": float("nan"),
            "decode_toks_per_s_metrics_delta": float("nan"),
            "swapin_blocks_delta": float("nan"),
            "swapout_blocks_delta": float("nan"),
            "recompute_tokens_delta": float("nan"),
            "restore_progress_stall_ms_delta": float("nan"),
            "metrics_elapsed_s": float("nan"),
            "metrics_samples": 0,
        }

    ts_vals = []
    pre_vals = []
    gpu_used_vals = []
    gpu_total_vals = []
    gpu_util_vals = []
    gpu_cache_vals = []
    avg_prompt_tp_vals = []
    avg_gen_tp_vals = []
    req_swapped_vals = []
    prompt_tok_vals = []
    gen_tok_vals = []
    swapin_vals = []
    swapout_vals = []
    recompute_vals = []
    stall_ms_vals = []

    in_snapshot = False
    cur_ts = None
    cur_pre = None
    cur_used = None
    cur_total = None
    cur_cache = None
    cur_avg_prompt_tp = None
    cur_avg_gen_tp = None
    cur_req_swapped = None
    cur_prompt_tok = None
    cur_gen_tok = None
    cur_swapin = None
    cur_swapout = None
    cur_recompute = None
    cur_stall_ms = None

    def flush_snapshot():
        nonlocal in_snapshot
        if not in_snapshot:
            return
        if cur_ts is not None:
            ts_vals.append(cur_ts)
        if cur_pre is not None:
            pre_vals.append(cur_pre)
        if cur_used is not None:
            gpu_used_vals.append(cur_used)
        if cur_total is not None:
            gpu_total_vals.append(cur_total)
        if cur_used is not None and cur_total is not None and cur_total > 0:
            gpu_util_vals.append((cur_used / cur_total) * 100.0)
        if cur_cache is not None:
            gpu_cache_vals.append(cur_cache)
        if cur_avg_prompt_tp is not None:
            avg_prompt_tp_vals.append(cur_avg_prompt_tp)
        if cur_avg_gen_tp is not None:
            avg_gen_tp_vals.append(cur_avg_gen_tp)
        if cur_req_swapped is not None:
            req_swapped_vals.append(cur_req_swapped)
        if cur_prompt_tok is not None:
            prompt_tok_vals.append(cur_prompt_tok)
        if cur_gen_tok is not None:
            gen_tok_vals.append(cur_gen_tok)
        if cur_swapin is not None:
            swapin_vals.append(cur_swapin)
        if cur_swapout is not None:
            swapout_vals.append(cur_swapout)
        if cur_recompute is not None:
            recompute_vals.append(cur_recompute)
        if cur_stall_ms is not None:
            stall_ms_vals.append(cur_stall_ms)

    with open(metrics_raw, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            m = TS_PAT.match(s)
            if m:
                flush_snapshot()
                in_snapshot = True
                cur_ts = float(m.group(1))
                cur_pre = None
                cur_used = None
                cur_total = None
                cur_cache = None
                cur_avg_prompt_tp = None
                cur_avg_gen_tp = None
                cur_req_swapped = None
                cur_prompt_tok = None
                cur_gen_tok = None
                cur_swapin = None
                cur_swapout = None
                cur_recompute = None
                cur_stall_ms = None
                continue

            mm = GPU_USED_PAT.match(s)
            if mm:
                cur_used = float(mm.group(1))
                continue

            mm = GPU_TOTAL_PAT.match(s)
            if mm:
                cur_total = float(mm.group(1))
                continue

            mm = METRIC_PAT.match(s)
            if not mm:
                continue
            name = mm.group(1)
            val = float(mm.group(2))
            if name == PREEMPT_METRIC:
                cur_pre = val
            elif name == GPU_CACHE_METRIC:
                cur_cache = val
            elif name == AVG_PROMPT_TP_METRIC:
                cur_avg_prompt_tp = val
            elif name == AVG_GEN_TP_METRIC:
                cur_avg_gen_tp = val
            elif name == REQ_SWAPPED_METRIC:
                cur_req_swapped = val
            elif name == PROMPT_TOK_METRIC:
                cur_prompt_tok = val
            elif name == GEN_TOK_METRIC:
                cur_gen_tok = val
            elif name in SWAPIN_COUNTER_CANDIDATES:
                cur_swapin = val
            elif name in SWAPOUT_COUNTER_CANDIDATES:
                cur_swapout = val
            elif name in RECOMPUTE_COUNTER_CANDIDATES:
                cur_recompute = val
            elif name in STALL_MS_COUNTER_CANDIDATES:
                cur_stall_ms = val

    flush_snapshot()

    pre_delta = delta_or_nan(pre_vals, default_nan=False)
    metrics_elapsed_s = delta_or_nan(ts_vals)
    prompt_tok_delta = delta_or_nan(prompt_tok_vals)
    gen_tok_delta = delta_or_nan(gen_tok_vals)
    decode_toks_per_s_metrics_delta = float("nan")
    if np.isfinite(metrics_elapsed_s) and metrics_elapsed_s > 0 and np.isfinite(gen_tok_delta):
        decode_toks_per_s_metrics_delta = gen_tok_delta / metrics_elapsed_s

    return {
        "preempt_sum_delta": pre_delta,
        "gpu_mem_used_mb_p50": percentile_or_nan(gpu_used_vals, 0.50),
        "gpu_mem_used_mb_p90": percentile_or_nan(gpu_used_vals, 0.90),
        "gpu_mem_used_mb_p99": percentile_or_nan(gpu_used_vals, 0.99),
        "gpu_mem_used_mb_max": max_or_nan(gpu_used_vals),
        "gpu_mem_total_mb_p50": percentile_or_nan(gpu_total_vals, 0.50),
        "gpu_mem_util_perc_p50": percentile_or_nan(gpu_util_vals, 0.50),
        "gpu_mem_util_perc_p90": percentile_or_nan(gpu_util_vals, 0.90),
        "gpu_mem_util_perc_p99": percentile_or_nan(gpu_util_vals, 0.99),
        "gpu_mem_util_perc_max": max_or_nan(gpu_util_vals),
        "gpu_cache_usage_perc_p50": percentile_or_nan(gpu_cache_vals, 0.50),
        "gpu_cache_usage_perc_p90": percentile_or_nan(gpu_cache_vals, 0.90),
        "gpu_cache_usage_perc_p99": percentile_or_nan(gpu_cache_vals, 0.99),
        "gpu_cache_usage_perc_max": max_or_nan(gpu_cache_vals),
        "avg_prompt_toks_per_s_p50": percentile_or_nan(avg_prompt_tp_vals, 0.50),
        "avg_prompt_toks_per_s_p90": percentile_or_nan(avg_prompt_tp_vals, 0.90),
        "avg_prompt_toks_per_s_p99": percentile_or_nan(avg_prompt_tp_vals, 0.99),
        "avg_prompt_toks_per_s_max": max_or_nan(avg_prompt_tp_vals),
        "avg_generation_toks_per_s_p50": percentile_or_nan(avg_gen_tp_vals, 0.50),
        "avg_generation_toks_per_s_p90": percentile_or_nan(avg_gen_tp_vals, 0.90),
        "avg_generation_toks_per_s_p99": percentile_or_nan(avg_gen_tp_vals, 0.99),
        "avg_generation_toks_per_s_max": max_or_nan(avg_gen_tp_vals),
        "num_requests_swapped_p50": percentile_or_nan(req_swapped_vals, 0.50),
        "num_requests_swapped_p90": percentile_or_nan(req_swapped_vals, 0.90),
        "num_requests_swapped_p99": percentile_or_nan(req_swapped_vals, 0.99),
        "num_requests_swapped_max": max_or_nan(req_swapped_vals),
        "prompt_tokens_delta": prompt_tok_delta,
        "generation_tokens_delta": gen_tok_delta,
        "decode_toks_per_s_metrics_delta": decode_toks_per_s_metrics_delta,
        "swapin_blocks_delta": delta_or_nan(swapin_vals),
        "swapout_blocks_delta": delta_or_nan(swapout_vals),
        "recompute_tokens_delta": delta_or_nan(recompute_vals),
        "restore_progress_stall_ms_delta": delta_or_nan(stall_ms_vals),
        "metrics_elapsed_s": metrics_elapsed_s,
        "metrics_samples": max(
            len(pre_vals),
            len(gpu_used_vals),
            len(gpu_cache_vals),
            len(avg_gen_tp_vals),
            len(req_swapped_vals),
        ),
    }


def read_client_metrics(client_csv: str):
    ok_200 = 0
    total_rows = 0
    total_out_tok_ok = 0
    tt = []
    tpot = []
    stall_gap = []
    send_abs_vals = []
    done_abs_vals = []

    with open(client_csv, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        has_stall_col = "stall_gap_max_s" in (r.fieldnames or [])
        for row in r:
            total_rows += 1
            status = row.get("http_status", "")
            try:
                out_tok = int(float(row.get("out_tok", "0")))
            except Exception:
                out_tok = 0
            if status == "200":
                ok_200 += 1
                total_out_tok_ok += max(0, out_tok)
            try:
                tt.append(float(row["ttft_s"]))
            except Exception:
                pass
            try:
                t_first = float(row.get("t_first_token_abs", "nan"))
                t_done = float(row.get("t_done_abs", "nan"))
                if out_tok > 1 and np.isfinite(t_first) and np.isfinite(t_done):
                    tpot.append((t_done - t_first) / max(1, out_tok - 1))
            except Exception:
                pass
            try:
                tsend = float(row.get("t_send_abs", "nan"))
                if np.isfinite(tsend):
                    send_abs_vals.append(tsend)
            except Exception:
                pass
            try:
                tdone = float(row.get("t_done_abs", "nan"))
                if np.isfinite(tdone):
                    done_abs_vals.append(tdone)
            except Exception:
                pass
            if has_stall_col:
                try:
                    s = float(row.get("stall_gap_max_s", "nan"))
                    if np.isfinite(s):
                        stall_gap.append(s)
                except Exception:
                    pass

    arr_tt = np.array(tt, dtype=float) if tt else np.array([], dtype=float)
    arr_tpot = np.array(tpot, dtype=float) if tpot else np.array([], dtype=float)
    arr_stall = np.array(stall_gap, dtype=float) if stall_gap else np.array([], dtype=float)

    send_min = float(np.min(np.array(send_abs_vals, dtype=float))) if send_abs_vals else float("nan")
    done_max = float(np.max(np.array(done_abs_vals, dtype=float))) if done_abs_vals else float("nan")
    client_span_s = float("nan")
    decode_toks_per_s_client_window = float("nan")
    if np.isfinite(send_min) and np.isfinite(done_max) and done_max > send_min:
        client_span_s = done_max - send_min
        decode_toks_per_s_client_window = float(total_out_tok_ok) / client_span_s

    return {
        "ok_200": ok_200,
        "total_rows": total_rows,
        "ttft_p50_s": percentile_or_nan(arr_tt, 0.50),
        "ttft_p90_s": percentile_or_nan(arr_tt, 0.90),
        "ttft_p99_s": percentile_or_nan(arr_tt, 0.99),
        "ttft_p999_s": percentile_or_nan(arr_tt, 0.999),
        "ttft_max_s": max_or_nan(arr_tt),
        "tpot_p50_s": percentile_or_nan(arr_tpot, 0.50),
        "tpot_p90_s": percentile_or_nan(arr_tpot, 0.90),
        "tpot_p99_s": percentile_or_nan(arr_tpot, 0.99),
        "tpot_p999_s": percentile_or_nan(arr_tpot, 0.999),
        "tpot_max_s": max_or_nan(arr_tpot),
        "stall_gap_p50_s": percentile_or_nan(arr_stall, 0.50),
        "stall_gap_p90_s": percentile_or_nan(arr_stall, 0.90),
        "stall_gap_p99_s": percentile_or_nan(arr_stall, 0.99),
        "stall_gap_max_s": max_or_nan(arr_stall),
        "total_out_tok_ok": total_out_tok_ok,
        "client_span_s": client_span_s,
        "decode_toks_per_s_client_window": decode_toks_per_s_client_window,
    }


def main():
    if len(sys.argv) != 7:
        print(
            "Usage: summarize_one_run_sweep_trace.py COND_DIR LAMBDA_RPS T_S WALL_ELAPSED_S META_JSON_PATH OUT_CSV_APPEND",
            file=sys.stderr,
        )
        sys.exit(2)

    cond_dir = sys.argv[1]
    lam = float(sys.argv[2])
    T_s = float(sys.argv[3])
    wall = float(sys.argv[4])
    meta_json_path = sys.argv[5]
    out_csv = sys.argv[6]

    client_csv = os.path.join(cond_dir, "client_results.csv")
    metrics_raw = os.path.join(cond_dir, "metrics_raw.txt")

    if not os.path.exists(meta_json_path):
        raise FileNotFoundError(meta_json_path)
    meta = json.load(open(meta_json_path, "r", encoding="utf-8"))
    N = int(meta.get("N", 0))

    cli = read_client_metrics(client_csv)
    metr = read_metrics_summary(metrics_raw)

    decode_toks_per_s_client_wall = float("nan")
    if wall > 0:
        decode_toks_per_s_client_wall = float(cli["total_out_tok_ok"]) / wall

    row = [
        cond_dir,
        f"{lam:.6f}",
        f"{T_s:.3f}",
        str(N),
        str(int(cli["ok_200"])),
        str(int(cli["total_rows"])),
        f"{wall:.6f}",
        f"{cli['ttft_p50_s']:.6f}",
        f"{cli['ttft_p90_s']:.6f}",
        f"{cli['ttft_p99_s']:.6f}",
        f"{cli['ttft_p999_s']:.6f}",
        f"{cli['ttft_max_s']:.6f}",
        f"{cli['tpot_p50_s']:.6f}",
        f"{cli['tpot_p90_s']:.6f}",
        f"{cli['tpot_p99_s']:.6f}",
        f"{cli['tpot_p999_s']:.6f}",
        f"{cli['tpot_max_s']:.6f}",
        f"{decode_toks_per_s_client_wall:.6f}",
        f"{cli['decode_toks_per_s_client_window']:.6f}",
        str(int(cli["total_out_tok_ok"])),
        f"{cli['client_span_s']:.6f}",
        f"{cli['stall_gap_p50_s']:.6f}",
        f"{cli['stall_gap_p90_s']:.6f}",
        f"{cli['stall_gap_p99_s']:.6f}",
        f"{cli['stall_gap_max_s']:.6f}",
        f"{metr['preempt_sum_delta']:.6f}",
        f"{metr['gpu_mem_used_mb_p50']:.6f}",
        f"{metr['gpu_mem_used_mb_p90']:.6f}",
        f"{metr['gpu_mem_used_mb_p99']:.6f}",
        f"{metr['gpu_mem_used_mb_max']:.6f}",
        f"{metr['gpu_mem_total_mb_p50']:.6f}",
        f"{metr['gpu_mem_util_perc_p50']:.6f}",
        f"{metr['gpu_mem_util_perc_p90']:.6f}",
        f"{metr['gpu_mem_util_perc_p99']:.6f}",
        f"{metr['gpu_mem_util_perc_max']:.6f}",
        f"{metr['gpu_cache_usage_perc_p50']:.6f}",
        f"{metr['gpu_cache_usage_perc_p90']:.6f}",
        f"{metr['gpu_cache_usage_perc_p99']:.6f}",
        f"{metr['gpu_cache_usage_perc_max']:.6f}",
        f"{metr['decode_toks_per_s_metrics_delta']:.6f}",
        f"{metr['avg_generation_toks_per_s_p50']:.6f}",
        f"{metr['avg_generation_toks_per_s_p90']:.6f}",
        f"{metr['avg_generation_toks_per_s_p99']:.6f}",
        f"{metr['avg_generation_toks_per_s_max']:.6f}",
        f"{metr['avg_prompt_toks_per_s_p50']:.6f}",
        f"{metr['avg_prompt_toks_per_s_p90']:.6f}",
        f"{metr['avg_prompt_toks_per_s_p99']:.6f}",
        f"{metr['avg_prompt_toks_per_s_max']:.6f}",
        f"{metr['num_requests_swapped_p50']:.6f}",
        f"{metr['num_requests_swapped_p90']:.6f}",
        f"{metr['num_requests_swapped_p99']:.6f}",
        f"{metr['num_requests_swapped_max']:.6f}",
        f"{metr['prompt_tokens_delta']:.6f}",
        f"{metr['generation_tokens_delta']:.6f}",
        f"{metr['swapin_blocks_delta']:.6f}",
        f"{metr['swapout_blocks_delta']:.6f}",
        f"{metr['recompute_tokens_delta']:.6f}",
        f"{metr['restore_progress_stall_ms_delta']:.6f}",
        f"{metr['metrics_elapsed_s']:.6f}",
        str(int(metr["metrics_samples"])),
    ]

    header = [
        "cond_dir",
        "lambda_rps",
        "T_s",
        "N",
        "ok_200",
        "total_rows",
        "wall_elapsed_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "ttft_p99_s",
        "ttft_p999_s",
        "ttft_max_s",
        "tpot_p50_s",
        "tpot_p90_s",
        "tpot_p99_s",
        "tpot_p999_s",
        "tpot_max_s",
        "decode_toks_per_s_client_wall",
        "decode_toks_per_s_client_window",
        "total_out_tok_ok",
        "client_span_s",
        "stall_gap_p50_s",
        "stall_gap_p90_s",
        "stall_gap_p99_s",
        "stall_gap_max_s",
        "preempt_sum_delta",
        "gpu_mem_used_mb_p50",
        "gpu_mem_used_mb_p90",
        "gpu_mem_used_mb_p99",
        "gpu_mem_used_mb_max",
        "gpu_mem_total_mb_p50",
        "gpu_mem_util_perc_p50",
        "gpu_mem_util_perc_p90",
        "gpu_mem_util_perc_p99",
        "gpu_mem_util_perc_max",
        "gpu_cache_usage_perc_p50",
        "gpu_cache_usage_perc_p90",
        "gpu_cache_usage_perc_p99",
        "gpu_cache_usage_perc_max",
        "decode_toks_per_s_metrics_delta",
        "avg_generation_toks_per_s_p50",
        "avg_generation_toks_per_s_p90",
        "avg_generation_toks_per_s_p99",
        "avg_generation_toks_per_s_max",
        "avg_prompt_toks_per_s_p50",
        "avg_prompt_toks_per_s_p90",
        "avg_prompt_toks_per_s_p99",
        "avg_prompt_toks_per_s_max",
        "num_requests_swapped_p50",
        "num_requests_swapped_p90",
        "num_requests_swapped_p99",
        "num_requests_swapped_max",
        "prompt_tokens_delta",
        "generation_tokens_delta",
        "swapin_blocks_delta",
        "swapout_blocks_delta",
        "recompute_tokens_delta",
        "restore_progress_stall_ms_delta",
        "metrics_elapsed_s",
        "metrics_samples",
    ]

    target_out_csv = out_csv
    write_header = (not os.path.exists(target_out_csv)) or (os.path.getsize(target_out_csv) == 0)
    if not write_header:
        try:
            with open(target_out_csv, "r", newline="", encoding="utf-8") as fi:
                rd = csv.reader(fi)
                old_header = next(rd, [])
                old_rows = list(rd)
            if old_header != header:
                old_idx = {name: i for i, name in enumerate(old_header)}
                tmp_csv = f"{target_out_csv}.tmp"
                with open(tmp_csv, "w", newline="", encoding="utf-8") as fo:
                    w = csv.writer(fo)
                    w.writerow(header)
                    for old_row in old_rows:
                        new_row = []
                        for col in header:
                            idx = old_idx.get(col, -1)
                            if idx >= 0 and idx < len(old_row):
                                new_row.append(old_row[idx])
                            else:
                                new_row.append("")
                        w.writerow(new_row)
                os.replace(tmp_csv, target_out_csv)
                print(
                    "[WARN] existing summary header mismatch, "
                    f"migrated in-place to new schema: {target_out_csv}"
                )
                write_header = False
        except Exception as e:
            print(
                "[WARN] failed to migrate existing summary header in-place, "
                f"append to {target_out_csv} as-is: {repr(e)}"
            )

    with open(target_out_csv, "a", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print(f"[OK] appended to {target_out_csv}:", ",".join(row))
    print(
        "[KEY] "
        f"lambda={lam:.3f} "
        f"decode_toks/s(client_wall)={decode_toks_per_s_client_wall:.3f} "
        f"decode_toks/s(metrics_delta)={metr['decode_toks_per_s_metrics_delta']:.3f} "
        f"ttft_p99_s={cli['ttft_p99_s']:.3f} "
        f"tpot_p99_s={cli['tpot_p99_s']:.3f} "
        f"stall_gap_p99_s={cli['stall_gap_p99_s']:.3f} "
        f"swapped_p99={metr['num_requests_swapped_p99']:.3f} "
        f"swapin_delta={metr['swapin_blocks_delta']:.3f} "
        f"swapout_delta={metr['swapout_blocks_delta']:.3f} "
        f"recompute_delta={metr['recompute_tokens_delta']:.3f} "
        f"restore_stall_ms_delta={metr['restore_progress_stall_ms_delta']:.3f}"
    )


if __name__ == "__main__":
    main()
