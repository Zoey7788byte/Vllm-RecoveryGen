#!/usr/bin/env python3
"""Compare Phase0/Phase1 outputs and validate Phase1 effectiveness gates.

Usage:
  python scripts/plot_analyze_phase0_phase1_effectiveness.py

Optional explicit paths:
  python scripts/plot_analyze_phase0_phase1_effectiveness.py \
    --phase0-summary <.../Phase0_or_Baseline/.../summary.csv> \
    --phase1-summary <.../Phase1_validation/.../summary.csv> \
    --phase1-recovery-ts <.../recovery(_obs)/recovery_ts.csv> \
    --phase1-events <.../recovery(_obs)/recovery_events.jsonl>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def latest_from_patterns(patterns: List[str]) -> Optional[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def to_float(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def read_summary(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    # keyed by lambda_rps string
    out: Dict[str, Dict[str, float]] = {}
    by_lam: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_lam.setdefault(r["lambda_rps"], []).append(r)
    for lam, grp in by_lam.items():
        ok_rates = [
            to_float(x.get("ok_200", "0")) / max(1.0, to_float(x.get("total_rows", "1")))
            for x in grp
        ]
        out[lam] = {
            "n_segments": float(len(grp)),
            "ttft_p90_mean": statistics.fmean(to_float(x["ttft_p90_s"]) for x in grp),
            "ttft_p99_mean": statistics.fmean(to_float(x["ttft_p99_s"]) for x in grp),
            "preempt_sum_mean": statistics.fmean(
                to_float(x["preempt_sum_delta"]) for x in grp
            ),
            "preempt_sum_total": sum(to_float(x["preempt_sum_delta"]) for x in grp),
            "ok_rate_mean": statistics.fmean(ok_rates),
        }
    return out


@dataclass
class ReqSwapStats:
    preempt_cnt: int = 0
    swap_out_blocks_total: int = 0
    swap_in_blocks_total: int = 0
    swap_out_events: List[Tuple[int, int]] = field(default_factory=list)  # (ts_ns, blocks)
    swap_in_events: List[Tuple[int, int]] = field(default_factory=list)  # (ts_ns, blocks)
    swap_in_micro_events: List[Tuple[int, int, int]] = field(default_factory=list)  # (ts_ns, k_req, k_done)
    remaining_host_events: List[Tuple[int, float]] = field(default_factory=list)  # (ts_ns, n_host)
    stalled_events: List[int] = field(default_factory=list)  # ts_ns
    swap_in_skip_restored_total: int = 0
    commit_events: List[Tuple[int, int, int]] = field(default_factory=list)  # (ts_ns, s, e)


def parse_commit_range(detail: dict) -> Optional[Tuple[int, int]]:
    # tolerant parsing: different versions may use different keys
    if "range" in detail:
        rng = detail.get("range")
        if isinstance(rng, list) and len(rng) == 2:
            try:
                return int(rng[0]), int(rng[1])
            except Exception:
                pass
    cand_pairs = [
        ("start_block", "end_block"),
        ("start", "end"),
        ("from", "to"),
        ("left", "right"),
    ]
    for a, b in cand_pairs:
        if a in detail and b in detail:
            try:
                s, e = int(detail[a]), int(detail[b])
                return (s, e)
            except Exception:
                pass
    if "restore_frontier" in detail:
        try:
            e = int(detail["restore_frontier"])
            return (0, e)
        except Exception:
            pass
    return None


def read_events(path: str) -> Tuple[Dict[str, ReqSwapStats], Dict[str, int]]:
    by_req: Dict[str, ReqSwapStats] = {}
    event_cnt: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                event_cnt["<bad_json>"] = event_cnt.get("<bad_json>", 0) + 1
                continue
            event = str(obj.get("event", ""))
            event_cnt[event] = event_cnt.get(event, 0) + 1
            req_id = obj.get("req_id")
            if req_id is None:
                seq_id = obj.get("seq_id")
                req_id = f"seq:{seq_id}"
            req_id = str(req_id)
            ts_ns = int(obj.get("ts_ns", 0))
            detail = obj.get("detail", {}) or {}
            st = by_req.setdefault(req_id, ReqSwapStats())

            if event == "PREEMPT_TRIGGERED":
                st.preempt_cnt += 1
            elif event == "SWAP_OUT":
                blocks = int(detail.get("blocks_count", 0) or 0)
                st.swap_out_blocks_total += blocks
                st.swap_out_events.append((ts_ns, blocks))
            elif event == "SWAP_IN":
                blocks = int(detail.get("blocks_count", 0) or 0)
                st.swap_in_blocks_total += blocks
                st.swap_in_events.append((ts_ns, blocks))
            elif event == "SWAP_IN_MICRO":
                k_req = int(detail.get("k_req", 0) or 0)
                k_done = int(detail.get("k_done", 0) or 0)
                st.swap_in_micro_events.append((ts_ns, k_req, k_done))
            elif event == "SWAP_IN_SKIP_ALREADY_RESTORED":
                st.swap_in_skip_restored_total += int(detail.get("count", 0) or 0)
            elif event == "RECOVERY_REMAINING_HOST_BLOCKS":
                try:
                    n_host = float(detail.get("n_host", "nan"))
                except Exception:
                    n_host = float("nan")
                st.remaining_host_events.append((ts_ns, n_host))
            elif event == "RECOVERY_STALLED":
                st.stalled_events.append(ts_ns)
            elif "COMMIT" in event:
                rng = parse_commit_range(detail)
                if rng is not None:
                    st.commit_events.append((ts_ns, rng[0], rng[1]))
    return by_req, event_cnt


def read_recovery_ts(path: str) -> Dict[str, float]:
    rows = 0
    pre = 0
    swi = 0
    swo = 0
    stall_vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows += 1
            pre += int(to_float(r.get("on_preempt_count_delta", "0"), 0.0))
            swi += int(to_float(r.get("swapin_blocks", "0"), 0.0))
            swo += int(to_float(r.get("swapout_blocks", "0"), 0.0))
            stall_vals.append(to_float(r.get("restore_progress_stall_ms", "0"), 0.0))
    return {
        "rows": float(rows),
        "preempt_total": float(pre),
        "swapin_total": float(swi),
        "swapout_total": float(swo),
        "stall_max_ms": max(stall_vals) if stall_vals else float("nan"),
        "stall_p99_ms": percentile(stall_vals, 99) if stall_vals else float("nan"),
    }


def read_recovery_ts_series(path: Optional[str]) -> Dict[str, List[float]]:
    out = {
        "t_rel_s": [],
        "preempt_cum": [],
        "swapin_cum": [],
        "swapout_cum": [],
        "stall_ms": [],
    }
    if not path or not os.path.isfile(path):
        return out

    pre = 0.0
    swi = 0.0
    swo = 0.0
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            pre += to_float(r.get("on_preempt_count_delta", "0"), 0.0)
            swi += to_float(r.get("swapin_blocks", "0"), 0.0)
            swo += to_float(r.get("swapout_blocks", "0"), 0.0)
            out["t_rel_s"].append(to_float(r.get("t_rel_s", "0"), 0.0))
            out["preempt_cum"].append(pre)
            out["swapin_cum"].append(swi)
            out["swapout_cum"].append(swo)
            out["stall_ms"].append(to_float(r.get("restore_progress_stall_ms", "0"), 0.0))
    return out


def _stats_lookup(stats: Dict[str, Dict[str, float]], lam: float, key: str) -> float:
    for cand in (f"{lam:.6f}", f"{lam:g}", str(lam)):
        if cand in stats:
            return stats[cand].get(key, float("nan"))
    for k, v in stats.items():
        try:
            if abs(float(k) - lam) < 1e-9:
                return v.get(key, float("nan"))
        except Exception:
            continue
    return float("nan")


def write_plots(
    p0_stats: Dict[str, Dict[str, float]],
    p1_stats: Dict[str, Dict[str, float]],
    by_req: Dict[str, ReqSwapStats],
    ts_series: Dict[str, List[float]],
    out_dir: str,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] skip plot output: matplotlib unavailable ({e})")
        return []

    os.makedirs(out_dir, exist_ok=True)
    out_files: List[str] = []

    lams = sorted({float(x) for x in set(p0_stats.keys()) | set(p1_stats.keys())})
    x = list(range(len(lams)))
    labels = [f"{v:.2f}" for v in lams]

    def get_vals(stats: Dict[str, Dict[str, float]], key: str) -> List[float]:
        return [_stats_lookup(stats, lam, key) for lam in lams]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(x, get_vals(p0_stats, "ttft_p90_mean"), marker="o", label="new_Phase0")
    ax1.plot(x, get_vals(p1_stats, "ttft_p90_mean"), marker="o", label="new_Phase1")
    ax1.set_title("TTFT p90 mean")
    ax1.set_xticks(x, labels)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x, get_vals(p0_stats, "ttft_p99_mean"), marker="o", label="new_Phase0")
    ax2.plot(x, get_vals(p1_stats, "ttft_p99_mean"), marker="o", label="new_Phase1")
    ax2.set_title("TTFT p99 mean")
    ax2.set_xticks(x, labels)
    ax2.grid(alpha=0.3)

    ax3.plot(x, get_vals(p0_stats, "preempt_sum_total"), marker="o", label="new_Phase0")
    ax3.plot(x, get_vals(p1_stats, "preempt_sum_total"), marker="o", label="new_Phase1")
    ax3.set_title("preempt_sum_total")
    ax3.set_xticks(x, labels)
    ax3.grid(alpha=0.3)

    ax4.plot(x, get_vals(p0_stats, "ok_rate_mean"), marker="o", label="new_Phase0")
    ax4.plot(x, get_vals(p1_stats, "ok_rate_mean"), marker="o", label="new_Phase1")
    ax4.set_title("ok_rate_mean")
    ax4.set_xticks(x, labels)
    ax4.grid(alpha=0.3)

    fig.supxlabel("lambda_rps")
    fig.tight_layout()
    out1 = os.path.join(out_dir, "new_Phase0_new_Phase1_summary_compare.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    out_files.append(out1)

    k_done: List[int] = []
    out_vals: List[int] = []
    in_vals: List[int] = []
    for st in by_req.values():
        if st.swap_in_micro_events:
            k_done.extend([k_done for _, _, k_done in st.swap_in_micro_events if k_done > 0])
        else:
            k_done.extend([b for _, b in st.swap_in_events if b > 0])
        if st.swap_out_blocks_total > 0 or st.swap_in_blocks_total > 0:
            out_vals.append(st.swap_out_blocks_total)
            in_vals.append(st.swap_in_blocks_total)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    ax21, ax22 = axes2
    if k_done:
        ax21.hist(k_done, bins=min(20, max(5, len(set(k_done)))), color="#3b82f6", alpha=0.8)
        ax21.axvline(sum(k_done) / len(k_done), color="red", linestyle="--", label="mean")
        ax21.set_title("new_Phase1 SWAP_IN blocks_count distribution")
        ax21.set_xlabel("k_done (blocks per SWAP_IN)")
        ax21.set_ylabel("count")
        ax21.grid(alpha=0.3)
        ax21.legend()
    else:
        ax21.text(0.5, 0.5, "No SWAP_IN events", ha="center", va="center")
        ax21.set_axis_off()

    if out_vals:
        ax22.scatter(out_vals, in_vals, alpha=0.8)
        mx = max(out_vals + in_vals + [1])
        ax22.plot([0, mx], [0, mx], linestyle="--", color="gray", label="y=x")
        ax22.set_title("Per-request swapin vs swapout blocks")
        ax22.set_xlabel("swapout total blocks")
        ax22.set_ylabel("swapin total blocks")
        ax22.grid(alpha=0.3)
        ax22.legend()
    else:
        ax22.text(0.5, 0.5, "No per-request swap stats", ha="center", va="center")
        ax22.set_axis_off()

    fig2.tight_layout()
    out2 = os.path.join(out_dir, "new_Phase1_micro_swap_distribution.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    out_files.append(out2)

    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax31, ax32 = axes3
    t = ts_series.get("t_rel_s", [])
    if t:
        ax31.plot(t, ts_series.get("preempt_cum", []), label="preempt_cum", linewidth=1.2)
        ax31.plot(t, ts_series.get("swapin_cum", []), label="swapin_cum_blocks", linewidth=1.2)
        ax31.plot(t, ts_series.get("swapout_cum", []), label="swapout_cum_blocks", linewidth=1.2)
        ax31.set_ylabel("cumulative")
        ax31.set_title("new_Phase1 recovery timeline (cumulative)")
        ax31.grid(alpha=0.3)
        ax31.legend()

        ax32.plot(t, ts_series.get("stall_ms", []), label="restore_progress_stall_ms", linewidth=1.0)
        ax32.set_xlabel("t_rel_s")
        ax32.set_ylabel("ms")
        ax32.set_title("new_Phase1 restore stall proxy")
        ax32.grid(alpha=0.3)
        ax32.legend()
    else:
        ax31.text(0.5, 0.5, "No recovery_ts.csv data", ha="center", va="center")
        ax31.set_axis_off()
        ax32.set_axis_off()

    fig3.tight_layout()
    out3 = os.path.join(out_dir, "new_Phase1_recovery_timeline.png")
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    out_files.append(out3)

    return out_files


def _infer_server_run_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    marker = "/server_logs/"
    if marker not in path:
        return None
    left, right = path.split(marker, 1)
    run_id = right.split("/", 1)[0]
    if not run_id:
        return None
    return os.path.join(left, "server_logs", run_id)


def detect_budget(
    summary_path: Optional[str],
    default_budget: int,
    recovery_ts_path: Optional[str] = None,
    recovery_events_path: Optional[str] = None,
) -> int:
    for p in (summary_path, recovery_ts_path, recovery_events_path):
        if not p:
            continue
        m = re.search(r"rbudget=(\d+)", p)
        if m:
            return int(m.group(1))

    for p in (recovery_ts_path, recovery_events_path):
        run_dir = _infer_server_run_dir(p)
        if not run_dir:
            continue
        cfg = os.path.join(run_dir, "server_config.json")
        if not os.path.isfile(cfg):
            continue
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                obj = json.load(f)
            raw = obj.get("recovery_budget")
            if raw is not None:
                return int(float(raw))
        except Exception:
            continue

    env_budget = os.getenv("VLLM_RECOVERY_BUDGET")
    if env_budget:
        try:
            return int(float(env_budget))
        except Exception:
            pass
    return default_budget


def gate_eval(
    by_req: Dict[str, ReqSwapStats],
    ts_stats: Optional[Dict[str, float]],
    budget: int,
    stall_gap_thr_ms: float,
    amplification_thr: float,
) -> Dict[str, dict]:
    def effective_swapin_events(st: ReqSwapStats) -> List[Tuple[int, int]]:
        if st.swap_in_micro_events:
            return [(ts, k_done) for ts, _, k_done in st.swap_in_micro_events if k_done > 0]
        return [(ts, b) for ts, b in st.swap_in_events if b > 0]

    reqs = [
        r for r in by_req.values() if (
            r.preempt_cnt > 0
            or r.swap_out_blocks_total > 0
            or r.swap_in_blocks_total > 0
            or len(r.swap_in_micro_events) > 0
        )
    ]

    # Gate 1: micro-task effective (prefer SWAP_IN_MICRO).
    k_done = []
    swapout_sizes = []
    multi_step_acc = 0
    for st in reqs:
        if st.swap_out_blocks_total > 0:
            swapout_sizes.append(st.swap_out_blocks_total)
        evs = sorted(effective_swapin_events(st), key=lambda x: x[0])
        in_total = sum(b for _, b in evs)
        k_done.extend([b for _, b in evs])
        if len(evs) >= 2:
            if st.swap_out_blocks_total > 0:
                if in_total >= 0.9 * st.swap_out_blocks_total:
                    multi_step_acc += 1
            elif in_total > 0:
                multi_step_acc += 1
    med_k = statistics.median(k_done) if k_done else float("nan")
    p90_k = percentile([float(x) for x in k_done], 90) if k_done else float("nan")
    max_k = max(k_done) if k_done else float("nan")
    med_out = statistics.median(swapout_sizes) if swapout_sizes else float("nan")
    ratio = (med_k / med_out) if (k_done and swapout_sizes and med_out > 0) else float("nan")
    ratio_ok = (ratio < 0.5) if math.isfinite(ratio) else True
    gate1_pass = bool(
        reqs
        and k_done
        and ratio_ok
        and p90_k <= max(32.0, float(2 * budget))
        and multi_step_acc >= 1
    )

    # Gate 2: continuation correctness.
    longest_gap_ms = 0.0
    unfinished = 0
    stalled_event_reqs = 0
    stalled_event_total = 0
    for st in reqs:
        evs = sorted(effective_swapin_events(st), key=lambda x: x[0])
        if st.stalled_events:
            stalled_event_reqs += 1
            stalled_event_total += len(st.stalled_events)
        last_host = float("nan")
        if st.remaining_host_events:
            for _, n_host in sorted(st.remaining_host_events, key=lambda x: x[0]):
                if math.isfinite(n_host):
                    last_host = n_host
            if math.isfinite(last_host) and last_host > 0:
                unfinished += 1
        elif st.swap_out_blocks_total > 0:
            if sum(v for _, v in evs) < 0.9 * st.swap_out_blocks_total:
                unfinished += 1
        for i in range(1, len(evs)):
            gap_ms = (evs[i][0] - evs[i - 1][0]) / 1e6
            if gap_ms > longest_gap_ms:
                longest_gap_ms = gap_ms
    ts_stall_max = ts_stats["stall_max_ms"] if ts_stats else float("nan")
    stall_ok = math.isnan(ts_stall_max) or ts_stall_max <= stall_gap_thr_ms
    gate2_pass = bool(
        reqs and unfinished == 0 and longest_gap_ms <= stall_gap_thr_ms and stall_ok
    )

    # Gate 3: repeated work controlled
    total_swi = 0
    for st in reqs:
        evs = effective_swapin_events(st)
        if evs:
            total_swi += sum(v for _, v in evs)
        else:
            total_swi += st.swap_in_blocks_total
    total_swo = sum(st.swap_out_blocks_total for st in reqs)
    amplification = (float(total_swi) / float(total_swo)) if total_swo > 0 else float("nan")
    commit_present = any(st.commit_events for st in reqs)
    commit_violation = 0
    skip_already_restored_total = sum(st.swap_in_skip_restored_total for st in reqs)
    if commit_present:
        for st in reqs:
            if not st.commit_events:
                continue
            events = sorted(st.commit_events, key=lambda x: x[0])
            prev_s, prev_e = -1, -1
            for _, s, e in events:
                if s > e or s < prev_s or e < prev_e:
                    commit_violation += 1
                    break
                prev_s, prev_e = s, e
    commit_ok = None if not commit_present else (commit_violation == 0)
    gate3_pass = bool(
        reqs and total_swo > 0 and amplification <= amplification_thr and (commit_ok is None or commit_ok)
    )

    return {
        "gate1_micro_task_effective": {
            "pass": gate1_pass,
            "reqs_in_scope": len(reqs),
            "median_k_done": med_k,
            "p90_k_done": p90_k,
            "max_k_done": max_k,
            "median_swapout_blocks": med_out,
            "k_done_vs_swapout_ratio": ratio,
            "multi_step_accumulated_reqs": multi_step_acc,
            "budget_blocks": budget,
        },
        "gate2_continuation_correctness": {
            "pass": gate2_pass,
            "unfinished_reqs": unfinished,
            "longest_gap_ms_between_swapins": longest_gap_ms,
            "restore_progress_stall_max_ms": ts_stall_max,
            "recovery_stalled_reqs": stalled_event_reqs,
            "recovery_stalled_events_total": stalled_event_total,
            "threshold_ms": stall_gap_thr_ms,
        },
        "gate3_redundant_work_controlled": {
            "pass": gate3_pass,
            "total_swapin_blocks": total_swi,
            "total_swapout_blocks": total_swo,
            "swapin_amplification_ratio": amplification,
            "amplification_threshold": amplification_thr,
            "commit_events_present": commit_present,
            "commit_monotonic": commit_ok,
            "commit_violation_count": commit_violation,
            "swap_in_skip_already_restored_total": skip_already_restored_total,
        },
    }


def print_summary_compare(p0_stats: Dict[str, Dict[str, float]], p1_stats: Dict[str, Dict[str, float]]) -> None:
    lams = sorted(set(p0_stats.keys()) | set(p1_stats.keys()), key=lambda x: float(x))
    print("=== Phase0 vs Phase1 Summary Compare ===")
    for lam in lams:
        s0 = p0_stats.get(lam)
        s1 = p1_stats.get(lam)
        if s0 is None or s1 is None:
            print(f"lambda={lam}: missing in one phase (phase0={s0 is not None}, phase1={s1 is not None})")
            continue
        print(
            f"lambda={lam} | "
            f"ttft_p90 {s0['ttft_p90_mean']:.3f}->{s1['ttft_p90_mean']:.3f} | "
            f"ttft_p99 {s0['ttft_p99_mean']:.3f}->{s1['ttft_p99_mean']:.3f} | "
            f"preempt_sum_total {s0['preempt_sum_total']:.1f}->{s1['preempt_sum_total']:.1f} | "
            f"ok_rate {s0['ok_rate_mean']:.4f}->{s1['ok_rate_mean']:.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-summary", dest="phase0_summary", default=None)
    ap.add_argument("--phase1-summary", dest="phase1_summary", default=None)
    ap.add_argument("--phase1-recovery-ts", dest="phase1_recovery_ts", default=None)
    ap.add_argument("--phase1-events", dest="phase1_events", default=None)
    # Backward-compatible aliases.
    ap.add_argument("--new_Phase0-summary", dest="phase0_summary", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--new_Phase1-summary", dest="phase1_summary", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--new_Phase1-recovery-ts", dest="phase1_recovery_ts", default=None, help=argparse.SUPPRESS
    )
    ap.add_argument("--new_Phase1-events", dest="phase1_events", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--stall-gap-threshold-ms", type=float, default=5000.0)
    ap.add_argument("--amplification-threshold", type=float, default=1.2)
    ap.add_argument("--json-out", default=None, help="optional output json path")
    ap.add_argument(
        "--plot-dir",
        default="logs/RecoveryGen/Phase1_validation/plots",
        help="output directory for plot images",
    )
    ap.add_argument("--no-plots", action="store_true", help="disable plot output")
    args = ap.parse_args()

    p0_summary = args.phase0_summary or latest_from_patterns([
        "logs/RecoveryGen/new_Phase0/client_logs/*_pmode=*/summary.csv",
        "logs/RecoveryGen/new_Phase0/client_logs/*/summary.csv",
        "logs/RecoveryGen/Baseline/client_logs/*/summary.csv",
        "logs/RecoveryGen/Baseline_chunked/client_logs/*/summary.csv",
        "logs/RecoveryGen/Phase0_validation/client_logs/*/summary.csv",
    ])
    p1_summary = args.phase1_summary or latest_from_patterns([
        "logs/RecoveryGen/new_Phase1/client_logs/*_pmode=*/summary.csv",
        "logs/RecoveryGen/new_Phase1/client_logs/*/summary.csv",
        "logs/RecoveryGen/Phase1_validation/client_logs/*/summary.csv",
        "logs/RecoveryGen/Phase1/client_logs/*/summary.csv",
    ])
    p1_ts = args.phase1_recovery_ts or latest_from_patterns([
        "logs/RecoveryGen/new_Phase1/server_logs/*/recovery/recovery_ts.csv",
        "logs/RecoveryGen/new_Phase1/server_logs/*/recovery_obs/recovery_ts.csv",
        "logs/RecoveryGen/Phase1_validation/server_logs/*/recovery/recovery_ts.csv",
        "logs/RecoveryGen/Phase1_validation/server_logs/*/recovery_obs/recovery_ts.csv",
        "logs/RecoveryGen/Phase1/server_logs/*/recovery/recovery_ts.csv",
        "logs/RecoveryGen/Phase1/server_logs/*/recovery_obs/recovery_ts.csv",
    ])
    p1_events = args.phase1_events or latest_from_patterns([
        "logs/RecoveryGen/new_Phase1/server_logs/*/recovery/recovery_events.jsonl",
        "logs/RecoveryGen/new_Phase1/server_logs/*/recovery_obs/recovery_events.jsonl",
        "logs/RecoveryGen/Phase1_validation/server_logs/*/recovery/recovery_events.jsonl",
        "logs/RecoveryGen/Phase1_validation/server_logs/*/recovery_obs/recovery_events.jsonl",
        "logs/RecoveryGen/Phase1/server_logs/*/recovery/recovery_events.jsonl",
        "logs/RecoveryGen/Phase1/server_logs/*/recovery_obs/recovery_events.jsonl",
    ])

    if not p1_summary or not os.path.isfile(p1_summary):
        raise SystemExit("Phase1 summary.csv not found.")

    new_Phase0_rows = read_summary(p0_summary) if (p0_summary and os.path.isfile(p0_summary)) else []
    new_Phase1_rows = read_summary(p1_summary)
    p0_stats = summary_stats(new_Phase0_rows)
    p1_stats = summary_stats(new_Phase1_rows)
    print(f"phase0 summary: {p0_summary if p0_summary else '<missing>'}")
    print(f"phase1 summary: {p1_summary}")
    if p0_stats:
        print_summary_compare(p0_stats, p1_stats)
    else:
        print("phase0 summary unavailable: skip cross-phase summary comparison.")

    ts_stats = None
    if p1_ts and os.path.isfile(p1_ts):
        ts_stats = read_recovery_ts(p1_ts)
        print(f"\nphase1 recovery_ts: {p1_ts}")
        print(
            "recovery_ts totals: "
            f"rows={int(ts_stats['rows'])}, "
            f"preempt_total={int(ts_stats['preempt_total'])}, "
            f"swapin_total={int(ts_stats['swapin_total'])}, "
            f"swapout_total={int(ts_stats['swapout_total'])}, "
            f"stall_max_ms={ts_stats['stall_max_ms']:.3f}"
        )
    else:
        print("\nphase1 recovery_ts: <missing>")

    event_cnt: Dict[str, int] = {}
    by_req: Dict[str, ReqSwapStats] = {}
    if p1_events and os.path.isfile(p1_events):
        by_req, event_cnt = read_events(p1_events)
        print(f"phase1 events: {p1_events}")
        print("event counts:", ", ".join(f"{k}={v}" for k, v in sorted(event_cnt.items())))
    else:
        print("phase1 events: <missing>")

    budget = detect_budget(p1_summary, args.budget, p1_ts, p1_events)
    gates = gate_eval(
        by_req,
        ts_stats,
        budget=budget,
        stall_gap_thr_ms=args.stall_gap_threshold_ms,
        amplification_thr=args.amplification_threshold,
    )

    print("\n=== Phase1 Effectiveness Gates ===")
    g1 = gates["gate1_micro_task_effective"]
    print(
        f"[{'PASS' if g1['pass'] else 'FAIL'}] micro-task 生效 | "
        f"reqs={g1['reqs_in_scope']}, "
        f"median_k_done={g1['median_k_done']}, p90_k_done={g1['p90_k_done']}, "
        f"median_swapout={g1['median_swapout_blocks']}, ratio={g1['k_done_vs_swapout_ratio']}, "
        f"multi_step_reqs={g1['multi_step_accumulated_reqs']}"
    )
    g2 = gates["gate2_continuation_correctness"]
    print(
        f"[{'PASS' if g2['pass'] else 'FAIL'}] 续作正确 | "
        f"unfinished_reqs={g2['unfinished_reqs']}, "
        f"longest_gap_ms={g2['longest_gap_ms_between_swapins']:.3f}, "
        f"stall_max_ms={g2['restore_progress_stall_max_ms']:.3f}, "
        f"stalled_events={g2['recovery_stalled_events_total']}, "
        f"threshold={g2['threshold_ms']:.1f}"
    )
    g3 = gates["gate3_redundant_work_controlled"]
    commit_status = (
        "UNKNOWN(no commit events)"
        if g3["commit_monotonic"] is None
        else ("PASS" if g3["commit_monotonic"] else "FAIL")
    )
    print(
        f"[{'PASS' if g3['pass'] else 'FAIL'}] 重复工作受控 | "
        f"swapin={g3['total_swapin_blocks']}, swapout={g3['total_swapout_blocks']}, "
        f"amp_ratio={g3['swapin_amplification_ratio']:.3f}, commit_monotonic={commit_status}"
    )

    all_pass = bool(g1["pass"] and g2["pass"] and g3["pass"])
    print(f"\nphase1 gate result: {'PASS' if all_pass else 'FAIL'}")

    plot_outputs: List[str] = []
    if not args.no_plots and args.plot_dir:
        ts_series = read_recovery_ts_series(p1_ts)
        plot_outputs = write_plots(
            p0_stats=p0_stats,
            p1_stats=p1_stats,
            by_req=by_req,
            ts_series=ts_series,
            out_dir=args.plot_dir,
        )
        for p in plot_outputs:
            print(f"plot: {p}")

    if args.json_out:
        payload = {
            "phase0_summary": p0_summary,
            "phase1_summary": p1_summary,
            "phase1_recovery_ts": p1_ts,
            "phase1_events": p1_events,
            "phase0_stats_by_lambda": p0_stats,
            "phase1_stats_by_lambda": p1_stats,
            "phase1_event_counts": event_cnt,
            "phase1_recovery_ts_stats": ts_stats,
            "new_Phase0_summary": p0_summary,
            "new_Phase1_summary": p1_summary,
            "new_Phase1_recovery_ts": p1_ts,
            "new_Phase1_events": p1_events,
            "new_Phase0_stats_by_lambda": p0_stats,
            "new_Phase1_stats_by_lambda": p1_stats,
            "new_Phase1_event_counts": event_cnt,
            "new_Phase1_recovery_ts_stats": ts_stats,
            "gates": gates,
            "all_pass": all_pass,
            "plots": plot_outputs,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"json report: {args.json_out}")


if __name__ == "__main__":
    main()
