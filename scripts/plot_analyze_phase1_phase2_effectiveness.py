#!/usr/bin/env python3
"""Compare Phase1 vs Phase2 and evaluate whether Phase2 is effective.

Usage:
  python scripts/analyze_phase1_phase2_effectiveness.py

Optional:
  python scripts/analyze_phase1_phase2_effectiveness.py \
    --phase1-summary <.../summary.csv> \
    --phase2-summary <.../summary.csv> \
    --phase1-ts <.../recovery_ts.csv> \
    --phase2-ts <.../recovery_ts.csv> \
    --phase1-events <.../recovery_events.jsonl> \
    --phase2-events <.../recovery_events.jsonl> \
    --plot-dir logs/RecoveryGen/new_Phase2/plots \
    --json-out logs/RecoveryGen/new_Phase2/phase1_vs_phase2_eval.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def to_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def mean(xs: Iterable[float], default: float = float("nan")) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    if not vals:
        return default
    return statistics.fmean(vals)


def percentile(values: List[float], q: float) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    if q <= 0:
        return min(vals)
    if q >= 100:
        return max(vals)
    s = sorted(vals)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _finite_ratio(values: List[float]) -> float:
    if not values:
        return float("nan")
    finite = sum(1 for v in values if not math.isnan(v))
    return finite / len(values)


def _series_tail_stats(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {
            "n": 0.0,
            "tail_cv": float("nan"),
            "jump_p99": float("nan"),
            "changed_steps": 0.0,
        }

    win = max(30, int(0.1 * len(vals)))
    tail = vals[-win:]
    tail_mean = mean(tail, float("nan"))
    if len(tail) >= 2:
        tail_std = statistics.pstdev(tail)
        tail_cv = tail_std / max(1e-9, abs(tail_mean))
    else:
        tail_cv = 0.0

    jumps: List[float] = []
    changed = 0
    for i in range(1, len(vals)):
        prev = vals[i - 1]
        cur = vals[i]
        d = cur - prev
        if abs(d) > 1e-12:
            changed += 1
        jumps.append(abs(d) / max(1e-9, abs(prev)))

    return {
        "n": float(len(vals)),
        "tail_cv": tail_cv,
        "jump_p99": percentile(jumps, 99),
        "changed_steps": float(changed),
    }


def summary_by_lambda(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[r.get("lambda_rps", "")].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for lam, grp in groups.items():
        ok = [to_float(x.get("ok_200", 0), 0.0) for x in grp]
        total = [max(1.0, to_float(x.get("total_rows", 1), 1.0)) for x in grp]
        wall = [max(1e-9, to_float(x.get("wall_elapsed_s", 0), 1e-9)) for x in grp]

        throughput = [ok_i / wall_i for ok_i, wall_i in zip(ok, wall)]
        ok_rate = [ok_i / total_i for ok_i, total_i in zip(ok, total)]

        out[lam] = {
            "segments": float(len(grp)),
            "throughput_mean": mean(throughput),
            "ok_rate_mean": mean(ok_rate),
            "ttft_p90_mean": mean(to_float(x.get("ttft_p90_s", "nan")) for x in grp),
            "ttft_p99_mean": mean(to_float(x.get("ttft_p99_s", "nan")) for x in grp),
            "preempt_sum_total": sum(to_float(x.get("preempt_sum_delta", 0), 0.0) for x in grp),
        }
    return out


def ts_stats(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    s_ms = [to_float(r.get("S_ms", 0), 0.0) for r in rows]
    b_ms = [to_float(r.get("B_ms", float("nan")), float("nan")) for r in rows]
    wait = [to_float(r.get("on_waiting_len", 0), 0.0) for r in rows]
    pre = [to_float(r.get("on_preempt_count_delta", 0), 0.0) for r in rows]
    restore_stall = [to_float(r.get("restore_progress_stall_ms", 0), 0.0) for r in rows]

    x_swap = [to_float(r.get("x_swap", 0), 0.0) for r in rows]
    x_rec = [to_float(r.get("x_rec", 0), 0.0) for r in rows]
    swapin = [to_float(r.get("swapin_blocks", 0), 0.0) for r in rows]
    rec_tok = [to_float(r.get("recompute_tokens", 0), 0.0) for r in rows]

    eps = 1e-6
    no_slack_idx = [i for i, v in enumerate(s_ms) if v <= eps]
    no_slack_recovery_cycles = 0
    for i in no_slack_idx:
        if i < len(x_swap) and i < len(x_rec) and i < len(swapin) and i < len(rec_tok):
            if (x_swap[i] + x_rec[i]) > 0 or swapin[i] > 0 or rec_tok[i] > 0:
                no_slack_recovery_cycles += 1

    # burst count in fixed 5s windows
    t_rel = [to_float(r.get("t_rel_s", 0), 0.0) for r in rows]
    burst_windows: Dict[int, float] = defaultdict(float)
    for t, p in zip(t_rel, pre):
        if p > 0:
            burst_windows[int(t // 5)] += p
    burst_windows_ge2 = sum(1 for v in burst_windows.values() if v >= 2.0)
    burst_windows_ge3 = sum(1 for v in burst_windows.values() if v >= 3.0)

    # waiting high duration (waiting_len >= 1)
    high_wait_cycles = sum(1 for w in wait if w >= 1.0)

    return {
        "rows": len(rows),
        "S_zero_ratio": (len(no_slack_idx) / len(s_ms)) if s_ms else float("nan"),
        "no_slack_cycles": len(no_slack_idx),
        "no_slack_recovery_cycles": no_slack_recovery_cycles,
        "no_slack_recovery_ratio": (
            no_slack_recovery_cycles / len(no_slack_idx) if no_slack_idx else float("nan")
        ),
        "wait_p99": percentile(wait, 99),
        "wait_p95": percentile(wait, 95),
        "wait_high_cycles": high_wait_cycles,
        "wait_high_ratio": (high_wait_cycles / len(wait)) if wait else float("nan"),
        "preempt_total": sum(pre),
        "preempt_burst_windows_5s": len(burst_windows),
        "preempt_burst_windows_5s_ge2": burst_windows_ge2,
        "preempt_burst_windows_5s_ge3": burst_windows_ge3,
        "restore_stall_p99_ms": percentile(restore_stall, 99),
        "restore_stall_max_ms": max(restore_stall) if restore_stall else float("nan"),
        "B_ms_mean": mean(b_ms),
        "B_ms_p95": percentile([x for x in b_ms if not math.isnan(x)], 95),
        "has_B_ms": any(not math.isnan(x) for x in b_ms),
        "series": {
            "t_rel_s": t_rel,
            "S_ms": s_ms,
            "B_ms": b_ms,
            "on_waiting_len": wait,
            "on_preempt_count_delta": pre,
        },
    }


def budget_stability(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    updates = [
        e for e in events if e.get("event") in
        ("RECOVERY_BUDGET_UPDATE", "RECOVERY_BUDGET_DECISION")
    ]
    if not updates:
        return {
            "has_budget_updates": False,
            "hysteresis_pass": False,
            "zigzag_pass": False,
            "trend_consistency_pass": False,
        }

    # Use target-to-target delta across consecutive updates.
    # Do not mix B_ms(prev effective budget) with target budget.
    targets: List[float] = []
    trends: List[str] = []
    for e in updates:
        detail = e.get("detail", {}) or {}
        tgt = to_float(
            detail.get(
                "B_target_ms",
                detail.get(
                    "budget_ms_target",
                    detail.get("recovery_budget_target_ms", "nan"),
                ),
            ))
        trend = str(detail.get("pressure_state",
                               detail.get("pressure_trend", "")))
        if math.isnan(tgt):
            continue
        targets.append(tgt)
        trends.append(trend)

    deltas: List[float] = []
    trend_pair: List[Tuple[str, float]] = []
    for i in range(1, len(targets)):
        d = targets[i] - targets[i - 1]
        deltas.append(d)
        trend_pair.append((trends[i], d))

    pos = [d for d in deltas if d > 0]
    neg = [abs(d) for d in deltas if d < 0]
    avg_pos = mean(pos, 0.0)
    avg_neg = mean(neg, 0.0)

    non_zero = [d for d in deltas if abs(d) > 1e-9]
    sign_changes = 0
    for i in range(1, len(non_zero)):
        if non_zero[i] * non_zero[i - 1] < 0:
            sign_changes += 1
    zigzag_ratio = sign_changes / max(1, len(non_zero) - 1)

    trend_hits = 0
    trend_total = 0
    for trend, d in trend_pair:
        if trend == "up":
            trend_total += 1
            if d <= 0:
                trend_hits += 1
        elif trend == "down":
            trend_total += 1
            if d >= 0:
                trend_hits += 1

    trend_consistency = trend_hits / trend_total if trend_total else float("nan")

    hysteresis_pass = avg_neg >= avg_pos
    zigzag_pass = zigzag_ratio <= 0.60
    trend_consistency_pass = (not math.isnan(trend_consistency)) and trend_consistency >= 0.65

    return {
        "has_budget_updates": True,
        "updates": len(updates),
        "avg_budget_step_up_ms": avg_pos,
        "avg_budget_step_down_ms": avg_neg,
        "hysteresis_pass": hysteresis_pass,
        "zigzag_ratio": zigzag_ratio,
        "zigzag_pass": zigzag_pass,
        "trend_consistency": trend_consistency,
        "trend_consistency_pass": trend_consistency_pass,
    }


def swap_timelines(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_req_out: Dict[str, int] = defaultdict(int)
    by_req_in: Dict[str, int] = defaultdict(int)
    by_req_in_ts: Dict[str, List[int]] = defaultdict(list)
    by_req_preempt_ts: Dict[str, List[int]] = defaultdict(list)
    commits: Dict[str, List[int]] = defaultdict(list)
    stalled_events = 0

    for e in events:
        et = e.get("event")
        detail = e.get("detail", {}) or {}
        req = e.get("req_id")
        if req is None:
            seq = e.get("seq_id")
            req = f"seq:{seq}"
        req = str(req)

        if et == "SWAP_OUT":
            by_req_out[req] += int(to_float(detail.get("blocks_count", 0), 0.0))
        elif et == "SWAP_IN":
            by_req_in[req] += int(to_float(detail.get("blocks_count", 0), 0.0))
            by_req_in_ts[req].append(int(to_float(e.get("ts_ns", 0), 0.0)))
        elif et == "PREEMPT_TRIGGERED":
            by_req_preempt_ts[req].append(int(to_float(e.get("ts_ns", 0), 0.0)))
        elif et == "RECOVERY_PROGRESS_COMMIT":
            frontier = detail.get("new_frontier")
            if frontier is None and isinstance(detail.get("range"), list) and len(detail["range"]) == 2:
                frontier = detail["range"][1]
            if frontier is not None:
                commits[req].append(int(to_float(frontier, 0.0)))
        elif et == "RECOVERY_STALLED":
            stalled_events += 1

    # max gap between consecutive swap-in events per request
    gap_ms_all: List[float] = []
    for req, ts_list in by_req_in_ts.items():
        ts_sorted = sorted(ts_list)
        for i in range(1, len(ts_sorted)):
            gap_ms_all.append((ts_sorted[i] - ts_sorted[i - 1]) / 1e6)

    # Active recovery gap: only measure inside each recovery epoch
    # [PREEMPT_i, PREEMPT_{i+1}) for the same request. This avoids counting
    # cross-round intervals (request already restored, then preempted again).
    gap_ms_active: List[float] = []
    for req, pre_list in by_req_preempt_ts.items():
        pre_sorted = sorted([x for x in pre_list if x > 0])
        if not pre_sorted:
            continue
        in_sorted = sorted([x for x in by_req_in_ts.get(req, []) if x > 0])
        if not in_sorted:
            continue
        for i, start in enumerate(pre_sorted):
            end = pre_sorted[i + 1] if i + 1 < len(pre_sorted) else None
            ep = [t for t in in_sorted if t >= start and (end is None or t < end)]
            if not ep:
                continue
            # Include "preempt -> first swapin" as starvation-sensitive delay.
            gap_ms_active.append((ep[0] - start) / 1e6)
            for j in range(1, len(ep)):
                gap_ms_active.append((ep[j] - ep[j - 1]) / 1e6)

    unfinished = 0
    reqs = 0
    for req, out_b in by_req_out.items():
        if out_b <= 0:
            continue
        reqs += 1
        in_b = by_req_in.get(req, 0)
        if in_b < 0.9 * out_b:
            unfinished += 1

    commit_monotonic_bad = 0
    commit_total_reqs = 0
    for req, arr in commits.items():
        if not arr:
            continue
        commit_total_reqs += 1
        prev = -10**18
        for x in arr:
            if x < prev:
                commit_monotonic_bad += 1
                break
            prev = x

    return {
        "swap_reqs": reqs,
        "unfinished_reqs": unfinished,
        "unfinished_ratio": (unfinished / reqs) if reqs else float("nan"),
        "swapin_gap_max_ms": max(gap_ms_all) if gap_ms_all else float("nan"),
        "swapin_gap_p99_ms": percentile(gap_ms_all, 99),
        "swapin_gap_active_max_ms": max(gap_ms_active) if gap_ms_active else float("nan"),
        "swapin_gap_active_p99_ms": percentile(gap_ms_active, 99),
        "commit_reqs": commit_total_reqs,
        "commit_monotonic": (commit_monotonic_bad == 0) if commit_total_reqs else None,
        "commit_monotonic_bad_reqs": commit_monotonic_bad,
        "stalled_events": stalled_events,
    }


def pick_high_lambda(common_lams: List[str]) -> Optional[str]:
    if not common_lams:
        return None
    return sorted(common_lams, key=lambda x: to_float(x, -1e18))[-1]


def eval_p2_m1_strict(ts_rows: List[Dict[str, str]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    t_cyc = [to_float(r.get("T_cyc_ms", float("nan"))) for r in ts_rows]
    t_on = [to_float(r.get("T_on_ms", float("nan"))) for r in ts_rows]
    s_ms = [to_float(r.get("S_ms", float("nan"))) for r in ts_rows]
    wait = [to_float(r.get("on_waiting_len", 0.0), 0.0) for r in ts_rows]
    pre = [to_float(r.get("on_preempt_count_delta", 0.0), 0.0) for r in ts_rows]
    x_swap = [to_float(r.get("x_swap", 0.0), 0.0) for r in ts_rows]
    x_rec = [to_float(r.get("x_rec", 0.0), 0.0) for r in ts_rows]
    swapin = [to_float(r.get("swapin_blocks", 0.0), 0.0) for r in ts_rows]
    rec_tok = [to_float(r.get("recompute_tokens", 0.0), 0.0) for r in ts_rows]
    eps = 1e-6
    n = len(ts_rows)

    slack_events = [e for e in events if e.get("event") == "CYCLE_SLACK_COMPUTED"]
    forced_zero = [int(to_float(r.get("slack_forced_zero", 0.0), 0.0)) for r in ts_rows]
    # Backward-compatible fallback: if CSV does not carry slack_forced_zero,
    # recover it from cycle events.
    if n > 0 and sum(forced_zero) == 0 and slack_events:
        by_cycle = {}
        for e in slack_events:
            detail = e.get("detail", {}) if isinstance(e, dict) else {}
            if not isinstance(detail, dict):
                continue
            cid = int(to_float(e.get("cycle_id", detail.get("cycle_id", -1)),
                               -1.0))
            if cid < 0:
                continue
            by_cycle[cid] = int(to_float(detail.get("slack_forced_zero", 0.0), 0.0))
        if by_cycle:
            for i, r in enumerate(ts_rows):
                cid = int(to_float(r.get("cycle_id", i + 1), float(i + 1)))
                forced_zero[i] = by_cycle.get(cid, forced_zero[i])

    slack_event_coverage = len(slack_events) / max(1, n)

    # Check consistency on non-forced cycles only:
    # S_ms ~= max(0, T_cyc_ms - T_on_ms). Forced-zero cycles intentionally
    # override this relation to enforce online-first hard boundary.
    s_err: List[float] = []
    for tc, to, s, fz in zip(t_cyc, t_on, s_ms, forced_zero):
        if math.isnan(tc) or math.isnan(to) or math.isnan(s):
            continue
        if fz != 0:
            continue
        s_ref = max(0.0, tc - to)
        s_err.append(abs(s - s_ref))
    slack_formula_err_p99 = percentile(s_err, 99)
    non_forced_cycles = sum(1 for fz in forced_zero if fz == 0)
    forced_zero_ratio = (
        sum(1 for fz in forced_zero if fz != 0) / n if n > 0 else float("nan"))

    no_slack_idx = [i for i, s in enumerate(s_ms) if (not math.isnan(s)) and s <= eps]
    no_slack_recovery = 0
    for i in no_slack_idx:
        if (x_swap[i] + x_rec[i]) > 0 or swapin[i] > 0 or rec_tok[i] > 0:
            no_slack_recovery += 1
    no_slack_recovery_ratio = (
        no_slack_recovery / len(no_slack_idx) if no_slack_idx else float("nan")
    )

    high_load_idx = [i for i, (w, p) in enumerate(zip(wait, pre)) if w >= 1.0 or p > 0.0]
    high_load_s_zero_ratio = (
        sum(1 for i in high_load_idx if (not math.isnan(s_ms[i])) and s_ms[i] <= eps) / len(high_load_idx)
        if high_load_idx else float("nan")
    )

    coverage_ok = _finite_ratio(t_on) >= 0.95 and slack_event_coverage >= 0.95
    formula_ok = math.isnan(slack_formula_err_p99) or slack_formula_err_p99 <= 2.0
    no_slack_ok = math.isnan(no_slack_recovery_ratio) or no_slack_recovery_ratio <= 0.01
    # In high-load region, slack should become zero with noticeable frequency.
    high_load_ok = (
        math.isnan(high_load_s_zero_ratio)
        or high_load_s_zero_ratio >= 0.10
    )

    return {
        "pass": bool(coverage_ok and formula_ok and no_slack_ok and high_load_ok),
        "rows": n,
        "slack_event_count": len(slack_events),
        "slack_event_coverage": slack_event_coverage,
        "T_on_coverage": _finite_ratio(t_on),
        "slack_formula_err_p99_ms": slack_formula_err_p99,
        "slack_formula_eval_cycles": non_forced_cycles,
        "slack_forced_zero_ratio": forced_zero_ratio,
        "no_slack_recovery_ratio": no_slack_recovery_ratio,
        "high_load_cycles": len(high_load_idx),
        "high_load_s_zero_ratio": high_load_s_zero_ratio,
        "coverage_ok": coverage_ok,
        "formula_ok": formula_ok,
        "no_slack_ok": no_slack_ok,
        "high_load_ok": high_load_ok,
    }


def eval_p2_m2_strict(ts_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    bar_swap = [
        to_float(
            r.get(
                "barT_swap",
                r.get("barT_swap_ms_per_block", float("nan")),
            ))
        for r in ts_rows
    ]
    bar_rec = [
        to_float(
            r.get(
                "barT_rec",
                r.get("barT_rec_ms_per_token", float("nan")),
            ))
        for r in ts_rows
    ]
    x_swap = [to_float(r.get("x_swap", 0.0), 0.0) for r in ts_rows]
    x_rec = [to_float(r.get("x_rec", 0.0), 0.0) for r in ts_rows]
    swapin = [to_float(r.get("swapin_blocks", 0.0), 0.0) for r in ts_rows]
    rec_tok = [to_float(r.get("recompute_tokens", 0.0), 0.0) for r in ts_rows]

    swap_active_cycles = sum(1 for a, b in zip(x_swap, swapin) if a > 0 or b > 0)
    rec_active_cycles = sum(1 for a, b in zip(x_rec, rec_tok) if a > 0 or b > 0)

    swap_stats = _series_tail_stats(bar_swap)
    rec_stats = _series_tail_stats(bar_rec)

    swap_cov = _finite_ratio(bar_swap)
    rec_cov = _finite_ratio(bar_rec)
    swap_positive = mean(1.0 if (not math.isnan(v) and v > 0) else 0.0 for v in bar_swap)
    rec_positive = mean(1.0 if (not math.isnan(v) and v > 0) else 0.0 for v in bar_rec)

    # swap cost is required in swap workload.
    swap_ok = (
        swap_active_cycles > 0
        and swap_cov >= 0.95
        and swap_positive >= 0.95
        and (math.isnan(swap_stats["tail_cv"]) or swap_stats["tail_cv"] <= 0.50)
        and (math.isnan(swap_stats["jump_p99"]) or swap_stats["jump_p99"] <= 1.00)
        and swap_stats["changed_steps"] >= 1
    )

    # recompute cost is optional when no recompute task is active.
    rec_applicable = rec_active_cycles > 0
    if rec_applicable:
        rec_ok = (
            rec_cov >= 0.95
            and rec_positive >= 0.95
            and (math.isnan(rec_stats["tail_cv"]) or rec_stats["tail_cv"] <= 0.50)
            and (math.isnan(rec_stats["jump_p99"]) or rec_stats["jump_p99"] <= 1.00)
            and rec_stats["changed_steps"] >= 1
        )
    else:
        rec_ok = True

    return {
        "pass": bool(swap_ok and rec_ok),
        "swap_active_cycles": swap_active_cycles,
        "rec_active_cycles": rec_active_cycles,
        "swap_coverage": swap_cov,
        "rec_coverage": rec_cov,
        "swap_positive_ratio": swap_positive,
        "rec_positive_ratio": rec_positive,
        "swap_tail_cv": swap_stats["tail_cv"],
        "swap_jump_p99": swap_stats["jump_p99"],
        "swap_changed_steps": swap_stats["changed_steps"],
        "rec_tail_cv": rec_stats["tail_cv"],
        "rec_jump_p99": rec_stats["jump_p99"],
        "rec_changed_steps": rec_stats["changed_steps"],
        "rec_applicable": rec_applicable,
        "swap_ok": swap_ok,
        "rec_ok": rec_ok,
    }


def eval_gates(
    p1_sum: Dict[str, Dict[str, float]],
    p2_sum: Dict[str, Dict[str, float]],
    p1_ts: Dict[str, Any],
    p2_ts: Dict[str, Any],
    p1_sw: Dict[str, Any],
    p2_sw: Dict[str, Any],
    bctl: Dict[str, Any],
    high_lambda: Optional[str],
) -> Dict[str, Any]:
    g: Dict[str, Any] = {}

    # Gate 1: online-first
    thr_ok = False
    ttft_ok = False
    if high_lambda and high_lambda in p1_sum and high_lambda in p2_sum:
        th1 = p1_sum[high_lambda]["throughput_mean"]
        th2 = p2_sum[high_lambda]["throughput_mean"]
        tt1 = p1_sum[high_lambda]["ttft_p99_mean"]
        tt2 = p2_sum[high_lambda]["ttft_p99_mean"]
        thr_ok = th2 >= 0.90 * th1
        ttft_ok = tt2 <= 1.20 * tt1

    no_slack_intrusion_ok = (
        math.isnan(p2_ts["no_slack_recovery_ratio"])
        or p2_ts["no_slack_recovery_ratio"] <= 0.01
    )

    online_first_pass = bool(thr_ok and ttft_ok and no_slack_intrusion_ok)
    g["online_first"] = {
        "pass": online_first_pass,
        "high_lambda": high_lambda,
        "phase1_throughput": p1_sum.get(high_lambda, {}).get("throughput_mean") if high_lambda else None,
        "phase2_throughput": p2_sum.get(high_lambda, {}).get("throughput_mean") if high_lambda else None,
        "phase1_ttft_p99": p1_sum.get(high_lambda, {}).get("ttft_p99_mean") if high_lambda else None,
        "phase2_ttft_p99": p2_sum.get(high_lambda, {}).get("ttft_p99_mean") if high_lambda else None,
        "S_zero_ratio_phase2": p2_ts["S_zero_ratio"],
        "no_slack_recovery_ratio_phase2": p2_ts["no_slack_recovery_ratio"],
        "throughput_ok": thr_ok,
        "ttft_ok": ttft_ok,
        "no_slack_intrusion_ok": no_slack_intrusion_ok,
    }

    # Gate 2: budget stable
    budget_pass = bool(
        bctl.get("has_budget_updates")
        and bctl.get("hysteresis_pass")
        and bctl.get("zigzag_pass")
        and bctl.get("trend_consistency_pass")
    )
    g["budget_stable"] = {
        "pass": budget_pass,
        **bctl,
    }

    # Gate 3: oscillation suppression
    # Burst is about concentrated spikes, not sparse single preempt events.
    # Use >=2 preempts within a 5s window as primary burst signal, with a
    # small tolerance to reduce single-run randomness sensitivity.
    p1_burst_ge2 = p1_ts.get("preempt_burst_windows_5s_ge2",
                              p1_ts["preempt_burst_windows_5s"])
    p2_burst_ge2 = p2_ts.get("preempt_burst_windows_5s_ge2",
                              p2_ts["preempt_burst_windows_5s"])
    preempt_burst_ok = p2_burst_ge2 <= (p1_burst_ge2 + 1)
    waiting_peak_ok = p2_ts["wait_p99"] <= p1_ts["wait_p99"]
    waiting_dur_ok = p2_ts["wait_high_ratio"] <= p1_ts["wait_high_ratio"]
    oscillation_pass = bool(preempt_burst_ok and waiting_peak_ok and waiting_dur_ok)
    g["oscillation_suppressed"] = {
        "pass": oscillation_pass,
        "phase1_preempt_burst_windows_5s": p1_ts["preempt_burst_windows_5s"],
        "phase2_preempt_burst_windows_5s": p2_ts["preempt_burst_windows_5s"],
        "phase1_preempt_burst_windows_5s_ge2": p1_burst_ge2,
        "phase2_preempt_burst_windows_5s_ge2": p2_burst_ge2,
        "phase1_wait_p99": p1_ts["wait_p99"],
        "phase2_wait_p99": p2_ts["wait_p99"],
        "phase1_wait_high_ratio": p1_ts["wait_high_ratio"],
        "phase2_wait_high_ratio": p2_ts["wait_high_ratio"],
        "preempt_burst_ok": preempt_burst_ok,
        "waiting_peak_ok": waiting_peak_ok,
        "waiting_duration_ok": waiting_dur_ok,
    }

    # Gate 4: no starvation
    commit_ok = p2_sw["commit_monotonic"] is True and p2_sw["commit_reqs"] > 0
    gap_ok = (
        math.isnan(p1_sw["swapin_gap_active_max_ms"])
        or math.isnan(p2_sw["swapin_gap_active_max_ms"])
        or p2_sw["swapin_gap_active_max_ms"] <= 1.10 * p1_sw["swapin_gap_active_max_ms"]
    )
    stalled_reduced_ok = p2_sw["stalled_events"] <= p1_sw["stalled_events"]
    no_starve_pass = bool(commit_ok and gap_ok and stalled_reduced_ok)
    g["no_starvation"] = {
        "pass": no_starve_pass,
        "phase1_unfinished_ratio": p1_sw["unfinished_ratio"],
        "phase2_unfinished_ratio": p2_sw["unfinished_ratio"],
        "phase1_swapin_gap_max_ms": p1_sw["swapin_gap_max_ms"],
        "phase2_swapin_gap_max_ms": p2_sw["swapin_gap_max_ms"],
        "phase1_swapin_gap_active_max_ms": p1_sw["swapin_gap_active_max_ms"],
        "phase2_swapin_gap_active_max_ms": p2_sw["swapin_gap_active_max_ms"],
        "phase1_stalled_events": p1_sw["stalled_events"],
        "phase2_stalled_events": p2_sw["stalled_events"],
        "phase2_commit_reqs": p2_sw["commit_reqs"],
        "phase2_commit_monotonic": p2_sw["commit_monotonic"],
        "commit_ok": commit_ok,
        "gap_ok": gap_ok,
        "stalled_reduced_ok": stalled_reduced_ok,
    }

    g["all_pass"] = bool(
        g["online_first"]["pass"]
        and g["budget_stable"]["pass"]
        and g["oscillation_suppressed"]["pass"]
        and g["no_starvation"]["pass"]
    )

    return g


def maybe_plot(plot_dir: str, p1_ts: Dict[str, Any], p2_ts: Dict[str, Any], p1_sum: Dict[str, Dict[str, float]], p2_sum: Dict[str, Dict[str, float]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] skip plot output: matplotlib unavailable ({e})")
        print("[HINT] install matplotlib, e.g. `python3 -m pip install matplotlib`")
        return []

    os.makedirs(plot_dir, exist_ok=True)
    out_paths: List[str] = []

    # Plot 1: Phase2 B_ms vs S_ms
    t2 = p2_ts["series"]["t_rel_s"]
    s2 = p2_ts["series"]["S_ms"]
    b2 = p2_ts["series"]["B_ms"]
    if t2 and s2:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t2, s2, label="S_ms (slack)", linewidth=1.0)
        if any(not math.isnan(x) for x in b2):
            ax.plot(t2, b2, label="B_ms (budget)", linewidth=1.0)
        ax.set_title("Phase2 slack vs budget over time")
        ax.set_xlabel("t_rel_s")
        ax.set_ylabel("ms")
        ax.legend()
        ax.grid(alpha=0.25)
        p = os.path.join(plot_dir, "phase2_budget_slack_timeseries.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p)

    # Plot 2: preempt + waiting compare
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    axes[0].plot(p1_ts["series"]["t_rel_s"], p1_ts["series"]["on_preempt_count_delta"], label="Phase1")
    axes[0].plot(p2_ts["series"]["t_rel_s"], p2_ts["series"]["on_preempt_count_delta"], label="Phase2")
    axes[0].set_title("Preempt delta per cycle")
    axes[0].set_ylabel("delta")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(p1_ts["series"]["t_rel_s"], p1_ts["series"]["on_waiting_len"], label="Phase1")
    axes[1].plot(p2_ts["series"]["t_rel_s"], p2_ts["series"]["on_waiting_len"], label="Phase2")
    axes[1].set_title("Waiting length per cycle")
    axes[1].set_xlabel("t_rel_s")
    axes[1].set_ylabel("waiting_len")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    p = os.path.join(plot_dir, "phase1_phase2_preempt_wait_compare.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)

    # Plot 3: high-level summary by lambda
    lams = sorted(set(p1_sum.keys()) & set(p2_sum.keys()), key=lambda x: to_float(x, 0.0))
    if lams:
        x = list(range(len(lams)))
        width = 0.35
        p1_thr = [p1_sum[l]["throughput_mean"] for l in lams]
        p2_thr = [p2_sum[l]["throughput_mean"] for l in lams]
        p1_t99 = [p1_sum[l]["ttft_p99_mean"] for l in lams]
        p2_t99 = [p2_sum[l]["ttft_p99_mean"] for l in lams]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar([i - width / 2 for i in x], p1_thr, width=width, label="Phase1")
        axes[0].bar([i + width / 2 for i in x], p2_thr, width=width, label="Phase2")
        axes[0].set_xticks(x, lams)
        axes[0].set_title("Throughput (ok_200 / wall_elapsed_s)")
        axes[0].set_xlabel("lambda_rps")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar([i - width / 2 for i in x], p1_t99, width=width, label="Phase1")
        axes[1].bar([i + width / 2 for i in x], p2_t99, width=width, label="Phase2")
        axes[1].set_xticks(x, lams)
        axes[1].set_title("TTFT p99")
        axes[1].set_xlabel("lambda_rps")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.25)

        p = os.path.join(plot_dir, "phase1_phase2_summary_compare.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p)

    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-summary", default=None)
    ap.add_argument("--phase2-summary", default=None)
    ap.add_argument("--phase1-ts", default=None)
    ap.add_argument("--phase2-ts", default=None)
    ap.add_argument("--phase1-events", default=None)
    ap.add_argument("--phase2-events", default=None)
    ap.add_argument("--high-lambda", default=None, help="override high-load lambda_rps")
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--plot-dir",
        default="logs/RecoveryGen/new_Phase2/plots",
        help="output directory for analysis plots (default enabled)",
    )
    ap.add_argument("--no-plots", action="store_true", help="disable plot output")
    ap.add_argument("--strict-m1-m2", action="store_true",
                    help="include P2-M1/P2-M2 strict checks into final ALL PASS")
    args = ap.parse_args()

    p1_summary = args.phase1_summary or latest_file("logs/RecoveryGen/new_Phase1/client_logs/*_pmode=*/summary.csv")
    p2_summary = args.phase2_summary or latest_file("logs/RecoveryGen/new_Phase2/client_logs/*_pmode=*/summary.csv")
    p1_ts_path = args.phase1_ts or latest_file("logs/RecoveryGen/new_Phase1/server_logs/*/recovery/recovery_ts.csv")
    p2_ts_path = args.phase2_ts or latest_file("logs/RecoveryGen/new_Phase2/server_logs/*/recovery/recovery_ts.csv")
    p1_events_path = args.phase1_events or latest_file("logs/RecoveryGen/new_Phase1/server_logs/*/recovery/recovery_events.jsonl")
    p2_events_path = args.phase2_events or latest_file("logs/RecoveryGen/new_Phase2/server_logs/*/recovery/recovery_events.jsonl")

    required = {
        "phase1_summary": p1_summary,
        "phase2_summary": p2_summary,
        "phase1_ts": p1_ts_path,
        "phase2_ts": p2_ts_path,
        "phase1_events": p1_events_path,
        "phase2_events": p2_events_path,
    }
    missing = [k for k, v in required.items() if not v or not os.path.isfile(v)]
    if missing:
        raise SystemExit(f"missing required inputs: {', '.join(missing)}")

    p1_sum_rows = read_csv(p1_summary)
    p2_sum_rows = read_csv(p2_summary)
    p1_ts_rows = read_csv(p1_ts_path)
    p2_ts_rows = read_csv(p2_ts_path)
    p1_events = read_jsonl(p1_events_path)
    p2_events = read_jsonl(p2_events_path)

    p1_sum = summary_by_lambda(p1_sum_rows)
    p2_sum = summary_by_lambda(p2_sum_rows)
    p1_ts = ts_stats(p1_ts_rows)
    p2_ts = ts_stats(p2_ts_rows)
    p1_sw = swap_timelines(p1_events)
    p2_sw = swap_timelines(p2_events)
    bctl = budget_stability(p2_events)

    common_lams = sorted(set(p1_sum.keys()) & set(p2_sum.keys()), key=lambda x: to_float(x, 0.0))
    high_lambda = args.high_lambda or pick_high_lambda(common_lams)

    gates = eval_gates(
        p1_sum=p1_sum,
        p2_sum=p2_sum,
        p1_ts=p1_ts,
        p2_ts=p2_ts,
        p1_sw=p1_sw,
        p2_sw=p2_sw,
        bctl=bctl,
        high_lambda=high_lambda,
    )
    strict_m1 = eval_p2_m1_strict(p2_ts_rows, p2_events)
    strict_m2 = eval_p2_m2_strict(p2_ts_rows)
    gates["strict_m1_online_first_slack"] = strict_m1
    gates["strict_m2_cost_model"] = strict_m2
    gates["all_pass_base"] = gates["all_pass"]
    gates["all_pass_strict_m1_m2"] = bool(
        gates["all_pass"] and strict_m1["pass"] and strict_m2["pass"]
    )
    if args.strict_m1_m2:
        gates["all_pass"] = gates["all_pass_strict_m1_m2"]

    print("=== Phase1 vs Phase2 Effectiveness ===")
    print(f"phase1_summary: {p1_summary}")
    print(f"phase2_summary: {p2_summary}")
    print(f"phase1_ts     : {p1_ts_path}")
    print(f"phase2_ts     : {p2_ts_path}")
    print(f"phase1_events : {p1_events_path}")
    print(f"phase2_events : {p2_events_path}")

    print("\n--- Gate 1: online-first ---")
    g1 = gates["online_first"]
    print(
        f"[{'PASS' if g1['pass'] else 'FAIL'}] high_lambda={g1['high_lambda']} "
        f"thr {g1['phase1_throughput']:.3f}->{g1['phase2_throughput']:.3f} "
        f"ttft_p99 {g1['phase1_ttft_p99']:.3f}->{g1['phase2_ttft_p99']:.3f} "
        f"S_zero_ratio={g1['S_zero_ratio_phase2']:.3f} "
        f"no_slack_recovery_ratio={g1['no_slack_recovery_ratio_phase2']:.3f}"
    )

    print("\n--- Gate 2: budget stable ---")
    g2 = gates["budget_stable"]
    print(
        f"[{'PASS' if g2['pass'] else 'FAIL'}] updates={g2.get('updates', 0)} "
        f"step_up={g2.get('avg_budget_step_up_ms', float('nan')):.3f} "
        f"step_down={g2.get('avg_budget_step_down_ms', float('nan')):.3f} "
        f"zigzag={g2.get('zigzag_ratio', float('nan')):.3f} "
        f"trend_consistency={g2.get('trend_consistency', float('nan')):.3f}"
    )

    print("\n--- Gate 3: oscillation suppressed ---")
    g3 = gates["oscillation_suppressed"]
    print(
        f"[{'PASS' if g3['pass'] else 'FAIL'}] "
        f"preempt_windows_ge2 {g3['phase1_preempt_burst_windows_5s_ge2']}->{g3['phase2_preempt_burst_windows_5s_ge2']} "
        f"(all>0 {g3['phase1_preempt_burst_windows_5s']}->{g3['phase2_preempt_burst_windows_5s']}) "
        f"wait_p99 {g3['phase1_wait_p99']:.3f}->{g3['phase2_wait_p99']:.3f} "
        f"wait_high_ratio {g3['phase1_wait_high_ratio']:.3f}->{g3['phase2_wait_high_ratio']:.3f}"
    )

    print("\n--- Gate 4: no starvation ---")
    g4 = gates["no_starvation"]
    cm = g4["phase2_commit_monotonic"]
    cm_s = "UNKNOWN" if cm is None else ("PASS" if cm else "FAIL")
    print(
        f"[{'PASS' if g4['pass'] else 'FAIL'}] "
        f"unfinished_ratio {g4['phase1_unfinished_ratio']:.3f}->{g4['phase2_unfinished_ratio']:.3f} "
        f"max_swapin_gap_ms {g4['phase1_swapin_gap_max_ms']:.3f}->{g4['phase2_swapin_gap_max_ms']:.3f} "
        f"stalled_events {g4['phase1_stalled_events']}->{g4['phase2_stalled_events']} "
        f"commit_reqs={g4['phase2_commit_reqs']} commit_monotonic={cm_s}"
    )

    print("\n--- Gate 5: strict P2-M1 (online-first + slack boundary) ---")
    g5 = gates["strict_m1_online_first_slack"]
    print(
        f"[{'PASS' if g5['pass'] else 'FAIL'}] "
        f"slack_event_cov={g5['slack_event_coverage']:.3f} "
        f"T_on_cov={g5['T_on_coverage']:.3f} "
        f"slack_err_p99_ms={g5['slack_formula_err_p99_ms']:.3f} "
        f"slack_forced_zero_ratio={g5['slack_forced_zero_ratio']:.3f} "
        f"no_slack_recovery_ratio={g5['no_slack_recovery_ratio']:.3f} "
        f"high_load_s_zero_ratio={g5['high_load_s_zero_ratio']:.3f}"
    )

    print("\n--- Gate 6: strict P2-M2 (cost model) ---")
    g6 = gates["strict_m2_cost_model"]
    rec_flag = "N/A" if not g6["rec_applicable"] else ("PASS" if g6["rec_ok"] else "FAIL")
    print(
        f"[{'PASS' if g6['pass'] else 'FAIL'}] "
        f"swap_active={g6['swap_active_cycles']} swap_cov={g6['swap_coverage']:.3f} "
        f"swap_cv={g6['swap_tail_cv']:.3f} swap_jump_p99={g6['swap_jump_p99']:.3f} "
        f"swap_changed={int(g6['swap_changed_steps'])} rec_ok={rec_flag}"
    )

    if args.strict_m1_m2:
        print(f"\nALL PASS (strict m1+m2): {'YES' if gates['all_pass'] else 'NO'}")
    else:
        print(f"\nALL PASS: {'YES' if gates['all_pass'] else 'NO'}")
        print(f"ALL PASS (strict m1+m2): {'YES' if gates['all_pass_strict_m1_m2'] else 'NO'}")

    plots: List[str] = []
    if (not args.no_plots) and args.plot_dir:
        plots = maybe_plot(args.plot_dir, p1_ts, p2_ts, p1_sum, p2_sum)
        if plots:
            print("\nplots:")
            for p in plots:
                print(f"  - {p}")
        else:
            print("\nplots: skipped (matplotlib unavailable)")

    payload = {
        "inputs": required,
        "common_lambdas": common_lams,
        "high_lambda": high_lambda,
        "phase1_summary_by_lambda": p1_sum,
        "phase2_summary_by_lambda": p2_sum,
        "phase1_ts_stats": {k: v for k, v in p1_ts.items() if k != "series"},
        "phase2_ts_stats": {k: v for k, v in p2_ts.items() if k != "series"},
        "phase1_swap_timeline": p1_sw,
        "phase2_swap_timeline": p2_sw,
        "phase2_budget_stability": bctl,
        "gates": gates,
        "plots": plots,
    }

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"json_out: {args.json_out}")


if __name__ == "__main__":
    main()
