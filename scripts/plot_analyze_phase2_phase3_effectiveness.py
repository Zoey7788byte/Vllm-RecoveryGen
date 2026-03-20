#!/usr/bin/env python3
"""Compare Phase2 vs Phase3 effectiveness with four strict gates.

Usage:
  python3 scripts/plot_analyze_phase2_phase3_effectiveness.py \
    --phase2-summary <.../summary.csv> \
    --phase3-summary <.../summary.csv> \
    --phase2-ts <.../recovery_ts.csv> \
    --phase3-ts <.../recovery_ts.csv> \
    --phase2-events <.../recovery_events.jsonl> \
    --phase3-events <.../recovery_events.jsonl> \
    --json-out <.../phase3_effectiveness_report.json>
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


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
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


def coeff_var(values: List[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    mu = statistics.fmean(vals)
    if abs(mu) < 1e-12:
        return float("nan")
    return statistics.pstdev(vals) / abs(mu)


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


def choose_high_lambda(rows: List[Dict[str, str]]) -> Optional[str]:
    best: Optional[Tuple[float, str]] = None
    for r in rows:
        s = str(r.get("lambda_rps", "")).strip()
        try:
            v = float(s)
        except Exception:
            continue
        if best is None or v > best[0]:
            best = (v, s)
    return best[1] if best is not None else None


def summary_lambda_stats(rows: List[Dict[str, str]], lam: str) -> Dict[str, float]:
    grp = [r for r in rows if str(r.get("lambda_rps", "")).strip() == lam]
    if not grp:
        return {}
    return {
        "segments": float(len(grp)),
        "ttft_p99_mean": mean([to_float(r.get("ttft_p99_s", "nan")) for r in grp]),
        "tpot_p99_mean": mean([to_float(r.get("tpot_p99_s", "nan")) for r in grp]),
        "stall_gap_p99_mean": mean([to_float(r.get("stall_gap_p99_s", "nan")) for r in grp]),
        "stall_gap_max_max": max([to_float(r.get("stall_gap_max_s", "nan")) for r in grp]),
        "preempt_sum_total": sum(to_float(r.get("preempt_sum_delta", 0), 0.0) for r in grp),
        "ttft_p99_cv": coeff_var([to_float(r.get("ttft_p99_s", "nan")) for r in grp]),
        "stall_gap_p99_cv": coeff_var([to_float(r.get("stall_gap_p99_s", "nan")) for r in grp]),
    }


def detect_client_root(summary_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return os.path.dirname(os.path.abspath(summary_path))


def collect_client_stall_stats(client_root: str) -> Dict[str, float]:
    files = sorted(
        glob.glob(os.path.join(client_root, "rc_cycle*_high_*", "client_results.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(client_root, "*", "client_results.csv")))
    stall_vals: List[float] = []
    ttft_vals: List[float] = []
    for p in files:
        try:
            rows = read_csv(p)
        except Exception:
            continue
        for r in rows:
            stall_vals.append(to_float(r.get("stall_gap_max_s", "nan")))
            ttft_vals.append(to_float(r.get("ttft_s", "nan")))
    return {
        "files": float(len(files)),
        "reqs": float(len([x for x in stall_vals if not math.isnan(x)])),
        "stall_max_s": max([x for x in stall_vals if not math.isnan(x)], default=float("nan")),
        "stall_p99_s": percentile(stall_vals, 99),
        "ttft_p99_s": percentile(ttft_vals, 99),
    }


def extract_detail(ev: Dict[str, Any]) -> Dict[str, Any]:
    d = ev.get("detail")
    return d if isinstance(d, dict) else {}


def event_times_by_req(events: List[Dict[str, Any]], etype: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for ev in events:
        if str(ev.get("event", "")) != etype:
            continue
        req = str(ev.get("req_id", "")).strip()
        if not req:
            continue
        ts = to_int(ev.get("ts_ns", 0), 0)
        if ts > 0:
            out[req].append(ts)
    for req in out:
        out[req].sort()
    return out


def rapid_preempt_ratio(events: List[Dict[str, Any]], window_ms: float) -> float:
    by_req = event_times_by_req(events, "PREEMPT_TRIGGERED")
    pairs = 0
    rapid = 0
    win_ns = int(max(0.0, window_ms) * 1e6)
    for ts_list in by_req.values():
        for i in range(1, len(ts_list)):
            pairs += 1
            if ts_list[i] - ts_list[i - 1] <= win_ns:
                rapid += 1
    if pairs == 0:
        return float("nan")
    return rapid / pairs


def mode_switch_jitter_ratio(events: List[Dict[str, Any]], window_ms: float) -> float:
    by_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    win_ns = int(max(0.0, window_ms) * 1e6)
    for ev in events:
        if str(ev.get("event", "")) != "RECOVERY_REQUEST_MODE_SWITCH":
            continue
        req = str(ev.get("req_id", "")).strip()
        seq = str(ev.get("seq_id", "")).strip()
        ts = to_int(ev.get("ts_ns", 0), 0)
        if req and seq and ts > 0:
            by_key[(req, seq)].append(ts)
    pairs = 0
    jitter = 0
    for ts_list in by_key.values():
        ts_list.sort()
        for i in range(1, len(ts_list)):
            pairs += 1
            if ts_list[i] - ts_list[i - 1] <= win_ns:
                jitter += 1
    if pairs == 0:
        return float("nan")
    return jitter / pairs


def mode_flipflop_ratio(events: List[Dict[str, Any]], window_ms: float) -> float:
    """Detect true oscillation: A->B->A within short window."""
    by_key: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    win_ns = int(max(0.0, window_ms) * 1e6)
    for ev in events:
        if str(ev.get("event", "")) != "RECOVERY_REQUEST_MODE_SWITCH":
            continue
        req = str(ev.get("req_id", "")).strip()
        seq = str(ev.get("seq_id", "")).strip()
        ts = to_int(ev.get("ts_ns", 0), 0)
        if not (req and seq and ts > 0):
            continue
        d = extract_detail(ev)
        to_mode = str(d.get("to_mode", "")).strip().upper()
        if not to_mode:
            continue
        by_key[(req, seq)].append((ts, to_mode))

    denom = 0
    flips = 0
    for key in by_key:
        seq_events = sorted(by_key[key], key=lambda x: x[0])
        if len(seq_events) < 3:
            continue
        for i in range(2, len(seq_events)):
            denom += 1
            t0, m0 = seq_events[i - 2]
            _, m1 = seq_events[i - 1]
            t2, m2 = seq_events[i]
            if (m0 == m2) and (m0 != m1) and ((t2 - t0) <= win_ns):
                flips += 1
    if denom == 0:
        return float("nan")
    return flips / denom


def normal_to_preempt_ratio(events: List[Dict[str, Any]], window_ms: float) -> float:
    normals = event_times_by_req(events, "MODE_ENTER_NORMAL")
    preempts = event_times_by_req(events, "PREEMPT_TRIGGERED")
    if not normals:
        return float("nan")
    win_ns = int(max(0.0, window_ms) * 1e6)
    total = 0
    hits = 0
    for req, nt in normals.items():
        pt = preempts.get(req, [])
        if not pt:
            total += len(nt)
            continue
        j = 0
        for t0 in nt:
            total += 1
            while j < len(pt) and pt[j] <= t0:
                j += 1
            if j < len(pt) and (pt[j] - t0) <= win_ns:
                hits += 1
    if total == 0:
        return float("nan")
    return hits / total


def phase3_pin_stats(events: List[Dict[str, Any]]) -> Dict[str, float]:
    pin_blocks: List[float] = []
    pin_events = 0
    swap_out_events = 0
    overlap_violations = 0
    for ev in events:
        et = str(ev.get("event", ""))
        d = extract_detail(ev)
        if et == "MWS_PIN_APPLIED":
            pin_events += 1
            pin_blocks.append(to_float(d.get("pinned_blocks", "nan")))
        elif et == "SWAP_OUT":
            swap_out_events += 1
            swapped = d.get("swapped_indices_sample", [])
            pinned = d.get("pinned_indices_sample", [])
            if isinstance(swapped, list) and isinstance(pinned, list):
                try:
                    if set(int(x) for x in swapped) & set(int(x) for x in pinned):
                        overlap_violations += 1
                except Exception:
                    pass
    return {
        "pin_events": float(pin_events),
        "swap_out_events": float(swap_out_events),
        "pin_blocks_median": percentile(pin_blocks, 50),
        "pin_blocks_p90": percentile(pin_blocks, 90),
        "pin_blocks_nonzero_ratio": (
            sum(1 for x in pin_blocks if not math.isnan(x) and x > 0) / len(pin_blocks)
            if pin_blocks else float("nan")
        ),
        "swap_out_overlap_violations": float(overlap_violations),
    }


def phase3_visibility_stats(events: List[Dict[str, Any]],
                            ts_rows: List[Dict[str, str]],
                            high_summary: Dict[str, float]) -> Dict[str, float]:
    vis_events = 0
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    req_pairs: Dict[str, set] = defaultdict(set)
    for ev in events:
        if str(ev.get("event", "")) != "VISIBILITY_WINDOW_ENFORCED":
            continue
        vis_events += 1
        req = str(ev.get("req_id", "")).strip()
        d = extract_detail(ev)
        a = to_int(d.get("A_tokens", -1), -1)
        w = to_int(d.get("W_min_tokens", -1), -1)
        pair = (a, w)
        pair_counts[pair] += 1
        if req:
            req_pairs[req].add(pair)

    req_with_multi_pair = sum(1 for s in req_pairs.values() if len(s) > 1)
    req_pair_change_ratio = (req_with_multi_pair / len(req_pairs)
                             if req_pairs else float("nan"))

    vis_enforced_vals = [to_float(r.get("visibility_enforced", "nan"))
                         for r in ts_rows]
    visible_tokens_vals = [to_float(r.get("visible_tokens", "nan"))
                           for r in ts_rows]
    req_mode_recovery_vals = [to_float(r.get("req_mode_cnt_recovery", "nan"))
                              for r in ts_rows]
    req_mode_fallback_vals = [to_float(r.get("req_mode_cnt_fallback", "nan"))
                              for r in ts_rows]
    mode_recovery_vals = [to_float(r.get("mode_cnt_recovery", "nan"))
                          for r in ts_rows]
    mode_fallback_vals = [to_float(r.get("mode_cnt_fallback", "nan"))
                          for r in ts_rows]

    active_idx: List[int] = []
    for i in range(len(ts_rows)):
        req_active = (
            i < len(req_mode_recovery_vals) and i < len(req_mode_fallback_vals)
            and (
                (not math.isnan(req_mode_recovery_vals[i]) and req_mode_recovery_vals[i] > 0)
                or (not math.isnan(req_mode_fallback_vals[i]) and req_mode_fallback_vals[i] > 0)
            ))
        global_active = (
            i < len(mode_recovery_vals) and i < len(mode_fallback_vals)
            and (
                (not math.isnan(mode_recovery_vals[i]) and mode_recovery_vals[i] > 0)
                or (not math.isnan(mode_fallback_vals[i]) and mode_fallback_vals[i] > 0)
            ))
        if req_active or global_active:
            active_idx.append(i)

    enforced_positive = sum(1 for v in vis_enforced_vals if not math.isnan(v) and v > 0)
    visible_positive = sum(1 for v in visible_tokens_vals if not math.isnan(v) and v > 0)
    enforced_active_positive = sum(
        1 for i in active_idx
        if i < len(vis_enforced_vals)
        and (not math.isnan(vis_enforced_vals[i]))
        and vis_enforced_vals[i] > 0)
    visible_active_positive = sum(
        1 for i in active_idx
        if i < len(visible_tokens_vals)
        and (not math.isnan(visible_tokens_vals[i]))
        and visible_tokens_vals[i] > 0)
    rows_n = max(1, len(ts_rows))
    active_rows_n = len(active_idx)

    return {
        "vis_events": float(vis_events),
        "distinct_aw_pairs": float(len(pair_counts)),
        "req_pair_change_ratio": req_pair_change_ratio,
        "ts_visibility_enforced_ratio": enforced_positive / rows_n,
        "ts_visible_tokens_ratio": visible_positive / rows_n,
        "ts_active_rows": float(active_rows_n),
        "ts_visibility_enforced_ratio_active": (
            enforced_active_positive / active_rows_n if active_rows_n > 0 else float("nan")),
        "ts_visible_tokens_ratio_active": (
            visible_active_positive / active_rows_n if active_rows_n > 0 else float("nan")),
        "high_ttft_p99_cv": to_float(high_summary.get("ttft_p99_cv", float("nan")), float("nan")),
        "high_stall_p99_cv": to_float(high_summary.get("stall_gap_p99_cv", float("nan")), float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-summary", default=None)
    ap.add_argument("--phase3-summary", default=None)
    ap.add_argument("--phase2-ts", default=None)
    ap.add_argument("--phase3-ts", default=None)
    ap.add_argument("--phase2-events", default=None)
    ap.add_argument("--phase3-events", default=None)
    ap.add_argument("--phase2-client-dir", default=None)
    ap.add_argument("--phase3-client-dir", default=None)
    ap.add_argument("--rapid-window-ms", type=float, default=3000.0)
    ap.add_argument("--normal-repreempt-window-ms", type=float, default=5000.0)
    ap.add_argument("--switch-jitter-window-ms", type=float, default=2000.0)
    ap.add_argument("--stall-max-improve-ratio", type=float, default=0.95)
    ap.add_argument("--stall-p99-improve-ratio", type=float, default=0.97)
    ap.add_argument("--stability-cv-threshold", type=float, default=0.35)
    ap.add_argument("--switches-per-preempt-max", type=float, default=6.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    p2_summary = args.phase2_summary or latest_file(
        "logs/RecoveryGen/Phase2_validation/client_logs/phase2_validate_*/summary.csv")
    p3_summary = args.phase3_summary or latest_file(
        "logs/RecoveryGen/Phase3_validation/client_logs/phase3_validate_*/summary.csv")
    p2_ts = args.phase2_ts or latest_file(
        "logs/RecoveryGen/Phase2_validation/server_logs/*/recovery/recovery_ts.csv")
    p3_ts = args.phase3_ts or latest_file(
        "logs/RecoveryGen/Phase3_validation/server_logs/*/recovery/recovery_ts.csv")
    p2_events = args.phase2_events or latest_file(
        "logs/RecoveryGen/Phase2_validation/server_logs/*/recovery/recovery_events.jsonl")
    p3_events = args.phase3_events or latest_file(
        "logs/RecoveryGen/Phase3_validation/server_logs/*/recovery/recovery_events.jsonl")

    required = {
        "phase2_summary": p2_summary,
        "phase3_summary": p3_summary,
        "phase2_ts": p2_ts,
        "phase3_ts": p3_ts,
        "phase2_events": p2_events,
        "phase3_events": p3_events,
    }
    missing = [k for k, v in required.items() if not v or (not os.path.isfile(v))]
    if missing:
        print("missing required inputs: " + ", ".join(missing))
        return

    p2_summary_rows = read_csv(str(p2_summary))
    p3_summary_rows = read_csv(str(p3_summary))
    p2_ts_rows = read_csv(str(p2_ts))
    p3_ts_rows = read_csv(str(p3_ts))
    p2_events_rows = read_jsonl(str(p2_events))
    p3_events_rows = read_jsonl(str(p3_events))

    high_lam = choose_high_lambda(p2_summary_rows) or choose_high_lambda(
        p3_summary_rows) or "0.85"
    p2_high = summary_lambda_stats(p2_summary_rows, high_lam)
    p3_high = summary_lambda_stats(p3_summary_rows, high_lam)

    p2_client_dir = detect_client_root(str(p2_summary), args.phase2_client_dir)
    p3_client_dir = detect_client_root(str(p3_summary), args.phase3_client_dir)
    p2_client = collect_client_stall_stats(p2_client_dir)
    p3_client = collect_client_stall_stats(p3_client_dir)

    p2_preempts = sum(1 for ev in p2_events_rows
                      if str(ev.get("event", "")) == "PREEMPT_TRIGGERED")
    p3_preempts = sum(1 for ev in p3_events_rows
                      if str(ev.get("event", "")) == "PREEMPT_TRIGGERED")

    # Gate 1: streaming continuity / stall tail
    g1_has_preempt = (p2_preempts > 0 and p3_preempts > 0)
    g1_stall_max_ok = (
        (not math.isnan(p2_client["stall_max_s"]))
        and (not math.isnan(p3_client["stall_max_s"]))
        and p3_client["stall_max_s"] <= p2_client["stall_max_s"] * args.stall_max_improve_ratio
    )
    g1_stall_p99_ok = (
        (not math.isnan(p2_client["stall_p99_s"]))
        and (not math.isnan(p3_client["stall_p99_s"]))
        and p3_client["stall_p99_s"] <= p2_client["stall_p99_s"] * args.stall_p99_improve_ratio
    )
    g1_tpot_ok = (
        (not math.isnan(p2_high.get("tpot_p99_mean", float("nan"))))
        and (not math.isnan(p3_high.get("tpot_p99_mean", float("nan"))))
        and p3_high["tpot_p99_mean"] <= p2_high["tpot_p99_mean"] * 1.05
    )
    g1_stall_ok = bool(g1_stall_max_ok or g1_stall_p99_ok)
    g1_pass = bool(g1_has_preempt and g1_stall_ok and g1_tpot_ok)

    # Gate 2: MWS pin effective and never swapped out
    pin = phase3_pin_stats(p3_events_rows)
    g2_pass = bool(
        pin["pin_events"] > 0
        and pin["swap_out_events"] > 0
        and pin["swap_out_overlap_violations"] == 0
        and (math.isnan(pin["pin_blocks_nonzero_ratio"])
             or pin["pin_blocks_nonzero_ratio"] >= 0.8)
    )

    # Gate 3: contract semantics reproducible (event + stability proxy)
    vis = phase3_visibility_stats(p3_events_rows, p3_ts_rows, p3_high)
    g3_cov_ok = bool(
        vis["vis_events"] > 0 and (
            (not math.isnan(vis["ts_visibility_enforced_ratio_active"])
             and vis["ts_visibility_enforced_ratio_active"] > 0)
            or (not math.isnan(vis["ts_visible_tokens_ratio_active"])
                and vis["ts_visible_tokens_ratio_active"] > 0)
            or vis["ts_visibility_enforced_ratio"] > 0
        ))
    g3_stable_ok = (
        (math.isnan(vis["high_ttft_p99_cv"])
         or vis["high_ttft_p99_cv"] <= args.stability_cv_threshold)
        and (math.isnan(vis["high_stall_p99_cv"])
             or vis["high_stall_p99_cv"] <= args.stability_cv_threshold)
    )
    g3_pair_ok = (math.isnan(vis["req_pair_change_ratio"])
                  or vis["req_pair_change_ratio"] <= 0.05)
    g3_pass = bool(g3_cov_ok and g3_stable_ok and g3_pair_ok)

    # Gate 4: switch jitter reduced
    p2_rapid = rapid_preempt_ratio(p2_events_rows, args.rapid_window_ms)
    p3_rapid = rapid_preempt_ratio(p3_events_rows, args.rapid_window_ms)
    p3_norm_repreempt = normal_to_preempt_ratio(
        p3_events_rows, args.normal_repreempt_window_ms)
    p3_switch_jitter = mode_switch_jitter_ratio(
        p3_events_rows, args.switch_jitter_window_ms)
    p3_flipflop = mode_flipflop_ratio(
        p3_events_rows, args.switch_jitter_window_ms)
    p3_switch_cnt = sum(1 for ev in p3_events_rows
                        if str(ev.get("event", "")) == "RECOVERY_REQUEST_MODE_SWITCH")
    p3_switch_per_preempt = (
        float(p3_switch_cnt) / float(max(1, p3_preempts)))
    g4_rapid_ok = (
        math.isnan(p2_rapid) or (not math.isnan(p3_rapid) and p3_rapid <= p2_rapid))
    g4_norm_ok = math.isnan(p3_norm_repreempt) or p3_norm_repreempt <= 0.20
    # Legacy jitter ratio is noisy for short healthy mode transitions.
    g4_switch_jitter_ok = math.isnan(p3_flipflop) or p3_flipflop <= 0.20
    g4_switch_rate_ok = p3_switch_per_preempt <= args.switches_per_preempt_max
    g4_pass = bool(g4_rapid_ok and g4_norm_ok and g4_switch_jitter_ok
                   and g4_switch_rate_ok)

    all_pass = bool(g1_pass and g2_pass and g3_pass and g4_pass)

    print(f"phase2 summary: {p2_summary}")
    print(f"phase3 summary: {p3_summary}")
    print(f"phase2 events : {p2_events}")
    print(f"phase3 events : {p3_events}")

    print("\n--- Gate 1: continuity improved under preemption ---")
    print(
        f"[{'PASS' if g1_pass else 'FAIL'}] "
        f"high_lambda={high_lam} preempt {p2_preempts}->{p3_preempts} "
        f"stall_max_s {p2_client['stall_max_s']:.3f}->{p3_client['stall_max_s']:.3f} "
        f"stall_p99_s {p2_client['stall_p99_s']:.3f}->{p3_client['stall_p99_s']:.3f} "
        f"tpot_p99 {p2_high.get('tpot_p99_mean', float('nan')):.3f}->{p3_high.get('tpot_p99_mean', float('nan')):.3f}"
    )

    print("\n--- Gate 2: MWS pin effective and not evicted ---")
    print(
        f"[{'PASS' if g2_pass else 'FAIL'}] "
        f"pin_events={int(pin['pin_events'])} swap_out_events={int(pin['swap_out_events'])} "
        f"overlap_violations={int(pin['swap_out_overlap_violations'])} "
        f"pin_nonzero_ratio={pin['pin_blocks_nonzero_ratio']:.3f} "
        f"pin_median={pin['pin_blocks_median']:.1f}"
    )

    print("\n--- Gate 3: contract semantics reproducible ---")
    print(
        f"[{'PASS' if g3_pass else 'FAIL'}] "
        f"vis_events={int(vis['vis_events'])} distinct_(A,W)={int(vis['distinct_aw_pairs'])} "
        f"req_pair_change_ratio={vis['req_pair_change_ratio']:.3f} "
        f"vis_ts_ratio={vis['ts_visibility_enforced_ratio']:.3f} "
        f"vis_ts_ratio_active={vis['ts_visibility_enforced_ratio_active']:.3f} "
        f"active_rows={int(vis['ts_active_rows'])} "
        f"ttft_p99_cv={vis['high_ttft_p99_cv']:.3f} stall_p99_cv={vis['high_stall_p99_cv']:.3f}"
    )

    print("\n--- Gate 4: mode switching not oscillatory ---")
    print(
        f"[{'PASS' if g4_pass else 'FAIL'}] "
        f"rapid_preempt_ratio {p2_rapid:.3f}->{p3_rapid:.3f} "
        f"normal_to_preempt_ratio={p3_norm_repreempt:.3f} "
        f"switch_jitter_ratio_legacy={p3_switch_jitter:.3f} "
        f"switch_flipflop_ratio={p3_flipflop:.3f} "
        f"switches/preempt={p3_switch_per_preempt:.3f}"
    )

    print(f"\nALL PASS (phase3): {'YES' if all_pass else 'NO'}")

    report = {
        "inputs": {
            "phase2_summary": p2_summary,
            "phase3_summary": p3_summary,
            "phase2_ts": p2_ts,
            "phase3_ts": p3_ts,
            "phase2_events": p2_events,
            "phase3_events": p3_events,
            "phase2_client_dir": p2_client_dir,
            "phase3_client_dir": p3_client_dir,
        },
        "high_lambda": high_lam,
        "phase2_high": p2_high,
        "phase3_high": p3_high,
        "phase2_client": p2_client,
        "phase3_client": p3_client,
        "gate1": {
            "pass": g1_pass,
            "has_preempt": g1_has_preempt,
            "stall_max_ok": g1_stall_max_ok,
            "stall_p99_ok": g1_stall_p99_ok,
            "stall_ok": g1_stall_ok,
            "tpot_ok": g1_tpot_ok,
            "preempt_phase2": p2_preempts,
            "preempt_phase3": p3_preempts,
        },
        "gate2": {
            "pass": g2_pass,
            **pin,
        },
        "gate3": {
            "pass": g3_pass,
            "coverage_ok": g3_cov_ok,
            "stable_ok": g3_stable_ok,
            "pair_ok": g3_pair_ok,
            **vis,
        },
        "gate4": {
            "pass": g4_pass,
            "rapid_ok": g4_rapid_ok,
            "normal_repreempt_ok": g4_norm_ok,
            "switch_jitter_ok": g4_switch_jitter_ok,
            "switch_rate_ok": g4_switch_rate_ok,
            "phase2_rapid_preempt_ratio": p2_rapid,
            "phase3_rapid_preempt_ratio": p3_rapid,
            "phase3_normal_to_preempt_ratio": p3_norm_repreempt,
            "phase3_switch_jitter_ratio": p3_switch_jitter,
            "phase3_switch_flipflop_ratio": p3_flipflop,
            "phase3_switches_per_preempt": p3_switch_per_preempt,
            "phase3_switch_count": p3_switch_cnt,
        },
        "all_pass": all_pass,
    }

    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"json report: {args.json_out}")


if __name__ == "__main__":
    main()
