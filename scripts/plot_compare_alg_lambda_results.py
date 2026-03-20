#!/usr/bin/env python3
"""Compare run_compare_alg.sh outputs with full metric instrumentation.

Compared methods:
1) vllm
2) RecoverGen
3) vllm (no_chunk)
4) RecoverGen (no_chunk) [optional]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


DISPLAY_METHOD_ORDER = [
    "vllm (no_chunk)",
    "vllm",
    "RecoverGen (no_chunk)",
    "RecoverGen",
]


def _method_display_rank(name: str) -> int:
    try:
        return DISPLAY_METHOD_ORDER.index(name)
    except ValueError:
        return len(DISPLAY_METHOD_ORDER)


def _ordered_methods(methods: List["MethodData"]) -> List["MethodData"]:
    return sorted(methods, key=lambda m: (_method_display_rank(m.name), m.name))


def _baseline_method(methods: List["MethodData"]) -> Optional["MethodData"]:
    for m in methods:
        if m.name == "vllm":
            return m
    return methods[0] if methods else None


def _gap_metric_spec(gap_metric_mode: str) -> Tuple[str, str, str]:
    # title, unit, key in request_by_lambda
    if gap_metric_mode == "long_gap_ratio":
        return ("Long-gap Ratio (>=1s)", "ratio", "long_gap_ge_1s_ratio")
    return ("Max Stall Gap", "s", "stall_gap_max_s_req")


def _to_float(v: Any, default: float = float("nan")) -> float:
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _finite(xs: Iterable[float]) -> List[float]:
    return [x for x in xs if not math.isnan(x)]


def _safe_mean(xs: Iterable[float]) -> float:
    vals = _finite(list(xs))
    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


def _safe_sum(xs: Iterable[float]) -> float:
    vals = _finite(list(xs))
    if not vals:
        return float("nan")
    return sum(vals)


def _safe_max(xs: Iterable[float]) -> float:
    vals = _finite(list(xs))
    if not vals:
        return float("nan")
    return max(vals)


def _safe_min(xs: Iterable[float]) -> float:
    vals = _finite(list(xs))
    if not vals:
        return float("nan")
    return min(vals)


def _percentile(values: List[float], q: float) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    if q <= 0:
        return min(vals)
    if q >= 100:
        return max(vals)
    s = sorted(vals)
    k = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _pct_change(new: float, old: float) -> float:
    if math.isnan(new) or math.isnan(old) or abs(old) < 1e-12:
        return float("nan")
    return (new - old) / abs(old) * 100.0


def _fmt(v: float, digits: int = 3) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
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


def _find_latest_summary(client_root: str) -> str:
    pattern = os.path.join(client_root, "*/summary.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No summary.csv under: {client_root}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _find_all_summaries(client_root: str) -> List[str]:
    pattern = os.path.join(client_root, "*/summary.csv")
    candidates = glob.glob(pattern)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates


def _try_find_latest_summary(client_root: Optional[str]) -> Optional[str]:
    if not client_root:
        return None
    try:
        return _find_latest_summary(client_root)
    except FileNotFoundError:
        return None


def _extract_run_ts(path: str) -> Optional[str]:
    all_hits = re.findall(r"(\d{8}_\d{6})", path)
    return all_hits[-1] if all_hits else None


def _normalize_recovery_dir(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return os.path.dirname(p)
    if os.path.basename(p) == "recovery":
        return p
    candidate = os.path.join(p, "recovery")
    if os.path.isdir(candidate):
        return candidate
    return p


def _infer_recovery_dir(summary_path: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        p = _normalize_recovery_dir(explicit)
        return p if os.path.isdir(p) else None

    summary_abs = os.path.abspath(summary_path)
    # .../<method>/client_logs/<run_tag>/summary.csv
    run_ts = _extract_run_ts(summary_abs)
    if not run_ts:
        return None
    run_dir = os.path.dirname(summary_abs)
    client_logs_dir = os.path.dirname(run_dir)
    method_root = os.path.dirname(client_logs_dir)
    candidate = os.path.join(method_root, "server_logs", run_ts, "recovery")
    return candidate if os.path.isdir(candidate) else None


def _parse_lambda_from_seg_name(seg_name: str) -> float:
    m = re.search(r"lam(\d+)p(\d+)", seg_name)
    if not m:
        return float("nan")
    return float(f"{int(m.group(1))}.{m.group(2)}")


def _summary_lambdas(summary_path: str) -> List[float]:
    rows = _read_csv(summary_path)
    vals: List[float] = []
    for r in rows:
        lam = _to_float(r.get("lambda_rps", "nan"))
        if math.isnan(lam):
            continue
        vals.append(lam)
    if not vals:
        return []
    uniq = sorted(set(vals))
    return uniq


def _select_lambdas(
    available_lambdas: List[float],
    target_lambda: Optional[float],
    lambda_tol: float,
    scope_name: str,
) -> List[float]:
    if target_lambda is None:
        return sorted(available_lambdas)
    if not available_lambdas:
        return []
    matched = [lam for lam in available_lambdas if abs(lam - target_lambda) <= lambda_tol]
    if matched:
        return [sorted(matched)[0]]
    raise ValueError(
        f"target lambda={target_lambda} not found in {scope_name}; "
        f"available={[round(x, 4) for x in sorted(available_lambdas)]}"
    )


def _dict_for_lambda(
    by_lambda: Dict[float, Dict[str, float]],
    lam: float,
    lambda_tol: float,
) -> Dict[str, float]:
    for key, val in by_lambda.items():
        if abs(key - lam) <= lambda_tol:
            return val
    return {}


def _parse_target_lambdas(text: Optional[str]) -> List[float]:
    if not text:
        return []
    tokens = re.split(r"[,\s]+", text.strip())
    out: List[float] = []
    for tok in tokens:
        if not tok:
            continue
        out.append(float(tok))
    return out


def _resolve_summary_paths(
    method_name: str,
    explicit_summary: Optional[str],
    client_root: Optional[str],
    target_lambdas: List[float],
    lambda_tol: float,
    optional: bool = False,
) -> List[str]:
    if explicit_summary:
        if not os.path.isfile(explicit_summary):
            raise FileNotFoundError(f"{method_name}: summary not found: {explicit_summary}")
        paths = [os.path.abspath(explicit_summary)]
    else:
        if not client_root:
            if optional:
                return []
            raise FileNotFoundError(f"{method_name}: client root is empty")
        all_paths = _find_all_summaries(client_root)
        if not all_paths:
            if optional:
                return []
            raise FileNotFoundError(f"{method_name}: no summary.csv under: {client_root}")

        if not target_lambdas:
            paths = [os.path.abspath(all_paths[0])]
        else:
            selected: List[str] = []
            covered: List[float] = []
            avail_union: List[float] = []
            for p in all_paths:
                try:
                    lams = _summary_lambdas(p)
                except Exception:
                    continue
                for x in lams:
                    if not any(abs(x - y) <= lambda_tol for y in avail_union):
                        avail_union.append(x)
                need = [
                    t for t in target_lambdas
                    if not any(abs(t - y) <= lambda_tol for y in covered)
                ]
                if not need:
                    break
                hit = any(any(abs(t - x) <= lambda_tol for x in lams) for t in need)
                if hit:
                    selected.append(os.path.abspath(p))
                    for t in need:
                        if any(abs(t - x) <= lambda_tol for x in lams):
                            covered.append(t)
            missing = [
                t for t in target_lambdas
                if not any(abs(t - y) <= lambda_tol for y in covered)
            ]
            if missing and not optional:
                raise ValueError(
                    f"{method_name}: missing target lambdas={missing}; "
                    f"available={sorted([round(x, 4) for x in avail_union])}"
                )
            paths = selected

    # de-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _event_req_key(ev: Dict[str, Any]) -> str:
    rid = str(ev.get("req_id", "")).strip()
    if rid:
        return rid
    sid = str(ev.get("seq_id", "")).strip()
    if sid:
        return f"seq:{sid}"
    return ""


def _event_detail(ev: Dict[str, Any]) -> Dict[str, Any]:
    d = ev.get("detail")
    return d if isinstance(d, dict) else {}


def _first_num(d: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        if k in d:
            v = _to_float(d.get(k), float("nan"))
            if not math.isnan(v):
                return v
    return float("nan")


def _windowed_rates(
    t_rel_s: List[float],
    deltas: List[float],
    window_s: float,
) -> List[float]:
    if window_s <= 0:
        return []
    pairs = [
        (t, max(0.0, d))
        for t, d in zip(t_rel_s, deltas)
        if (not math.isnan(t)) and (not math.isnan(d))
    ]
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[0])
    t0 = pairs[0][0]
    bins: DefaultDict[int, float] = defaultdict(float)
    max_bin = 0
    for t, d in pairs:
        b = int(max(0.0, (t - t0)) // window_s)
        bins[b] += d
        if b > max_bin:
            max_bin = b
    rates: List[float] = []
    for b in range(max_bin + 1):
        rates.append(bins.get(b, 0.0) / window_s)
    return rates


@dataclass
class MethodData:
    name: str
    summary_path: str
    summary_paths: List[str]
    client_run_dir: str
    recovery_dir: Optional[str]
    summary_rows: List[Dict[str, str]]
    summary_by_lambda: Dict[float, Dict[str, float]]
    request_by_lambda: Dict[float, Dict[str, float]]
    preempt_window_by_lambda: Dict[float, Dict[str, float]]
    ts_metrics: Dict[str, float]
    event_metrics: Dict[str, float]


def _aggregate_summary_by_lambda(rows: List[Dict[str, str]]) -> Dict[float, Dict[str, float]]:
    groups: DefaultDict[float, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        lam = _to_float(r.get("lambda_rps", "nan"))
        if math.isnan(lam):
            continue
        groups[lam].append(r)

    out: Dict[float, Dict[str, float]] = {}
    for lam, grp in groups.items():
        ok_sum = _safe_sum([_to_float(r.get("ok_200", "nan")) for r in grp])
        total_sum = _safe_sum([_to_float(r.get("total_rows", "nan")) for r in grp])
        ok_rate = float("nan")
        if not math.isnan(ok_sum) and not math.isnan(total_sum) and total_sum > 0:
            ok_rate = ok_sum / total_sum

        out[lam] = {
            "segments": float(len(grp)),
            "ok_rate": ok_rate,
            "ttft_p50_s": _safe_mean([_to_float(r.get("ttft_p50_s")) for r in grp]),
            "ttft_p90_s": _safe_mean([_to_float(r.get("ttft_p90_s")) for r in grp]),
            "ttft_p99_s": _safe_mean([_to_float(r.get("ttft_p99_s")) for r in grp]),
            "ttft_p999_s": _safe_mean([_to_float(r.get("ttft_p999_s")) for r in grp]),
            "ttft_max_s": _safe_mean([_to_float(r.get("ttft_max_s")) for r in grp]),
            "tpot_p50_s": _safe_mean([_to_float(r.get("tpot_p50_s")) for r in grp]),
            "tpot_p90_s": _safe_mean([_to_float(r.get("tpot_p90_s")) for r in grp]),
            "tpot_p99_s": _safe_mean([_to_float(r.get("tpot_p99_s")) for r in grp]),
            "tpot_p999_s": _safe_mean([_to_float(r.get("tpot_p999_s")) for r in grp]),
            "tpot_max_s": _safe_mean([_to_float(r.get("tpot_max_s")) for r in grp]),
            "stall_gap_p50_s": _safe_mean([_to_float(r.get("stall_gap_p50_s")) for r in grp]),
            "stall_gap_p90_s": _safe_mean([_to_float(r.get("stall_gap_p90_s")) for r in grp]),
            "stall_gap_p99_s": _safe_mean([_to_float(r.get("stall_gap_p99_s")) for r in grp]),
            "stall_gap_max_s": _safe_mean([_to_float(r.get("stall_gap_max_s")) for r in grp]),
            "decode_tps_wall": _safe_mean(
                [_to_float(r.get("decode_toks_per_s_client_wall")) for r in grp]),
            "preempt_sum_delta_mean": _safe_mean(
                [_to_float(r.get("preempt_sum_delta")) for r in grp]),
            "preempt_sum_delta_total": _safe_sum(
                [_to_float(r.get("preempt_sum_delta")) for r in grp]),
        }
    return out


def _collect_segment_files(client_run_dir: str) -> List[Tuple[float, str, str]]:
    out: List[Tuple[float, str, str]] = []
    for seg_dir in sorted(glob.glob(os.path.join(client_run_dir, "rc_cycle*"))):
        if not os.path.isdir(seg_dir):
            continue
        lam = _parse_lambda_from_seg_name(os.path.basename(seg_dir))
        out.append((
            lam,
            os.path.join(seg_dir, "client_results.csv"),
            os.path.join(seg_dir, "metrics_timeseries.csv"),
        ))
    return out


def _request_metrics_by_lambda(client_run_dir: str) -> Dict[float, Dict[str, float]]:
    stall_map: DefaultDict[float, List[float]] = defaultdict(list)
    ttft_map: DefaultDict[float, List[float]] = defaultdict(list)
    tpot_req_map: DefaultDict[float, List[float]] = defaultdict(list)

    for lam, client_results_path, _ in _collect_segment_files(client_run_dir):
        if math.isnan(lam) or not os.path.isfile(client_results_path):
            continue
        try:
            rows = _read_csv(client_results_path)
        except Exception:
            continue
        for r in rows:
            ttft = _to_float(r.get("ttft_s"))
            stall = _to_float(r.get("stall_gap_max_s"))
            out_tok = _to_float(r.get("out_tok"))
            t1 = _to_float(r.get("t_first_token_abs"))
            t2 = _to_float(r.get("t_done_abs"))

            if not math.isnan(ttft):
                ttft_map[lam].append(ttft)
            if not math.isnan(stall):
                stall_map[lam].append(stall)

            # Request-level TPOT proxy: average per generated token for each request.
            if (
                (not math.isnan(out_tok))
                and out_tok > 1.0
                and (not math.isnan(t1))
                and (not math.isnan(t2))
                and t2 > t1
            ):
                tpot_req_map[lam].append((t2 - t1) / (out_tok - 1.0))

    out: Dict[float, Dict[str, float]] = {}
    for lam in sorted(set(stall_map) | set(ttft_map) | set(tpot_req_map)):
        stall = stall_map.get(lam, [])
        ttft = ttft_map.get(lam, [])
        tpot_req = tpot_req_map.get(lam, [])
        n = float(len(stall))
        thr200 = [x for x in stall if x >= 0.2]
        thr1000 = [x for x in stall if x >= 1.0]

        out[lam] = {
            "req_count": n,
            "ttft_mean_s_req": _safe_mean(ttft),
            "ttft_p50_s_req": _percentile(ttft, 50),
            "ttft_p90_s_req": _percentile(ttft, 90),
            "ttft_p99_s_req": _percentile(ttft, 99),
            "ttft_p999_s_req": _percentile(ttft, 99.9),
            "ttft_max_s_req": _safe_max(ttft),
            "tpot_p50_s_req": _percentile(tpot_req, 50),
            "tpot_p90_s_req": _percentile(tpot_req, 90),
            "tpot_p99_s_req": _percentile(tpot_req, 99),
            "tpot_p999_s_req": _percentile(tpot_req, 99.9),
            "tpot_max_s_req": _safe_max(tpot_req),
            "stall_gap_p50_s_req": _percentile(stall, 50),
            "stall_gap_p90_s_req": _percentile(stall, 90),
            "stall_gap_p99_s_req": _percentile(stall, 99),
            "stall_gap_max_s_req": _safe_max(stall),
            "long_gap_ge_200ms_count": float(len(thr200)),
            "long_gap_ge_200ms_ratio": (
                float(len(thr200)) / n if n > 0 else float("nan")),
            "long_gap_ge_200ms_total_s_proxy": (
                _safe_sum(thr200) if thr200 else 0.0),
            "long_gap_ge_200ms_excess_s_proxy": (
                _safe_sum([x - 0.2 for x in thr200]) if thr200 else 0.0),
            "long_gap_ge_1s_count": float(len(thr1000)),
            "long_gap_ge_1s_ratio": (
                float(len(thr1000)) / n if n > 0 else float("nan")),
            "long_gap_ge_1s_total_s_proxy": (
                _safe_sum(thr1000) if thr1000 else 0.0),
            "long_gap_ge_1s_excess_s_proxy": (
                _safe_sum([x - 1.0 for x in thr1000]) if thr1000 else 0.0),
        }
    return out


def _preempt_window_by_lambda(client_run_dir: str, window_s: float) -> Dict[float, Dict[str, float]]:
    rates_map: DefaultDict[float, List[float]] = defaultdict(list)

    for lam, _, metrics_ts_path in _collect_segment_files(client_run_dir):
        if math.isnan(lam) or not os.path.isfile(metrics_ts_path):
            continue
        try:
            rows = _read_csv(metrics_ts_path)
        except Exception:
            continue
        t_rel: List[float] = []
        deltas: List[float] = []
        prev_preempt_total = float("nan")
        for r in rows:
            t = _to_float(r.get("t_rel_s"))
            d = _to_float(r.get("on_preempt_count_delta"), float("nan"))
            if math.isnan(d):
                cur = _to_float(r.get("preempt_total"), float("nan"))
                if (not math.isnan(prev_preempt_total)) and (not math.isnan(cur)):
                    d = max(0.0, cur - prev_preempt_total)
                elif not math.isnan(cur):
                    d = 0.0
                prev_preempt_total = cur
            t_rel.append(t)
            deltas.append(d)
        rates = _windowed_rates(t_rel, deltas, window_s)
        if rates:
            rates_map[lam].extend(rates)

    out: Dict[float, Dict[str, float]] = {}
    for lam, rates in rates_map.items():
        out[lam] = {
            "preempt_rate_win_p50": _percentile(rates, 50),
            "preempt_rate_win_p90": _percentile(rates, 90),
            "preempt_rate_win_p99": _percentile(rates, 99),
            "preempt_rate_win_max": _safe_max(rates),
        }
    return out


def _event_metrics(events: List[Dict[str, Any]]) -> Dict[str, float]:
    preempt_by_req: DefaultDict[str, List[int]] = defaultdict(list)
    swapin_by_req: DefaultDict[str, List[int]] = defaultdict(list)
    commits_by_req: DefaultDict[str, List[int]] = defaultdict(list)
    normals_by_req: DefaultDict[str, List[int]] = defaultdict(list)
    switches_by_key: DefaultDict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)

    preempt_global: List[int] = []
    mode_switch_count = 0
    preempt_count = 0

    swapin_events = 0
    swapout_events = 0
    swapin_blocks = 0.0
    swapout_blocks = 0.0
    swapin_bytes = 0.0
    swapout_bytes = 0.0
    mws_pin_events = 0
    mws_pin_blocks: List[float] = []

    total_blocks_snapshot: List[float] = []
    watermark_blocks_snapshot: List[float] = []

    for ev in events:
        et = str(ev.get("event", ""))
        ts = _to_int(ev.get("ts_ns"), 0)
        req = _event_req_key(ev)
        detail = _event_detail(ev)

        if et == "PREEMPT_TRIGGERED":
            preempt_count += 1
            if req and ts > 0:
                preempt_by_req[req].append(ts)
                preempt_global.append(ts)
        elif et == "SWAP_IN":
            swapin_events += 1
            blk = _first_num(detail, ["blocks", "blocks_count"]) if detail else float("nan")
            if not math.isnan(blk):
                swapin_blocks += blk
            b = _first_num(detail, ["bytes"])
            if not math.isnan(b):
                swapin_bytes += b
            if req and ts > 0:
                swapin_by_req[req].append(ts)
        elif et == "SWAP_OUT":
            swapout_events += 1
            blk = _first_num(detail, ["blocks", "blocks_count"]) if detail else float("nan")
            if not math.isnan(blk):
                swapout_blocks += blk
            b = _first_num(detail, ["bytes"])
            if not math.isnan(b):
                swapout_bytes += b
        elif et == "RECOVERY_PROGRESS_COMMIT":
            if req and ts > 0:
                commits_by_req[req].append(ts)
        elif et == "RECOVERY_REQUEST_MODE_SWITCH":
            mode_switch_count += 1
            seq = str(ev.get("seq_id", "")).strip()
            to_mode = str(detail.get("to_mode", "")).strip().upper()
            if req and seq and ts > 0 and to_mode:
                switches_by_key[(req, seq)].append((ts, to_mode))
        elif et == "MODE_ENTER_NORMAL":
            if req and ts > 0:
                normals_by_req[req].append(ts)
        elif et == "MWS_PIN_APPLIED":
            mws_pin_events += 1
            pb = _first_num(detail, ["pinned_blocks"])
            if not math.isnan(pb):
                mws_pin_blocks.append(pb)

        mem = ev.get("mem_snapshot")
        if isinstance(mem, dict):
            tot = _first_num(mem, ["total_gpu_blocks", "gpu_total_blocks"])
            wm = _first_num(mem, ["watermark_blocks", "gpu_watermark_blocks"])
            if not math.isnan(tot):
                total_blocks_snapshot.append(tot)
            if not math.isnan(wm):
                watermark_blocks_snapshot.append(wm)

    for arr in preempt_by_req.values():
        arr.sort()
    for arr in swapin_by_req.values():
        arr.sort()
    for arr in commits_by_req.values():
        arr.sort()
    for arr in normals_by_req.values():
        arr.sort()

    rapid_win_ns = int(3000.0 * 1e6)
    normal_repreempt_win_ns = int(5000.0 * 1e6)
    switch_win_ns = int(2000.0 * 1e6)
    episode_win_ns = int(3000.0 * 1e6)
    cluster_win_ns = int(5000.0 * 1e6)

    # rapid preempt ratio
    pair_cnt = 0
    rapid_cnt = 0
    for ts_list in preempt_by_req.values():
        for i in range(1, len(ts_list)):
            pair_cnt += 1
            if ts_list[i] - ts_list[i - 1] <= rapid_win_ns:
                rapid_cnt += 1
    rapid_preempt_ratio = (rapid_cnt / pair_cnt) if pair_cnt > 0 else float("nan")

    # mode jitter ratio
    switch_pairs = 0
    switch_jitter = 0
    for key in switches_by_key:
        seq_arr = sorted(switches_by_key[key], key=lambda x: x[0])
        for i in range(1, len(seq_arr)):
            switch_pairs += 1
            if seq_arr[i][0] - seq_arr[i - 1][0] <= switch_win_ns:
                switch_jitter += 1
    switch_jitter_ratio = (switch_jitter / switch_pairs) if switch_pairs > 0 else float("nan")

    # mode flipflop ratio (A->B->A in short window)
    flip_denom = 0
    flip_cnt = 0
    for key in switches_by_key:
        seq_arr = sorted(switches_by_key[key], key=lambda x: x[0])
        if len(seq_arr) < 3:
            continue
        for i in range(2, len(seq_arr)):
            flip_denom += 1
            t0, m0 = seq_arr[i - 2]
            _, m1 = seq_arr[i - 1]
            t2, m2 = seq_arr[i]
            if (m0 == m2) and (m0 != m1) and (t2 - t0 <= switch_win_ns):
                flip_cnt += 1
    switch_flipflop_ratio = (flip_cnt / flip_denom) if flip_denom > 0 else float("nan")

    # normal -> preempt short window ratio
    n_total = 0
    n_hits = 0
    for req, nt in normals_by_req.items():
        pt = preempt_by_req.get(req, [])
        j = 0
        for t0 in nt:
            n_total += 1
            while j < len(pt) and pt[j] <= t0:
                j += 1
            if j < len(pt) and (pt[j] - t0 <= normal_repreempt_win_ns):
                n_hits += 1
    normal_to_preempt_ratio = (n_hits / n_total) if n_total > 0 else float("nan")

    # preempt cluster ratio in short windows
    preempt_global.sort()
    cluster_sizes: List[int] = []
    if preempt_global:
        cur = 1
        for i in range(1, len(preempt_global)):
            if preempt_global[i] - preempt_global[i - 1] <= cluster_win_ns:
                cur += 1
            else:
                cluster_sizes.append(cur)
                cur = 1
        cluster_sizes.append(cur)
    clustered_preempts = sum(c for c in cluster_sizes if c >= 2)
    clustered_preempt_ratio = (
        clustered_preempts / len(preempt_global) if preempt_global else float("nan"))

    # preempt->restore episodes in short windows
    episode_hits = 0
    if preempt_count > 0:
        for req, pt in preempt_by_req.items():
            it = swapin_by_req.get(req, [])
            if not it:
                continue
            j = 0
            for t0 in pt:
                while j < len(it) and it[j] < t0:
                    j += 1
                if j < len(it) and (it[j] - t0 <= episode_win_ns):
                    episode_hits += 1
    preempt_restore_episode_ratio = (
        episode_hits / preempt_count if preempt_count > 0 else float("nan"))

    # swap-in gap continuity
    gap_all_ms: List[float] = []
    for req, ts_list in swapin_by_req.items():
        for i in range(1, len(ts_list)):
            gap_all_ms.append((ts_list[i] - ts_list[i - 1]) / 1e6)

    gap_active_ms: List[float] = []
    for req, pre_list in preempt_by_req.items():
        p = sorted([x for x in pre_list if x > 0])
        if not p:
            continue
        ins = sorted([x for x in swapin_by_req.get(req, []) if x > 0])
        if not ins:
            continue
        for i, start in enumerate(p):
            end = p[i + 1] if i + 1 < len(p) else None
            ep = [t for t in ins if t >= start and (end is None or t < end)]
            if not ep:
                continue
            gap_active_ms.append((ep[0] - start) / 1e6)
            for j in range(1, len(ep)):
                gap_active_ms.append((ep[j] - ep[j - 1]) / 1e6)

    # progress commit intervals
    commit_intervals_ms: List[float] = []
    for req, ts_list in commits_by_req.items():
        for i in range(1, len(ts_list)):
            commit_intervals_ms.append((ts_list[i] - ts_list[i - 1]) / 1e6)

    switches_per_preempt = mode_switch_count / max(1, preempt_count)
    total_blocks = _percentile(total_blocks_snapshot, 50)
    watermark_blocks = _percentile(watermark_blocks_snapshot, 50)

    return {
        "events_total": float(len(events)),
        "preempt_events": float(preempt_count),
        "swapin_events": float(swapin_events),
        "swapout_events": float(swapout_events),
        "swapin_blocks_events": swapin_blocks,
        "swapout_blocks_events": swapout_blocks,
        "swapin_bytes_events": swapin_bytes if swapin_bytes > 0 else float("nan"),
        "swapout_bytes_events": swapout_bytes if swapout_bytes > 0 else float("nan"),
        "rapid_preempt_ratio_3s": rapid_preempt_ratio,
        "normal_to_preempt_ratio_5s": normal_to_preempt_ratio,
        "switch_jitter_ratio_2s": switch_jitter_ratio,
        "switch_flipflop_ratio_2s": switch_flipflop_ratio,
        "switches_per_preempt": switches_per_preempt,
        "clustered_preempt_ratio_5s": clustered_preempt_ratio,
        "preempt_restore_episode_ratio_3s": preempt_restore_episode_ratio,
        "swapin_gap_p99_ms": _percentile(gap_all_ms, 99),
        "swapin_gap_max_ms": _safe_max(gap_all_ms),
        "swapin_gap_active_p99_ms": _percentile(gap_active_ms, 99),
        "swapin_gap_active_max_ms": _safe_max(gap_active_ms),
        "progress_commit_interval_p99_ms": _percentile(commit_intervals_ms, 99),
        "progress_commit_interval_max_ms": _safe_max(commit_intervals_ms),
        "mws_pin_events": float(mws_pin_events),
        "mws_pin_blocks_p50": _percentile(mws_pin_blocks, 50),
        "mws_pin_blocks_p90": _percentile(mws_pin_blocks, 90),
        "kv_total_blocks_from_events": total_blocks,
        "kv_watermark_blocks_from_events": watermark_blocks,
    }


def _ts_metrics(ts_rows: List[Dict[str, str]], ev_metrics: Dict[str, float]) -> Dict[str, float]:
    if not ts_rows:
        return {}

    t_rel = [_to_float(r.get("t_rel_s"), float("nan")) for r in ts_rows]
    if all(math.isnan(x) for x in t_rel):
        ts_ns = [_to_float(r.get("ts_ns"), float("nan")) for r in ts_rows]
        base = _safe_min(ts_ns)
        if not math.isnan(base):
            t_rel = [(x - base) / 1e9 if not math.isnan(x) else float("nan") for x in ts_ns]

    pre_delta = [_to_float(r.get("on_preempt_count_delta"), float("nan")) for r in ts_rows]
    pre_window_rates = _windowed_rates(t_rel, pre_delta, 5.0)

    swapin_blocks = [_to_float(r.get("swapin_blocks"), float("nan")) for r in ts_rows]
    swapout_blocks = [_to_float(r.get("swapout_blocks"), float("nan")) for r in ts_rows]
    recompute_tokens = [_to_float(r.get("recompute_tokens"), float("nan")) for r in ts_rows]
    kv_free_blocks = [_to_float(r.get("gpu_kv_blocks_free"), float("nan")) for r in ts_rows]

    cycle_wall_ms = [_to_float(r.get("cycle_wall_ms"), float("nan")) for r in ts_rows]
    ton_ms = [_to_float(r.get("T_on_ms"), float("nan")) for r in ts_rows]
    s_ms = [_to_float(r.get("S_ms"), float("nan")) for r in ts_rows]
    b_ms = [_to_float(r.get("B_ms"), float("nan")) for r in ts_rows]

    sched_overhead_ms: List[float] = []
    for cw, to in zip(cycle_wall_ms, ton_ms):
        if not math.isnan(cw) and not math.isnan(to):
            sched_overhead_ms.append(max(0.0, cw - to))

    t_min = _safe_min(t_rel)
    t_max = _safe_max(t_rel)
    duration_s = (t_max - t_min) if (not math.isnan(t_min) and not math.isnan(t_max)) else float("nan")
    if not math.isnan(duration_s):
        duration_s = max(duration_s, 1e-9)

    swapin_sum = _safe_sum(swapin_blocks)
    swapout_sum = _safe_sum(swapout_blocks)
    recompute_sum = _safe_sum(recompute_tokens)
    swap_total = 0.0
    if not math.isnan(swapin_sum):
        swap_total += swapin_sum
    if not math.isnan(swapout_sum):
        swap_total += swapout_sum
    if swap_total <= 0:
        swap_total = float("nan")

    watermark = _to_float(ev_metrics.get("kv_watermark_blocks_from_events"), float("nan"))
    low_headroom_ratio = float("nan")
    if not math.isnan(watermark):
        valid = [x for x in kv_free_blocks if not math.isnan(x)]
        if valid:
            low_headroom_ratio = sum(1 for x in valid if x <= watermark) / float(len(valid))

    total_blocks = _to_float(ev_metrics.get("kv_total_blocks_from_events"), float("nan"))
    kv_used_ratio: List[float] = []
    if not math.isnan(total_blocks) and total_blocks > 0:
        for f in kv_free_blocks:
            if math.isnan(f):
                continue
            kv_used_ratio.append(max(0.0, min(1.0, 1.0 - (f / total_blocks))))

    preempt_cnt = _to_float(ev_metrics.get("preempt_events"), float("nan"))
    swap_blocks_per_preempt = float("nan")
    recompute_per_preempt = float("nan")
    if not math.isnan(preempt_cnt) and preempt_cnt > 0:
        if not math.isnan(swap_total):
            swap_blocks_per_preempt = swap_total / preempt_cnt
        if not math.isnan(recompute_sum):
            recompute_per_preempt = recompute_sum / preempt_cnt

    swap_blocks_per_s = float("nan")
    if (not math.isnan(duration_s)) and duration_s > 0 and (not math.isnan(swap_total)):
        swap_blocks_per_s = swap_total / duration_s

    bytes_in = _to_float(ev_metrics.get("swapin_bytes_events"), float("nan"))
    bytes_out = _to_float(ev_metrics.get("swapout_bytes_events"), float("nan"))
    bytes_total = 0.0
    has_bytes = False
    if not math.isnan(bytes_in):
        bytes_total += bytes_in
        has_bytes = True
    if not math.isnan(bytes_out):
        bytes_total += bytes_out
        has_bytes = True
    swap_bytes_per_s = float("nan")
    if has_bytes and (not math.isnan(duration_s)) and duration_s > 0:
        swap_bytes_per_s = bytes_total / duration_s

    return {
        "recovery_ts_rows": float(len(ts_rows)),
        "duration_s": duration_s,
        "preempt_rate_win5s_p50": _percentile(pre_window_rates, 50),
        "preempt_rate_win5s_p90": _percentile(pre_window_rates, 90),
        "preempt_rate_win5s_p99": _percentile(pre_window_rates, 99),
        "preempt_rate_win5s_max": _safe_max(pre_window_rates),
        "swapin_blocks_total_ts": swapin_sum,
        "swapout_blocks_total_ts": swapout_sum,
        "recompute_tokens_total_ts": recompute_sum,
        "swap_blocks_total_ts": swap_total,
        "swap_blocks_per_s": swap_blocks_per_s,
        "swap_bytes_per_s_events": swap_bytes_per_s,
        "swap_blocks_per_preempt": swap_blocks_per_preempt,
        "recompute_tokens_per_preempt": recompute_per_preempt,
        "gpu_kv_free_blocks_p50": _percentile(kv_free_blocks, 50),
        "gpu_kv_free_blocks_p90": _percentile(kv_free_blocks, 90),
        "gpu_kv_free_blocks_p99": _percentile(kv_free_blocks, 99),
        "gpu_kv_free_blocks_min": _safe_min(kv_free_blocks),
        "gpu_kv_low_headroom_ratio": low_headroom_ratio,
        "gpu_kv_used_ratio_p50": _percentile(kv_used_ratio, 50),
        "gpu_kv_used_ratio_p90": _percentile(kv_used_ratio, 90),
        "gpu_kv_used_ratio_p99": _percentile(kv_used_ratio, 99),
        "gpu_kv_used_ratio_max": _safe_max(kv_used_ratio),
        "cycle_wall_ms_p50": _percentile(cycle_wall_ms, 50),
        "cycle_wall_ms_p90": _percentile(cycle_wall_ms, 90),
        "cycle_wall_ms_p99": _percentile(cycle_wall_ms, 99),
        "cycle_wall_ms_max": _safe_max(cycle_wall_ms),
        "T_on_ms_p50": _percentile(ton_ms, 50),
        "T_on_ms_p90": _percentile(ton_ms, 90),
        "T_on_ms_p99": _percentile(ton_ms, 99),
        "S_ms_p50": _percentile(s_ms, 50),
        "S_ms_p90": _percentile(s_ms, 90),
        "S_ms_p99": _percentile(s_ms, 99),
        "B_ms_p50": _percentile(b_ms, 50),
        "B_ms_p90": _percentile(b_ms, 90),
        "B_ms_p99": _percentile(b_ms, 99),
        "scheduler_overhead_ms_p50": _percentile(sched_overhead_ms, 50),
        "scheduler_overhead_ms_p90": _percentile(sched_overhead_ms, 90),
        "scheduler_overhead_ms_p99": _percentile(sched_overhead_ms, 99),
        "scheduler_overhead_ms_max": _safe_max(sched_overhead_ms),
    }


def _build_method(
    name: str,
    summary_path: str,
    explicit_server_log: Optional[str],
    preempt_window_s: float,
) -> MethodData:
    summary_abs = os.path.abspath(summary_path)
    client_run_dir = os.path.dirname(summary_abs)
    summary_rows = _read_csv(summary_abs)
    recovery_dir = _infer_recovery_dir(summary_abs, explicit_server_log)

    summary_by_lambda = _aggregate_summary_by_lambda(summary_rows)
    req_by_lambda = _request_metrics_by_lambda(client_run_dir)
    preempt_win_by_lambda = _preempt_window_by_lambda(client_run_dir, preempt_window_s)

    ts_rows: List[Dict[str, str]] = []
    events: List[Dict[str, Any]] = []
    if recovery_dir:
        ts_path = os.path.join(recovery_dir, "recovery_ts.csv")
        ev_path = os.path.join(recovery_dir, "recovery_events.jsonl")
        if os.path.isfile(ts_path):
            try:
                ts_rows = _read_csv(ts_path)
            except Exception:
                ts_rows = []
        if os.path.isfile(ev_path):
            try:
                events = _read_jsonl(ev_path)
            except Exception:
                events = []

    ev_metrics = _event_metrics(events)
    ts_metrics = _ts_metrics(ts_rows, ev_metrics)
    return MethodData(
        name=name,
        summary_path=summary_abs,
        summary_paths=[summary_abs],
        client_run_dir=client_run_dir,
        recovery_dir=recovery_dir,
        summary_rows=summary_rows,
        summary_by_lambda=summary_by_lambda,
        request_by_lambda=req_by_lambda,
        preempt_window_by_lambda=preempt_win_by_lambda,
        ts_metrics=ts_metrics,
        event_metrics=ev_metrics,
    )


def _build_method_from_summaries(
    name: str,
    summary_paths: List[str],
    explicit_server_log: Optional[str],
    preempt_window_s: float,
    lambda_tol: float,
) -> MethodData:
    if not summary_paths:
        raise FileNotFoundError(f"{name}: no summaries to build method data")

    # Keep newest-first behavior from caller.
    parts = [
        _build_method(name, p, explicit_server_log, preempt_window_s)
        for p in summary_paths
    ]
    newest = parts[0]

    merged_summary: Dict[float, Dict[str, float]] = {}
    merged_req: Dict[float, Dict[str, float]] = {}
    merged_preempt: Dict[float, Dict[str, float]] = {}

    def merge_map(
        dst: Dict[float, Dict[str, float]],
        src: Dict[float, Dict[str, float]],
    ) -> None:
        for lam, val in src.items():
            if any(abs(lam - k) <= lambda_tol for k in dst.keys()):
                continue
            dst[lam] = val

    for m in parts:
        merge_map(merged_summary, m.summary_by_lambda)
        merge_map(merged_req, m.request_by_lambda)
        merge_map(merged_preempt, m.preempt_window_by_lambda)

    return MethodData(
        name=name,
        summary_path=newest.summary_path,
        summary_paths=[m.summary_path for m in parts],
        client_run_dir=newest.client_run_dir,
        recovery_dir=newest.recovery_dir,
        summary_rows=newest.summary_rows,
        summary_by_lambda=merged_summary,
        request_by_lambda=merged_req,
        preempt_window_by_lambda=merged_preempt,
        ts_metrics=newest.ts_metrics,
        event_metrics=newest.event_metrics,
    )


def _print_responsiveness(
    methods: List[MethodData],
    target_lambda: Optional[float],
    lambda_tol: float,
) -> None:
    methods = _ordered_methods(methods)
    all_lambdas = sorted({lam for m in methods for lam in m.summary_by_lambda})
    lambdas = _select_lambdas(all_lambdas, target_lambda, lambda_tol, "summary")
    print("\n=== Responsiveness and Pacing (summary.csv, mean over segments) ===")
    print(",".join([
        "lambda", "method",
        "ttft_p50_s", "ttft_p90_s", "ttft_p99_s", "ttft_p999_s", "ttft_max_s",
        "tpot_p50_s", "tpot_p90_s", "tpot_p99_s", "tpot_p999_s", "tpot_max_s",
        "decode_tps_wall", "ok_rate", "segments",
    ]))
    for lam in lambdas:
        for m in methods:
            d = _dict_for_lambda(m.summary_by_lambda, lam, lambda_tol)
            print(",".join([
                _fmt(lam, 2), m.name,
                _fmt(d.get("ttft_p50_s", float("nan")), 4),
                _fmt(d.get("ttft_p90_s", float("nan")), 4),
                _fmt(d.get("ttft_p99_s", float("nan")), 4),
                _fmt(d.get("ttft_p999_s", float("nan")), 4),
                _fmt(d.get("ttft_max_s", float("nan")), 4),
                _fmt(d.get("tpot_p50_s", float("nan")), 4),
                _fmt(d.get("tpot_p90_s", float("nan")), 4),
                _fmt(d.get("tpot_p99_s", float("nan")), 4),
                _fmt(d.get("tpot_p999_s", float("nan")), 4),
                _fmt(d.get("tpot_max_s", float("nan")), 4),
                _fmt(d.get("decode_tps_wall", float("nan")), 3),
                _fmt(d.get("ok_rate", float("nan")), 3),
                _fmt(d.get("segments", float("nan")), 0),
            ]))


def _print_streaming(
    methods: List[MethodData],
    target_lambda: Optional[float],
    lambda_tol: float,
) -> None:
    methods = _ordered_methods(methods)
    all_lambdas = sorted({lam for m in methods for lam in m.request_by_lambda})
    lambdas = _select_lambdas(all_lambdas, target_lambda, lambda_tol, "request")
    print("\n=== Streaming Stability (client_results stall_gap_max_s) ===")
    print(",".join([
        "lambda", "method",
        "stall_gap_p50_s", "stall_gap_p90_s", "stall_gap_p99_s", "stall_gap_max_s",
        "long_gap_ge_200ms_count", "long_gap_ge_200ms_ratio",
        "long_gap_ge_200ms_total_s_proxy",
        "long_gap_ge_1s_count", "long_gap_ge_1s_ratio",
        "long_gap_ge_1s_total_s_proxy",
        "req_count",
    ]))
    for lam in lambdas:
        for m in methods:
            d = _dict_for_lambda(m.request_by_lambda, lam, lambda_tol)
            print(",".join([
                _fmt(lam, 2), m.name,
                _fmt(d.get("stall_gap_p50_s_req", float("nan")), 4),
                _fmt(d.get("stall_gap_p90_s_req", float("nan")), 4),
                _fmt(d.get("stall_gap_p99_s_req", float("nan")), 4),
                _fmt(d.get("stall_gap_max_s_req", float("nan")), 4),
                _fmt(d.get("long_gap_ge_200ms_count", float("nan")), 0),
                _fmt(d.get("long_gap_ge_200ms_ratio", float("nan")), 3),
                _fmt(d.get("long_gap_ge_200ms_total_s_proxy", float("nan")), 3),
                _fmt(d.get("long_gap_ge_1s_count", float("nan")), 0),
                _fmt(d.get("long_gap_ge_1s_ratio", float("nan")), 3),
                _fmt(d.get("long_gap_ge_1s_total_s_proxy", float("nan")), 3),
                _fmt(d.get("req_count", float("nan")), 0),
            ]))
    print(
        "NOTE,long-gap total time is computed as per-request max-gap proxy "
        "(sum of stall_gap_max_s above threshold)."
    )


def _print_preempt_recovery(methods: List[MethodData]) -> None:
    methods = _ordered_methods(methods)
    print("\n=== Preemption, Recovery, and Oscillation (server recovery logs) ===")
    print(",".join([
        "method",
        "preempt_events",
        "preempt_rate_win5s_p99",
        "preempt_rate_win5s_max",
        "rapid_preempt_ratio_3s",
        "normal_to_preempt_ratio_5s",
        "switch_jitter_ratio_2s",
        "switch_flipflop_ratio_2s",
        "switches_per_preempt",
        "clustered_preempt_ratio_5s",
        "preempt_restore_episode_ratio_3s",
        "swapin_gap_active_p99_ms",
        "swapin_gap_active_max_ms",
        "progress_commit_interval_p99_ms",
        "progress_commit_interval_max_ms",
        "swap_blocks_total_ts",
        "recompute_tokens_total_ts",
        "swap_blocks_per_preempt",
        "recompute_tokens_per_preempt",
    ]))
    for m in methods:
        e = m.event_metrics
        t = m.ts_metrics
        print(",".join([
            m.name,
            _fmt(e.get("preempt_events", float("nan")), 0),
            _fmt(t.get("preempt_rate_win5s_p99", float("nan")), 4),
            _fmt(t.get("preempt_rate_win5s_max", float("nan")), 4),
            _fmt(e.get("rapid_preempt_ratio_3s", float("nan")), 3),
            _fmt(e.get("normal_to_preempt_ratio_5s", float("nan")), 3),
            _fmt(e.get("switch_jitter_ratio_2s", float("nan")), 3),
            _fmt(e.get("switch_flipflop_ratio_2s", float("nan")), 3),
            _fmt(e.get("switches_per_preempt", float("nan")), 3),
            _fmt(e.get("clustered_preempt_ratio_5s", float("nan")), 3),
            _fmt(e.get("preempt_restore_episode_ratio_3s", float("nan")), 3),
            _fmt(e.get("swapin_gap_active_p99_ms", float("nan")), 3),
            _fmt(e.get("swapin_gap_active_max_ms", float("nan")), 3),
            _fmt(e.get("progress_commit_interval_p99_ms", float("nan")), 3),
            _fmt(e.get("progress_commit_interval_max_ms", float("nan")), 3),
            _fmt(t.get("swap_blocks_total_ts", float("nan")), 1),
            _fmt(t.get("recompute_tokens_total_ts", float("nan")), 1),
            _fmt(t.get("swap_blocks_per_preempt", float("nan")), 3),
            _fmt(t.get("recompute_tokens_per_preempt", float("nan")), 3),
        ]))


def _print_resource_overhead(methods: List[MethodData]) -> None:
    methods = _ordered_methods(methods)
    print("\n=== Resource Utilization and Scheduler Overhead ===")
    print(",".join([
        "method",
        "gpu_kv_free_blocks_p50",
        "gpu_kv_free_blocks_p90",
        "gpu_kv_free_blocks_p99",
        "gpu_kv_free_blocks_min",
        "gpu_kv_low_headroom_ratio",
        "gpu_kv_used_ratio_p50",
        "gpu_kv_used_ratio_p90",
        "gpu_kv_used_ratio_p99",
        "gpu_kv_used_ratio_max",
        "swap_blocks_per_s",
        "swap_bytes_per_s_events",
        "cycle_wall_ms_p50",
        "cycle_wall_ms_p90",
        "cycle_wall_ms_p99",
        "cycle_wall_ms_max",
        "scheduler_overhead_ms_p99",
        "T_on_ms_p99",
        "S_ms_p99",
        "B_ms_p99",
    ]))
    for m in methods:
        t = m.ts_metrics
        print(",".join([
            m.name,
            _fmt(t.get("gpu_kv_free_blocks_p50", float("nan")), 2),
            _fmt(t.get("gpu_kv_free_blocks_p90", float("nan")), 2),
            _fmt(t.get("gpu_kv_free_blocks_p99", float("nan")), 2),
            _fmt(t.get("gpu_kv_free_blocks_min", float("nan")), 2),
            _fmt(t.get("gpu_kv_low_headroom_ratio", float("nan")), 3),
            _fmt(t.get("gpu_kv_used_ratio_p50", float("nan")), 3),
            _fmt(t.get("gpu_kv_used_ratio_p90", float("nan")), 3),
            _fmt(t.get("gpu_kv_used_ratio_p99", float("nan")), 3),
            _fmt(t.get("gpu_kv_used_ratio_max", float("nan")), 3),
            _fmt(t.get("swap_blocks_per_s", float("nan")), 3),
            _fmt(t.get("swap_bytes_per_s_events", float("nan")), 3),
            _fmt(t.get("cycle_wall_ms_p50", float("nan")), 3),
            _fmt(t.get("cycle_wall_ms_p90", float("nan")), 3),
            _fmt(t.get("cycle_wall_ms_p99", float("nan")), 3),
            _fmt(t.get("cycle_wall_ms_max", float("nan")), 3),
            _fmt(t.get("scheduler_overhead_ms_p99", float("nan")), 3),
            _fmt(t.get("T_on_ms_p99", float("nan")), 3),
            _fmt(t.get("S_ms_p99", float("nan")), 3),
            _fmt(t.get("B_ms_p99", float("nan")), 3),
        ]))


def _print_relative_change(
    methods: List[MethodData],
    target_lambda: Optional[float],
    lambda_tol: float,
) -> None:
    methods = _ordered_methods(methods)
    if len(methods) < 2:
        return
    base = _baseline_method(methods)
    if base is None:
        return
    all_lambdas = sorted({lam for m in methods for lam in m.summary_by_lambda})
    lambdas = _select_lambdas(all_lambdas, target_lambda, lambda_tol, "summary")
    print(f"\n=== Relative Change vs {base.name} (%, by lambda) ===")
    print(",".join([
        "lambda", "method",
        "ttft_p99_change_pct",
        "tpot_p99_change_pct",
        "stall_gap_p99_change_pct",
        "stall_gap_max_change_pct",
        "decode_tps_wall_change_pct",
        "long_gap_ge_1s_ratio_change_pct",
        "preempt_rate_win_p99_change_pct",
    ]))

    for lam in lambdas:
        bsum = _dict_for_lambda(base.summary_by_lambda, lam, lambda_tol)
        breq = _dict_for_lambda(base.request_by_lambda, lam, lambda_tol)
        bwin = _dict_for_lambda(base.preempt_window_by_lambda, lam, lambda_tol)
        for m in [x for x in methods if x.name != base.name]:
            dsum = _dict_for_lambda(m.summary_by_lambda, lam, lambda_tol)
            dreq = _dict_for_lambda(m.request_by_lambda, lam, lambda_tol)
            dwin = _dict_for_lambda(m.preempt_window_by_lambda, lam, lambda_tol)
            print(",".join([
                _fmt(lam, 2),
                m.name,
                _fmt(_pct_change(
                    dsum.get("ttft_p99_s", float("nan")),
                    bsum.get("ttft_p99_s", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    dsum.get("tpot_p99_s", float("nan")),
                    bsum.get("tpot_p99_s", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    dsum.get("stall_gap_p99_s", float("nan")),
                    bsum.get("stall_gap_p99_s", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    dsum.get("stall_gap_max_s", float("nan")),
                    bsum.get("stall_gap_max_s", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    dsum.get("decode_tps_wall", float("nan")),
                    bsum.get("decode_tps_wall", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    dreq.get("long_gap_ge_1s_ratio", float("nan")),
                    breq.get("long_gap_ge_1s_ratio", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    dwin.get("preempt_rate_win_p99", float("nan")),
                    bwin.get("preempt_rate_win_p99", float("nan")),
                ), 2),
            ]))


def _print_key_metrics_by_lambda(
    methods: List[MethodData],
    target_lambdas: List[float],
    lambda_tol: float,
    gap_metric_mode: str,
) -> None:
    methods = _ordered_methods(methods)
    if not target_lambdas:
        return

    baseline = _baseline_method(methods)
    gap_title, _, gap_key = _gap_metric_spec(gap_metric_mode)
    for lam in target_lambdas:
        print(f"\n=== Key Metrics Comparison @ lambda={lam:.2f} ===")
        print(",".join([
            "method",
            "effective_throughput_tok_s",
            "p99_ttft_s",
            "p99_tpot_s",
            gap_key,
        ]))
        for m in methods:
            s = _dict_for_lambda(m.summary_by_lambda, lam, lambda_tol)
            r = _dict_for_lambda(m.request_by_lambda, lam, lambda_tol)
            decode_tps = _to_float(s.get("decode_tps_wall", float("nan")), float("nan"))
            ok_rate = _to_float(s.get("ok_rate", float("nan")), float("nan"))
            eff_tps = (
                decode_tps * ok_rate
                if (not math.isnan(decode_tps) and not math.isnan(ok_rate))
                else float("nan")
            )
            print(",".join([
                m.name,
                _fmt(eff_tps, 3),
                _fmt(s.get("ttft_p99_s", float("nan")), 4),
                _fmt(s.get("tpot_p99_s", float("nan")), 4),
                _fmt(r.get(gap_key, float("nan")), 4 if gap_metric_mode == "stall_max" else 3),
            ]))

        if not baseline:
            continue
        bs = _dict_for_lambda(baseline.summary_by_lambda, lam, lambda_tol)
        br = _dict_for_lambda(baseline.request_by_lambda, lam, lambda_tol)
        b_decode_tps = _to_float(bs.get("decode_tps_wall", float("nan")), float("nan"))
        b_ok_rate = _to_float(bs.get("ok_rate", float("nan")), float("nan"))
        b_eff_tps = (
            b_decode_tps * b_ok_rate
            if (not math.isnan(b_decode_tps) and not math.isnan(b_ok_rate))
            else float("nan")
        )
        print(f"--- Relative Change vs {baseline.name} @ lambda={lam:.2f} (%) ---")
        print(",".join([
            "method",
            "effective_throughput_change_pct",
            "ttft_p99_change_pct",
            "tpot_p99_change_pct",
            "gap_metric_change_pct",
        ]))
        for m in [x for x in methods if baseline is not None and x.name != baseline.name]:
            s = _dict_for_lambda(m.summary_by_lambda, lam, lambda_tol)
            r = _dict_for_lambda(m.request_by_lambda, lam, lambda_tol)
            d_decode_tps = _to_float(s.get("decode_tps_wall", float("nan")), float("nan"))
            d_ok_rate = _to_float(s.get("ok_rate", float("nan")), float("nan"))
            d_eff_tps = (
                d_decode_tps * d_ok_rate
                if (not math.isnan(d_decode_tps) and not math.isnan(d_ok_rate))
                else float("nan")
            )
            print(",".join([
                m.name,
                _fmt(_pct_change(
                    d_eff_tps,
                    b_eff_tps,
                ), 2),
                _fmt(_pct_change(
                    s.get("ttft_p99_s", float("nan")),
                    bs.get("ttft_p99_s", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    s.get("tpot_p99_s", float("nan")),
                    bs.get("tpot_p99_s", float("nan")),
                ), 2),
                _fmt(_pct_change(
                    r.get(gap_key, float("nan")),
                    br.get(gap_key, float("nan")),
                ), 2),
            ]))


def _method_to_dict(m: MethodData) -> Dict[str, Any]:
    return {
        "name": m.name,
        "summary_path": m.summary_path,
        "summary_paths": m.summary_paths,
        "client_run_dir": m.client_run_dir,
        "recovery_dir": m.recovery_dir,
        "summary_by_lambda": m.summary_by_lambda,
        "request_by_lambda": m.request_by_lambda,
        "preempt_window_by_lambda": m.preempt_window_by_lambda,
        "event_metrics": m.event_metrics,
        "ts_metrics": m.ts_metrics,
    }


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _method_color(method_name: str) -> str:
    color_map = {
        "vllm": "#1f77b4",
        "RecoverGen": "#d62728",
        "vllm (no_chunk)": "#2ca02c",
        "RecoverGen (no_chunk)": "#ff7f0e",
    }
    return color_map.get(method_name, "#4c4c4c")


def _write_svg_grouped_bars(
    title: str,
    categories: List[str],
    methods: List[str],
    values_by_method: Dict[str, List[float]],
    y_label: str,
    out_path: str,
) -> bool:
    if not categories or not methods:
        return False

    all_vals: List[float] = []
    for m in methods:
        arr = values_by_method.get(m, [])
        all_vals.extend([v for v in arr if not math.isnan(v) and v >= 0])
    max_val = max(all_vals) if all_vals else 1.0
    if max_val <= 0:
        max_val = 1.0

    width = max(980, 220 + 180 * len(categories))
    height = 560
    left = 90
    right = 30
    top = 90
    bottom = 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    )
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="36" text-anchor="middle" '
        'font-size="22" font-family="sans-serif" fill="#111111">'
        f"{_svg_escape(title)}</text>"
    )
    lines.append(
        f'<text x="{left}" y="{top-16}" text-anchor="start" '
        'font-size="14" font-family="sans-serif" fill="#444444">'
        f"{_svg_escape(y_label)}</text>"
    )

    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        y = top + plot_h * (1.0 - frac)
        v = max_val * frac
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" '
            'stroke="#e5e5e5" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" '
            'font-size="12" font-family="monospace" fill="#555555">'
            f"{_fmt(v, 3)}</text>"
        )

    lines.append(
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" '
        'stroke="#333333" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" '
        'stroke="#333333" stroke-width="1.2"/>'
    )

    group_w = plot_w / float(len(categories))
    inner_w = group_w * 0.72
    bar_w = inner_w / float(max(1, len(methods)))

    for ci, cat in enumerate(categories):
        gx = left + ci * group_w + (group_w - inner_w) / 2.0
        for mi, m in enumerate(methods):
            arr = values_by_method.get(m, [])
            v = arr[ci] if ci < len(arr) else float("nan")
            x = gx + mi * bar_w + bar_w * 0.10
            bw = bar_w * 0.80
            if not math.isnan(v) and v >= 0:
                h = 0.0 if max_val <= 0 else (plot_h * v / max_val)
                y = top + plot_h - h
                lines.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{h:.2f}" '
                    f'fill="{_method_color(m)}" opacity="0.86"/>'
                )
                lines.append(
                    f'<text x="{x+bw/2:.2f}" y="{max(top+10, y-4):.2f}" text-anchor="middle" '
                    'font-size="10" font-family="monospace" fill="#333333">'
                    f"{_fmt(v, 3)}</text>"
                )
            else:
                y0 = top + plot_h - 6
                lines.append(
                    f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x+bw:.2f}" y2="{y0-8:.2f}" '
                    'stroke="#999999" stroke-width="1"/>'
                )
                lines.append(
                    f'<text x="{x+bw/2:.2f}" y="{y0-10:.2f}" text-anchor="middle" '
                    'font-size="9" font-family="monospace" fill="#888888">na</text>'
                )

        lines.append(
            f'<text x="{left + ci*group_w + group_w/2:.2f}" y="{top+plot_h+22:.2f}" '
            'text-anchor="middle" font-size="12" font-family="sans-serif" fill="#444444">'
            f"{_svg_escape(cat)}</text>"
        )

    legend_x = left + 6
    legend_y = top - 52
    for i, m in enumerate(methods):
        y = legend_y + i * 18
        lines.append(
            f'<rect x="{legend_x:.2f}" y="{y:.2f}" width="12" height="12" '
            f'fill="{_method_color(m)}" opacity="0.86"/>'
        )
        lines.append(
            f'<text x="{legend_x+18:.2f}" y="{y+10:.2f}" text-anchor="start" '
            'font-size="12" font-family="sans-serif" fill="#333333">'
            f"{_svg_escape(m)}</text>"
        )

    lines.append("</svg>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def _write_svg_method_bars(
    title: str,
    methods: List[str],
    values: List[float],
    y_label: str,
    out_path: str,
) -> bool:
    data = {m: [v] for m, v in zip(methods, values)}
    return _write_svg_grouped_bars(
        title=title,
        categories=["methods"],
        methods=methods,
        values_by_method=data,
        y_label=y_label,
        out_path=out_path,
    )


def _paper_style_metric_defs(
    lambda_tol: float,
    gap_metric_mode: str,
) -> List[Tuple[str, str, Any]]:
    _, gap_unit, gap_key = _gap_metric_spec(gap_metric_mode)
    return [
        ("Effective Thpt.", "tok/s", lambda md, lam: (
            _to_float(
                _dict_for_lambda(md.summary_by_lambda, lam, lambda_tol)
                .get("decode_tps_wall", float("nan")),
                float("nan"),
            )
            * _to_float(
                _dict_for_lambda(md.summary_by_lambda, lam, lambda_tol)
                .get("ok_rate", float("nan")),
                float("nan"),
            )
        )),
        ("P99 TTFT", "s", lambda md, lam: _to_float(
            _dict_for_lambda(md.summary_by_lambda, lam, lambda_tol)
            .get("ttft_p99_s", float("nan")),
            float("nan"),
        )),
        ("P99 TPOT", "s", lambda md, lam: _to_float(
            _dict_for_lambda(md.summary_by_lambda, lam, lambda_tol)
            .get("tpot_p99_s", float("nan")),
            float("nan"),
        )),
        ("Long-gap Ratio (>=1s)" if gap_metric_mode == "long_gap_ratio" else "Max Stall Gap",
         gap_unit,
         lambda md, lam: _to_float(
            _dict_for_lambda(md.request_by_lambda, lam, lambda_tol)
            .get(gap_key, float("nan")),
            float("nan"),
        )),
    ]


def _generate_paper_style_lambda_grid_svg(
    methods: List[MethodData],
    target_lambdas: List[float],
    lambda_tol: float,
    plot_dir: str,
    gap_metric_mode: str,
) -> List[str]:
    methods = _ordered_methods(methods)
    metric_defs = _paper_style_metric_defs(lambda_tol, gap_metric_mode)
    method_names = [m.name for m in methods]
    color_seq = [_method_color(n) for n in method_names]
    hatch_style = ["diag_fwd", "diag_back", "cross", "star", "cross", "diag_fwd"]

    n_rows = len(target_lambdas)
    n_cols = len(metric_defs)
    cell_w = 180
    cell_h = 130
    gap_x = 18
    gap_y = 18
    left = 96
    top = 70
    right = 20
    bottom = 92
    width = left + n_cols * cell_w + (n_cols - 1) * gap_x + right
    height = top + n_rows * cell_h + (n_rows - 1) * gap_y + bottom

    def _tick_fmt(v: float) -> str:
        if v == 0 or math.isnan(v):
            return "0"
        av = abs(v)
        if av >= 1000 or av < 0.01:
            exp = int(math.floor(math.log10(av)))
            base = v / (10 ** exp)
            return f"{base:.1f}e{exp}"
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append("<defs>")
    for i, col in enumerate(color_seq):
        pid = f"pat_{i}"
        lines.append(f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="8" height="8">')
        lines.append(f'<rect x="0" y="0" width="8" height="8" fill="{col}"/>')
        style = hatch_style[i % len(hatch_style)]
        if style == "diag_fwd":
            lines.append('<path d="M-2,8 L8,-2 M0,10 L10,0" stroke="#111111" stroke-width="1"/>')
        elif style == "diag_back":
            lines.append('<path d="M-2,0 L8,10 M0,-2 L10,8" stroke="#111111" stroke-width="1"/>')
        elif style == "cross":
            lines.append('<path d="M4,0 L4,8 M0,4 L8,4" stroke="#111111" stroke-width="1"/>')
        elif style == "star":
            lines.append('<path d="M4,0 L4,8 M0,4 L8,4 M1,1 L7,7 M7,1 L1,7" stroke="#111111" stroke-width="0.9"/>')
        lines.append("</pattern>")
    lines.append("</defs>")

    lines.append(
        f'<text x="{width/2:.1f}" y="34" text-anchor="middle" '
        'font-size="22" font-family="serif" fill="#111111">'
        "Performance Metrics Comparison by Lambda</text>"
    )

    col_ymax: List[float] = []
    for _, _, metric_fn in metric_defs:
        vals: List[float] = []
        for lam in target_lambdas:
            for md in methods:
                v = metric_fn(md, lam)
                if not math.isnan(v) and v >= 0:
                    vals.append(v)
        max_v = max(vals) if vals else 1.0
        if max_v <= 0:
            max_v = 1.0
        col_ymax.append(max_v)

    for c, (title, _, _) in enumerate(metric_defs):
        x0 = left + c * (cell_w + gap_x)
        lines.append(
            f'<text x="{x0 + cell_w/2:.1f}" y="{top - 14}" text-anchor="middle" '
            'font-size="17" font-family="sans-serif" fill="#111111">'
            f"{_svg_escape(title)}</text>"
        )

    for r, lam in enumerate(target_lambdas):
        row_y = top + r * (cell_h + gap_y)
        lines.append(
            f'<text x="34" y="{row_y + cell_h/2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 34 {row_y + cell_h/2:.1f})" '
            'font-size="17" font-family="sans-serif" fill="#111111">'
            f"lambda={lam:.2f}</text>"
        )
        for c, (_, unit, metric_fn) in enumerate(metric_defs):
            col_x = left + c * (cell_w + gap_x)
            lines.append(
                f'<rect x="{col_x}" y="{row_y}" width="{cell_w}" height="{cell_h}" '
                'fill="#f6f6f6" stroke="#888888" stroke-width="1"/>'
            )
            px = col_x + 36
            py = row_y + 16
            pw = cell_w - 44
            ph = cell_h - 30

            vals = [metric_fn(md, lam) for md in methods]
            ymax = col_ymax[c]

            for ti in range(4):
                frac = ti / 3.0
                yy = py + ph * (1.0 - frac)
                lines.append(
                    f'<line x1="{px}" y1="{yy:.2f}" x2="{px + pw}" y2="{yy:.2f}" '
                    'stroke="#d9d9d9" stroke-width="1"/>'
                )
                tv = ymax * frac
                lines.append(
                    f'<text x="{px - 4}" y="{yy + 4:.2f}" text-anchor="end" '
                    'font-size="9" font-family="monospace" fill="#555555">'
                    f"{_svg_escape(_tick_fmt(tv))}</text>"
                )

            lines.append(
                f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" '
                'stroke="#333333" stroke-width="1.1"/>'
            )
            lines.append(
                f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" '
                'stroke="#333333" stroke-width="1.1"/>'
            )

            slot = pw / max(1, len(methods))
            bw = slot * 0.72
            for i, v in enumerate(vals):
                bx = px + i * slot + (slot - bw) / 2.0
                if math.isnan(v) or v < 0:
                    yb = py + ph - 6
                    lines.append(
                        f'<line x1="{bx:.2f}" y1="{yb:.2f}" x2="{bx + bw:.2f}" y2="{yb - 8:.2f}" '
                        'stroke="#999999" stroke-width="1"/>'
                    )
                    continue
                bh = ph * (v / ymax) if ymax > 0 else 0.0
                by = py + ph - bh
                lines.append(
                    f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" '
                    f'fill="url(#pat_{i})" stroke="#111111" stroke-width="0.8"/>'
                )

            lines.append(
                f'<text x="{col_x + cell_w - 4}" y="{row_y + 12}" text-anchor="end" '
                'font-size="9" font-family="sans-serif" fill="#666666">'
                f"{_svg_escape(unit)}</text>"
            )

    legend_y = height - 52
    step = (width - 120) / max(1, len(method_names))
    for i, name in enumerate(method_names):
        lx = 62 + i * step
        lines.append(
            f'<rect x="{lx:.2f}" y="{legend_y}" width="20" height="12" '
            f'fill="url(#pat_{i})" stroke="#111111" stroke-width="0.8"/>'
        )
        lines.append(
            f'<text x="{lx + 26:.2f}" y="{legend_y + 10}" text-anchor="start" '
            'font-size="16" font-family="sans-serif" fill="#111111">'
            f"{_svg_escape(name)}</text>"
        )

    lines.append("</svg>")
    os.makedirs(plot_dir, exist_ok=True)
    out_svg = os.path.join(plot_dir, "paper_style_lambda_key_metrics_grid.svg")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return [out_svg]


def _generate_paper_style_lambda_grid_plot(
    methods: List[MethodData],
    target_lambdas: List[float],
    lambda_tol: float,
    plot_dir: str,
    gap_metric_mode: str,
) -> List[str]:
    methods = _ordered_methods(methods)
    if not methods or not target_lambdas:
        return []
    metric_defs = _paper_style_metric_defs(lambda_tol, gap_metric_mode)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import numpy as np
    except Exception:
        return _generate_paper_style_lambda_grid_svg(
            methods, target_lambdas, lambda_tol, plot_dir, gap_metric_mode)

    method_names = [m.name for m in methods]
    color_seq = [_method_color(n) for n in method_names]
    hatch_seq = ["//", "\\\\", "++", "**", "xx", ".."]

    n_rows = len(target_lambdas)
    n_cols = len(metric_defs)
    fig_w = max(9.0, 2.55 * n_cols + 0.6)
    fig_h = max(3.5, 1.95 * n_rows + 1.1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("#ffffff")

    col_ymax: List[float] = []
    for _, _, metric_fn in metric_defs:
        vals: List[float] = []
        for lam in target_lambdas:
            for md in methods:
                v = metric_fn(md, lam)
                if not math.isnan(v) and v >= 0:
                    vals.append(v)
        max_v = max(vals) if vals else 1.0
        if max_v <= 0:
            max_v = 1.0
        col_ymax.append(max_v)

    x = np.arange(len(methods))
    for r, lam in enumerate(target_lambdas):
        for c, (title, y_unit, metric_fn) in enumerate(metric_defs):
            ax = axes[r][c]
            vals = [metric_fn(md, lam) for md in methods]
            y_vals = [0.0 if (math.isnan(v) or v < 0) else v for v in vals]

            bars = ax.bar(
                x,
                y_vals,
                width=0.72,
                edgecolor="black",
                linewidth=0.8,
                color=color_seq,
            )
            for i, b in enumerate(bars):
                b.set_hatch(hatch_seq[i % len(hatch_seq)])

            if r == 0:
                ax.set_title(title, fontsize=10.5, pad=5)
            if c == 0:
                ax.set_ylabel(f"lambda={lam:.2f}", fontsize=10)
            else:
                ax.set_ylabel("")

            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=8)
            ax.set_ylim(0.0, col_ymax[c] * 1.02)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
            ax.set_axisbelow(True)
            ax.set_facecolor("#f7f7f7")
            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.8)

            ax.text(
                0.98,
                0.96,
                y_unit,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#555555",
            )

    handles = [
        Patch(
            facecolor=color_seq[i],
            edgecolor="black",
            hatch=hatch_seq[i % len(hatch_seq)],
            label=method_names[i],
        )
        for i in range(len(method_names))
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=True,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=9,
    )
    fig.suptitle(
        "Performance Metrics Comparison by Lambda",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))

    os.makedirs(plot_dir, exist_ok=True)
    out_png = os.path.join(plot_dir, "paper_style_lambda_key_metrics_grid.png")
    out_pdf = os.path.join(plot_dir, "paper_style_lambda_key_metrics_grid.pdf")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def _generate_plots(
    methods: List[MethodData],
    plot_dir: str,
    target_lambda: Optional[float],
    lambda_tol: float,
) -> List[str]:
    methods = _ordered_methods(methods)
    out_paths: List[str] = []
    method_names = [m.name for m in methods]

    def add_grouped_plot(
        file_name: str,
        title: str,
        y_label: str,
        source: str,
        metric_key: str,
    ) -> None:
        if source == "summary":
            all_lams = sorted({lam for m in methods for lam in m.summary_by_lambda})
            getter = lambda md, lam: _dict_for_lambda(
                md.summary_by_lambda, lam, lambda_tol).get(metric_key, float("nan"))
        elif source == "request":
            all_lams = sorted({lam for m in methods for lam in m.request_by_lambda})
            getter = lambda md, lam: _dict_for_lambda(
                md.request_by_lambda, lam, lambda_tol).get(metric_key, float("nan"))
        else:
            all_lams = sorted({lam for m in methods for lam in m.preempt_window_by_lambda})
            getter = lambda md, lam: _dict_for_lambda(
                md.preempt_window_by_lambda, lam, lambda_tol).get(metric_key, float("nan"))

        lams = _select_lambdas(all_lams, target_lambda, lambda_tol, source)
        if not lams:
            return
        categories = [f"{lam:.2f}" for lam in lams]
        values_by_method: Dict[str, List[float]] = {}
        for md in methods:
            values_by_method[md.name] = [_to_float(getter(md, lam)) for lam in lams]

        out = os.path.join(plot_dir, file_name)
        ok = _write_svg_grouped_bars(
            title=title,
            categories=categories,
            methods=method_names,
            values_by_method=values_by_method,
            y_label=y_label,
            out_path=out,
        )
        if ok:
            out_paths.append(out)

    def add_single_lambda_method_plot(
        file_name: str,
        title: str,
        y_label: str,
        source: str,
        metric_key: str,
    ) -> None:
        if target_lambda is None:
            return
        vals: List[float] = []
        for md in methods:
            if source == "summary":
                d = _dict_for_lambda(md.summary_by_lambda, target_lambda, lambda_tol)
            elif source == "request":
                d = _dict_for_lambda(md.request_by_lambda, target_lambda, lambda_tol)
            else:
                d = _dict_for_lambda(md.preempt_window_by_lambda, target_lambda, lambda_tol)
            vals.append(_to_float(d.get(metric_key, float("nan")), float("nan")))
        out = os.path.join(plot_dir, file_name)
        ok = _write_svg_method_bars(
            title=f"{title} @ lambda={target_lambda:.2f}",
            methods=method_names,
            values=vals,
            y_label=y_label,
            out_path=out,
        )
        if ok:
            out_paths.append(out)

    def add_method_plot(file_name: str, title: str, y_label: str, metric_key: str, from_ts: bool) -> None:
        vals: List[float] = []
        for md in methods:
            src = md.ts_metrics if from_ts else md.event_metrics
            vals.append(_to_float(src.get(metric_key), float("nan")))
        out = os.path.join(plot_dir, file_name)
        ok = _write_svg_method_bars(
            title=title,
            methods=method_names,
            values=vals,
            y_label=y_label,
            out_path=out,
        )
        if ok:
            out_paths.append(out)

    add_grouped_plot(
        "responsiveness_ttft_p99_by_lambda.svg",
        "TTFT p99 by Lambda",
        "seconds",
        "summary",
        "ttft_p99_s",
    )
    add_grouped_plot(
        "responsiveness_tpot_p99_by_lambda.svg",
        "TPOT p99 by Lambda",
        "seconds",
        "summary",
        "tpot_p99_s",
    )
    add_grouped_plot(
        "streaming_stall_p99_by_lambda.svg",
        "Stall Gap p99 by Lambda",
        "seconds",
        "request",
        "stall_gap_p99_s_req",
    )
    add_grouped_plot(
        "streaming_stall_max_by_lambda.svg",
        "Stall Gap Max by Lambda",
        "seconds",
        "request",
        "stall_gap_max_s_req",
    )
    add_grouped_plot(
        "streaming_long_gap_ratio_ge_1s_by_lambda.svg",
        "Long Gap (>=1s) Ratio by Lambda",
        "ratio",
        "request",
        "long_gap_ge_1s_ratio",
    )
    add_grouped_plot(
        "preemption_rate_win_p99_by_lambda.svg",
        "Preemption Rate Window p99 by Lambda",
        "events/s",
        "preempt",
        "preempt_rate_win_p99",
    )

    add_single_lambda_method_plot(
        "single_lambda_ttft_p99_by_method.svg",
        "TTFT p99 by Method",
        "seconds",
        "summary",
        "ttft_p99_s",
    )
    add_single_lambda_method_plot(
        "single_lambda_tpot_p99_by_method.svg",
        "TPOT p99 by Method",
        "seconds",
        "summary",
        "tpot_p99_s",
    )
    add_single_lambda_method_plot(
        "single_lambda_stall_p99_by_method.svg",
        "Stall Gap p99 by Method",
        "seconds",
        "request",
        "stall_gap_p99_s_req",
    )
    add_single_lambda_method_plot(
        "single_lambda_long_gap_ratio_ge_1s_by_method.svg",
        "Long Gap (>=1s) Ratio by Method",
        "ratio",
        "request",
        "long_gap_ge_1s_ratio",
    )
    add_single_lambda_method_plot(
        "single_lambda_preempt_rate_win_p99_by_method.svg",
        "Preemption Rate Window p99 by Method",
        "events/s",
        "preempt",
        "preempt_rate_win_p99",
    )

    add_method_plot(
        "preemption_events_by_method.svg",
        "Preemption Events",
        "count",
        "preempt_events",
        from_ts=False,
    )
    add_method_plot(
        "oscillation_switch_flipflop_ratio_by_method.svg",
        "Switch Flipflop Ratio",
        "ratio",
        "switch_flipflop_ratio_2s",
        from_ts=False,
    )
    add_method_plot(
        "recovery_swap_blocks_per_preempt_by_method.svg",
        "Swap Blocks per Preempt",
        "blocks/preempt",
        "swap_blocks_per_preempt",
        from_ts=True,
    )
    add_method_plot(
        "overhead_scheduler_p99_by_method.svg",
        "Scheduler Overhead p99",
        "ms",
        "scheduler_overhead_ms_p99",
        from_ts=True,
    )
    add_method_plot(
        "resource_kv_low_headroom_ratio_by_method.svg",
        "KV Low-Headroom Ratio",
        "ratio",
        "gpu_kv_low_headroom_ratio",
        from_ts=True,
    )
    add_method_plot(
        "resource_swap_blocks_per_s_by_method.svg",
        "Swap Blocks per Second",
        "blocks/s",
        "swap_blocks_per_s",
        from_ts=True,
    )
    return out_paths


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_base_root = os.path.join(repo_root, "logs/RecoveryGen/Baseline/client_logs")
    default_recover_root = os.path.join(repo_root, "logs/RecoveryGen/RecoverGen/client_logs")
    default_nochunk_root = os.path.join(repo_root, "logs/RecoveryGen/Baseline_no_chunked/client_logs")
    default_recover_nochunk_root = os.path.join(
        repo_root, "logs/RecoveryGen/RecoverGen_no_chunked/client_logs"
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-summary", default=None)
    ap.add_argument("--recovergen-summary", default=None)
    ap.add_argument(
        "--nochunk-summary",
        "--baseline-nochunk-summary",
        dest="baseline_nochunk_summary",
        default=None,
    )
    ap.add_argument("--recovergen-nochunk-summary", default=None)
    ap.add_argument("--baseline-root", default=default_base_root)
    ap.add_argument("--recovergen-root", default=default_recover_root)
    ap.add_argument(
        "--nochunk-root",
        "--baseline-nochunk-root",
        dest="baseline_nochunk_root",
        default=default_nochunk_root,
    )
    ap.add_argument("--recovergen-nochunk-root", default=default_recover_nochunk_root)
    ap.add_argument("--baseline-server-log", default=None)
    ap.add_argument("--recovergen-server-log", default=None)
    ap.add_argument(
        "--nochunk-server-log",
        "--baseline-nochunk-server-log",
        dest="baseline_nochunk_server_log",
        default=None,
    )
    ap.add_argument("--recovergen-nochunk-server-log", default=None)
    ap.add_argument("--preempt-window-s", type=float, default=5.0)
    ap.add_argument(
        "--target-lambda",
        type=float,
        default=None,
        help="Only compare metrics for this lambda across algorithms.",
    )
    ap.add_argument(
        "--target-lambdas",
        default=None,
        help="Comma/space separated lambdas, e.g. '0.6,0.9,1.2'.",
    )
    ap.add_argument(
        "--lambda-tol",
        type=float,
        default=1e-6,
        help="Tolerance when matching target lambda.",
    )
    ap.add_argument(
        "--full-report",
        action="store_true",
        help="When using multiple target lambdas, also print full detailed tables.",
    )
    ap.add_argument(
        "--gap-metric",
        choices=["stall_max", "long_gap_ratio"],
        default="stall_max",
        help="Key gap metric in key-metrics table and paper-style grid.",
    )
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    requested_lambdas = _parse_target_lambdas(args.target_lambdas)
    if args.target_lambda is not None:
        requested_lambdas.append(args.target_lambda)

    target_lambdas: List[float] = []
    for lam in requested_lambdas:
        if not any(abs(lam - x) <= args.lambda_tol for x in target_lambdas):
            target_lambdas.append(lam)

    try:
        baseline_summaries = _resolve_summary_paths(
            "vllm",
            args.baseline_summary,
            args.baseline_root,
            target_lambdas,
            args.lambda_tol,
            optional=False,
        )
        recovergen_summaries = _resolve_summary_paths(
            "RecoverGen",
            args.recovergen_summary,
            args.recovergen_root,
            target_lambdas,
            args.lambda_tol,
            optional=False,
        )
        baseline_nochunk_summaries = _resolve_summary_paths(
            "vllm (no_chunk)",
            args.baseline_nochunk_summary,
            args.baseline_nochunk_root,
            target_lambdas,
            args.lambda_tol,
            optional=False,
        )
        recovergen_nochunk_summaries = _resolve_summary_paths(
            "RecoverGen (no_chunk)",
            args.recovergen_nochunk_summary,
            args.recovergen_nochunk_root,
            target_lambdas,
            args.lambda_tol,
            optional=True,
        )

        methods = [
            _build_method_from_summaries(
                "vllm",
                baseline_summaries,
                args.baseline_server_log,
                args.preempt_window_s,
                args.lambda_tol,
            ),
            _build_method_from_summaries(
                "RecoverGen",
                recovergen_summaries,
                args.recovergen_server_log,
                args.preempt_window_s,
                args.lambda_tol,
            ),
            _build_method_from_summaries(
                "vllm (no_chunk)",
                baseline_nochunk_summaries,
                args.baseline_nochunk_server_log,
                args.preempt_window_s,
                args.lambda_tol,
            ),
        ]
        if recovergen_nochunk_summaries:
            methods.append(
                _build_method_from_summaries(
                    "RecoverGen (no_chunk)",
                    recovergen_nochunk_summaries,
                    args.recovergen_nochunk_server_log,
                    args.preempt_window_s,
                    args.lambda_tol,
                )
            )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[ERROR] {e}")

    methods = _ordered_methods(methods)

    all_summary_lams = sorted({lam for m in methods for lam in m.summary_by_lambda})
    if target_lambdas:
        try:
            resolved: List[float] = []
            for lam in target_lambdas:
                resolved_lams = _select_lambdas(
                    all_summary_lams, lam, args.lambda_tol, "summary")
                if resolved_lams:
                    resolved.append(resolved_lams[0])
            target_lambdas = resolved
        except ValueError as e:
            raise SystemExit(f"[ERROR] {e}")
    elif len(all_summary_lams) == 1:
        target_lambdas = [all_summary_lams[0]]
        print(f"[INFO] inferred target lambda from summaries: {target_lambdas[0]:.4f}")

    target_lambda: Optional[float] = None
    if len(target_lambdas) == 1:
        target_lambda = target_lambdas[0]

    print("Using run paths:")
    for m in methods:
        print(f"- {m.name}: summary={m.summary_path}")
        if len(m.summary_paths) > 1:
            print(f"  summary_count={len(m.summary_paths)}")
            for sp in m.summary_paths:
                print(f"  summary_item={sp}")
        print(f"  client_run={m.client_run_dir}")
        print(f"  recovery={m.recovery_dir or 'not_found'}")

    try:
        if target_lambdas:
            print("[INFO] comparing algorithms for target lambdas: "
                  + ", ".join([f"{x:.4f}" for x in target_lambdas]))
            _print_key_metrics_by_lambda(
                methods, target_lambdas, args.lambda_tol, args.gap_metric)

        # Single target lambda keeps previous detailed behavior.
        if target_lambda is not None:
            _print_responsiveness(methods, target_lambda, args.lambda_tol)
            _print_streaming(methods, target_lambda, args.lambda_tol)
            _print_preempt_recovery(methods)
            _print_resource_overhead(methods)
            _print_relative_change(methods, target_lambda, args.lambda_tol)
        elif not target_lambdas or args.full_report:
            _print_responsiveness(methods, None, args.lambda_tol)
            _print_streaming(methods, None, args.lambda_tol)
            _print_preempt_recovery(methods)
            _print_resource_overhead(methods)
            _print_relative_change(methods, None, args.lambda_tol)
    except ValueError as e:
        raise SystemExit(f"[ERROR] {e}")

    plot_paths: List[str] = []
    if args.plot_dir:
        plot_dir = os.path.abspath(args.plot_dir)
        os.makedirs(plot_dir, exist_ok=True)
        try:
            if target_lambdas and len(target_lambdas) > 1:
                for lam in target_lambdas:
                    lam_tag = f"{lam:.2f}".replace(".", "p")
                    lam_plot_dir = os.path.join(plot_dir, f"lambda_{lam_tag}")
                    plot_paths.extend(
                        _generate_plots(
                            methods, lam_plot_dir, lam, args.lambda_tol))
            else:
                plot_paths = _generate_plots(
                    methods, plot_dir, target_lambda, args.lambda_tol)
            grid_lambdas = target_lambdas if target_lambdas else all_summary_lams
            plot_paths.extend(
                _generate_paper_style_lambda_grid_plot(
                    methods, grid_lambdas, args.lambda_tol, plot_dir, args.gap_metric
                )
            )
        except ValueError as e:
            raise SystemExit(f"[ERROR] {e}")
        print("\nGenerated plots:")
        if plot_paths:
            for p in plot_paths:
                print(f"- {p}")
        else:
            print("- none")

    if args.json_out:
        payload = {
            "methods": [_method_to_dict(m) for m in methods],
            "plot_paths": plot_paths,
            "notes": {
                "long_gap_proxy": (
                    "long-gap frequency/time uses per-request stall_gap_max_s "
                    "thresholding from client_results.csv"
                ),
                "target_lambda": target_lambda,
                "target_lambdas": target_lambdas,
                "gap_metric_mode": args.gap_metric,
            },
        }
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        print(f"\njson report: {out_path}")


if __name__ == "__main__":
    main()
