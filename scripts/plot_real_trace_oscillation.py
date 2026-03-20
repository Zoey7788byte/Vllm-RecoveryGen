#!/usr/bin/env python3
"""Plot segment-aligned oscillation signals with shaded bands for lambda=0.9."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SeriesSpec:
    label: str
    client_run_dir: str
    recovery_ts_path: str
    color: str


@dataclass
class CycleSeries:
    xs: np.ndarray
    ys: np.ndarray


def _to_float(v: object) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def _read_csv_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _segment_dirs(run_dir: str) -> List[str]:
    out = []
    for entry in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, entry)
        if os.path.isdir(full) and ("_high_" in entry or "_low_" in entry):
            out.append(full)
    return out


def _interp_step(series: CycleSeries, grid_x: np.ndarray) -> np.ndarray:
    out = np.full_like(grid_x, np.nan, dtype=float)
    if len(series.xs) == 0:
        return out
    j = 0
    cur = float(series.ys[0])
    start_x = float(series.xs[0])
    end_x = float(series.xs[-1])
    for i, x in enumerate(grid_x):
        if x < start_x - 1e-9 or x > end_x + 1e-9:
            continue
        while j + 1 < len(series.xs) and series.xs[j + 1] <= x + 1e-9:
            j += 1
            cur = float(series.ys[j])
        out[i] = cur
    return out


def _aggregate_band(series_list: Sequence[CycleSeries], grid_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_x = min(float(max(s.xs[-1] for s in series_list if len(s.xs) > 0)), 180.0)
    n_bins = int(math.floor(max_x / grid_s)) + 1
    grid_x = np.arange(n_bins, dtype=float) * grid_s
    stacked = np.vstack([_interp_step(s, grid_x) for s in series_list])
    valid_cols = np.any(np.isfinite(stacked), axis=0)
    grid_x = grid_x[valid_cols]
    stacked = stacked[:, valid_cols]
    mean_y = np.nanmean(stacked, axis=0)
    min_y = np.nanmin(stacked, axis=0)
    max_y = np.nanmax(stacked, axis=0)
    return grid_x, mean_y, min_y, max_y


def _aggregate_sum_band(series_list: Sequence[CycleSeries], bin_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_x = min(float(max(s.xs[-1] for s in series_list if len(s.xs) > 0)), 180.0)
    n_bins = int(math.floor(max_x / bin_s)) + 1
    grid_x = np.arange(n_bins, dtype=float) * bin_s
    stacked = np.full((len(series_list), n_bins), np.nan, dtype=float)

    for i, series in enumerate(series_list):
        hist = np.zeros(n_bins, dtype=float)
        used = np.zeros(n_bins, dtype=bool)
        for x, y in zip(series.xs, series.ys):
            idx = int(x // bin_s)
            if 0 <= idx < n_bins and not math.isnan(y):
                hist[idx] += y
                used[idx] = True
        hist[~used] = np.nan
        stacked[i] = hist

    valid_cols = np.any(np.isfinite(stacked), axis=0)
    grid_x = grid_x[valid_cols]
    stacked = stacked[:, valid_cols]
    mean_y = np.nanmean(stacked, axis=0)
    min_y = np.nanmin(stacked, axis=0)
    max_y = np.nanmax(stacked, axis=0)
    return grid_x, mean_y, min_y, max_y


def _aggregate_max_band(series_list: Sequence[CycleSeries], bin_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_x = 180.0
    n_bins = int(math.floor(max_x / bin_s)) + 1
    grid_x = np.arange(n_bins, dtype=float) * bin_s
    stacked = np.zeros((len(series_list), n_bins), dtype=float)

    for i, series in enumerate(series_list):
        for x, y in zip(series.xs, series.ys):
            idx = int(x // bin_s)
            if 0 <= idx < n_bins and not math.isnan(y):
                stacked[i, idx] = max(stacked[i, idx], y)

    mean_y = np.mean(stacked, axis=0)
    min_y = np.min(stacked, axis=0)
    max_y = np.max(stacked, axis=0)
    return grid_x, mean_y, min_y, max_y


def _load_client_metric_cycles(run_dir: str, key: str, scale: float = 1.0) -> Tuple[List[CycleSeries], List[Tuple[float, float]]]:
    series = []
    windows = []
    for seg_dir in _segment_dirs(run_dir):
        path = os.path.join(seg_dir, "metrics_timeseries.csv")
        if not os.path.isfile(path):
            continue
        rows = _read_csv_rows(path)
        xs = []
        ys = []
        global_ts = []
        for row in rows:
            t_rel = _to_float(row.get("t_rel_s"))
            val = _to_float(row.get(key))
            if math.isnan(t_rel) or math.isnan(val):
                continue
            global_ts.append(t_rel)
            xs.append(t_rel)
            ys.append(val * scale)
        if not xs:
            continue
        start_t = min(global_ts)
        end_t = max(global_ts)
        local_x = np.array([x - start_t for x in xs], dtype=float)
        series.append(CycleSeries(xs=local_x, ys=np.array(ys, dtype=float)))
        windows.append((start_t, end_t))
    return series, windows


def _load_recovery_metric_cycles(
    recovery_ts_path: str,
    windows: Sequence[Tuple[float, float]],
    key: str,
    reducer: str,
    end_fill: str = "none",
) -> List[CycleSeries]:
    rows = _read_csv_rows(recovery_ts_path)
    out: List[CycleSeries] = []
    for start_t, end_t in windows:
        xs = []
        ys = []
        for row in rows:
            t_rel = _to_float(row.get("t_rel_s"))
            val = _to_float(row.get(key))
            if math.isnan(t_rel) or math.isnan(val):
                continue
            if t_rel < start_t - 1e-9 or t_rel > end_t + 1e-9:
                continue
            xs.append(t_rel - start_t)
            ys.append(val)
        if not xs:
            local_end = max(0.0, end_t - start_t)
            if local_end > 0.0:
                out.append(
                    CycleSeries(
                        xs=np.array([0.0, local_end], dtype=float),
                        ys=np.array([0.0, 0.0], dtype=float),
                    )
                )
            else:
                out.append(
                    CycleSeries(
                        xs=np.array([0.0], dtype=float),
                        ys=np.array([0.0], dtype=float),
                    )
                )
            continue
        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        order = np.argsort(xs_arr)
        xs_arr = xs_arr[order]
        ys_arr = ys_arr[order]
        if reducer == "nonneg":
            ys_arr = np.maximum(0.0, ys_arr)
        local_end = max(0.0, end_t - start_t)
        if end_fill != "none" and len(xs_arr) > 0 and xs_arr[-1] < local_end - 1e-9:
            xs_arr = np.append(xs_arr, local_end)
            if end_fill == "last":
                ys_arr = np.append(ys_arr, ys_arr[-1])
            elif end_fill == "zero":
                ys_arr = np.append(ys_arr, 0.0)
            else:
                raise ValueError(f"unsupported end_fill={end_fill}")
        out.append(CycleSeries(xs=xs_arr, ys=ys_arr))
    return out


def _plot_method_band(ax, grid_x: np.ndarray, mean_y: np.ndarray, p25_y: np.ndarray, p75_y: np.ndarray, color: str, label: str) -> None:
    ax.plot(grid_x, mean_y, linewidth=2.2, color=color, label=label)
    ax.fill_between(grid_x, p25_y, p75_y, color=color, alpha=0.18, linewidth=0.0)


def _write_summary(path: str, rows: Sequence[Sequence[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "n_segments",
                "preempt_bin_mean_peak",
                "preempt_bin_max_peak",
                "restore_stall_mean_peak_ms",
                "restore_stall_max_peak_ms",
                "visible_tokens_mean_peak",
                "visible_tokens_max_peak",
                "visible_positive_segments",
            ]
        )
        for row in rows:
            w.writerow(list(row))


def _plot(specs: Sequence[SeriesSpec], grid_s: float, out_prefix: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.8), sharex=True)
    summary_rows = []

    for spec in specs:
        _, windows = _load_client_metric_cycles(spec.client_run_dir, "req_waiting", scale=1.0)
        pre_cycles = _load_recovery_metric_cycles(
            spec.recovery_ts_path,
            windows,
            key="on_preempt_count_delta",
            reducer="nonneg",
            end_fill="zero",
        )
        stall_cycles = _load_recovery_metric_cycles(
            spec.recovery_ts_path,
            windows,
            key="restore_progress_stall_ms",
            reducer="nonneg",
            end_fill="zero",
        )
        visible_cycles = _load_recovery_metric_cycles(
            spec.recovery_ts_path,
            windows,
            key="visible_tokens",
            reducer="nonneg",
            end_fill="last",
        )

        if not pre_cycles or not stall_cycles or not visible_cycles:
            raise RuntimeError(f"incomplete aligned cycles for {spec.label}")

        x_pre, mean_pre, min_pre, max_pre = _aggregate_sum_band(pre_cycles, bin_s=5.0)
        x_stall, mean_stall, min_stall, max_stall = _aggregate_max_band(stall_cycles, grid_s)
        x_vis, mean_vis, min_vis, max_vis = _aggregate_max_band(visible_cycles, grid_s)

        _plot_method_band(axes[0], x_pre, mean_pre, min_pre, max_pre, spec.color, spec.label)
        _plot_method_band(axes[1], x_stall, mean_stall, min_stall, max_stall, spec.color, spec.label)
        _plot_method_band(axes[2], x_vis, mean_vis, min_vis, max_vis, spec.color, spec.label)

        visible_positive_segments = int(
            sum(
                1
                for cyc in visible_cycles
                if any(
                    (x <= 180.0 + 1e-9) and (y > 0.0)
                    for x, y in zip(cyc.xs, cyc.ys)
                )
            )
        )

        summary_rows.append(
            [
                spec.label,
                len(pre_cycles),
                float(np.nanmax(mean_pre)),
                float(np.nanmax(max_pre)),
                float(np.nanmax(mean_stall)),
                float(np.nanmax(max_stall)),
                float(np.nanmax(mean_vis)),
                float(np.nanmax(max_vis)),
                visible_positive_segments,
            ]
        )

    axes[0].set_ylabel("Preempts / 5s")
    axes[1].set_ylabel("Restore Stall (ms)")
    axes[2].set_ylabel("Visible tokens")
    axes[2].set_xlabel("Time since segment start (s)")

    axes[0].set_title("Preemption clustering")
    axes[1].set_title("Recovery stall")
    axes[2].set_title("Recovery visibility")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0.0, right=180.0)
    axes[0].legend(frameon=True, fontsize=9, loc="upper right")

    fig.suptitle("Segment-aligned mean with min-max band (lambda=0.9, all segments)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=220)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)

    _write_summary(out_prefix + "_summary.csv", summary_rows)
    print(f"[INFO] wrote {out_prefix}.png/.pdf")
    print(f"[INFO] wrote {out_prefix}_summary.csv")
    for row in summary_rows:
        print(
            f"[INFO] {row[0]}: n_segments={row[1]} "
            f"pre_mean_peak={row[2]:.3f} stall_mean_peak={row[4]:.3f} "
            f"visible_mean_peak={row[6]:.3f} visible_pos_segments={row[8]}"
        )


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument(
        "--nochunk-client-run",
        default=os.path.join(
            repo_root,
            "logs/RecoveryGen/Baseline_no_chunked/client_logs/baseline_nochunk_lam0p9_batch_20260315_003717_20260315_023802",
        ),
    )
    ap.add_argument(
        "--baseline-client-run",
        default=os.path.join(
            repo_root,
            "logs/RecoveryGen/Baseline/client_logs/baseline_lam0p9_batch_20260315_003717_20260315_003718",
        ),
    )
    ap.add_argument(
        "--recovergen-client-run",
        default=os.path.join(
            repo_root,
            "logs/RecoveryGen/RecoverGen/client_logs/recovergen_lam0p9_batch_20260315_003717_20260315_013744",
        ),
    )
    ap.add_argument(
        "--nochunk-ts",
        default=os.path.join(
            repo_root,
            "logs/RecoveryGen/Baseline_no_chunked/server_logs/20260315_023802/recovery/recovery_ts.csv",
        ),
    )
    ap.add_argument(
        "--baseline-ts",
        default=os.path.join(
            repo_root,
            "logs/RecoveryGen/Baseline/server_logs/20260315_003718/recovery/recovery_ts.csv",
        ),
    )
    ap.add_argument(
        "--recovergen-ts",
        default=os.path.join(
            repo_root,
            "logs/RecoveryGen/RecoverGen/server_logs/20260315_013744/recovery/recovery_ts.csv",
        ),
    )
    ap.add_argument(
        "--plot-dir",
        default=os.path.join(repo_root, "logs/RecoveryGen/real_trace_oscillation"),
    )
    ap.add_argument("--run-tag", default="lam0p9")
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    specs = [
        SeriesSpec("vllm (no_chunk)", args.nochunk_client_run, args.nochunk_ts, "#4C72B0"),
        SeriesSpec("vllm", args.baseline_client_run, args.baseline_ts, "#D62728"),
        SeriesSpec("RecoverGen", args.recovergen_client_run, args.recovergen_ts, "#2CA02C"),
    ]

    stem = "fig3_real_trace_oscillation"
    if args.run_tag:
        stem = f"{stem}_{args.run_tag}"
    out_prefix = os.path.join(args.plot_dir, stem)
    _plot(specs, grid_s=args.grid_s, out_prefix=out_prefix)


if __name__ == "__main__":
    main()
