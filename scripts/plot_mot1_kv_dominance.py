#!/usr/bin/env python3
"""Plot motivation Figure 1: KV-cache dominance under concurrency growth.

Usage:
  python3 scripts/plot_mot1_kv_dominance.py \
    --summary logs/RecoveryGen/Mot1_mem_concy/<run>/summary.csv \
    --out logs/RecoveryGen/Mot1_mem_concy/plot/kv_dominance.pdf \
    --out-xlsx logs/RecoveryGen/Mot1_mem_concy/plot/kv_dominance_data.xlsx

You may pass multiple --summary files (e.g., different context settings):
  python3 scripts/plot_mot1_kv_dominance.py \
    --summary run_ctx4k/summary.csv run_ctx6k/summary.csv run_ctx8k/summary.csv \
    --labels ctx4k ctx6k ctx8k \
    --out logs/RecoveryGen/Mot1_mem_concy/plot/kv_dominance.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

@dataclass
class RunCurve:
    label: str
    context_rank: float
    df: pd.DataFrame


def _to_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _infer_context_rank(label: str) -> float:
    pats = [
        r"(?:ctx|context|tok|max(?:_total)?_tokens)[=_-]?(\d+)",
        r"(\d+)k",
    ]
    for pat in pats:
        m = re.search(pat, label, flags=re.IGNORECASE)
        if m:
            val = _to_float(m.group(1), np.nan)
            if np.isfinite(val):
                if "k" in m.group(0).lower() and val < 1000:
                    return val * 1000.0
                return val
    return np.nan


def _normalize_cache_frac(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(s.to_numpy(dtype=float))) if len(s) else np.nan
    if np.isfinite(med) and med > 1.5:
        return (s / 100.0).clip(lower=0.0, upper=1.0)
    return s.clip(lower=0.0, upper=1.0)


def _load_one(summary_path: Path, label: str, stat: str) -> RunCurve:
    raw = pd.read_csv(summary_path)
    if "lambda_rps" not in raw.columns:
        raise ValueError(f"{summary_path}: missing column lambda_rps")

    used_col = f"gpu_mem_used_mb_{stat}"
    cache_col = f"gpu_cache_usage_perc_{stat}"
    util_col = f"gpu_mem_util_perc_{stat}"
    total_col = "gpu_mem_total_mb_p50"

    missing = [c for c in [used_col, cache_col, total_col] if c not in raw.columns]
    if missing:
        raise ValueError(f"{summary_path}: missing columns: {', '.join(missing)}")

    g = raw.groupby("lambda_rps", as_index=False).median(numeric_only=True).sort_values("lambda_rps")
    g["lambda_rps"] = pd.to_numeric(g["lambda_rps"], errors="coerce")
    g = g.dropna(subset=["lambda_rps"])

    used = pd.to_numeric(g[used_col], errors="coerce")
    total = pd.to_numeric(g[total_col], errors="coerce")
    cache_frac = _normalize_cache_frac(g[cache_col])

    g["gpu_mem_used_mb"] = used
    g["gpu_mem_total_mb"] = total
    g["gpu_cache_frac"] = cache_frac
    g["kv_est_mb"] = (used * cache_frac).clip(lower=0.0)
    g["non_kv_est_mb"] = (used - g["kv_est_mb"]).clip(lower=0.0)
    g["headroom_mb"] = (total - used).clip(lower=0.0)
    g["kv_share_used_pct"] = (100.0 * g["kv_est_mb"] / used.replace(0.0, np.nan)).clip(lower=0.0, upper=100.0)

    if util_col in g.columns:
        g["gpu_util_pct"] = pd.to_numeric(g[util_col], errors="coerce")
    else:
        g["gpu_util_pct"] = 100.0 * used / total.replace(0.0, np.nan)

    # Some summaries may store 0~1 in *_util_perc; normalize to 0~100.
    med_util = float(np.nanmedian(g["gpu_util_pct"].to_numpy(dtype=float)))
    if np.isfinite(med_util) and med_util <= 1.5:
        g["gpu_util_pct"] = 100.0 * g["gpu_util_pct"]

    rank = _infer_context_rank(label)
    return RunCurve(label=label, context_rank=rank, df=g)


def _pick_representative(runs: List[RunCurve]) -> RunCurve:
    finite = [r for r in runs if np.isfinite(r.context_rank)]
    if finite:
        return sorted(finite, key=lambda r: r.context_rank)[-1]
    return runs[-1]


def _is_non_decreasing(arr: np.ndarray, tol: float = 1e-6) -> bool:
    vals = arr[np.isfinite(arr)]
    if vals.size <= 1:
        return True
    return bool(np.all(np.diff(vals) >= -tol))


def _print_terminal_summary(runs: List[RunCurve], rep_idx: int) -> None:
    print("\n=== Plot Input Summary (for terminal check) ===")
    for i, run in enumerate(runs, start=1):
        d = run.df.copy()
        closure_cols = [
            "decode_toks_per_s_client_wall",
            "decode_toks_per_s_metrics_delta",
            "ttft_p99_s",
            "tpot_p99_s",
            "stall_gap_p99_s",
            "num_requests_swapped_p99",
            "swapin_blocks_delta",
            "swapout_blocks_delta",
            "recompute_tokens_delta",
            "restore_progress_stall_ms_delta",
        ]
        missing_closure = [c for c in closure_cols if c not in d.columns]
        view_cols = [
            "lambda_rps",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "kv_est_mb",
            "kv_share_used_pct",
            "gpu_util_pct",
            "decode_toks_per_s_client_wall",
            "decode_toks_per_s_metrics_delta",
            "ttft_p99_s",
            "tpot_p99_s",
            "stall_gap_p99_s",
            "preempt_sum_delta",
            "num_requests_swapped_p99",
            "swapin_blocks_delta",
            "swapout_blocks_delta",
            "recompute_tokens_delta",
            "restore_progress_stall_ms_delta",
            "ok_200",
            "total_rows",
        ]
        cols = [c for c in view_cols if c in d.columns]
        disp = d[cols].copy()
        for c in disp.columns:
            if c in ("ok_200", "total_rows"):
                disp[c] = pd.to_numeric(disp[c], errors="coerce").round(0).astype("Int64")
            else:
                disp[c] = pd.to_numeric(disp[c], errors="coerce").round(3)

        kv_last = float(d["kv_share_used_pct"].dropna().iloc[-1]) if d["kv_share_used_pct"].notna().any() else np.nan
        util_last = float(d["gpu_util_pct"].dropna().iloc[-1]) if d["gpu_util_pct"].notna().any() else np.nan
        used_last = float(d["gpu_mem_used_mb"].dropna().iloc[-1]) if d["gpu_mem_used_mb"].notna().any() else np.nan
        total_last = float(d["gpu_mem_total_mb"].dropna().iloc[-1]) if d["gpu_mem_total_mb"].notna().any() else np.nan
        headroom_last = (total_last - used_last) if (np.isfinite(total_last) and np.isfinite(used_last)) else np.nan
        kv_mon = _is_non_decreasing(d["kv_share_used_pct"].to_numpy(dtype=float))
        util_mon = _is_non_decreasing(d["gpu_util_pct"].to_numpy(dtype=float))

        tag = " (Representative for Panel A)" if (i - 1) == rep_idx else ""
        print(f"\n[Setting {i}]{tag} source_label={run.label}")
        if missing_closure:
            print("missing_closure_columns:", ", ".join(missing_closure))
        print(disp.to_string(index=False))
        print(
            "checks: "
            f"kv_share_non_decreasing={kv_mon}, "
            f"gpu_util_non_decreasing={util_mon}, "
            f"tail_kv_share_pct={kv_last:.2f}, "
            f"tail_gpu_util_pct={util_last:.2f}, "
            f"tail_headroom_mb={headroom_last:.2f}"
        )


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _excel_col_name(idx_1based: int) -> str:
    out = []
    n = idx_1based
    while n > 0:
        n, r = divmod(n - 1, 26)
        out.append(chr(ord("A") + r))
    return "".join(reversed(out))


def _sheet_xml_from_df(df: pd.DataFrame) -> str:
    rows_xml = []
    # Header row.
    header_cells = []
    for cidx, col in enumerate(df.columns, start=1):
        cell_ref = f"{_excel_col_name(cidx)}1"
        header_cells.append(
            f'<c r="{cell_ref}" t="inlineStr"><is><t>{_xml_escape(col)}</t></is></c>'
        )
    rows_xml.append(f'<row r="1">{"".join(header_cells)}</row>')

    # Data rows.
    for ridx, row in enumerate(df.itertuples(index=False), start=2):
        cells = []
        for cidx, val in enumerate(row, start=1):
            cell_ref = f"{_excel_col_name(cidx)}{ridx}"
            if pd.isna(val):
                continue
            if isinstance(val, (int, float, np.integer, np.floating)):
                if np.isfinite(float(val)):
                    cells.append(f'<c r="{cell_ref}"><v>{float(val)}</v></c>')
                continue
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{_xml_escape(str(val))}</t></is></c>'
            )
        rows_xml.append(f'<row r="{ridx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows_xml)}</sheetData>'
        "</worksheet>"
    )


def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    name = re.sub(r"[:\\\\/?*\\[\\]]", "_", name)
    name = name.strip() or "Sheet"
    name = name[:31]
    base = name
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = (base[: 31 - len(suffix)] + suffix)[:31]
        i += 1
    used.add(name)
    return name


def _write_xlsx_minimal(path: Path, sheets: List[tuple[str, pd.DataFrame]]) -> None:
    used_names: set[str] = set()
    safe_sheets = [(_sanitize_sheet_name(n, used_names), df) for n, df in sheets]

    workbook_sheets = []
    rel_items = []
    content_items = []
    for i, (name, _) in enumerate(safe_sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{_xml_escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        )
        rel_items.append(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
        content_items.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rel_items)}'
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{"".join(content_items)}'
        "</Types>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        "</styleSheet>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        for i, (_, df) in enumerate(safe_sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml_from_df(df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="+", required=True, help="One or more summary.csv paths.")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for summaries.")
    ap.add_argument("--stat", default="p50", choices=["p50", "p90", "p99", "max"])
    ap.add_argument(
        "--out",
        default="logs/RecoveryGen/Mot1_mem_concy/plot/kv_dominance.pdf",
    )
    ap.add_argument(
        "--out-png",
        default="logs/RecoveryGen/Mot1_mem_concy/plot/kv_dominance.png",
    )
    ap.add_argument(
        "--out-xlsx",
        default="logs/RecoveryGen/Mot1_mem_concy/plot/kv_dominance_data.xlsx",
    )
    ap.add_argument(
        "--print-summary",
        action="store_true",
        help="Print plotting data summary in terminal (enabled by default).",
    )
    ap.add_argument(
        "--no-print-summary",
        action="store_true",
        help="Disable terminal summary output.",
    )
    args = ap.parse_args()

    summary_paths = [Path(p) for p in args.summary]
    if args.labels and len(args.labels) not in (0, len(summary_paths)):
        raise SystemExit("--labels count must match --summary count.")

    # Keep artifacts next to the current run's summary to avoid cross-run confusion.
    # Even if user passes a different directory in --out/--out-png/--out-xlsx,
    # keep only basename and write under <summary_dir>/plot.
    out_dir = summary_paths[0].parent / "plot"
    out_pdf_name = Path(args.out).name if args.out else "kv_dominance.pdf"
    out_png_name = Path(args.out_png).name if args.out_png else "kv_dominance.png"
    out_xlsx_name = Path(args.out_xlsx).name if args.out_xlsx else "kv_dominance_data.xlsx"
    args.out = str(out_dir / out_pdf_name)
    args.out_png = str(out_dir / out_png_name) if args.out_png else ""
    args.out_xlsx = str(out_dir / out_xlsx_name)

    labels = args.labels if args.labels else [p.parent.name for p in summary_paths]
    runs = [_load_one(p, lab, args.stat) for p, lab in zip(summary_paths, labels)]
    runs = sorted(
        runs,
        key=lambda r: (np.inf if not np.isfinite(r.context_rank) else r.context_rank, r.label),
    )
    rep = _pick_representative(runs)

    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)
    ax1, ax2 = axes

    # Panel A: stacked memory decomposition for a representative run.
    xvals = rep.df["lambda_rps"].to_numpy(dtype=float)
    idx = np.arange(len(xvals))
    w = 0.75
    ax1.bar(idx, rep.df["non_kv_est_mb"], width=w, color="#9ecae1", label="Non-KV (estimated)")
    ax1.bar(
        idx,
        rep.df["kv_est_mb"],
        width=w,
        bottom=rep.df["non_kv_est_mb"],
        color="#1f77b4",
        label="KV-cache (estimated)",
    )
    ax1.plot(idx, rep.df["gpu_mem_total_mb"], color="#222222", linewidth=1.6, label="GPU total")
    ax1.plot(idx, rep.df["gpu_mem_used_mb"], color="#ff7f0e", linewidth=1.6, label="GPU used")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"{x:.2f}" for x in xvals])
    ax1.set_xlabel("Arrival rate (req/s)")
    ax1.set_ylabel("Memory (MB)")
    ax1.grid(alpha=0.25, axis="y")
    ax1.legend(fontsize=8, loc="upper left")

    # Panel B: KV dominance trend (and GPU util) across runs.
    cmap = plt.get_cmap("tab10")
    for i, run in enumerate(runs):
        x = run.df["lambda_rps"].to_numpy(dtype=float)
        kv_share = run.df["kv_share_used_pct"].to_numpy(dtype=float)
        run_name = f"Setting {i + 1}"
        ax2.plot(
            x,
            kv_share,
            marker="o",
            linewidth=1.6,
            color=cmap(i % 10),
            label=f"{run_name} KV share",
        )

    # Show GPU-util curve only for representative run to avoid clutter.
    rep_idx = runs.index(rep)
    ax2.plot(
        rep.df["lambda_rps"].to_numpy(dtype=float),
        rep.df["gpu_util_pct"].to_numpy(dtype=float),
        marker="s",
        linewidth=1.4,
        linestyle="--",
        color="#d62728",
        label=f"Setting {rep_idx + 1} GPU util",
    )
    ax2.axhline(100.0, color="#666666", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Arrival rate (req/s)")
    ax2.set_ylabel("Percent (%)")
    ax2.set_ylim(0.0, 105.0)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200)
    if args.out_png:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)

    # Export plotting data to Excel.
    rows = []
    for i, run in enumerate(runs, start=1):
        d = run.df.copy()
        d.insert(0, "setting_id", i)
        d.insert(1, "setting_name", f"Setting {i}")
        d.insert(2, "source_label", run.label)
        rows.append(d)
    all_df = pd.concat(rows, ignore_index=True)

    rep_df = rep.df.copy()
    rep_df.insert(0, "setting_id", rep_idx + 1)
    rep_df.insert(1, "setting_name", f"Setting {rep_idx + 1}")
    rep_df.insert(2, "source_label", rep.label)

    meta_rows = []
    for i, (p, lab) in enumerate(zip(summary_paths, labels), start=1):
        meta_rows.append({
            "setting_id": i,
            "setting_name": f"Setting {i}",
            "summary_path": str(p),
            "input_label": lab,
        })
    meta_df = pd.DataFrame(meta_rows)

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("openpyxl") is not None:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="plot_data_all", index=False)
            rep_df.to_excel(writer, sheet_name="panelA_representative", index=False)
            meta_df.to_excel(writer, sheet_name="meta", index=False)
    elif importlib.util.find_spec("xlsxwriter") is not None:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            all_df.to_excel(writer, sheet_name="plot_data_all", index=False)
            rep_df.to_excel(writer, sheet_name="panelA_representative", index=False)
            meta_df.to_excel(writer, sheet_name="meta", index=False)
    else:
        _write_xlsx_minimal(
            out_xlsx,
            [
                ("plot_data_all", all_df),
                ("panelA_representative", rep_df),
                ("meta", meta_df),
            ],
        )

    print(f"[OK] wrote figure: {out_pdf}")
    if args.out_png:
        print(f"[OK] wrote figure: {args.out_png}")
    print(f"[OK] wrote data: {out_xlsx}")
    print_summary = True
    if args.no_print_summary:
        print_summary = False
    if args.print_summary:
        print_summary = True
    if print_summary:
        _print_terminal_summary(runs, rep_idx)


if __name__ == "__main__":
    main()
