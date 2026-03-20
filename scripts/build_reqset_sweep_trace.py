#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import sys
from typing import Dict, Any, List, Tuple

def lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower().strip(): v for k, v in d.items()}

def parse_trace_window(
    trace_path: str,
    start_ts: float,
    duration_s: float,
    max_total_tokens: int,
) -> List[Tuple[float, int, int]]:
    rows: List[Tuple[float, int, int]] = []
    end_ts = start_ts + duration_s
    with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            d = lower_keys(row)
            ts = d.get("timestamp", None)
            it = d.get("request tokens", d.get("request_tokens", d.get("in_tok", d.get("prompt_tokens", None))))
            ot = d.get("response tokens", d.get("response_tokens", d.get("out_tok", d.get("completion_tokens", None))))
            if ts is None or it is None or ot is None:
                continue
            try:
                tsf = float(ts)
                itok = int(float(it))
                otok = int(float(ot))
            except Exception:
                continue
            if not (start_ts <= tsf < end_ts):
                continue
            if itok + otok > max_total_tokens:
                continue
            rows.append((tsf, itok, otok))
    rows.sort(key=lambda x: x[0])
    return rows

def main():
    if len(sys.argv) != 6:
        print("Usage: build_reqset_sweep_trace.py TRACE_PATH START_TS DURATION_S MAX_TOTAL_TOKENS OUT_JSONL", file=sys.stderr)
        sys.exit(2)

    trace_path = sys.argv[1]
    start_ts = float(sys.argv[2])
    duration_s = float(sys.argv[3])
    max_total_tokens = int(sys.argv[4])
    out_jsonl = sys.argv[5]

    rows = parse_trace_window(trace_path, start_ts, duration_s, max_total_tokens)
    if not rows:
        raise RuntimeError("No trace rows in selected window after filtering.")

    totals = sorted([it + ot for _, it, ot in rows])
    median_total = totals[len(totals) // 2]

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, (tsf, itok, otok) in enumerate(rows):
            rec = {
                "req_id": i,
                "ts": tsf,
                "in_tok": itok,
                "out_tok": otok,
                "total_tok": itok + otok,
                "is_long": 1 if (itok + otok) >= median_total else 0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] wrote reqset: rows={len(rows)} median_total={median_total} out={out_jsonl}")

if __name__ == "__main__":
    main()
