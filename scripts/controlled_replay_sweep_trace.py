#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, csv, json, time, random, asyncio
from typing import List, Dict, Any, Tuple

import numpy as np
import aiohttp
from transformers import AutoTokenizer

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if raw is None:
        return float(default)
    raw = raw.strip()
    if raw == "":
        return float(default)
    return float(raw)

def load_human_segments(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs: List[str] = []
    for item in data:
        conv = item.get("conversations", [])
        for turn in conv:
            frm = turn.get("from") or turn.get("role")
            val = turn.get("value") or turn.get("content") or ""
            if frm == "human" and isinstance(val, str):
                s = val.strip()
                if s:
                    segs.append(s)
    return segs

def build_prompt(tok, human_segs: List[str], target_tokens: int) -> str:
    # strictly from ShareGPT human-only, by truncate/concat
    for _ in range(50):
        s = random.choice(human_segs)
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) >= target_tokens:
            ids = ids[:target_tokens]
            return tok.decode(ids, skip_special_tokens=True)

    ids_all: List[int] = []
    sep = tok.encode("\n\n", add_special_tokens=False)
    for _ in range(200):
        s = random.choice(human_segs)
        ids = tok.encode(s, add_special_tokens=False)
        if not ids:
            continue
        ids_all.extend(ids)
        ids_all.extend(sep)
        if len(ids_all) >= target_tokens:
            ids_all = ids_all[:target_tokens]
            return tok.decode(ids_all, skip_special_tokens=True)

    return human_segs[0]

def load_reqset_jsonl(reqset_path: str, max_total_tokens: int) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = []
    with open(reqset_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            itok = int(r["in_tok"]); otok = int(r["out_tok"])
            if itok + otok > max_total_tokens:
                continue
            reqs.append({
                "req_id": int(r.get("req_id", len(reqs))),
                "in_tok": itok,
                "out_tok": otok,
                "is_long": int(r.get("is_long", 0)),
            })
    if not reqs:
        raise RuntimeError("Empty reqset after filtering.")
    return reqs

def gen_arrivals_poisson(lam: float, T: float, N: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    # Conditional on N arrivals in [0, T], the arrival times are i.i.d. Uniform(0, T)
    ts = list(rng.uniform(0.0, T, size=N))
    ts.sort()
    return ts

def gen_arrivals_burst(lam: float, T: float, N: int, b: int, seed: int, burst_window_s: float) -> Tuple[List[float], List[Tuple[float,float]]]:
    rng = np.random.default_rng(seed)
    b = max(1, int(b))
    N = int(N)
    num_clusters = int(np.ceil(N / b))
    # Conditional on K clusters in [0, T], cluster times are i.i.d. Uniform(0, T)
    cluster_starts = list(rng.uniform(0.0, T, size=num_clusters))
    cluster_starts.sort()

    eps = 1e-6  # tiny jitter to avoid identical timestamps
    arrivals: List[float] = []
    segs: List[Tuple[float,float]] = []
    k = 0
    for cs in cluster_starts:
        segs.append((max(0.0, cs), min(T, cs + burst_window_s)))
        for _ in range(b):
            if k >= N:
                break
            arrivals.append(float(cs + rng.uniform(0.0, eps)))
            k += 1
        if k >= N:
            break
    arrivals.sort()
    return arrivals, segs

async def main():
    idx = 1
    prompts_json = sys.argv[idx]; idx += 1
    tokenizer_name = sys.argv[idx]; idx += 1
    reqset_path = sys.argv[idx]; idx += 1
    seed_reqset = int(sys.argv[idx]); idx += 1  # used to shuffle reqset deterministically

    base_url = sys.argv[idx]; idx += 1
    model = sys.argv[idx]; idx += 1

    mode = sys.argv[idx]; idx += 1            # poisson | burst
    lam = float(sys.argv[idx]); idx += 1      # requests/s
    T = float(sys.argv[idx]); idx += 1
    b = int(sys.argv[idx]); idx += 1          # burst size (only used in burst)
    burst_window_s = float(sys.argv[idx]); idx += 1
    seed_arrival = int(sys.argv[idx]); idx += 1

    max_outstanding = int(sys.argv[idx]); idx += 1
    max_total_tokens = int(sys.argv[idx]); idx += 1

    out_csv = sys.argv[idx]; idx += 1
    meta_json = sys.argv[idx]; idx += 1
    burst_segments_csv = sys.argv[idx]; idx += 1

    req_timeout_s = _env_float("CLIENT_REQ_TIMEOUT_S", 180.0)
    req_sock_connect_timeout_s = _env_float(
        "CLIENT_REQ_SOCK_CONNECT_TIMEOUT_S", 10.0)
    req_sock_read_timeout_s = _env_float("CLIENT_REQ_SOCK_READ_TIMEOUT_S",
                                         45.0)

    print(f"[INFO] mode={mode} lam={lam} T={T} b={b} max_outstanding={max_outstanding} max_total_tokens={max_total_tokens}")
    print(f"[INFO] reqset={reqset_path} prompts={prompts_json} tokenizer={tokenizer_name}")
    print(f"[INFO] base_url={base_url} model={model}")
    print(
        "[INFO] request_timeout_s="
        f"{req_timeout_s} sock_connect_s={req_sock_connect_timeout_s} "
        f"sock_read_s={req_sock_read_timeout_s}"
    )

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    human_segs = load_human_segments(prompts_json)
    if not human_segs:
        raise RuntimeError("No human segments loaded from ShareGPT JSON.")

    reqs = load_reqset_jsonl(reqset_path, max_total_tokens)

    # deterministic shuffle to avoid ordering artifacts, but keep the same across conditions
    rnd = random.Random(seed_reqset)
    rnd.shuffle(reqs)

    # N fixed by lam*T to align sweep semantics
    N = max(1, int(round(lam * T)))
    if N <= len(reqs):
        reqs = reqs[:N]
    else:
        # reqset not enough: sample with replacement (deterministic by seed)
        rng_req = np.random.default_rng(seed_reqset)
        idxs = rng_req.integers(0, len(reqs), size=N)
        reqs = [reqs[i] for i in idxs]

    if mode == "poisson":
        arrivals = gen_arrivals_poisson(lam, T, N, seed_arrival)
        segs: List[Tuple[float,float]] = []
    elif mode == "burst":
        arrivals, segs = gen_arrivals_burst(lam, T, N, b, seed_arrival, burst_window_s)
    else:
        raise ValueError("mode must be 'poisson' or 'burst'")

    with open(burst_segments_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s","end_s","window_s","mode","b","lambda_rps"])
        if mode == "burst":
            for a, e in segs:
                w.writerow([f"{a:.6f}", f"{e:.6f}", f"{burst_window_s:.6f}", mode, str(b), f"{lam:.6f}"])

    out_fields = [
        "req_id","exp_send_s","is_long","in_tok_target","out_tok",
        "http_status","t_send_abs","t_first_token_abs","t_done_abs","ttft_s","stall_gap_max_s"
    ]
    fout = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=out_fields)
    writer.writeheader()
    fout.flush()

    sem = asyncio.Semaphore(max_outstanding)

    replay_start_wall = time.time()
    meta = {
        "mode": mode,
        "lambda_rps": lam,
        "T_s": T,
        "b": b,
        "burst_window_s": burst_window_s,
        "seed_reqset": seed_reqset,
        "seed_arrival": seed_arrival,
        "N": N,
        "reqset_path": reqset_path,
        "prompts_json": prompts_json,
        "tokenizer": tokenizer_name,
        "replay_start_wall": replay_start_wall,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    done_count = 0

    async def send_one(session: aiohttp.ClientSession, i: int):
        nonlocal done_count
        r = reqs[i]
        exp_send = float(arrivals[i])
        await asyncio.sleep(max(0.0, exp_send - (time.time() - replay_start_wall)))

        prompt = build_prompt(tok, human_segs, int(r["in_tok"]))
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": int(r["out_tok"]),
            "temperature": 0,
            "stream": True,
        }

        async with sem:
            t_send = time.time()
            http_status = "NA"
            t_first = None
            t_done = None
            prev_chunk_ts = None
            stall_gap_max_s = 0.0
            stream_chunk_cnt = 0
            try:
                req_timeout = aiohttp.ClientTimeout(
                    total=req_timeout_s,
                    sock_connect=req_sock_connect_timeout_s,
                    sock_read=req_sock_read_timeout_s,
                )
                async with session.post(base_url, json=payload, timeout=req_timeout) as resp:
                    http_status = str(resp.status)
                    async for raw in resp.content:
                        now = time.time()
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data = line[len("data:"):].strip()
                            if data == "[DONE]":
                                t_done = now
                                break
                            if t_first is None:
                                t_first = now
                            if prev_chunk_ts is not None:
                                gap = now - prev_chunk_ts
                                if gap > stall_gap_max_s:
                                    stall_gap_max_s = gap
                            prev_chunk_ts = now
                            stream_chunk_cnt += 1
            except asyncio.TimeoutError:
                http_status = "TIMEOUT"
                t_done = time.time()
            except Exception:
                if http_status == "NA":
                    http_status = "ERROR"
                t_done = time.time()

            if t_done is None:
                t_done = time.time()
            if t_first is None:
                t_first = t_done

            writer.writerow({
                "req_id": r["req_id"],
                "exp_send_s": f"{exp_send:.6f}",
                "is_long": int(r["is_long"]),
                "in_tok_target": int(r["in_tok"]),
                "out_tok": int(r["out_tok"]),
                "http_status": http_status,
                "t_send_abs": f"{t_send:.6f}",
                "t_first_token_abs": f"{t_first:.6f}",
                "t_done_abs": f"{t_done:.6f}",
                "ttft_s": f"{(t_first - t_send):.6f}",
                "stall_gap_max_s": f"{(stall_gap_max_s if stream_chunk_cnt > 1 else 0.0):.6f}",
            })

            done_count += 1
            if done_count % 100 == 0 or done_count == N:
                fout.flush()
                print(f"[PROGRESS] done={done_count}/{N}")

    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [asyncio.create_task(send_one(session, i)) for i in range(N)]
        t0 = time.time()
        await asyncio.gather(*tasks)
        print(f"[OK] done N={N} wall_s={(time.time()-t0):.3f}")

    fout.flush()
    fout.close()

if __name__ == "__main__":
    asyncio.run(main())
