import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from vllm import envs
from vllm.recovery.cost_model import get_cost_model_snapshot


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _load_json_flags(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


@dataclass(frozen=True)
class RecoveryConfig:
    obs_enabled: bool
    flags_json: Optional[str]
    log_dir: str
    ts_period_ms: int
    v2: int
    budget: int
    phase: int
    min_budget: int
    swap_ms_per_block: float
    target_cycle_ms: float
    stall_ms: float
    budget_init_ms: float
    budget_min_ms: float
    dplus_ms: float
    dminus_ms: float
    prefetch_max_blocks: int

    @classmethod
    def from_env_or_json(cls) -> "RecoveryConfig":
        env_cfg: Dict[str, Any] = {
            "obs_enabled": envs.VLLM_RECOVERY_OBS,
            "flags_json": envs.VLLM_RECOVERY_FLAGS_JSON,
            "log_dir": envs.VLLM_RECOVERY_LOG_DIR,
            "ts_period_ms": envs.VLLM_RECOVERY_TS_PERIOD_MS,
            "v2": envs.VLLM_RECOVERY_V2,
            "budget": envs.VLLM_RECOVERY_BUDGET,
            "phase": envs.VLLM_RECOVERY_PHASE,
            "min_budget": envs.VLLM_RECOVERY_MIN_BUDGET,
            "swap_ms_per_block": envs.VLLM_RECOVERY_SWAP_MS_PER_BLOCK,
            "target_cycle_ms": envs.REC_T_CYC_MS,
            "stall_ms": envs.REC_STALL_MS,
            "budget_init_ms": envs.REC_BUDGET_INIT_MS,
            "budget_min_ms": envs.REC_BUDGET_MIN_MS,
            "dplus_ms": envs.REC_DPLUS_MS,
            "dminus_ms": envs.REC_DMINUS_MS,
            "prefetch_max_blocks": envs.REC_PREFETCH_MAX_BLOCKS,
        }

        json_cfg: Dict[str, Any] = {}
        flags_path = env_cfg.get("flags_json")
        if isinstance(flags_path, str) and flags_path:
            if os.path.isfile(flags_path):
                json_cfg = _load_json_flags(flags_path)

        if json_cfg:
            # Support either flat keys or nested "recovery" dict.
            if isinstance(json_cfg.get("recovery"), dict):
                json_cfg = json_cfg["recovery"]

            if "obs_enabled" in json_cfg:
                v = _coerce_bool(json_cfg.get("obs_enabled"))
                if v is not None:
                    env_cfg["obs_enabled"] = v
            if "obs" in json_cfg:
                v = _coerce_bool(json_cfg.get("obs"))
                if v is not None:
                    env_cfg["obs_enabled"] = v
            if "log_dir" in json_cfg and isinstance(json_cfg.get("log_dir"), str):
                env_cfg["log_dir"] = json_cfg["log_dir"]
            if "ts_period_ms" in json_cfg:
                v = _coerce_int(json_cfg.get("ts_period_ms"))
                if v is not None:
                    env_cfg["ts_period_ms"] = v
            if "v2" in json_cfg:
                v = _coerce_int(json_cfg.get("v2"))
                if v is not None:
                    env_cfg["v2"] = v
            if "budget" in json_cfg:
                v = _coerce_int(json_cfg.get("budget"))
                if v is not None:
                    env_cfg["budget"] = v
            if "phase" in json_cfg:
                v = _coerce_int(json_cfg.get("phase"))
                if v is not None:
                    env_cfg["phase"] = v
            if "min_budget" in json_cfg:
                v = _coerce_int(json_cfg.get("min_budget"))
                if v is not None:
                    env_cfg["min_budget"] = v
            if "swap_ms_per_block" in json_cfg:
                v = json_cfg.get("swap_ms_per_block")
                try:
                    env_cfg["swap_ms_per_block"] = float(v)
                except Exception:
                    pass
            if "target_cycle_ms" in json_cfg:
                v = json_cfg.get("target_cycle_ms")
                try:
                    env_cfg["target_cycle_ms"] = float(v)
                except Exception:
                    pass
            if "t_cyc_ms" in json_cfg:
                v = json_cfg.get("t_cyc_ms")
                try:
                    env_cfg["target_cycle_ms"] = float(v)
                except Exception:
                    pass
            if "stall_ms" in json_cfg:
                v = json_cfg.get("stall_ms")
                try:
                    env_cfg["stall_ms"] = float(v)
                except Exception:
                    pass
            if "b_min" in json_cfg:
                v = _coerce_int(json_cfg.get("b_min"))
                if v is not None:
                    env_cfg["min_budget"] = v
            if "budget_init_ms" in json_cfg:
                v = json_cfg.get("budget_init_ms")
                try:
                    env_cfg["budget_init_ms"] = float(v)
                except Exception:
                    pass
            if "budget_min_ms" in json_cfg:
                v = json_cfg.get("budget_min_ms")
                try:
                    env_cfg["budget_min_ms"] = float(v)
                except Exception:
                    pass
            if "dplus_ms" in json_cfg:
                v = json_cfg.get("dplus_ms")
                try:
                    env_cfg["dplus_ms"] = float(v)
                except Exception:
                    pass
            if "dminus_ms" in json_cfg:
                v = json_cfg.get("dminus_ms")
                try:
                    env_cfg["dminus_ms"] = float(v)
                except Exception:
                    pass
            if "prefetch_max_blocks" in json_cfg:
                v = _coerce_int(json_cfg.get("prefetch_max_blocks"))
                if v is not None:
                    env_cfg["prefetch_max_blocks"] = v

        return cls(
            obs_enabled=bool(env_cfg["obs_enabled"]),
            flags_json=env_cfg["flags_json"],
            log_dir=str(env_cfg["log_dir"]),
            ts_period_ms=int(env_cfg["ts_period_ms"]),
            v2=int(env_cfg["v2"]),
            budget=int(env_cfg["budget"]),
            phase=int(env_cfg["phase"]),
            min_budget=max(0, int(env_cfg["min_budget"])),
            swap_ms_per_block=max(1e-6, float(env_cfg["swap_ms_per_block"])),
            target_cycle_ms=max(1.0, float(env_cfg["target_cycle_ms"])),
            stall_ms=max(1.0, float(env_cfg["stall_ms"])),
            budget_init_ms=max(0.0, float(env_cfg["budget_init_ms"])),
            budget_min_ms=max(0.0, float(env_cfg["budget_min_ms"])),
            dplus_ms=max(0.0, float(env_cfg["dplus_ms"])),
            dminus_ms=max(0.0, float(env_cfg["dminus_ms"])),
            prefetch_max_blocks=max(0, int(env_cfg["prefetch_max_blocks"])),
        )


@lru_cache(maxsize=1)
def get_recovery_config() -> RecoveryConfig:
    return RecoveryConfig.from_env_or_json()


def _build_mem_snapshot(block_manager: Any) -> Optional[Dict[str, Any]]:
    try:
        free_gpu = block_manager.get_num_free_gpu_blocks()
        free_cpu = block_manager.get_num_free_cpu_blocks()
    except Exception:
        return None

    snap: Dict[str, Any] = {
        "free_gpu_blocks": free_gpu,
        "free_cpu_blocks": free_cpu,
    }
    try:
        total_gpu = getattr(block_manager, "num_total_gpu_blocks", None)
        total_cpu = getattr(block_manager, "num_total_cpu_blocks", None)
        if total_gpu is not None:
            snap["total_gpu_blocks"] = total_gpu
            snap["used_gpu_blocks"] = max(0, total_gpu - free_gpu)
        if total_cpu is not None:
            snap["total_cpu_blocks"] = total_cpu
            snap["used_cpu_blocks"] = max(0, total_cpu - free_cpu)
    except Exception:
        pass
    try:
        watermark_blocks = getattr(block_manager, "watermark_blocks", None)
        if watermark_blocks is not None:
            snap["watermark_blocks"] = watermark_blocks
    except Exception:
        pass
    return snap


class _RecoveryEventLogger:
    # Phase0: keep JSONL sparse to avoid IO jitter on online path.
    _EVENT_ALLOWLIST = frozenset({
        "PREEMPT_TRIGGERED",
        "CYCLE_SLACK_COMPUTED",
        "RECOVERY_MODE_SWITCH",
        "RECOVERY_BUDGET_DECISION",
        "RECOVERY_STATE_CREATED",
        "RECOMPUTE_MICRO",
        "RECOVERY_PROGRESS_COMMIT",
        "SWAP_IN_MICRO",
        "SWAP_PREFETCH_READY",
        "SWAP_IN_SKIP_ALREADY_RESTORED",
        "RECOVERY_REMAINING_HOST_BLOCKS",
        "RECOVERY_STALLED",
        "SWAP_OUT",
        "SWAP_IN",
    })

    def __init__(self, cfg: RecoveryConfig) -> None:
        self._cfg = cfg
        self._path = os.path.join(cfg.log_dir, "recovery_events.jsonl")
        self._fh = None

    def _ensure_open(self) -> bool:
        if self._fh is not None:
            return True
        try:
            os.makedirs(self._cfg.log_dir, exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")
            return True
        except Exception:
            self._fh = None
            return False

    def log(self, payload: Dict[str, Any]) -> None:
        if not self._cfg.obs_enabled:
            return
        event = payload.get("event")
        if not isinstance(event, str) or event not in self._EVENT_ALLOWLIST:
            return
        if not self._ensure_open():
            return
        try:
            self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._fh.flush()
        except Exception:
            return


class _RecoveryCsvLogger:

    _COLUMNS = [
        "ts_ns",
        "t_rel_s",
        "cycle_id",
        "T_cyc_ms",
        "on_preempt_count_delta",
        "on_waiting_len",
        "T_on_ms",
        "S_ms",
        "B_ms",
        "barT_swap",
        "barT_rec",
        "gain_swap",
        "gain_rec",
        "x_swap",
        "x_rec",
        "over_budget_drop",
        "swapin_blocks",
        "swapout_blocks",
        "recompute_tokens",
        "gpu_kv_blocks_free",
        "mode_cnt_normal",
        "mode_cnt_recovery",
        "mode_cnt_fallback",
        "restore_progress_stall_ms",
        "cycle_wall_ms",
        "prefill_tokens",
        "decode_tokens",
    ]

    def __init__(self, cfg: RecoveryConfig) -> None:
        self._cfg = cfg
        self._path = os.path.join(cfg.log_dir, "recovery_ts.csv")
        self._fh = None
        self._buffer: List[str] = []
        # Prevent immediate flush on first row append.
        self._last_flush_ms = time.time() * 1000.0
        self._buffer_limit = 100
        self._start_ns = time.time_ns()

    def _ensure_open(self) -> bool:
        if self._fh is not None:
            return True
        try:
            os.makedirs(self._cfg.log_dir, exist_ok=True)
            need_header = not os.path.exists(self._path) or os.path.getsize(
                self._path) == 0
            self._fh = open(self._path, "a", encoding="utf-8")
            if need_header:
                self._fh.write(",".join(self._COLUMNS) + "\n")
                self._fh.flush()
            return True
        except Exception:
            self._fh = None
            return False

    def _maybe_flush(self, now_ms: float) -> None:
        if not self._fh:
            return
        if (now_ms - self._last_flush_ms >=
                float(self._cfg.ts_period_ms)) or (
                    len(self._buffer) >= self._buffer_limit):
            try:
                self._fh.writelines(self._buffer)
                self._fh.flush()
                self._buffer.clear()
                self._last_flush_ms = now_ms
            except Exception:
                return

    def append(self, row: Dict[str, Any]) -> None:
        if not self._cfg.obs_enabled:
            return
        if not self._ensure_open():
            return
        ts_ns = int(row.get("ts_ns", time.time_ns()))
        t_rel_s = (ts_ns - self._start_ns) / 1e9
        values = []
        for key in self._COLUMNS:
            if key == "ts_ns":
                values.append(str(ts_ns))
            elif key == "t_rel_s":
                values.append(f"{t_rel_s:.6f}")
            else:
                v = row.get(key, "")
                if v is None:
                    values.append("")
                else:
                    values.append(str(v))
        self._buffer.append(",".join(values) + "\n")
        now_ms = time.time() * 1000.0
        self._maybe_flush(now_ms)


@lru_cache(maxsize=1)
def _get_event_logger() -> _RecoveryEventLogger:
    return _RecoveryEventLogger(get_recovery_config())


def log_recovery_event(
    event: str,
    *,
    cycle_id: Optional[int] = None,
    req_id: Optional[str] = None,
    seq_id: Optional[int] = None,
    reason: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    mem_snapshot: Optional[Dict[str, Any]] = None,
    block_manager: Any = None,
) -> None:
    if mem_snapshot is None and block_manager is not None:
        mem_snapshot = _build_mem_snapshot(block_manager)
    payload: Dict[str, Any] = {
        "ts_ns": time.time_ns(),
        "event": event,
    }
    if cycle_id is not None:
        payload["cycle_id"] = cycle_id
    if req_id is not None:
        payload["req_id"] = req_id
    if seq_id is not None:
        payload["seq_id"] = seq_id
    if reason is not None:
        payload["reason"] = reason
    if detail is not None:
        payload["detail"] = detail
    if mem_snapshot is not None:
        payload["mem_snapshot"] = mem_snapshot
    _get_event_logger().log(payload)


@dataclass
class CycleCounters:
    swapin_blocks: int = 0
    swapout_blocks: int = 0
    swapin_bytes: int = 0
    swapout_bytes: int = 0
    restore_commit_blocks: int = 0
    x_swap: int = 0
    x_rec: int = 0
    recompute_tokens: int = 0
    over_budget_drop: int = 0

    def reset(self) -> None:
        self.swapin_blocks = 0
        self.swapout_blocks = 0
        self.swapin_bytes = 0
        self.swapout_bytes = 0
        self.restore_commit_blocks = 0
        self.x_swap = 0
        self.x_rec = 0
        self.recompute_tokens = 0
        self.over_budget_drop = 0

    def add_swap_in(self, blocks: int, bytes_count: Optional[int]) -> None:
        self.swapin_blocks += int(blocks)
        if bytes_count:
            self.swapin_bytes += int(bytes_count)

    def add_swap_out(self, blocks: int, bytes_count: Optional[int]) -> None:
        self.swapout_blocks += int(blocks)
        if bytes_count:
            self.swapout_bytes += int(bytes_count)

    def add_restore_commit(self, blocks: int) -> None:
        self.restore_commit_blocks += int(blocks)

    def add_swap_micro(self, count: int = 1) -> None:
        self.x_swap += int(count)

    def add_recompute(self, tokens: int = 0, tasks: int = 1) -> None:
        self.x_rec += int(tasks)
        self.recompute_tokens += int(tokens)

    def add_over_budget_drop(self, count: int = 1) -> None:
        self.over_budget_drop += int(count)


@lru_cache(maxsize=1)
def _get_cycle_counters() -> CycleCounters:
    return CycleCounters()


def reset_cycle_counters() -> None:
    _get_cycle_counters().reset()


def add_swap_in(blocks: int, bytes_count: Optional[int] = None) -> None:
    _get_cycle_counters().add_swap_in(blocks, bytes_count)


def add_swap_out(blocks: int, bytes_count: Optional[int] = None) -> None:
    _get_cycle_counters().add_swap_out(blocks, bytes_count)


def add_restore_commit(blocks: int) -> None:
    _get_cycle_counters().add_restore_commit(blocks)


def add_swap_micro(count: int = 1) -> None:
    _get_cycle_counters().add_swap_micro(count)


def add_recompute(tokens: int = 0, tasks: int = 1) -> None:
    _get_cycle_counters().add_recompute(tokens=tokens, tasks=tasks)


def add_over_budget_drop(count: int = 1) -> None:
    _get_cycle_counters().add_over_budget_drop(count)


def get_cycle_counters_snapshot() -> Dict[str, int]:
    c = _get_cycle_counters()
    return {
        "swapin_blocks": c.swapin_blocks,
        "swapout_blocks": c.swapout_blocks,
        "swapin_bytes": c.swapin_bytes,
        "swapout_bytes": c.swapout_bytes,
        "restore_commit_blocks_delta": c.restore_commit_blocks,
        "x_swap": c.x_swap,
        "x_rec": c.x_rec,
        "recompute_tokens": c.recompute_tokens,
        "over_budget_drop": c.over_budget_drop,
    }


class RecoveryObservability:

    def __init__(self) -> None:
        self._cfg = get_recovery_config()
        self._logger = _get_event_logger()
        self._csv_logger = _RecoveryCsvLogger(self._cfg)
        self._cycle_id = 0
        self._cycle_start_s: Optional[float] = None
        self._mode = "normal"
        self._mode_enter_ns = time.time_ns()
        self._mode_switches = 0
        self._prev_swapped_len = 0
        self._stall_cycles = 0
        self._stall_ms_proxy = 0.0

    @property
    def cycle_id(self) -> int:
        return self._cycle_id

    def _is_enabled(self) -> bool:
        return bool(self._cfg.obs_enabled)

    def log_request_event(
        self,
        event: str,
        *,
        req_id: str,
        seq_id: Optional[int] = None,
        reason: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        block_manager: Any = None,
    ) -> None:
        if not self._is_enabled():
            return
        log_recovery_event(
            event,
            cycle_id=self._cycle_id,
            req_id=req_id,
            seq_id=seq_id,
            reason=reason,
            detail=detail,
            block_manager=block_manager,
        )

    def _resolve_mode(self, cycle_context: Dict[str, Any]) -> str:
        preempted = int(cycle_context.get("preempted", 0))
        swapin_blocks = int(cycle_context.get("swapin_blocks", 0))
        swapout_blocks = int(cycle_context.get("swapout_blocks", 0))
        swapped_len = int(cycle_context.get("swapped_len", 0))
        if swapped_len > 0 and self._stall_ms_proxy >= self._cfg.stall_ms:
            return "fallback"
        if preempted > 0 or swapout_blocks > 0 or swapin_blocks > 0 or swapped_len > 0:
            return "recovery"
        return "normal"

    def on_cycle_begin(self, now_s: float) -> None:
        if not self._is_enabled():
            return
        self._cycle_id += 1
        self._cycle_start_s = now_s
        reset_cycle_counters()

    def on_cycle_end(self, now_s: float,
                     cycle_context: Optional[Dict[str, Any]] = None) -> None:
        if not self._is_enabled():
            return
        start_s = self._cycle_start_s
        self._cycle_start_s = None
        duration_ms = None
        if start_s is not None:
            duration_ms = max(0.0, (now_s - start_s) * 1000.0)
        payload: Dict[str, Any] = {
            "ts_ns": time.time_ns(),
            "event": "SCHED_CYCLE_SUMMARY",
            "cycle_id": self._cycle_id,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if cycle_context is None:
            cycle_context = {}
        cycle_context.update(get_cycle_counters_snapshot())
        cycle_context.update(get_cost_model_snapshot())
        if duration_ms is not None:
            cycle_context["cycle_wall_ms"] = duration_ms
        else:
            cycle_context.setdefault("cycle_wall_ms", 0.0)
        cycle_wall_ms = float(cycle_context.get("cycle_wall_ms") or 0.0)

        ts_ns = payload["ts_ns"]
        mode = self._resolve_mode(cycle_context)
        mode_switches_cycle = 0
        if mode != self._mode:
            prev_mode = self._mode
            prev_mode_dwell_ms = max(0.0, (ts_ns - self._mode_enter_ns) / 1e6)
            self._mode = mode
            self._mode_enter_ns = ts_ns
            self._mode_switches += 1
            mode_switches_cycle = 1
            log_recovery_event(
                "RECOVERY_MODE_SWITCH",
                cycle_id=self._cycle_id,
                reason="mode_change",
                detail={
                    "from_mode": prev_mode,
                    "to_mode": mode,
                    "prev_mode_dwell_ms": round(prev_mode_dwell_ms, 3),
                    "mode_switches": self._mode_switches,
                },
            )
        mode_resident_ms = max(0.0, (ts_ns - self._mode_enter_ns) / 1e6)
        cycle_context["mode"] = self._mode
        cycle_context["mode_resident_ms"] = round(mode_resident_ms, 3)
        cycle_context["mode_switches"] = self._mode_switches
        cycle_context["mode_switches_cycle"] = mode_switches_cycle

        swapped_len = int(cycle_context.get("swapped_len", 0))
        swapin_blocks = int(cycle_context.get("swapin_blocks", 0))
        has_recovery_pressure = swapped_len > 0 or self._prev_swapped_len > 0
        progressed = False
        stall_delta_ms = 0.0
        if has_recovery_pressure:
            progressed = (swapin_blocks > 0 or swapped_len < self._prev_swapped_len)
            if progressed:
                self._stall_cycles = 0
                self._stall_ms_proxy = 0.0
                log_recovery_event(
                    "RECOVERY_PROGRESS",
                    cycle_id=self._cycle_id,
                    detail={
                        "swapped_len": swapped_len,
                        "prev_swapped_len": self._prev_swapped_len,
                        "swapin_blocks": swapin_blocks,
                    },
                )
            elif swapped_len > 0:
                self._stall_cycles += 1
                self._stall_ms_proxy += cycle_wall_ms
                stall_delta_ms = cycle_wall_ms
                if self._stall_cycles == 1 or self._stall_cycles % 10 == 0:
                    log_recovery_event(
                        "RECOVERY_STALL",
                        cycle_id=self._cycle_id,
                        reason="no_progress",
                        detail={
                            "swapped_len": swapped_len,
                            "prev_swapped_len": self._prev_swapped_len,
                            "stall_cycles": self._stall_cycles,
                        },
                    )
        else:
            self._stall_cycles = 0
            self._stall_ms_proxy = 0.0
        self._prev_swapped_len = swapped_len
        cycle_context["recovery_progress"] = int(progressed)
        cycle_context["recovery_stall_cycles"] = self._stall_cycles
        cycle_context["restore_progress_stall_ms"] = round(self._stall_ms_proxy,
                                                           3)
        cycle_context["restore_progress_stall_ms_delta"] = round(stall_delta_ms,
                                                                 3)
        cycle_context["mode_cnt_normal"] = 1 if self._mode == "normal" else 0
        cycle_context["mode_cnt_recovery"] = 1 if self._mode == "recovery" else 0
        cycle_context["mode_cnt_fallback"] = 1 if self._mode == "fallback" else 0

        if cycle_context:
            payload["detail"] = cycle_context
        self._logger.log(payload)
        try:
            recovery_overhead_ms = float(cycle_context.get("recovery_overhead_ms",
                                                           0.0))
            t_on_ms = float(cycle_context.get(
                "T_on_ms", max(0.0, cycle_wall_ms - recovery_overhead_ms)))
            cycle_context["T_on_ms"] = round(t_on_ms, 3)
            s_ms = float(cycle_context.get("S_ms", max(0.0, cycle_wall_ms - t_on_ms)))
            row = {
                "ts_ns": payload["ts_ns"],
                "cycle_id": self._cycle_id,
                "T_cyc_ms": round(float(cycle_context.get("T_cyc_ms",
                                                          cycle_wall_ms)), 3),
                "on_preempt_count_delta": cycle_context.get(
                    "on_preempt_count_delta", cycle_context.get("preempted",
                                                                  0)),
                "on_waiting_len": cycle_context.get("on_waiting_len",
                                                    cycle_context.get(
                                                        "waiting_len", 0)),
                "T_on_ms": round(t_on_ms, 3),
                "S_ms": round(s_ms, 3),
                "B_ms": round(float(cycle_context.get("recovery_budget_ms",
                                                      0.0)), 3),
                "barT_swap": cycle_context.get("barT_swap_ms_per_block", 0.0),
                "barT_rec": cycle_context.get("barT_rec_ms_per_token", 0.0),
                "gain_swap": cycle_context.get("swap_gain_blocks_per_ms", 0.0),
                "gain_rec": cycle_context.get("rec_gain_tokens_per_ms", 0.0),
                "x_swap": cycle_context.get("x_swap", 0),
                "x_rec": cycle_context.get("x_rec", 0),
                "over_budget_drop": cycle_context.get("over_budget_drop", 0),
                "swapin_blocks": cycle_context.get("swapin_blocks", 0),
                "swapout_blocks": cycle_context.get("swapout_blocks", 0),
                "recompute_tokens": cycle_context.get("recompute_tokens", 0),
                "gpu_kv_blocks_free": cycle_context.get("gpu_kv_blocks_free",
                                                        0),
                "mode_cnt_normal": cycle_context.get("mode_cnt_normal", 1),
                "mode_cnt_recovery": cycle_context.get("mode_cnt_recovery", 0),
                "mode_cnt_fallback": cycle_context.get("mode_cnt_fallback", 0),
                "restore_progress_stall_ms": cycle_context.get(
                    "restore_progress_stall_ms", 0),
                "cycle_wall_ms": round(cycle_wall_ms, 3),
                "prefill_tokens": cycle_context.get("prefill_tokens", 0),
                "decode_tokens": cycle_context.get("decode_tokens", 0),
            }
            self._csv_logger.append(row)
        except Exception:
            pass
