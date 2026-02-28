import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vllm.logger import init_logger
from vllm.recovery.observability import RecoveryConfig, get_recovery_config

if TYPE_CHECKING:
    from vllm.core.scheduler import Scheduler, SchedulerOutputs

logger = init_logger(__name__)


@dataclass
class CycleCounters:
    preemptions: int = 0
    preemptions_recompute: int = 0
    preemptions_swap: int = 0
    swapin_blocks: int = 0
    swapout_blocks: int = 0
    recompute_tokens: int = 0

    def reset(self) -> None:
        self.preemptions = 0
        self.preemptions_recompute = 0
        self.preemptions_swap = 0
        self.swapin_blocks = 0
        self.swapout_blocks = 0
        self.recompute_tokens = 0


class RecoveryObservability:
    _LOW_FREQ_JSONL_EVENTS = {
        "PREEMPT_TRIGGERED",
        "SWAP_IN",
        "SWAP_OUT",
    }
    _TS_HEADER = ",".join([
        "t_rel_s",
        "cycle_id",
        "on_preempt_count_delta",
        "on_waiting_len",
        "T_on_ms",
        "S_ms",
        "swapin_blocks",
        "swapout_blocks",
        "recompute_tokens",
        "gpu_kv_blocks_free",
        "mode_cnt_normal",
        "mode_cnt_recovery",
        "mode_cnt_fallback",
        "restore_progress_stall_ms",
    ])

    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.enabled = config.obs_enabled
        self._start_ts = time.time()
        self._last_ts_emit_ts = 0.0
        self._last_cycle_end_ts: Optional[float] = None
        self._cycle_wall_begin_ts: Optional[float] = None
        self._cycle_id = 0

        self._mode = "normal"
        self._mode_switches_total = 0
        self._mode_resident_ms = {
            "normal": 0.0,
            "recovery": 0.0,
            "fallback": 0.0,
        }
        self._stall_active = False
        self._stall_ms_total = 0.0

        self._cycle_begin_waiting = 0
        self._cycle_begin_running = 0
        self._cycle_begin_swapped = 0
        self._cycle_counters = CycleCounters()

        self._ts_fp = None
        self._events_fp = None
        self._ts_row_buffer: List[str] = []
        self._events_row_buffer: List[str] = []
        self._last_ts_flush_ts = self._start_ts
        self._last_events_flush_ts = self._start_ts
        self._active_scheduler: Optional["Scheduler"] = None

        if not self.enabled:
            return
        self._init_output_files()

    @classmethod
    def from_config(cls, config: Optional[RecoveryConfig] = None
                    ) -> "RecoveryObservability":
        return cls(config or get_recovery_config())

    def _init_output_files(self) -> None:
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
            if self.config.emit_ts_csv:
                ts_path = os.path.join(self.config.log_dir, "recovery_ts.csv")
                ts_exists = os.path.exists(ts_path) and os.path.getsize(
                    ts_path) > 0
                self._ts_fp = open(ts_path,
                                   "a",
                                   encoding="utf-8",
                                   buffering=1)
                if not ts_exists:
                    self._ts_fp.write(f"{self._TS_HEADER}\n")
                    self._ts_fp.flush()
            if self.config.emit_events_jsonl:
                events_path = os.path.join(self.config.log_dir,
                                           "recovery_events.jsonl")
                self._events_fp = open(events_path,
                                       "a",
                                       encoding="utf-8",
                                       buffering=1)
        except Exception as exc:
            logger.warning("Disable recovery observability due to IO error: %r",
                           exc)
            self.enabled = False
            self._ts_fp = None
            self._events_fp = None

    def _flush_ts_rows(self) -> None:
        if self._ts_fp is None or not self._ts_row_buffer:
            return
        self._ts_fp.writelines(self._ts_row_buffer)
        self._ts_fp.flush()
        self._ts_row_buffer.clear()

    def _flush_events_rows(self) -> None:
        if self._events_fp is None or not self._events_row_buffer:
            return
        self._events_fp.writelines(self._events_row_buffer)
        self._events_fp.flush()
        self._events_row_buffer.clear()

    def close(self) -> None:
        self._flush_ts_rows()
        self._flush_events_rows()
        if self._ts_fp is not None:
            self._ts_fp.close()
            self._ts_fp = None
        if self._events_fp is not None:
            self._events_fp.close()
            self._events_fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _append_ts_row(self, now_s: float, row: str) -> None:
        if self._ts_fp is None:
            return
        self._ts_row_buffer.append(row)
        if (len(self._ts_row_buffer) >= self.config.ts_flush_rows
                or (now_s - self._last_ts_flush_ts) * 1000.0 >=
                self.config.ts_period_ms):
            self._flush_ts_rows()
            self._last_ts_flush_ts = now_s

    def _append_event_row(self, now_s: float, row: str) -> None:
        if self._events_fp is None:
            return
        self._events_row_buffer.append(row)
        if (len(self._events_row_buffer) >= self.config.events_flush_rows
                or (now_s - self._last_events_flush_ts) * 1000.0 >=
                self.config.events_flush_period_ms):
            self._flush_events_rows()
            self._last_events_flush_ts = now_s

    def _build_mem_snapshot(self) -> Optional[Dict[str, Any]]:
        scheduler = self._active_scheduler
        if scheduler is None:
            return None
        block_manager = scheduler.block_manager
        try:
            gpu_free = block_manager.get_num_free_gpu_blocks()
            cpu_free = block_manager.get_num_free_cpu_blocks()
        except Exception:
            return None

        snapshot: Dict[str, Any] = {
            "gpu_free_blocks": gpu_free,
            "cpu_free_blocks": cpu_free,
        }

        gpu_total = getattr(block_manager, "num_total_gpu_blocks", None)
        cpu_total = getattr(block_manager, "num_total_cpu_blocks", None)
        if isinstance(gpu_total, int) and gpu_total >= 0:
            snapshot["gpu_total_blocks"] = gpu_total
            snapshot["gpu_used_blocks"] = max(0, gpu_total - gpu_free)
        if isinstance(cpu_total, int) and cpu_total >= 0:
            snapshot["cpu_total_blocks"] = cpu_total
            snapshot["cpu_used_blocks"] = max(0, cpu_total - cpu_free)

        gpu_watermark_blocks = getattr(block_manager, "watermark_blocks", None)
        if isinstance(gpu_watermark_blocks, int):
            snapshot["gpu_watermark_blocks"] = gpu_watermark_blocks
            snapshot["gpu_free_le_watermark"] = (gpu_free <=
                                                 gpu_watermark_blocks)

        return snapshot

    def _emit_event(self,
                    event: str,
                    *,
                    req_id: Optional[str] = None,
                    seq_ids: Optional[List[int]] = None,
                    reason: Optional[str] = None,
                    detail: Optional[Dict[str, Any]] = None,
                    with_mem_snapshot: bool = True) -> None:
        if (not self.enabled or not self.config.emit_events_jsonl
                or self._events_fp is None):
            return
        if (self.config.events_low_freq_only
                and event not in self._LOW_FREQ_JSONL_EVENTS):
            return
        now_s = time.time()
        rec = {
            "ts_ns": int(now_s * 1e9),
            "cycle_id": self._cycle_id,
            "event": event,
            "recovery_phase": self.config.recovery_phase,
        }
        if req_id is not None:
            rec["req_id"] = req_id
        if seq_ids is not None:
            rec["seq_ids"] = seq_ids
        if reason is not None:
            rec["reason"] = reason
        if detail is not None:
            rec["detail"] = detail
        if with_mem_snapshot:
            mem_snapshot = self._build_mem_snapshot()
            if mem_snapshot is not None:
                rec["mem_snapshot"] = mem_snapshot
        self._append_event_row(now_s, f"{json.dumps(rec, ensure_ascii=True)}\n")

    def begin_cycle(self, scheduler: "Scheduler") -> None:
        self.on_cycle_begin(time.time(), scheduler)

    def on_cycle_begin(self, now_s: float, scheduler: "Scheduler") -> None:
        if not self.enabled:
            return
        self._active_scheduler = scheduler
        self._cycle_wall_begin_ts = now_s
        self._cycle_id += 1
        self._cycle_begin_waiting = len(scheduler.waiting)
        self._cycle_begin_running = len(scheduler.running)
        self._cycle_begin_swapped = len(scheduler.swapped)
        self._cycle_counters.reset()

    def note_preemption(self,
                        request_id: str,
                        mode: str,
                        reason: str,
                        seq_ids: Optional[List[int]] = None,
                        seq_group: Optional[Any] = None) -> None:
        if not self.enabled:
            return
        self._cycle_counters.preemptions += 1
        if mode == "recompute":
            self._cycle_counters.preemptions_recompute += 1
        elif mode == "swap":
            self._cycle_counters.preemptions_swap += 1
        if seq_group is not None:
            recovery_obs = getattr(seq_group, "recovery_obs", None)
            if recovery_obs is not None:
                recovery_obs.preempt_cnt += 1
        if self.config.emit_request_events:
            self._emit_event(
                "PREEMPT_TRIGGERED",
                req_id=request_id,
                seq_ids=seq_ids,
                reason=reason,
                detail={"mode": mode},
            )

    def note_swap_in(self,
                     request_id: str,
                     blocks: int,
                     bytes_count: Optional[int] = None,
                     seq_ids: Optional[List[int]] = None,
                     reason: Optional[str] = None) -> None:
        if not self.enabled:
            return
        blocks = max(0, blocks)
        if bytes_count is None:
            bytes_count = blocks * self.config.swap_bytes_per_block
        bytes_count = max(0, bytes_count)
        self._cycle_counters.swapin_blocks += blocks
        if self.config.emit_request_events:
            self._emit_event(
                "SWAP_IN",
                req_id=request_id,
                seq_ids=seq_ids,
                reason=reason,
                detail={
                    "blocks": blocks,
                    "bytes": bytes_count,
                },
            )

    def note_swap_out(self,
                      request_id: str,
                      blocks: int,
                      bytes_count: Optional[int] = None,
                      seq_ids: Optional[List[int]] = None,
                      reason: Optional[str] = None) -> None:
        if not self.enabled:
            return
        blocks = max(0, blocks)
        if bytes_count is None:
            bytes_count = blocks * self.config.swap_bytes_per_block
        bytes_count = max(0, bytes_count)
        self._cycle_counters.swapout_blocks += blocks
        if self.config.emit_request_events:
            self._emit_event(
                "SWAP_OUT",
                req_id=request_id,
                seq_ids=seq_ids,
                reason=reason,
                detail={
                    "blocks": blocks,
                    "bytes": bytes_count,
                },
            )

    def note_recompute_tokens(self, request_id: str, tokens: int) -> None:
        if not self.enabled:
            return
        delta = max(0, tokens)
        self._cycle_counters.recompute_tokens += delta
        if self.config.emit_request_events:
            self._emit_event(
                "RECOMPUTE_ACCOUNTING",
                req_id=request_id,
                detail={"tokens": delta},
            )

    def on_block_swap_event(self,
                            event: str,
                            seq_group: Any,
                            seq_id: int,
                            blocks: int,
                            reason: Optional[str] = None,
                            bytes_count: Optional[int] = None) -> None:
        request_id = getattr(seq_group, "request_id", "")
        recovery_obs = getattr(seq_group, "recovery_obs", None)
        if event == "swap_in":
            if recovery_obs is not None:
                recovery_obs.swapin_blocks_total += max(0, blocks)
            self.note_swap_in(request_id=request_id,
                              blocks=blocks,
                              bytes_count=bytes_count,
                              seq_ids=[seq_id],
                              reason=reason)
        elif event == "swap_out":
            if recovery_obs is not None:
                recovery_obs.swapout_blocks_total += max(0, blocks)
            self.note_swap_out(request_id=request_id,
                               blocks=blocks,
                               bytes_count=bytes_count,
                               seq_ids=[seq_id],
                               reason=reason)

    def end_cycle(self, scheduler: "Scheduler", scheduler_outputs: "SchedulerOutputs",
                  scheduler_time_s: float) -> None:
        self.on_cycle_end(
            time.time(),
            {
                "scheduler": scheduler,
                "scheduler_outputs": scheduler_outputs,
                "scheduler_time_s": scheduler_time_s,
            },
        )

    def on_cycle_end(self, now_s: float, cycle_context: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        scheduler = cycle_context["scheduler"]
        scheduler_outputs = cycle_context["scheduler_outputs"]
        scheduler_time_s = float(cycle_context.get("scheduler_time_s", 0.0))
        now = now_s
        if self._last_cycle_end_ts is None:
            cycle_dt_ms = scheduler_time_s * 1000.0
        else:
            cycle_dt_ms = max(0.0, (now - self._last_cycle_end_ts) * 1000.0)
        self._last_cycle_end_ts = now
        if self._cycle_wall_begin_ts is not None:
            cycle_wall_ms = max(0.0, (now - self._cycle_wall_begin_ts) * 1000.0)
        else:
            cycle_wall_ms = scheduler_time_s * 1000.0

        mode = self._derive_mode(scheduler)
        mode_switched = 0
        if mode != self._mode:
            mode_switched = 1
            self._mode_switches_total += 1
            if self.config.emit_cycle_events:
                self._emit_event(
                    "MODE_SWITCH",
                    detail={
                        "from_mode": self._mode,
                        "to_mode": mode,
                        "mode_switches_total": self._mode_switches_total,
                    },
                )
        self._mode_resident_ms[mode] += cycle_dt_ms
        self._mode = mode

        swapped_now = len(scheduler.swapped)
        waiting_now = len(scheduler.waiting)
        progress_score = max(0, self._cycle_begin_swapped - swapped_now)
        progress_score += max(0, self._cycle_begin_waiting - waiting_now)
        progress_score += self._cycle_counters.swapin_blocks
        stall_cycle_ms = 0.0
        if swapped_now > 0 and progress_score == 0:
            stall_cycle_ms = cycle_dt_ms
            self._stall_ms_total += stall_cycle_ms
            if not self._stall_active and self.config.emit_cycle_events:
                self._emit_event("RECOVERY_STALL_START")
            self._stall_active = True
        elif self._stall_active:
            if self.config.emit_cycle_events:
                self._emit_event("RECOVERY_STALL_END")
            self._stall_active = False

        running_now = len(scheduler.running)
        online_load_est = running_now + waiting_now + swapped_now
        seq_slack_est = max(0, scheduler.scheduler_config.max_num_seqs -
                            running_now)
        token_slack_est = max(
            0, scheduler.scheduler_config.max_num_batched_tokens -
            scheduler_outputs.num_batched_tokens)
        gpu_kv_blocks_free = scheduler.block_manager.get_num_free_gpu_blocks()
        cpu_kv_blocks_free = scheduler.block_manager.get_num_free_cpu_blocks()
        # Phase-0: recovery overhead isolation is not enabled yet.
        recovery_overhead_ms = 0.0
        t_on_ms = max(0.0, cycle_wall_ms - recovery_overhead_ms)
        s_ms = max(0.0, cycle_wall_ms - t_on_ms)
        prefill_tokens = 0
        decode_tokens = 0
        num_prefill_groups = scheduler_outputs.num_prefill_groups
        for idx, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            if idx < num_prefill_groups:
                prefill_tokens += scheduled_seq_group.token_chunk_size
            else:
                decode_tokens += scheduled_seq_group.token_chunk_size
        num_decode_groups = max(0, len(scheduler_outputs.scheduled_seq_groups) -
                                num_prefill_groups)
        # Phase-0 reserve fields. Keep fixed schema for downstream parsers.
        recompute_tokens_for_ts = 0
        mode_cnt_normal = 1
        mode_cnt_recovery = 0
        mode_cnt_fallback = 0

        scheduler_outputs.recovery_swapin_blocks = self._cycle_counters.swapin_blocks
        scheduler_outputs.recovery_swapout_blocks = self._cycle_counters.swapout_blocks
        scheduler_outputs.recovery_recompute_tokens = 0
        scheduler_outputs.recovery_restore_progress_stall_ms = stall_cycle_ms
        scheduler_outputs.recovery_mode = self._mode
        scheduler_outputs.recovery_mode_switches_cycle = mode_switched
        scheduler_outputs.recovery_mode_switches_total = (
            self._mode_switches_total)
        scheduler_outputs.recovery_online_load_est = online_load_est
        scheduler_outputs.recovery_seq_slack_est = seq_slack_est
        scheduler_outputs.recovery_token_slack_est = token_slack_est
        scheduler_outputs.recovery_gpu_kv_blocks_free = gpu_kv_blocks_free
        scheduler_outputs.recovery_cpu_kv_blocks_free = cpu_kv_blocks_free

        if self.config.emit_cycle_events:
            self._emit_event(
                "SCHED_CYCLE_SUMMARY",
                detail={
                    "scheduler_time_ms": scheduler_time_s * 1000.0,
                    "cycle_wall_ms": cycle_wall_ms,
                    "recovery_overhead_ms": recovery_overhead_ms,
                    "mode": mode,
                    "queue_waiting_begin": self._cycle_begin_waiting,
                    "queue_running_begin": self._cycle_begin_running,
                    "queue_swapped_begin": self._cycle_begin_swapped,
                    "req_waiting": waiting_now,
                    "req_running": running_now,
                    "req_swapped": swapped_now,
                    "num_prefill_groups": num_prefill_groups,
                    "num_decode_groups": num_decode_groups,
                    "num_scheduled_groups":
                    len(scheduler_outputs.scheduled_seq_groups),
                    "num_batched_tokens": scheduler_outputs.num_batched_tokens,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "preemptions_cycle": self._cycle_counters.preemptions,
                    "preemptions_cycle_recompute":
                    self._cycle_counters.preemptions_recompute,
                    "preemptions_cycle_swap": self._cycle_counters.preemptions_swap,
                    "swapin_blocks_cycle": self._cycle_counters.swapin_blocks,
                    "swapout_blocks_cycle": self._cycle_counters.swapout_blocks,
                    "recompute_tokens_cycle": self._cycle_counters.recompute_tokens,
                    "on_preempt_count_delta": self._cycle_counters.preemptions,
                    "on_waiting_len": waiting_now,
                    "T_on_ms": t_on_ms,
                    "S_ms": s_ms,
                    "swapin_blocks": self._cycle_counters.swapin_blocks,
                    "swapout_blocks": self._cycle_counters.swapout_blocks,
                    "recompute_tokens": recompute_tokens_for_ts,
                    "mode_cnt_normal": mode_cnt_normal,
                    "mode_cnt_recovery": mode_cnt_recovery,
                    "mode_cnt_fallback": mode_cnt_fallback,
                    "restore_progress_stall_ms": stall_cycle_ms,
                    "restore_progress_stall_ms_cycle": stall_cycle_ms,
                    "restore_progress_stall_ms_total": self._stall_ms_total,
                    "online_load_est": online_load_est,
                    "seq_slack_est": seq_slack_est,
                    "token_slack_est": token_slack_est,
                    "gpu_kv_blocks_free": gpu_kv_blocks_free,
                    "cpu_kv_blocks_free": cpu_kv_blocks_free,
                },
            )

        if self.config.emit_ts_csv and self._should_emit_ts(now):
            if self._ts_fp is not None:
                row = [
                    f"{(now - self._start_ts):.6f}",
                    str(self._cycle_id),
                    str(self._cycle_counters.preemptions),
                    str(waiting_now),
                    f"{t_on_ms:.3f}",
                    f"{s_ms:.3f}",
                    str(self._cycle_counters.swapin_blocks),
                    str(self._cycle_counters.swapout_blocks),
                    str(recompute_tokens_for_ts),
                    str(gpu_kv_blocks_free),
                    str(mode_cnt_normal),
                    str(mode_cnt_recovery),
                    str(mode_cnt_fallback),
                    f"{stall_cycle_ms:.3f}",
                ]
                self._append_ts_row(now, ",".join(row) + "\n")
            self._last_ts_emit_ts = now
        if self._ts_row_buffer and (now - self._last_ts_flush_ts
                                    ) * 1000.0 >= self.config.ts_period_ms:
            self._flush_ts_rows()
            self._last_ts_flush_ts = now
        if self._events_row_buffer and (
                now - self._last_events_flush_ts
        ) * 1000.0 >= self.config.events_flush_period_ms:
            self._flush_events_rows()
            self._last_events_flush_ts = now
        self._active_scheduler = None

    def _should_emit_ts(self, now: float) -> bool:
        if self._last_ts_emit_ts <= 0:
            return True
        return ((now - self._last_ts_emit_ts) * 1000.0
                >= self.config.ts_period_ms)

    def _derive_mode(self, scheduler: "Scheduler") -> str:
        if self._cycle_counters.preemptions > 0 or len(scheduler.swapped) > 0:
            return "recovery"
        return "normal"
