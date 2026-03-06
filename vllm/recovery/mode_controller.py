import time
from dataclasses import dataclass
from typing import Dict

from vllm.recovery.observability import RecoveryConfig


@dataclass(frozen=True)
class ModeSignals:
    has_recovery_work: bool
    waiting_len: int
    preempt_delta: int
    stalled: bool
    free_kv_blocks: int
    total_kv_blocks: int
    watermark_hi_blocks: int
    now_ns: int


class RecoveryModeController:
    """Phase3 global mode controller (NORMAL/RECOVERY/FALLBACK)."""

    def __init__(self, cfg: RecoveryConfig) -> None:
        self._cfg = cfg
        self._mode = "normal"
        self._mode_enter_ns = time.time_ns()
        self._preempt_streak = 0
        self._global_safe_since_ns = self._mode_enter_ns

    @property
    def mode(self) -> str:
        return self._mode

    def update(self, signals: ModeSignals) -> Dict[str, float]:
        now_ns = int(signals.now_ns)
        waiting_len = max(0, int(signals.waiting_len))
        preempt_delta = max(0, int(signals.preempt_delta))
        has_recovery_work = bool(signals.has_recovery_work)
        stalled = bool(signals.stalled)
        free_kv_blocks = max(0, int(signals.free_kv_blocks))
        total_kv_blocks = max(0, int(signals.total_kv_blocks))
        watermark_hi_blocks = max(0, int(signals.watermark_hi_blocks))
        if watermark_hi_blocks == 0 and total_kv_blocks > 0:
            # Fallback safe band when allocator watermark is unavailable.
            watermark_hi_blocks = max(1, total_kv_blocks // 20)
        mem_safe = (watermark_hi_blocks == 0
                    or free_kv_blocks >= watermark_hi_blocks)
        if mem_safe:
            if self._global_safe_since_ns <= 0:
                self._global_safe_since_ns = now_ns
        else:
            self._global_safe_since_ns = 0
        global_safe_ms = (max(0.0, (now_ns - self._global_safe_since_ns) / 1e6)
                          if self._global_safe_since_ns > 0 else 0.0)

        if preempt_delta > 0:
            # Count only recent/consecutive preempt pressure.
            self._preempt_streak += preempt_delta
            self._preempt_streak = min(self._preempt_streak, 64)
        else:
            # Drop stale history to avoid fallback oscillation triggered by
            # old bursts.
            self._preempt_streak = 0

        dwell_ms = max(0.0, (now_ns - self._mode_enter_ns) / 1e6)
        can_switch = dwell_ms >= float(self._cfg.mode_stable_window_ms)
        global_stable = global_safe_ms >= float(self._cfg.mode_stable_window_ms)
        target = self._mode
        reason = "hold"

        if self._mode == "normal":
            if has_recovery_work:
                target = "recovery"
                reason = "recovery_work_detected"
        elif self._mode == "recovery":
            if (not has_recovery_work) and mem_safe and global_stable:
                target = "normal"
                reason = "recovery_drained_global_stable"
            elif (not has_recovery_work) and (not mem_safe):
                reason = "wait_mem_safe_band"
            elif (not has_recovery_work) and (not global_stable):
                reason = "wait_global_stable_window"
            elif (stalled and dwell_ms >= float(self._cfg.fallback_stall_ms)):
                target = "fallback"
                reason = "stalled_recovery"
            elif (self._preempt_streak >=
                  max(1, int(self._cfg.fallback_preempt_threshold))
                  and waiting_len > 0):
                target = "fallback"
                reason = "preempt_streak"
        else:
            # fallback
            if (not has_recovery_work) and mem_safe and global_stable:
                target = "normal"
                reason = "fallback_drained_global_stable"
            elif (not has_recovery_work) and (not mem_safe):
                reason = "fallback_wait_mem_safe"
            elif (not has_recovery_work) and (not global_stable):
                reason = "fallback_wait_global_stable"
            elif (dwell_ms >= float(self._cfg.fallback_residence_ms)
                  and preempt_delta == 0 and (not stalled)):
                target = "recovery"
                reason = "fallback_residence_elapsed"

        switched = False
        if target != self._mode:
            # Entering recovery/fallback should be responsive; normal is slower.
            immediate_downward = (target == "fallback"
                                  or (self._mode == "normal"
                                      and target == "recovery"))
            if immediate_downward or can_switch:
                self._mode = target
                self._mode_enter_ns = now_ns
                switched = True
                dwell_ms = 0.0

        return {
            "mode": self._mode,
            "switched": float(1 if switched else 0),
            "mode_dwell_ms": round(dwell_ms, 3),
            "preempt_streak": float(self._preempt_streak),
            "mem_safe": float(1 if mem_safe else 0),
            "global_safe_ms": round(global_safe_ms, 3),
            "watermark_hi_blocks": float(watermark_hi_blocks),
            "allow_normal": float(1 if (mem_safe and global_stable) else 0),
            "reason": reason,
        }
