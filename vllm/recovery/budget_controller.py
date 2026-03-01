from dataclasses import dataclass
from typing import Dict

from vllm.recovery.observability import RecoveryConfig


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


@dataclass(frozen=True)
class BudgetSignals:
    slack_ms: float
    preempt_delta: int
    free_kv_blocks: int
    total_kv_blocks: int
    waiting_len: int
    stalled: bool


class BudgetController:

    def __init__(self, cfg: RecoveryConfig) -> None:
        self._b_min = max(0.0, float(cfg.budget_min_ms))
        self._b_max = max(self._b_min, float(cfg.target_cycle_ms))
        init_ms = max(self._b_min, float(cfg.budget_init_ms))
        self._b_target_ms = _clamp(init_ms, self._b_min, self._b_max)
        self._d_plus = max(0.0, float(cfg.dplus_ms))
        # Enforce hysteresis asymmetry to suppress oscillation.
        self._d_minus = max(float(cfg.dminus_ms), self._d_plus + 1e-6)

        self._free_ema = 1.0
        self._preempt_ema = 0.0
        self._alpha = 0.2
        self._low_free_ratio = 0.10
        self._recover_free_ratio = 0.20

    def update(self, signals: BudgetSignals) -> Dict[str, float]:
        slack_ms = max(0.0, float(signals.slack_ms))
        preempt_delta = max(0, int(signals.preempt_delta))
        free_blocks = max(0, int(signals.free_kv_blocks))
        total_blocks = max(1, int(signals.total_kv_blocks))
        waiting_len = max(0, int(signals.waiting_len))
        free_ratio = float(free_blocks) / float(total_blocks)

        self._free_ema = ((1.0 - self._alpha) * self._free_ema +
                          self._alpha * free_ratio)
        self._preempt_ema = ((1.0 - self._alpha) * self._preempt_ema +
                             self._alpha * float(preempt_delta))

        pressure_up = (waiting_len > 0 or preempt_delta > 0 or
                       self._preempt_ema > 0.5 or
                       free_ratio <= self._low_free_ratio or
                       self._free_ema <= self._low_free_ratio)
        pressure_down = (waiting_len == 0 and preempt_delta == 0 and
                         self._preempt_ema < 0.1 and
                         free_ratio >= self._recover_free_ratio and
                         self._free_ema >= self._recover_free_ratio)

        if pressure_up:
            delta = -self._d_minus
            pressure_state = "up"
        elif pressure_down:
            delta = self._d_plus
            pressure_state = "down"
        else:
            delta = 0.0
            pressure_state = "hold"

        self._b_target_ms = _clamp(self._b_target_ms + delta, self._b_min,
                                   self._b_max)
        if signals.stalled:
            self._b_target_ms = max(self._b_target_ms, self._b_min)

        b_ms = min(slack_ms, self._b_target_ms)
        return {
            "B_target_ms": round(self._b_target_ms, 3),
            "B_ms": round(b_ms, 3),
            "pressure_state": pressure_state,
            "delta_ms": round(delta, 3),
            "free_ratio": round(free_ratio, 6),
            "free_ema": round(self._free_ema, 6),
            "preempt_ema": round(self._preempt_ema, 6),
        }
