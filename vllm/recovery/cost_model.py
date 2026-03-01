import time
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Dict

from vllm import envs


@dataclass
class _EmaState:
    value: float
    alpha: float = 0.1
    clip_low_ratio: float = 0.25
    clip_high_ratio: float = 4.0

    def update(self, sample: float) -> float:
        sample = float(sample)
        if sample <= 0.0:
            return self.value
        if self.value <= 0.0:
            self.value = sample
            return self.value
        lo = self.value * self.clip_low_ratio
        hi = self.value * self.clip_high_ratio
        clipped = min(max(sample, lo), hi)
        self.value = ((1.0 - self.alpha) * self.value) + (self.alpha * clipped)
        return self.value


class RecoveryCostModel:

    def __init__(self) -> None:
        init_swap = max(1e-6, float(envs.VLLM_RECOVERY_SWAP_MS_PER_BLOCK))
        self._swap = _EmaState(value=init_swap)
        # Conservative bootstrap for recompute token cost.
        self._rec = _EmaState(value=0.05)
        self._lock = Lock()
        self._updated_at_ns = time.time_ns()

    def observe_swap_task(self, task_ms: float, blocks_done: int) -> None:
        if blocks_done <= 0:
            return
        per_block = float(task_ms) / float(blocks_done)
        with self._lock:
            self._swap.update(per_block)
            self._updated_at_ns = time.time_ns()

    def observe_recompute_task(self, task_ms: float, tokens_done: int) -> None:
        if tokens_done <= 0:
            return
        per_token = float(task_ms) / float(tokens_done)
        with self._lock:
            self._rec.update(per_token)
            self._updated_at_ns = time.time_ns()

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            bar_t_swap = max(1e-6, self._swap.value)
            bar_t_rec = max(1e-6, self._rec.value)
            return {
                "barT_swap_ms_per_block": round(bar_t_swap, 6),
                "barT_rec_ms_per_token": round(bar_t_rec, 6),
                "swap_gain_blocks_per_ms": round(1.0 / bar_t_swap, 6),
                "rec_gain_tokens_per_ms": round(1.0 / bar_t_rec, 6),
                "cost_model_updated_at_ns": self._updated_at_ns,
            }


@lru_cache(maxsize=1)
def get_recovery_cost_model() -> RecoveryCostModel:
    return RecoveryCostModel()


def observe_swap_task(task_ms: float, blocks_done: int) -> None:
    get_recovery_cost_model().observe_swap_task(task_ms, blocks_done)


def observe_recompute_task(task_ms: float, tokens_done: int) -> None:
    get_recovery_cost_model().observe_recompute_task(task_ms, tokens_done)


def get_cost_model_snapshot() -> Dict[str, float]:
    return get_recovery_cost_model().snapshot()
