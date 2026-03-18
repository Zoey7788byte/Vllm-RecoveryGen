import enum
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.recovery.budget_controller import BudgetController, BudgetSignals
from vllm.recovery.cost_model import (get_cost_model_snapshot,
                                      observe_recompute_task)
from vllm.recovery.mode_controller import ModeSignals, RecoveryModeController
from vllm.recovery.observability import (RecoveryObservability, add_over_budget_drop,
                                         add_recompute, get_recovery_config,
                                         log_recovery_event)
from vllm.sequence import (RecoveryMode, RecoveryStorageHint, Sequence,
                           SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500
RECOVERY_RECOMPUTE_DELTA_TOKENS = 128
# Small stalled threshold for recovery candidate priority (Patch 2).
RECOVERY_PRIORITY_STALL_MS = 200.0
RECOVERY_SWAP_UNLOCK_RESERVE_BLOCKS = 1


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    # Number of cached tokens in the batch.
    _num_cached_tokens: int = 0
    # Number of actual non-cached tokens in the batch.
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        # We allow num_new_tokens to be 0 when the entire sequence has
        # been cached.
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self,
                               req_id: str,
                               num_batched_tokens: int,
                               num_cached_tokens: int = 0):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                # Sort sequence groups so that all prefills come before all
                # decodes as required by chunked prefill.
                return (not group.seq_group.is_prefill(), *key)
            return key

        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups,
                                           key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    # Optimization for fast-access to seq_group lists
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[ScheduledSequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


@dataclass
class _RecoveryTask:
    kind: str
    seq_group: SequenceGroup
    seq: Optional[Sequence]
    est_cost_ms: float
    gain_density: float
    priority: float
    delta_tokens: int = 0
    remaining_host_blocks: int = 0
    stalled: bool = False


def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id="",
                                 is_prompt=False,
                                 seq_data={},
                                 sampling_params=None,
                                 block_tables={})


def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[],
                                   prefill_seq_groups=[],
                                   preempted=[],
                                   swapped_out=[],
                                   blocks_to_swap_out=[],
                                   blocks_to_copy=[],
                                   num_lookahead_slots=0,
                                   prefill_seq_groups_list=[],
                                   decode_seq_groups_list=[])


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup),
                                  token_chunk_size=0)
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "selfattn"
        if (self.scheduler_config.runner_type == "pooling"
                or self.cache_config.is_attention_free):
            version = "placeholder"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0
        self._recovery_obs = RecoveryObservability()
        self._recovery_stall_logged_ns: Dict[int, int] = {}
        self._recovery_cfg = get_recovery_config()
        self._recovery_last_cycle_wall_ms: float = 0.0
        self._recovery_last_budget_blocks: int = 0
        self._recovery_last_budget_ms: float = 0.0
        self._recovery_last_slack_ms_est: float = 0.0
        self._recovery_last_seq_slack_est: int = 0
        self._recovery_last_token_slack_est: int = 0
        self._recovery_last_online_load_est: int = 0
        self._recovery_budget_ms_state: float = 0.0
        self._recovery_online_only_pass: bool = False
        self._recovery_budget_controller = BudgetController(self._recovery_cfg)
        self._recovery_last_budget_target_ms: float = 0.0
        self._recovery_mode_controller = RecoveryModeController(self._recovery_cfg)
        self._recovery_mode: str = "normal"
        self._recovery_cycle_idx: int = 0
        self._recovery_pause_decode_active: bool = False
        self._recovery_admission_throttle_cycle: int = 0
        self._recovery_mws_pinned_blocks_total: int = 0
        self._recovery_mws_limit_blocks: int = 0
        # Per-request anti-starvation state for tail recovery tasks.
        self._recovery_req_no_progress_cycles: Dict[str, int] = {}
        # Use persistent waiting pressure instead of single-cycle spikes.
        self._recovery_waiting_streak: int = 0
        self._recovery_waiting_streak_required: int = 2
        self._recovery_waiting_pressure_headroom_margin: int = 8
        self._recovery_force_progress_cycles: int = max(
            2,
            int(float(self._recovery_cfg.stall_ms) / max(
                1.0, float(self._recovery_cfg.target_cycle_ms))),
        )
        self._recovery_force_max_tasks_per_cycle: int = 1
        # Keep forced recompute chunks small enough to fit tight fallback
        # budgets (e.g., 1.0~1.5ms) and guarantee progress.
        self._recovery_force_recompute_tokens: int = 16
        self._recovery_prev_preempt_delta: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = \
            self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            num_uncached_new_tokens, _ = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.RUNNING, enable_chunking,
                    budget))

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if self.use_async_output_proc and seq_group.seqs[0].get_len(
            ) > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    victim_seq_group: Optional[SequenceGroup] = None
                    # Under phase3 swap preemption, prefer victims that can
                    # actually swap out with MWS pin constraints; otherwise a
                    # single unlucky victim can turn preempt into recompute-only
                    # and collapse swap/visibility signals.
                    if (self._recovery_cfg.phase >= 3
                            and self.user_specified_preemption_mode == "swap"):
                        for cand in reversed(running_queue):
                            if self._can_swap_out_with_phase3_pin(cand):
                                victim_seq_group = cand
                                running_queue.remove(cand)
                                break
                    if victim_seq_group is None:
                        # Preempt the lowest-priority sequence group.
                        victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(
                        victim_seq_group,
                        blocks_to_swap_out,
                        reason="no_append_slots",
                    )
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped
        k_budget_total = self._estimate_recovery_swap_budget(budget)
        k_budget_remaining = k_budget_total
        if (not self._recovery_online_only_pass) and k_budget_total <= 0 and len(
                swapped_queue) > 0:
            add_over_budget_drop(len(swapped_queue))

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]
            needs_recovery = False
            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                if seq.recovery_state.needs_recovery():
                    needs_recovery = True
                    break
            if needs_recovery and k_budget_remaining <= 0:
                if self._can_run_with_visibility_contract(seq_group):
                    needs_recovery = False
                else:
                    if self._recovery_online_only_pass:
                        try:
                            prefetch_fn = getattr(self.block_manager,
                                                  "prefetch_swap_in", None)
                            if callable(prefetch_fn):
                                prefetch_fn(seq_group, k_prefetch_max=4)
                        except Exception:
                            pass
                    swapped_queue.popleft()
                    leftover_swapped.appendleft(seq_group)
                    if not self._recovery_online_only_pass:
                        add_over_budget_drop(1)
                    continue

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            k_swap_plan_max = (k_budget_remaining if needs_recovery else 0)
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking),
                k_swap_max=k_swap_plan_max)
            use_unlock_swap = False
            if (alloc_status == AllocStatus.LATER and needs_recovery
                    and k_budget_remaining > 0
                    and self._is_seq_group_stalled_for_recovery_priority(
                        seq_group)):
                alloc_status = self.block_manager.can_swap_in(
                    seq_group,
                    self._get_num_lookahead_slots(is_prefill, enable_chunking),
                    k_swap_max=1,
                    stalled=True,
                    reserve_blocks=RECOVERY_SWAP_UNLOCK_RESERVE_BLOCKS,
                )
                if alloc_status != AllocStatus.NEVER:
                    use_unlock_swap = True
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            k_swap_exec_max = (1 if use_unlock_swap else
                               (k_budget_remaining if needs_recovery else 0))
            k_done, fully_restored = self._swap_in(seq_group,
                                                   blocks_to_swap_in,
                                                   k_swap_exec_max)
            k_budget_remaining = max(0, k_budget_remaining - max(0, k_done))
            if not fully_restored:
                if self._can_run_with_visibility_contract(seq_group):
                    try:
                        seq0 = seq_group.seqs[0] if seq_group.seqs else None
                        log_recovery_event(
                            "VISIBILITY_CONTRACT_ADMIT",
                            req_id=seq_group.request_id,
                            seq_id=(seq0.seq_id if seq0 is not None else None),
                            reason="mws_ready_partial_restore",
                            detail={
                                "k_done": k_done,
                                "k_budget_remaining": k_budget_remaining,
                            },
                            block_manager=self.block_manager,
                        )
                    except Exception:
                        pass
                else:
                    if k_done == 0:
                        break
                    swapped_queue.popleft()
                    leftover_swapped.appendleft(seq_group)
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.SWAPPED, enable_chunking,
                    budget))

            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                swapped_queue.popleft()
                leftover_swapped.appendleft(seq_group)
                continue

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                seq.status = SequenceStatus.RUNNING
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group,
                        token_chunk_size=num_new_tokens_uncached +
                        num_new_tokens_cached,
                    ))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        if (not self._recovery_online_only_pass) and k_budget_remaining == 0:
            dropped = len(swapped_queue) + len(leftover_swapped)
            if dropped > 0:
                add_over_budget_drop(dropped)
        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _estimate_recovery_swap_budget(self, budget: SchedulingBudget) -> int:
        cfg = self._recovery_cfg
        fallback_budget = cfg.budget if cfg.budget > 0 else 1
        manual_budget_cap = cfg.budget if cfg.budget > 1 else 0
        phase2_enabled = (cfg.budget == 1) or (cfg.phase >= 2 and cfg.budget != 0)
        online_load = len(self.waiting) + len(self.running) + len(self.swapped)
        seq_slack = max(0, self.scheduler_config.max_num_seqs - budget.num_curr_seqs)
        token_slack = max(0,
                          self.scheduler_config.max_num_batched_tokens -
                          budget.num_batched_tokens)
        self._recovery_last_online_load_est = online_load
        self._recovery_last_seq_slack_est = seq_slack
        self._recovery_last_token_slack_est = token_slack
        if self._recovery_online_only_pass:
            self._recovery_last_budget_blocks = 0
            self._recovery_last_budget_ms = 0.0
            self._recovery_last_slack_ms_est = 0.0
            return 0

        # Phase0/1 compatibility: fixed swap budget behavior.
        if not phase2_enabled:
            self._recovery_last_budget_blocks = fallback_budget
            self._recovery_last_budget_ms = fallback_budget * cfg.swap_ms_per_block
            self._recovery_last_slack_ms_est = 0.0
            return fallback_budget

        if len(self.swapped) == 0:
            self._recovery_last_budget_blocks = 0
            self._recovery_last_budget_ms = 0.0
            self._recovery_last_slack_ms_est = 0.0
            return 0

        # Phase2: derive recovery budget from slack and apply smooth hysteresis.
        last_cycle_ms = self._recovery_last_cycle_wall_ms
        if last_cycle_ms <= 0:
            last_cycle_ms = cfg.target_cycle_ms * 0.8
        slack_ms = max(0.0, cfg.target_cycle_ms - last_cycle_ms)

        target_budget_ms = max(cfg.budget_min_ms, slack_ms)
        if seq_slack <= 0 or token_slack <= 0:
            target_budget_ms = cfg.budget_min_ms

        if self._recovery_budget_ms_state <= 0:
            self._recovery_budget_ms_state = max(cfg.budget_init_ms,
                                                 cfg.budget_min_ms)

        if target_budget_ms > self._recovery_budget_ms_state:
            self._recovery_budget_ms_state += min(cfg.dplus_ms,
                                                  target_budget_ms -
                                                  self._recovery_budget_ms_state)
        else:
            self._recovery_budget_ms_state -= min(cfg.dminus_ms,
                                                  self._recovery_budget_ms_state -
                                                  target_budget_ms)
        self._recovery_budget_ms_state = max(cfg.budget_min_ms,
                                             self._recovery_budget_ms_state)

        est_blocks = int(self._recovery_budget_ms_state / cfg.swap_ms_per_block)
        est_blocks = max(cfg.min_budget, est_blocks)
        if manual_budget_cap > 0:
            est_blocks = min(est_blocks, manual_budget_cap)
        est_blocks = max(0, est_blocks)
        budget_ms = est_blocks * cfg.swap_ms_per_block

        self._recovery_last_budget_blocks = est_blocks
        self._recovery_last_budget_ms = budget_ms
        self._recovery_last_slack_ms_est = slack_ms
        try:
            log_recovery_event(
                "RECOVERY_BUDGET_DECISION",
                detail={
                    "phase": cfg.phase,
                    "phase2_enabled": int(phase2_enabled),
                    "k_swap_max": est_blocks,
                    "budget_ms": round(budget_ms, 3),
                    "budget_ms_target": round(target_budget_ms, 3),
                    "budget_ms_state": round(self._recovery_budget_ms_state, 3),
                    "slack_ms_est": round(slack_ms, 3),
                    "last_cycle_wall_ms": round(self._recovery_last_cycle_wall_ms,
                                                3),
                    "target_cycle_ms": round(cfg.target_cycle_ms, 3),
                    "swap_ms_per_block": cfg.swap_ms_per_block,
                    "min_budget": cfg.min_budget,
                    "budget_min_ms": cfg.budget_min_ms,
                    "budget_init_ms": cfg.budget_init_ms,
                    "dplus_ms": cfg.dplus_ms,
                    "dminus_ms": cfg.dminus_ms,
                    "manual_budget_cap": manual_budget_cap,
                    "online_load_est": online_load,
                    "seq_slack_est": seq_slack,
                    "token_slack_est": token_slack,
                },
                block_manager=self.block_manager,
            )
        except Exception:
            pass
        return est_blocks

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled and \
                not self.scheduler_config.is_multi_step:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _get_priority(self,
                      seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        """ Get the priority of the sequence group.
        Highest preference to user-defined priority, followed by arrival time.
        Args:
            seq_group: The sequence group input.
        Returns:
            The priority of the sequence group.
        """
        return seq_group.priority, seq_group.arrival_time

    def _schedule_priority_preemption(
        self,
        budget: SchedulingBudget,
    ) -> int:
        """Sorts waiting and running queue. Also, force preempt requests
        from the running queue if their priority is lower.
        Priority-based preemption is used with the priority policy.
        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
        Returns:
            A count of priority-based preemptions.
        """

        waiting_queue = self.waiting

        running_queue = deque(sorted(self.running, key=self._get_priority))

        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0

        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, _ = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, False, budget))

            #Only preempt if priority inversion exists
            while running_queue and self._get_priority(
                    running_queue[-1]) > self._get_priority(seq_group):
                #Only preempt if waiting sequence cannot be allocated
                can_allocate = self.block_manager.can_allocate(seq_group)
                if (num_new_tokens_uncached > 0
                        and can_allocate == AllocStatus.OK
                        and budget.can_schedule(
                            num_new_tokens=num_new_tokens_uncached,
                            num_new_seqs=num_new_seqs,
                        )):
                    break

                #Adjust budget to remove the victim sequence group
                vseq_group = running_queue.pop()
                num_running_tokens_uncached, _ = (
                    self._get_num_new_uncached_and_cached_tokens(
                        vseq_group, SequenceStatus.RUNNING, False, budget))
                budget.subtract_num_batched_tokens(
                    vseq_group.request_id, num_running_tokens_uncached)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id,
                                         num_running_seqs)

                #Preempt out the victim sequence group
                self._preempt(
                    vseq_group,
                    blocks_to_swap_out,
                    reason="priority_preemption",
                )
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            #Put the sequence back into the waiting queue
            waiting_queue.appendleft(seq_group)

        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))

        self.waiting = waiting_queue
        self.running = running_queue
        return force_preemption_count

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            throttle_mws, throttle_detail = self._should_throttle_mws_admission(
                seq_group)
            if throttle_mws:
                self._recovery_admission_throttle_cycle += 1
                self._recovery_mws_pinned_blocks_total = int(
                    throttle_detail.get("mws_pinned_blocks_total", 0))
                self._recovery_mws_limit_blocks = int(
                    throttle_detail.get("mws_limit_blocks", 0))
                try:
                    seq0 = seq_group.seqs[0] if seq_group.seqs else None
                    log_recovery_event(
                        "ADMISSION_THROTTLE_MWS",
                        req_id=seq_group.request_id,
                        seq_id=(seq0.seq_id if seq0 is not None else None),
                        reason="mws_capacity_guard",
                        detail=throttle_detail,
                        block_manager=self.block_manager,
                    )
                except Exception:
                    pass
                break

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            recompute_needed = (
                waiting_seqs[0].recovery_state.recovery_enabled and
                waiting_seqs[0].recovery_state.has_missing_recompute() and
                (not waiting_seqs[0].recovery_state.has_host_stored()))
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached
            if recompute_needed:
                num_new_tokens_uncached = min(num_new_tokens_uncached,
                                              RECOVERY_RECOMPUTE_DELTA_TOKENS)
                # Phase1 recompute micro-task: advance by bounded uncached
                # tokens only, independent of cache-hit accounting.
                num_new_tokens_cached = 0
                num_new_tokens = num_new_tokens_uncached

            if not enable_chunking and not recompute_needed:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if (not recompute_needed) and num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        if len(prefills.seq_groups
               ) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        prefills = self._schedule_prefills(budget,
                                           curr_loras,
                                           enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        # Put prefills first due to Attention backend ordering assumption.
        scheduled_seq_groups = (prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups)
        num_prefill_groups = (len(prefills.seq_groups) +
                              len(swapped_in.prefill_seq_groups) +
                              len(running_scheduled.prefill_seq_groups))
        # If all prompts, then we set num_lookahead_slots to 0
        # this allows us to go through the `no_spec` path in
        # `spec_decode_worker.py`
        all_prefills = (len(scheduled_seq_groups) == num_prefill_groups)
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup,
                          enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        if is_prefill and num_lookahead_slots > 0:
            # Appending prefill slots only happens multi-step and
            # chunked-prefill are enabled together.
            assert self.scheduler_config.is_multi_step and enable_chunking

        return self.block_manager.can_append_slots(
            seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        # async_output_proc is allowed only when we have a single sequence
        # in the sequence group
        no_single_seq = seq_group.sampling_params is None or (
            seq_group.sampling_params.n == 1)
        return no_single_seq

    def schedule(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        cycle_start_wall = time.time()
        self._recovery_obs.on_cycle_begin(cycle_start_wall)
        self._recovery_cycle_idx += 1
        self._recovery_admission_throttle_cycle = 0
        self._recovery_mws_pinned_blocks_total = self._current_mws_pinned_blocks_total()
        self._recovery_mws_limit_blocks = self._mws_capacity_limit_blocks()
        online_first_phase2 = self._is_phase2_online_first_enabled()
        self._recovery_online_only_pass = online_first_phase2
        # Keep fallback decode throttling local to admission decisions.
        # A full-cycle global pause introduces burstiness and can distort
        # mode-switch/preempt ratios under short pressure bursts.
        self._recovery_pause_decode_active = False
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        visibility_enforced_cnt = 0
        visible_tokens_sum = 0
        visible_recent_tokens_sum = 0
        visibility_counted_seq_ids: Set[int] = set()
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                raw_block_table = self.block_manager.get_block_table(seq)
                masked_block_table, vis_detail = self._mask_block_table_for_visibility(
                    seq, raw_block_table)
                block_tables[seq_id] = masked_block_table
                if int(vis_detail.get("enforced", 0)) == 1:
                    visibility_counted_seq_ids.add(seq.seq_id)
                    visibility_enforced_cnt += 1
                    visible_tokens_sum += int(vis_detail.get("visible_tokens", 0))
                    visible_recent_tokens_sum += int(vis_detail.get(
                        "W_min_tokens", 0))
                    try:
                        log_recovery_event(
                            "VISIBILITY_WINDOW_ENFORCED",
                            req_id=seq_group.request_id,
                            seq_id=seq.seq_id,
                            reason="recovery_visibility_contract",
                            detail={
                                "A_tokens": int(vis_detail.get("A_tokens", 0)),
                                "W_min_tokens": int(
                                    vis_detail.get("W_min_tokens", 0)),
                                "visible_tokens": int(
                                    vis_detail.get("visible_tokens", 0)),
                                "visible_blocks": int(
                                    vis_detail.get("visible_blocks", 0)),
                                "masked_blocks": int(
                                    vis_detail.get("masked_blocks", 0)),
                            },
                            block_manager=self.block_manager,
                        )
                    except Exception:
                        pass
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    multi_modal_placeholders=seq_group.multi_modal_placeholders
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group)

            if is_prompt and (not online_first_phase2):
                for seq in seq_group.get_seqs():
                    if not (seq.recovery_state.recovery_enabled
                            and seq.recovery_state.has_missing_recompute()
                            and (not seq.recovery_state.has_host_stored())):
                        continue
                    self._execute_recompute_micro_task(seq_group, seq,
                                                       token_chunk_size)

        if self._recovery_cfg.phase >= 3:
            for seq_group in list(self.waiting) + list(self.running) + list(
                    self.swapped):
                for seq in seq_group.seqs:
                    if seq.seq_id in visibility_counted_seq_ids:
                        continue
                    if (not seq.recovery_state.recovery_enabled
                            or seq.recovery_state.mode == RecoveryMode.NORMAL):
                        continue
                    raw_block_table = self.block_manager.get_block_table(seq)
                    if len(raw_block_table) == 0:
                        continue
                    _, vis_summary = self._visibility_contract_summary(
                        seq, len(raw_block_table))
                    visibility_enforced_cnt += 1
                    visible_tokens_sum += int(vis_summary.get(
                        "visible_tokens", 0))
                    visible_recent_tokens_sum += int(vis_summary.get(
                        "W_min_tokens", 0))

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Phase2 hard boundary: run online scheduling first, then use slack for
        # recovery micro-tasks only in tail stage.
        t_on_ms = max(0.0, (time.time() - cycle_start_wall) * 1000.0)
        t_cyc_target_ms = float(self._recovery_cfg.target_cycle_ms)
        raw_s_ms = max(0.0, t_cyc_target_ms - t_on_ms)
        slack_forced_zero = 0
        slack_escape_hatch = 0
        stalled = False
        mode_switched = 0
        mode_reason = "phase2_default"
        has_recovery_work = False
        req_mode_normal = 0
        req_mode_recovery = 0
        req_mode_fallback = 0
        req_mode_switches = 0
        req_mode_residence_ms_avg = 0.0
        mode_mem_safe = 1
        mode_global_safe_ms = 0.0
        mode_watermark_hi_blocks = 0
        mode_allow_normal = 1
        fresh_preempt_edge = 0
        fallback_budget_allowed = False
        fallback_budget_reason = "not_fallback"
        fallback_floor_ms = 0.0
        waiting_persistent = 0
        waiting_headroom_tight = 0
        hard_low_headroom = 0
        hard_pressure = 0
        high_load_pressure = 0
        free_gpu_blocks = 0
        total_gpu_blocks = getattr(self.block_manager, "num_total_gpu_blocks", 0)
        mode_watermark_hi_blocks = int(
            getattr(self.block_manager, "watermark_blocks", 0) or 0)
        try:
            free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        except Exception:
            free_gpu_blocks = 0
        phase3_enabled = self._recovery_cfg.phase >= 3
        waiting_len = len(self.waiting)
        swapped_len = len(self.swapped)
        waiting_relief_threshold = 2
        hard_low_headroom = int(free_gpu_blocks <= mode_watermark_hi_blocks)
        waiting_headroom_tight = int(
            free_gpu_blocks <=
            (mode_watermark_hi_blocks +
             self._recovery_waiting_pressure_headroom_margin))
        if waiting_len >= waiting_relief_threshold:
            self._recovery_waiting_streak += 1
        else:
            self._recovery_waiting_streak = 0
        waiting_persistent = int(
            self._recovery_waiting_streak >= self._recovery_waiting_streak_required)
        hard_pressure = int((scheduler_outputs.preempted > 0)
                            or hard_low_headroom)
        high_load_pressure = int(waiting_persistent and waiting_headroom_tight)
        if online_first_phase2:
            now_ns = time.time_ns()
            stalled = self._has_stalled_recovery(now_ns)
            has_recovery_work = self._has_recovery_work()
            if phase3_enabled:
                mode_update = self._recovery_mode_controller.update(
                    ModeSignals(
                        has_recovery_work=has_recovery_work,
                        waiting_len=waiting_len,
                        preempt_delta=scheduler_outputs.preempted,
                        stalled=stalled,
                        free_kv_blocks=free_gpu_blocks,
                        total_kv_blocks=total_gpu_blocks,
                        watermark_hi_blocks=mode_watermark_hi_blocks,
                        now_ns=now_ns,
                    ))
                self._recovery_mode = str(mode_update["mode"])
                mode_reason = str(mode_update["reason"])
                mode_switched = int(mode_update["switched"])
                mode_mem_safe = int(mode_update.get("mem_safe", 1))
                mode_global_safe_ms = float(mode_update.get("global_safe_ms", 0.0))
                mode_watermark_hi_blocks = int(
                    mode_update.get("watermark_hi_blocks", mode_watermark_hi_blocks))
                mode_allow_normal = int(mode_update.get("allow_normal", 1))
                req_mode_counts = self._update_request_modes_phase3(
                    now_ns,
                    global_mode=self._recovery_mode,
                    global_reason=mode_reason,
                    global_allow_normal=bool(mode_allow_normal),
                )
                req_mode_normal = int(req_mode_counts.get("normal", 0))
                req_mode_recovery = int(req_mode_counts.get("recovery", 0))
                req_mode_fallback = int(req_mode_counts.get("fallback", 0))
                req_mode_switches = int(req_mode_counts.get("switches", 0))
                req_mode_residence_ms_avg = float(
                    req_mode_counts.get("residence_ms_avg", 0.0))
            else:
                self._recovery_mode = ("recovery"
                                       if has_recovery_work else "normal")
        # In high-load pressure cycles, apply headroom-tiered throttling
        # instead of hard zeroing recovery budget.
        if (online_first_phase2 and self._recovery_mode != "fallback"
                and (hard_pressure or high_load_pressure)):
            fresh_preempt_edge = int(
                scheduler_outputs.preempted > 0
                and self._recovery_prev_preempt_delta <= 0)
            watermark_hi_blocks = max(0, int(mode_watermark_hi_blocks))
            watermark_lo_blocks = max(0, watermark_hi_blocks - 1)
            tiny_budget_ms = max(1e-3, float(self._recovery_cfg.swap_ms_per_block))
            low_budget_ms = tiny_budget_ms

            if free_gpu_blocks > watermark_hi_blocks:
                s_ms = raw_s_ms
            elif free_gpu_blocks > watermark_lo_blocks:
                s_ms = min(raw_s_ms, low_budget_ms)
            else:
                s_ms = min(raw_s_ms, tiny_budget_ms)

            if s_ms <= 0.0:
                slack_forced_zero = 1
            elif s_ms < raw_s_ms:
                slack_escape_hatch = 1
        elif online_first_phase2 and self._recovery_mode == "fallback":
            # Fallback uses soft suppression under sustained waiting pressure.
            # Avoid full freezes that can collapse swap/visibility signals.
            fallback_floor_base = (self._recovery_cfg.budget_min_ms +
                                   self._recovery_cfg.fallback_budget_boost_ms)
            fallback_floor_ms = fallback_floor_base
            if (not stalled) and waiting_len > 0 and high_load_pressure:
                fallback_budget_allowed = False
                fallback_budget_reason = "fallback_online_pressure_hold"
            elif stalled:
                fallback_budget_allowed = True
                fallback_budget_reason = "stalled_recovery"
            elif waiting_len == 0:
                fallback_budget_allowed = True
                fallback_budget_reason = "fallback_no_waiting_pressure"
            elif has_recovery_work:
                fallback_budget_allowed = True
                fallback_floor_ms = float(self._recovery_cfg.budget_min_ms)
                fallback_budget_reason = "fallback_soft_pressure"
            else:
                fallback_budget_allowed = False
                fallback_budget_reason = "fallback_no_recovery_work"
            if fallback_budget_allowed:
                s_ms = min(raw_s_ms, max(1e-3, fallback_floor_ms))
                slack_escape_hatch = 1 if s_ms > 0.0 else 0
            else:
                s_ms = 0.0
                slack_forced_zero = 1
        else:
            s_ms = raw_s_ms
        recovery_overhead_ms = 0.0
        budget_target_ms = self._recovery_last_budget_target_ms
        budget_exec_ms = self._recovery_last_budget_ms
        if online_first_phase2:
            decision = self._recovery_budget_controller.update(
                BudgetSignals(
                    slack_ms=s_ms,
                    preempt_delta=scheduler_outputs.preempted,
                    free_kv_blocks=free_gpu_blocks,
                    total_kv_blocks=total_gpu_blocks,
                    waiting_len=waiting_len,
                    stalled=stalled,
                ))
            budget_target_ms = float(decision["B_target_ms"])
            budget_exec_ms = float(decision["B_ms"])
            if self._recovery_mode == "fallback":
                if fallback_budget_allowed:
                    budget_target_ms = max(budget_target_ms, fallback_floor_ms)
                    budget_exec_ms = min(max(0.0, s_ms), budget_target_ms)
                else:
                    budget_target_ms = 0.0
                    budget_exec_ms = 0.0
            self._recovery_last_budget_target_ms = budget_target_ms
            self._recovery_last_budget_ms = budget_exec_ms
            try:
                log_recovery_event(
                    "CYCLE_SLACK_COMPUTED",
                    cycle_id=self._recovery_obs.cycle_id,
                    detail={
                        "T_on_ms": round(t_on_ms, 3),
                        "raw_S_ms": round(raw_s_ms, 3),
                        "S_ms": round(s_ms, 3),
                        "T_cyc_ms": round(t_cyc_target_ms, 3),
                        "slack_forced_zero": slack_forced_zero,
                        "slack_escape_hatch": slack_escape_hatch,
                        "recovery_mode": self._recovery_mode,
                        "mode_switched": mode_switched,
                        "mode_reason": mode_reason,
                        "mode_mem_safe": mode_mem_safe,
                        "mode_global_safe_ms": round(mode_global_safe_ms, 3),
                        "mode_allow_normal": mode_allow_normal,
                        "mode_watermark_hi_blocks": mode_watermark_hi_blocks,
                        "req_mode_normal": req_mode_normal,
                        "req_mode_recovery": req_mode_recovery,
                        "req_mode_fallback": req_mode_fallback,
                        "req_mode_switches": req_mode_switches,
                        "req_mode_residence_ms_avg": round(
                            req_mode_residence_ms_avg, 3),
                        "mode_switch_delta": req_mode_switches,
                        "fresh_preempt_edge": fresh_preempt_edge,
                        "hard_pressure": hard_pressure,
                        "high_load_pressure": high_load_pressure,
                        "hard_low_headroom": hard_low_headroom,
                        "waiting_headroom_tight": waiting_headroom_tight,
                        "waiting_persistent": waiting_persistent,
                        "waiting_streak": int(self._recovery_waiting_streak),
                        "swapped_len": swapped_len,
                        "fallback_budget_allowed": int(fallback_budget_allowed),
                        "fallback_budget_reason": fallback_budget_reason,
                        "admission_throttle": int(
                            self._recovery_admission_throttle_cycle),
                        "mws_pinned_blocks_total": int(
                            self._recovery_mws_pinned_blocks_total),
                        "mws_limit_blocks": int(self._recovery_mws_limit_blocks),
                    },
                    block_manager=self.block_manager,
                )
            except Exception:
                pass
            try:
                log_recovery_event(
                    "RECOVERY_BUDGET_DECISION",
                    cycle_id=self._recovery_obs.cycle_id,
                    detail={
                        "phase": self._recovery_cfg.phase,
                        "B_target_ms": budget_target_ms,
                        "B_ms": budget_exec_ms,
                        "S_ms": round(s_ms, 3),
                        "raw_S_ms": round(raw_s_ms, 3),
                        "slack_forced_zero": slack_forced_zero,
                        "slack_escape_hatch": slack_escape_hatch,
                        "recovery_mode": self._recovery_mode,
                        "mode_switched": mode_switched,
                        "mode_reason": mode_reason,
                        "mode_mem_safe": mode_mem_safe,
                        "mode_global_safe_ms": round(mode_global_safe_ms, 3),
                        "mode_allow_normal": mode_allow_normal,
                        "mode_watermark_hi_blocks": mode_watermark_hi_blocks,
                        "req_mode_normal": req_mode_normal,
                        "req_mode_recovery": req_mode_recovery,
                        "req_mode_fallback": req_mode_fallback,
                        "req_mode_switches": req_mode_switches,
                        "req_mode_residence_ms_avg": round(
                            req_mode_residence_ms_avg, 3),
                        "mode_switch_delta": req_mode_switches,
                        "fresh_preempt_edge": fresh_preempt_edge,
                        "hard_pressure": hard_pressure,
                        "high_load_pressure": high_load_pressure,
                        "hard_low_headroom": hard_low_headroom,
                        "waiting_headroom_tight": waiting_headroom_tight,
                        "waiting_persistent": waiting_persistent,
                        "waiting_streak": int(self._recovery_waiting_streak),
                        "swapped_len": swapped_len,
                        "fallback_budget_allowed": int(fallback_budget_allowed),
                        "fallback_budget_reason": fallback_budget_reason,
                        "admission_throttle": int(
                            self._recovery_admission_throttle_cycle),
                        "mws_pinned_blocks_total": int(
                            self._recovery_mws_pinned_blocks_total),
                        "mws_limit_blocks": int(self._recovery_mws_limit_blocks),
                        "preempt_delta": scheduler_outputs.preempted,
                        "waiting_len": waiting_len,
                        "free_kv_blocks": free_gpu_blocks,
                        "total_kv_blocks": total_gpu_blocks,
                        "stalled": int(stalled),
                        "pressure_state": decision["pressure_state"],
                        "delta_ms": decision["delta_ms"],
                        "free_ratio": decision["free_ratio"],
                        "free_ema": decision["free_ema"],
                        "preempt_ema": decision["preempt_ema"],
                    },
                    block_manager=self.block_manager,
                )
            except Exception:
                pass
            if budget_exec_ms > 0.0:
                _, recovery_overhead_ms = self._run_recovery_tail_micro_tasks(
                    budget_exec_ms, scheduler_outputs.blocks_to_swap_in)
        self._recovery_online_only_pass = False

        # Observability: cycle boundary summary (Phase0/1/2)
        cycle_end_wall = time.time()
        cycle_ctx: Dict[str, Union[int, float, str]] = {}
        try:
            num_scheduled = len(scheduler_outputs.scheduled_seq_groups)
            num_prefill = scheduler_outputs.num_prefill_groups
            num_decode = max(0, num_scheduled - num_prefill)
            prefill_tokens = 0
            decode_tokens = 0
            for scheduled in scheduler_outputs.scheduled_seq_groups:
                if scheduled.seq_group.is_prefill():
                    prefill_tokens += scheduled.token_chunk_size
                else:
                    decode_tokens += scheduled.token_chunk_size
            swap_frontier_sum = 0
            swap_frontier_count = 0
            for seq_group in list(self.waiting) + list(self.running) + list(
                    self.swapped):
                for seq in seq_group.seqs:
                    if not seq.recovery_state.recovery_enabled:
                        continue
                    swap_frontier_sum += seq.recovery_state.swap_frontier
                    swap_frontier_count += 1
            swap_frontier_avg = None
            if swap_frontier_count > 0:
                swap_frontier_avg = (swap_frontier_sum /
                                     swap_frontier_count)
            mws_pinned_blocks_total = self._current_mws_pinned_blocks_total()
            self._recovery_mws_pinned_blocks_total = mws_pinned_blocks_total
            cycle_ctx = {
                "prefill_groups": num_prefill,
                "decode_groups": num_decode,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "cycle_wall_ms": max(0.0, (cycle_end_wall - cycle_start_wall) * 1000.0),
                "num_batched_tokens": scheduler_outputs.num_batched_tokens,
                "running_queue_size": scheduler_outputs.running_queue_size,
                "waiting_len": len(self.waiting),
                "running_len": len(self.running),
                "swapped_len": len(self.swapped),
                "preempted": scheduler_outputs.preempted,
                "T_cyc_ms": round(t_cyc_target_ms, 3),
                "T_on_ms": round(t_on_ms, 3),
                "S_ms": round(s_ms, 3),
                "raw_S_ms": round(raw_s_ms, 3),
                "slack_forced_zero": slack_forced_zero,
                "slack_escape_hatch": slack_escape_hatch,
                "recovery_mode": self._recovery_mode,
                "mode_switched": mode_switched,
                "mode_reason": mode_reason,
                "mode_mem_safe": mode_mem_safe,
                "mode_global_safe_ms": round(mode_global_safe_ms, 3),
                "mode_allow_normal": mode_allow_normal,
                "mode_watermark_hi_blocks": mode_watermark_hi_blocks,
                "req_mode_normal": req_mode_normal,
                "req_mode_recovery": req_mode_recovery,
                "req_mode_fallback": req_mode_fallback,
                "req_mode_switches": req_mode_switches,
                "req_mode_residence_ms_avg": round(req_mode_residence_ms_avg,
                                                   3),
                "mode_switch_delta": req_mode_switches,
                "pause_decode_active": int(self._recovery_pause_decode_active),
                "admission_throttle": int(self._recovery_admission_throttle_cycle),
                "mws_pinned_blocks_total": int(mws_pinned_blocks_total),
                "mws_limit_blocks": int(self._recovery_mws_limit_blocks),
                "visibility_enforced": int(visibility_enforced_cnt),
                "visible_tokens": int(visible_tokens_sum),
                "visible_recent_tokens": int(visible_recent_tokens_sum),
                "recovery_overhead_ms": round(recovery_overhead_ms, 3),
                "recovery_budget_target_ms": round(budget_target_ms, 3),
                "recovery_budget_blocks": self._recovery_last_budget_blocks,
                "recovery_budget_ms": round(budget_exec_ms, 3),
                "recovery_slack_ms_est": round(self._recovery_last_slack_ms_est, 3),
                "recovery_online_load_est": self._recovery_last_online_load_est,
                "recovery_seq_slack_est": self._recovery_last_seq_slack_est,
                "recovery_token_slack_est": self._recovery_last_token_slack_est,
            }
            if swap_frontier_avg is not None:
                cycle_ctx["swap_frontier_avg"] = swap_frontier_avg
            self._recovery_obs.on_cycle_end(cycle_end_wall, cycle_ctx)
        except Exception:
            self._recovery_obs.on_cycle_end(cycle_end_wall, None)
        self._recovery_last_cycle_wall_ms = float(cycle_ctx.get("cycle_wall_ms", 0.0)
                                                  or 0.0)

        # Export cycle metrics into scheduler outputs for /metrics scraping.
        try:
            scheduler_outputs.recovery_swapin_blocks = int(
                cycle_ctx.get("swapin_blocks", 0))
            scheduler_outputs.recovery_swapout_blocks = int(
                cycle_ctx.get("swapout_blocks", 0))
            scheduler_outputs.recovery_recompute_tokens = int(
                cycle_ctx.get("recompute_tokens", 0))
            scheduler_outputs.recovery_restore_progress_stall_ms = float(
                cycle_ctx.get("restore_progress_stall_ms_delta", 0.0))
            scheduler_outputs.recovery_mode = str(cycle_ctx.get("mode", "normal"))
            scheduler_outputs.recovery_mode_switches_cycle = int(
                cycle_ctx.get("mode_switches_cycle", 0))
            scheduler_outputs.recovery_online_load_est = int(
                cycle_ctx.get("recovery_online_load_est", 0))
            scheduler_outputs.recovery_seq_slack_est = int(
                cycle_ctx.get("recovery_seq_slack_est", 0))
            scheduler_outputs.recovery_token_slack_est = int(
                cycle_ctx.get("recovery_token_slack_est", 0))
        except Exception:
            pass

        self._recovery_prev_preempt_delta = int(scheduler_outputs.preempted)

        # Recovery stalled detection (Phase1 diagnostic only).
        try:
            now_ns = time.time_ns()
            stall_threshold_ns = 10_000_000_000
            for seq_group in list(self.waiting) + list(self.running) + list(
                    self.swapped):
                for seq in seq_group.seqs:
                    if not seq.recovery_state.recovery_enabled:
                        continue
                    last_progress = seq.recovery_state.last_progress_ts_ns
                    if last_progress == 0:
                        continue
                    if now_ns - last_progress < stall_threshold_ns:
                        continue
                    last_logged = self._recovery_stall_logged_ns.get(
                        seq.seq_id, 0)
                    if now_ns - last_logged < stall_threshold_ns:
                        continue
                    self._recovery_stall_logged_ns[seq.seq_id] = now_ns
                    log_recovery_event(
                        "RECOVERY_STALLED",
                        req_id=seq_group.request_id,
                        seq_id=seq.seq_id,
                        detail={
                            "stall_ms": max(
                                0.0, (now_ns - last_progress) / 1e6),
                        },
                    )
        except Exception:
            pass

        # Return results
        return (seq_group_metadata_list, scheduler_outputs,
                allow_async_output_proc)

    def _run_recovery_micro_tasks(self) -> None:
        """Select and execute small recovery tasks per cycle.

        Phase1: placeholder hook for future swap/recompute planning.
        """
        return

    def _is_phase2_online_first_enabled(self) -> bool:
        cfg = self._recovery_cfg
        return (cfg.budget == 1) or (cfg.phase >= 2 and cfg.budget != 0)

    def _execute_recompute_micro_task(self, seq_group: SequenceGroup, seq: Sequence,
                                      delta_tokens: int) -> Tuple[int, float]:
        rec_task_start = time.perf_counter()
        before, after, delta_done = seq.recovery_state.recompute_step(
            delta_tokens,
            seq.block_size,
            seq.get_len(),
        )
        add_recompute(tokens=delta_done, tasks=1)
        try:
            log_recovery_event(
                "RECOMPUTE_MICRO",
                req_id=seq_group.request_id,
                seq_id=seq.seq_id,
                detail={
                    "delta_req": delta_tokens,
                    "delta_done": delta_done,
                    "rec_frontier_before": before,
                    "rec_frontier_after": after,
                },
                block_manager=self.block_manager,
            )
        except Exception:
            pass
        if delta_done > 0:
            # Keep progress timestamp fresh even if commit logging fails.
            seq.recovery_state.note_progress()
        try:
            if delta_done > 0:
                start_block = before // seq.block_size
                end_block = ((after - 1) // seq.block_size
                             if after > 0 else start_block)
                commit_start, commit_end, new_frontier = (
                    seq.recovery_state.commit_progress(
                        "recompute",
                        start_block,
                        end_block + 1,
                        token_after=after,
                    ))
                log_recovery_event(
                    "RECOVERY_PROGRESS_COMMIT",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "kind": "recompute",
                        "range": [commit_start, commit_end],
                        "tokens_committed": delta_done,
                        "new_frontier": new_frontier,
                    },
                    block_manager=self.block_manager,
                )
        except Exception:
            pass
        rec_task_ms = max(0.0, (time.perf_counter() - rec_task_start) * 1000.0)
        if delta_done > 0:
            observe_recompute_task(rec_task_ms, delta_done)
        return delta_done, rec_task_ms

    def _select_recovery_tasks(self, budget_ms: float) -> List[_RecoveryTask]:
        now_ns = time.time_ns()
        stall_threshold_ns = int(max(1.0, RECOVERY_PRIORITY_STALL_MS) * 1e6)
        phase3_enabled = self._recovery_cfg.phase >= 3
        waiting_pressure = len(self.waiting) > 0
        recent_preempt_pressure = self._recovery_prev_preempt_delta > 0
        defer_deep_restore = (phase3_enabled
                              and (waiting_pressure or recent_preempt_pressure))
        snapshot = get_cost_model_snapshot()
        bar_t_swap = max(1e-6, float(snapshot.get("barT_swap_ms_per_block", 1.0)))
        bar_t_rec = max(1e-6, float(snapshot.get("barT_rec_ms_per_token", 0.05)))
        gain_swap = 1.0 / bar_t_swap
        gain_rec = 1.0 / bar_t_rec
        delta_tokens = RECOVERY_RECOMPUTE_DELTA_TOKENS

        tasks: List[_RecoveryTask] = []
        swap_group_task_idx: Dict[int, int] = {}
        seen_groups: Set[int] = set()
        for seq_group in list(self.waiting) + list(self.running) + list(self.swapped):
            gid = id(seq_group)
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            for seq in seq_group.seqs:
                if not seq.recovery_state.recovery_enabled:
                    continue
                if not seq.recovery_state.needs_recovery():
                    continue
                last_progress = seq.recovery_state.last_progress_ts_ns
                stalled = (last_progress > 0 and
                           (now_ns - last_progress) >= stall_threshold_ns)
                preempt_cnt = 0
                try:
                    preempt_cnt = int(seq.recovery_obs.preempt_cnt)
                except Exception:
                    preempt_cnt = 0
                # Longer stalls are prioritized; frequently re-preempted requests
                # are restored more conservatively.
                stalled_boost = 2.0 if stalled else 1.0
                preempt_penalty = 1.0 / (1.0 + 0.1 * max(0, preempt_cnt))
                weight = stalled_boost * preempt_penalty
                mws_need = False
                mws_host = 0
                if phase3_enabled:
                    p_blocks, r_blocks = self._mws_blocks_for_seq(seq)
                    _, mws_host, mws_missing = seq.recovery_state.count_mws_hints(
                        p_blocks, r_blocks)
                    mws_need = (mws_host > 0 or mws_missing > 0)
                    if seq.recovery_state.mode == RecoveryMode.FALLBACK:
                        # Fallback should prioritize fast contract readiness.
                        weight *= 1.5
                        if mws_need:
                            weight *= 1.25

                if (seq.status == SequenceStatus.SWAPPED
                        and seq.recovery_state.has_host_stored()):
                    remaining_host = 0
                    try:
                        _, remaining_host, _ = seq.recovery_state.count_hints()
                    except Exception:
                        remaining_host = 0
                    host_target = remaining_host
                    if (phase3_enabled
                            and seq.recovery_state.mode == RecoveryMode.FALLBACK
                            and mws_host > 0):
                        host_target = mws_host
                    if (phase3_enabled
                            and seq.recovery_state.mode in
                        (RecoveryMode.RECOVERY, RecoveryMode.FALLBACK)):
                        if mws_host > 0:
                            # Visibility contract debt is always first-class.
                            host_target = mws_host
                        elif defer_deep_restore:
                            # Under online pressure, defer non-visible deep
                            # restore to avoid inflating decode stalls.
                            continue
                        else:
                            # Keep eventual convergence without aggressive
                            # full-history restore bursts.
                            host_target = min(remaining_host, 1)
                    if host_target <= 0:
                        continue
                    # Prefer finishing near-complete requests to reduce long
                    # recovery gaps on tail blocks.
                    finish_boost = 1.0 + (1.0 / max(1.0, float(host_target)))
                    new_task = _RecoveryTask(
                        kind="swap",
                        seq_group=seq_group,
                        seq=seq,
                        est_cost_ms=bar_t_swap,
                        gain_density=gain_swap,
                        priority=weight * gain_swap * finish_boost,
                        delta_tokens=0,
                        remaining_host_blocks=max(0, host_target),
                        stalled=stalled,
                    )
                    if gid not in swap_group_task_idx:
                        swap_group_task_idx[gid] = len(tasks)
                        tasks.append(
                            new_task)
                    else:
                        old_idx = swap_group_task_idx[gid]
                        old_task = tasks[old_idx]
                        if ((new_task.stalled and not old_task.stalled)
                                or (new_task.priority > old_task.priority)
                                or (new_task.remaining_host_blocks <
                                    old_task.remaining_host_blocks)):
                            tasks[old_idx] = new_task
                    continue

                if (seq.recovery_state.has_missing_recompute()
                        and (not seq.recovery_state.has_host_stored())):
                    req_delta_tokens = delta_tokens
                    # Keep recompute tasks executable under tight budgets by
                    # adapting chunk size to remaining cycle slack.
                    budget_cap_tokens = max(
                        1,
                        int(
                            math.floor(
                                max(1e-6, float(budget_ms)) / bar_t_rec)),
                    )
                    req_delta_tokens = min(req_delta_tokens, budget_cap_tokens)
                    if (phase3_enabled
                            and seq.recovery_state.mode == RecoveryMode.FALLBACK
                            and mws_need):
                        # Fallback focuses on visible contract readiness first.
                        fallback_tokens = max(1, delta_tokens // 2)
                        req_delta_tokens = min(req_delta_tokens,
                                               fallback_tokens)
                    est_cost_ms = max(1e-6, bar_t_rec * float(req_delta_tokens))
                    tasks.append(
                        _RecoveryTask(
                            kind="recompute",
                            seq_group=seq_group,
                            seq=seq,
                            est_cost_ms=est_cost_ms,
                            gain_density=gain_rec,
                            priority=weight * gain_rec,
                            delta_tokens=req_delta_tokens,
                            stalled=stalled,
                        ))

        tasks.sort(
            key=lambda t: (
                0 if t.stalled else 1,
                -t.priority,
                t.est_cost_ms,
            ))
        if budget_ms <= 0.0:
            return tasks
        return tasks

    def _has_stalled_recovery(self, now_ns: int) -> bool:
        threshold_ns = int(max(1.0, float(self._recovery_cfg.stall_ms)) * 1e6)
        for seq_group in list(self.waiting) + list(self.running) + list(
                self.swapped):
            for seq in seq_group.seqs:
                if not seq.recovery_state.recovery_enabled:
                    continue
                if not seq.recovery_state.needs_recovery():
                    continue
                last_progress = seq.recovery_state.last_progress_ts_ns
                if last_progress <= 0:
                    continue
                if now_ns - last_progress >= threshold_ns:
                    return True
        return False

    def _is_seq_group_stalled_for_recovery_priority(
            self,
            seq_group: SequenceGroup,
            now_ns: Optional[int] = None) -> bool:
        threshold_ns = int(max(1.0, RECOVERY_PRIORITY_STALL_MS) * 1e6)
        ts_now = int(now_ns if now_ns is not None else time.time_ns())
        for seq in seq_group.seqs:
            if not seq.recovery_state.recovery_enabled:
                continue
            if not seq.recovery_state.needs_recovery():
                continue
            last_progress = int(seq.recovery_state.last_progress_ts_ns)
            if last_progress <= 0:
                continue
            if ts_now - last_progress >= threshold_ns:
                return True
        return False

    def _iter_all_sequence_groups(self) -> Iterable[SequenceGroup]:
        seen: Set[int] = set()
        for seq_group in list(self.waiting) + list(self.running) + list(
                self.swapped):
            gid = id(seq_group)
            if gid in seen:
                continue
            seen.add(gid)
            yield seq_group

    def _mws_blocks_for_seq(self, seq: Sequence) -> Tuple[int, int]:
        block_size = max(1, int(seq.block_size))
        prefix_tokens = max(0, int(self._recovery_cfg.mws_prefix_tokens))
        recent_tokens = max(0, int(self._recovery_cfg.mws_recent_tokens))
        prefix_blocks = ((prefix_tokens + block_size - 1) // block_size
                         if prefix_tokens > 0 else 0)
        recent_blocks = ((recent_tokens + block_size - 1) // block_size
                         if recent_tokens > 0 else 0)
        return prefix_blocks, recent_blocks

    def _can_swap_out_with_phase3_pin(self, seq_group: SequenceGroup) -> bool:
        block_tables = getattr(self.block_manager, "block_tables", None)
        if block_tables is None:
            return True
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            table = block_tables.get(seq.seq_id)
            if table is None:
                continue
            blocks = table.blocks
            if len(blocks) <= 0:
                continue
            try:
                seq.recovery_state.align_to_num_blocks(len(blocks))
            except Exception:
                pass
            p_blocks, r_blocks = self._mws_blocks_for_seq(seq)
            pinned = set(
                seq.recovery_state.get_visible_indices_mws(p_blocks, r_blocks))
            for idx in range(len(blocks)):
                if idx in pinned:
                    continue
                if (seq.recovery_state.storage_hint.get(idx) ==
                        RecoveryStorageHint.HOST_STORED):
                    continue
                return True
        return False

    def _visibility_contract_summary(
        self,
        seq: Sequence,
        num_blocks: int,
    ) -> Tuple[Set[int], Dict[str, int]]:
        total_blocks = max(0, int(num_blocks))
        try:
            seq.recovery_state.align_to_num_blocks(total_blocks)
        except Exception:
            pass
        seq_len = int(max(0, seq.get_len()))
        a_tokens = min(seq_len, max(0, int(self._recovery_cfg.mws_prefix_tokens)))
        remaining_tokens = max(0, seq_len - a_tokens)
        w_min_tokens = min(remaining_tokens,
                           max(0, int(self._recovery_cfg.mws_recent_tokens)))
        visible_tokens = min(seq_len, a_tokens + w_min_tokens)
        prefix_blocks, recent_blocks = self._mws_blocks_for_seq(seq)
        visible_indices = set(
            seq.recovery_state.get_visible_indices_mws(prefix_blocks,
                                                       recent_blocks))
        visible_blocks = min(total_blocks, len(visible_indices))
        return visible_indices, {
            "A_tokens": int(a_tokens),
            "W_min_tokens": int(w_min_tokens),
            "visible_tokens": int(visible_tokens),
            "visible_blocks": int(visible_blocks),
            "masked_blocks": int(max(0, total_blocks - visible_blocks)),
        }

    def _mask_block_table_for_visibility(
        self,
        seq: Sequence,
        block_table: List[int],
    ) -> Tuple[List[int], Dict[str, Union[int, float]]]:
        detail: Dict[str, Union[int, float]] = {
            "enforced": 0,
            "A_tokens": 0,
            "W_min_tokens": 0,
            "visible_tokens": int(seq.get_len()),
            "visible_blocks": len(block_table),
            "masked_blocks": 0,
        }
        if self._recovery_cfg.phase < 3:
            return block_table, detail
        if (not seq.recovery_state.recovery_enabled or
                seq.recovery_state.mode == RecoveryMode.NORMAL):
            return block_table, detail
        if len(block_table) == 0:
            return block_table, detail
        null_id_fn = getattr(self.block_manager, "get_null_block_id", None)
        if not callable(null_id_fn):
            return block_table, detail
        null_block_id = null_id_fn()
        if null_block_id is None:
            return block_table, detail

        visible_indices, vis_summary = self._visibility_contract_summary(
            seq, len(block_table))
        if len(visible_indices) >= len(block_table):
            # Contract is active and currently satisfied without masking.
            detail.update({
                "enforced": 1,
                "A_tokens": int(vis_summary.get("A_tokens", 0)),
                "W_min_tokens": int(vis_summary.get("W_min_tokens", 0)),
                "visible_tokens": int(vis_summary.get("visible_tokens", 0)),
                "visible_blocks": len(block_table),
                "masked_blocks": 0,
                "null_block_id": int(null_block_id),
            })
            return block_table, detail

        masked = list(block_table)
        masked_blocks = 0
        for idx in range(len(masked)):
            if idx in visible_indices:
                continue
            masked[idx] = int(null_block_id)
            masked_blocks += 1

        detail.update({
            "enforced": 1 if masked_blocks > 0 else 0,
            "A_tokens": int(vis_summary.get("A_tokens", 0)),
            "W_min_tokens": int(vis_summary.get("W_min_tokens", 0)),
            "visible_tokens": int(vis_summary.get("visible_tokens", 0)),
            "visible_blocks": int(vis_summary.get("visible_blocks", 0)),
            "masked_blocks": masked_blocks,
            "null_block_id": int(null_block_id),
        })
        return masked, detail

    def _can_run_with_visibility_contract(self, seq_group: SequenceGroup) -> bool:
        if self._recovery_cfg.phase < 3:
            return False
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if not seq.recovery_state.recovery_enabled:
                return False
            p_blocks, r_blocks = self._mws_blocks_for_seq(seq)
            if seq.recovery_state.needs_recovery_mws(p_blocks, r_blocks):
                return False
        return True

    def _estimate_mws_blocks_for_seq(self, seq: Sequence) -> int:
        block_size = max(1, int(seq.block_size))
        mws_tokens = (max(0, int(self._recovery_cfg.mws_prefix_tokens)) +
                      max(0, int(self._recovery_cfg.mws_recent_tokens)))
        if mws_tokens <= 0:
            return 0
        mws_blocks = (mws_tokens + block_size - 1) // block_size
        return max(0, min(int(seq.n_blocks), int(mws_blocks)))

    def _current_mws_pinned_blocks_total(self) -> int:
        total = 0
        seen_seq_ids: Set[int] = set()
        for seq_group in self._iter_all_sequence_groups():
            for seq in seq_group.seqs:
                sid = int(seq.seq_id)
                if sid in seen_seq_ids:
                    continue
                seen_seq_ids.add(sid)
                pinned = len(seq.recovery_state.pinned_blocks)
                if pinned > 0:
                    total += pinned
                    continue
                if seq.recovery_state.recovery_enabled:
                    total += self._estimate_mws_blocks_for_seq(seq)
        return max(0, int(total))

    def _mws_capacity_limit_blocks(self) -> int:
        total_gpu_blocks = int(
            max(0, getattr(self.block_manager, "num_total_gpu_blocks", 0)))
        rho = float(self._recovery_cfg.mws_admit_rho)
        rho = max(0.0, min(1.0, rho))
        return max(0, int(rho * float(total_gpu_blocks)))

    def _estimate_mws_blocks_for_group(self, seq_group: SequenceGroup) -> int:
        total = 0
        for seq in seq_group.seqs:
            total += self._estimate_mws_blocks_for_seq(seq)
        return max(0, int(total))

    def _is_new_admission_candidate(self, seq_group: SequenceGroup) -> bool:
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            if seq.recovery_state.recovery_enabled:
                return False
        return True

    def _should_throttle_mws_admission(
            self, seq_group: SequenceGroup) -> Tuple[bool, Dict[str, Union[int,
                                                                            float]]]:
        if self._recovery_cfg.phase < 3:
            return False, {}
        if not self._is_new_admission_candidate(seq_group):
            return False, {}
        limit_blocks = self._mws_capacity_limit_blocks()
        if limit_blocks <= 0:
            return False, {}
        pinned_total = self._current_mws_pinned_blocks_total()
        incoming_blocks = self._estimate_mws_blocks_for_group(seq_group)
        projected_total = pinned_total + incoming_blocks
        detail: Dict[str, Union[int, float]] = {
            "mws_pinned_blocks_total": int(pinned_total),
            "mws_incoming_blocks": int(incoming_blocks),
            "mws_projected_blocks_total": int(projected_total),
            "mws_limit_blocks": int(limit_blocks),
            "mws_admit_rho": float(self._recovery_cfg.mws_admit_rho),
            "kv_total_blocks": int(
                max(0, getattr(self.block_manager, "num_total_gpu_blocks", 0))),
        }
        if projected_total > limit_blocks:
            return True, detail
        return False, detail

    def _apply_mws_hard_pin_for_group(self, seq_group: SequenceGroup, *,
                                      reason: str) -> int:
        if self._recovery_cfg.phase < 3:
            return 0
        pin_fn = getattr(self.block_manager, "apply_mws_hard_pin", None)
        if not callable(pin_fn):
            return 0
        try:
            return int(
                pin_fn(
                    seq_group,
                    int(self._recovery_cfg.mws_prefix_tokens),
                    int(self._recovery_cfg.mws_recent_tokens),
                    reason=reason,
                ))
        except Exception:
            return 0

    def _has_fallback_decode_protected_group(self) -> bool:
        threshold = int(self._recovery_cfg.fallback_protect_priority_gte)
        if threshold < 0:
            return False
        for seq_group in self._iter_all_sequence_groups():
            if int(getattr(seq_group, "priority", 0)) >= threshold:
                return True
        return False

    def _has_request_fallback_mode(self) -> bool:
        for seq_group in self._iter_all_sequence_groups():
            for seq in seq_group.seqs:
                if seq.recovery_state.mode == RecoveryMode.FALLBACK:
                    return True
        return False

    def _is_seq_stalled_phase3(self, seq: Sequence, now_ns: int) -> bool:
        last_progress = int(seq.recovery_state.last_progress_ts_ns)
        if last_progress <= 0:
            return False
        threshold_ns = int(max(1.0, float(self._recovery_cfg.fallback_stall_ms))
                           * 1e6)
        return (now_ns - last_progress) >= threshold_ns

    def _update_request_modes_phase3(
            self, now_ns: int, *, global_mode: str, global_reason: str,
            global_allow_normal: bool) -> Dict[str, Union[int, float]]:
        counts = {
            "normal": 0,
            "recovery": 0,
            "fallback": 0,
            "switches": 0,
            "residence_ms_sum": 0.0,
            "residence_ms_count": 0,
            "residence_ms_avg": 0.0,
        }
        stable_window_ms = float(self._recovery_cfg.mode_stable_window_ms)
        pin_applied_groups: Set[int] = set()
        for seq_group in self._iter_all_sequence_groups():
            gid = id(seq_group)
            for seq in seq_group.seqs:
                state = seq.recovery_state
                state.ensure_mode_initialized(now_ns)
                prefix_blocks, recent_blocks = self._mws_blocks_for_seq(seq)
                needs_full = state.needs_recovery()
                needs_mws = state.needs_recovery_mws(prefix_blocks,
                                                     recent_blocks)
                mws_gpu, mws_host, mws_missing = state.count_mws_hints(
                    prefix_blocks, recent_blocks)
                mws_total = max(0, mws_gpu + mws_host + mws_missing)
                mws_ready_ratio = (1.0 if mws_total == 0 else
                                   (float(mws_gpu) / float(mws_total)))
                mws_admit_rho = max(0.0, float(self._recovery_cfg.mws_admit_rho))
                stalled = self._is_seq_stalled_phase3(seq, now_ns)
                stalled_visible = bool(stalled and needs_mws)
                progressed_since_mode_enter = (
                    int(seq.recovery_state.last_progress_ts_ns)
                    > int(state.mode_enter_ts_ns))
                fallback_exit_ready = (
                    mws_ready_ratio >= mws_admit_rho
                    and progressed_since_mode_enter)

                has_recovery_need = bool(needs_full or needs_mws)
                if not state.recovery_enabled:
                    # Requests never preempted should stay NORMAL and must not
                    # be pulled into RECOVERY by global hysteresis.
                    target = RecoveryMode.NORMAL
                    reason = "no_recovery_state"
                elif not has_recovery_need:
                    if global_allow_normal:
                        target = RecoveryMode.NORMAL
                        reason = "drained_global_stable"
                    else:
                        # If drained but global guard is not yet stable, hold
                        # current mode to suppress high-frequency toggling.
                        target = state.mode
                        reason = "drained_wait_global_stable"
                elif state.mode == RecoveryMode.FALLBACK:
                    # Keep FALLBACK sticky until request-level readiness holds.
                    if stalled_visible:
                        target = RecoveryMode.FALLBACK
                        reason = "stalled_recovery"
                    elif not needs_mws:
                        # MWS contract already satisfied; don't keep toggling
                        # FALLBACK for background deep-restore debt.
                        target = RecoveryMode.RECOVERY
                        reason = "mws_clear_recovery"
                    elif global_mode == "fallback":
                        target = RecoveryMode.FALLBACK
                        reason = "global_fallback_hold"
                    elif not fallback_exit_ready:
                        target = RecoveryMode.FALLBACK
                        # Avoid FALLBACK->RECOVERY->FALLBACK flip-flop when
                        # readiness is transient but no real recovery progress
                        # has been made in this fallback residency.
                        if mws_ready_ratio < mws_admit_rho:
                            reason = "mws_unready"
                        else:
                            reason = "fallback_wait_progress"
                    else:
                        target = RecoveryMode.RECOVERY
                        reason = "mws_ready_recovery"
                elif global_mode == "fallback":
                    if not needs_mws:
                        target = RecoveryMode.RECOVERY
                        reason = "mws_clear_recovery"
                    elif (not stalled_visible) and fallback_exit_ready:
                        target = RecoveryMode.RECOVERY
                        reason = "mws_ready_recovery"
                    else:
                        target = RecoveryMode.FALLBACK
                        if stalled_visible:
                            reason = "stalled_recovery"
                        elif needs_mws:
                            reason = "mws_unready"
                        elif (not progressed_since_mode_enter):
                            reason = "fallback_wait_progress"
                        else:
                            reason = str(global_reason)
                else:
                    target = RecoveryMode.RECOVERY
                    reason = "needs_recovery"

                prev_mode = state.mode
                prev_mode_residence_ms = max(
                    0.0, (now_ns - state.mode_enter_ts_ns) / 1e6)
                # Only fallback uses forced fast switch here. Recovery fast
                # path is handled at preempt-trigger site to avoid cycle-level
                # NORMAL<->RECOVERY ping-pong.
                force_switch = (target == RecoveryMode.FALLBACK
                                and reason == "stalled_recovery")
                switched = state.set_mode(
                    target,
                    reason=reason,
                    now_ns=now_ns,
                    stable_window_ms=stable_window_ms,
                    force=force_switch,
                )
                residence_ms = max(0.0,
                                   (now_ns - state.mode_enter_ts_ns) / 1e6)
                counts["residence_ms_sum"] += residence_ms
                counts["residence_ms_count"] += 1
                seq.recovery_obs.mode = state.mode
                seq.recovery_obs.mode_enter_ts_ns = state.mode_enter_ts_ns
                if switched:
                    counts["switches"] += 1
                    log_recovery_event(
                        "RECOVERY_REQUEST_MODE_SWITCH",
                        req_id=seq_group.request_id,
                        seq_id=seq.seq_id,
                        reason=reason,
                        detail={
                            "from_mode": prev_mode.value,
                            "to_mode": state.mode.value,
                            "mode_switches": state.mode_switches,
                            "global_mode": global_mode,
                            "mws_prefix_blocks": prefix_blocks,
                            "mws_recent_blocks": recent_blocks,
                            "mws_gpu": mws_gpu,
                            "mws_host": mws_host,
                            "mws_missing": mws_missing,
                            "mws_ready": int(not needs_mws),
                            "mws_ready_ratio": round(mws_ready_ratio, 4),
                            "mws_admit_rho": mws_admit_rho,
                            "global_allow_normal": int(global_allow_normal),
                            "mode_enter_ts_ns": state.mode_enter_ts_ns,
                            "mode_residence_ms": round(residence_ms, 3),
                            "prev_mode_residence_ms": round(
                                prev_mode_residence_ms, 3),
                        },
                        block_manager=self.block_manager,
                    )
                    mode_enter_event = None
                    if state.mode == RecoveryMode.RECOVERY:
                        mode_enter_event = "MODE_ENTER_RECOVERY"
                    elif state.mode == RecoveryMode.NORMAL:
                        mode_enter_event = "MODE_ENTER_NORMAL"
                    elif state.mode == RecoveryMode.FALLBACK:
                        mode_enter_event = "MODE_ENTER_FALLBACK"
                    if mode_enter_event is not None:
                        log_recovery_event(
                            mode_enter_event,
                            req_id=seq_group.request_id,
                            seq_id=seq.seq_id,
                            reason=reason,
                            detail={
                                "from_mode": prev_mode.value,
                                "to_mode": state.mode.value,
                                "num_mode_switches": state.mode_switches,
                                "mode_enter_ts_ns": state.mode_enter_ts_ns,
                                "mode_residence_ms": round(residence_ms, 3),
                            },
                            block_manager=self.block_manager,
                        )
                    if state.mode == RecoveryMode.FALLBACK:
                        log_recovery_event(
                            "ENTER_FALLBACK",
                            req_id=seq_group.request_id,
                            seq_id=seq.seq_id,
                            reason=reason,
                            detail={
                                "from_mode": prev_mode.value,
                                "to_mode": state.mode.value,
                                "num_mode_switches": state.mode_switches,
                            },
                            block_manager=self.block_manager,
                        )
                    elif prev_mode == RecoveryMode.FALLBACK:
                        log_recovery_event(
                            "EXIT_FALLBACK",
                            req_id=seq_group.request_id,
                            seq_id=seq.seq_id,
                            reason=reason,
                            detail={
                                "from_mode": prev_mode.value,
                                "to_mode": state.mode.value,
                                "num_mode_switches": state.mode_switches,
                            },
                            block_manager=self.block_manager,
                        )
                    if (state.mode == RecoveryMode.RECOVERY
                            and gid not in pin_applied_groups):
                        pin_applied_groups.add(gid)
                        self._apply_mws_hard_pin_for_group(
                            seq_group, reason="mode_enter_recovery")
                if state.mode == RecoveryMode.NORMAL:
                    counts["normal"] += 1
                elif state.mode == RecoveryMode.RECOVERY:
                    counts["recovery"] += 1
                else:
                    counts["fallback"] += 1

        residence_count = int(counts["residence_ms_count"])
        if residence_count > 0:
            counts["residence_ms_avg"] = (
                float(counts["residence_ms_sum"]) / float(residence_count))
        return counts

    def _has_recovery_work(self) -> bool:
        for seq_group in list(self.waiting) + list(self.running) + list(
                self.swapped):
            for seq in seq_group.seqs:
                if not seq.recovery_state.recovery_enabled:
                    continue
                if seq.recovery_state.needs_recovery():
                    return True
        return False

    def _active_recovery_request_ids(self) -> Set[str]:
        out: Set[str] = set()
        for seq_group in list(self.waiting) + list(self.running) + list(
                self.swapped):
            req_id = str(getattr(seq_group, "request_id", "")).strip()
            if not req_id:
                continue
            for seq in seq_group.seqs:
                if (seq.recovery_state.recovery_enabled
                        and seq.recovery_state.needs_recovery()):
                    out.add(req_id)
                    break
        return out

    def _update_recovery_no_progress_cycles(self,
                                            progressed_req_ids: Set[str]) -> None:
        active_req_ids = self._active_recovery_request_ids()
        # Drop stale entries for finished/drained requests.
        stale = [
            req_id for req_id in self._recovery_req_no_progress_cycles
            if req_id not in active_req_ids
        ]
        for req_id in stale:
            self._recovery_req_no_progress_cycles.pop(req_id, None)

        for req_id in active_req_ids:
            if req_id in progressed_req_ids:
                self._recovery_req_no_progress_cycles[req_id] = 0
                continue
            prev = int(self._recovery_req_no_progress_cycles.get(req_id, 0))
            self._recovery_req_no_progress_cycles[req_id] = min(10_000, prev + 1)

    def _try_force_progress_for_starved(
        self,
        tasks: List[_RecoveryTask],
        remaining_ms: float,
        progressed_req_ids: Set[str],
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> Tuple[bool, float, Optional[str]]:
        threshold = max(1, self._recovery_force_progress_cycles)
        starved_tasks = [
            t for t in tasks
            if (t.seq_group.request_id not in progressed_req_ids)
            and self._recovery_req_no_progress_cycles.get(
                t.seq_group.request_id, 0) >= threshold
        ]
        starved_tasks.sort(
            key=lambda t: (
                self._recovery_req_no_progress_cycles.get(
                    t.seq_group.request_id, 0),
                1 if t.stalled else 0,
                t.priority,
                -t.est_cost_ms,
            ),
            reverse=True,
        )
        for task in starved_tasks:
            req_id = task.seq_group.request_id
            no_progress_cycles = int(
                self._recovery_req_no_progress_cycles.get(req_id, 0))
            if task.kind == "swap":
                req_id = task.seq_group.request_id
                k_swap_target = 1
                if no_progress_cycles >= threshold:
                    k_swap_target = 2
                force_cost_ms = max(1e-6,
                                    float(task.est_cost_ms) *
                                    float(k_swap_target))
                if remaining_ms < force_cost_ms:
                    if remaining_ms < max(1e-6, float(task.est_cost_ms)):
                        continue
                    k_swap_target = 1
                    force_cost_ms = max(1e-6, float(task.est_cost_ms))
                is_prefill = task.seq_group.is_prefill()
                alloc_status = self.block_manager.can_swap_in(
                    task.seq_group,
                    self._get_num_lookahead_slots(is_prefill,
                                                  enable_chunking=False),
                    k_swap_max=k_swap_target)
                if (alloc_status == AllocStatus.LATER and task.stalled
                        and remaining_ms > 0.0):
                    alloc_status_unlock = self.block_manager.can_swap_in(
                        task.seq_group,
                        self._get_num_lookahead_slots(
                            is_prefill, enable_chunking=False),
                        k_swap_max=1,
                        stalled=True,
                        reserve_blocks=RECOVERY_SWAP_UNLOCK_RESERVE_BLOCKS,
                    )
                    if alloc_status_unlock != AllocStatus.NEVER:
                        alloc_status = alloc_status_unlock
                        k_swap_target = 1
                        force_cost_ms = max(1e-6, float(task.est_cost_ms))
                if alloc_status == AllocStatus.NEVER:
                    continue
                k_done, _ = self._swap_in(task.seq_group, blocks_to_swap_in,
                                          k_swap_target)
                if k_done <= 0:
                    continue
                remaining_ms -= max(1e-6,
                                    float(task.est_cost_ms) * float(k_done))
                try:
                    log_recovery_event(
                        "RECOVERY_FORCE_PROGRESS",
                        req_id=req_id,
                        seq_id=(task.seq.seq_id
                                if task.seq is not None else None),
                        reason="no_progress_guard_swap",
                        detail={
                            "no_progress_cycles": no_progress_cycles,
                            "threshold_cycles": threshold,
                            "k_swap_target": int(k_swap_target),
                            "k_swap_done": int(k_done),
                            "remaining_ms_after": round(remaining_ms, 6),
                        },
                        block_manager=self.block_manager,
                    )
                except Exception:
                    pass
                return True, remaining_ms, req_id
            if task.kind == "recompute" and task.seq is not None:
                force_tokens = max(
                    1,
                    min(int(task.delta_tokens),
                        self._recovery_force_recompute_tokens),
                )
                force_cost_ms = max(
                    1e-6,
                    float(task.est_cost_ms) * float(force_tokens) /
                    float(max(1, int(task.delta_tokens))),
                )
                if remaining_ms < force_cost_ms:
                    continue
                delta_done, task_ms = self._execute_recompute_micro_task(
                    task.seq_group,
                    task.seq,
                    force_tokens,
                )
                if delta_done <= 0:
                    continue
                remaining_ms -= max(force_cost_ms, task_ms)
                try:
                    log_recovery_event(
                        "RECOVERY_FORCE_PROGRESS",
                        req_id=req_id,
                        seq_id=task.seq.seq_id,
                        reason="no_progress_guard_recompute",
                        detail={
                            "no_progress_cycles": no_progress_cycles,
                            "threshold_cycles": threshold,
                            "force_tokens": int(force_tokens),
                            "remaining_ms_after": round(remaining_ms, 6),
                        },
                        block_manager=self.block_manager,
                    )
                except Exception:
                    pass
                return True, remaining_ms, req_id
        return False, remaining_ms, None

    def _run_recovery_tail_micro_tasks(
            self, budget_ms: float,
            blocks_to_swap_in: List[Tuple[int, int]]) -> Tuple[int, float]:
        # Phase2: allow tail-stage recompute even when swapped queue is empty.
        if budget_ms <= 0.0:
            self._update_recovery_no_progress_cycles(set())
            return 0, 0.0
        start = time.perf_counter()
        remaining_ms = max(0.0, float(budget_ms))
        executed_tasks = 0
        progressed_req_ids: Set[str] = set()
        forced_tasks = 0

        while remaining_ms > 0.0:
            tasks = self._select_recovery_tasks(remaining_ms)
            if not tasks:
                break

            executed_one = False
            budget_blocked = 0
            for task in tasks:
                est_cost_ms = max(1e-6, float(task.est_cost_ms))
                if remaining_ms < est_cost_ms:
                    budget_blocked += 1
                    continue
                if task.kind == "swap":
                    req_id = task.seq_group.request_id
                    no_progress_cycles = int(
                        self._recovery_req_no_progress_cycles.get(req_id, 0))
                    k_swap_target = 1
                    if task.stalled or (
                            no_progress_cycles >=
                            self._recovery_force_progress_cycles):
                        k_swap_target = 2
                    swap_cost_ms = est_cost_ms * float(k_swap_target)
                    if remaining_ms < swap_cost_ms:
                        if remaining_ms < est_cost_ms:
                            budget_blocked += 1
                            continue
                        k_swap_target = 1
                        swap_cost_ms = est_cost_ms
                    is_prefill = task.seq_group.is_prefill()
                    alloc_status = self.block_manager.can_swap_in(
                        task.seq_group,
                        self._get_num_lookahead_slots(
                            is_prefill, enable_chunking=False),
                        k_swap_max=k_swap_target)
                    if (alloc_status == AllocStatus.LATER and task.stalled
                            and remaining_ms > 0.0):
                        alloc_status_unlock = self.block_manager.can_swap_in(
                            task.seq_group,
                            self._get_num_lookahead_slots(
                                is_prefill, enable_chunking=False),
                            k_swap_max=1,
                            stalled=True,
                            reserve_blocks=RECOVERY_SWAP_UNLOCK_RESERVE_BLOCKS,
                        )
                        if alloc_status_unlock != AllocStatus.NEVER:
                            alloc_status = alloc_status_unlock
                            k_swap_target = 1
                    # For stalled recovery, allow a best-effort swap even if
                    # allocator reports LATER; this helps shrink long gaps on
                    # tail blocks without changing normal-path aggressiveness.
                    if alloc_status == AllocStatus.NEVER:
                        continue
                    if alloc_status == AllocStatus.LATER and (not task.stalled):
                        continue
                    k_done, _ = self._swap_in(task.seq_group, blocks_to_swap_in,
                                              k_swap_target)
                    if k_done <= 0:
                        continue
                    remaining_ms -= max(est_cost_ms, est_cost_ms * float(k_done))
                    executed_tasks += 1
                    progressed_req_ids.add(task.seq_group.request_id)
                    executed_one = True
                    break
                if task.kind == "recompute" and task.seq is not None:
                    delta_done, task_ms = self._execute_recompute_micro_task(
                        task.seq_group, task.seq, task.delta_tokens)
                    if delta_done <= 0:
                        continue
                    remaining_ms -= max(est_cost_ms, task_ms)
                    executed_tasks += 1
                    progressed_req_ids.add(task.seq_group.request_id)
                    executed_one = True
                    break

            # Fairness guard: even if the loop already made progress, allow one
            # extra micro-task for a different starved request per cycle.
            if forced_tasks < self._recovery_force_max_tasks_per_cycle:
                forced_done, remaining_ms, forced_req_id = (
                    self._try_force_progress_for_starved(
                        tasks,
                        remaining_ms,
                        progressed_req_ids,
                        blocks_to_swap_in,
                    ))
                if forced_done and forced_req_id is not None:
                    executed_tasks += 1
                    progressed_req_ids.add(forced_req_id)
                    forced_tasks += 1
                    if not executed_one:
                        continue

            if not executed_one:
                # Count only budget-limited drops. If tasks are blocked by
                # allocator state rather than remaining budget, avoid inflating
                # the over_budget_drop metric.
                if budget_blocked > 0:
                    add_over_budget_drop(budget_blocked)
                break

        self._update_recovery_no_progress_cycles(progressed_req_ids)
        pending = len(self._select_recovery_tasks(0.0))
        if pending > 0 and remaining_ms <= 0.0:
            add_over_budget_drop(pending)

        return executed_tasks, max(0.0, (time.perf_counter() - start) * 1000.0)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            # Free cross-attention block table, if it exists
            self._free_seq_group_cross_attn_blocks(seq_group)

            # Add the finished requests to the finished requests list.
            # This list will be used to update the Mamba cache in the
            # next step.
            self._finished_requests_ids.append(seq_group.request_id)

        # Free finished seqs
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        self.running = remaining

        # Handle async stopped sequence groups
        # (ones that reached max model len)
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)

                # Free finished seqs
                self._free_finished_seqs(seq_group)

            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(self,
                      seq_group: SequenceGroup,
                      blocks_to_copy: List[Tuple[int, int]],
                      enable_chunking: bool = False) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking)

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        reason: Optional[str] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        try:
            seq_id = None
            if seq_group.seqs:
                seq_id = seq_group.seqs[0].seq_id
            log_recovery_event(
                "PREEMPT_TRIGGERED",
                req_id=seq_group.request_id,
                seq_id=seq_id,
                reason=reason or "no_kv_space",
                detail={"preemption_mode": preemption_mode.name},
                block_manager=self.block_manager,
            )
        except Exception:
            pass
        try:
            for seq in seq_group.seqs:
                seq.ensure_recovery_blocks()
                seq.recovery_state.recovery_enabled = True
                seq.recovery_state.set_mode(
                    RecoveryMode.RECOVERY,
                    reason="preempt_triggered",
                    force=True,
                )
                seq.recovery_state.num_recovery_attempts += 1
                seq.recovery_state.swap_frontier = 0
                seq.recovery_state.rec_frontier = 0
                seq.recovery_state.reset_restored_map(
                    seq.recovery_state.num_blocks())
                seq.recovery_state.note_progress()
                log_recovery_event(
                    "RECOVERY_STATE_CREATED",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "swap_frontier": seq.recovery_state.swap_frontier,
                        "rec_frontier": seq.recovery_state.rec_frontier,
                        "num_blocks": seq.recovery_state.num_blocks(),
                    },
                    block_manager=self.block_manager,
                )
        except Exception:
            pass
        if preemption_mode == PreemptionMode.SWAP:
            self._apply_mws_hard_pin_for_group(
                seq_group, reason="preempt_enter_recovery")
            try:
                if self._recovery_cfg.phase >= 3:
                    for seq in seq_group.seqs:
                        raw_block_table = self.block_manager.get_block_table(
                            seq)
                        if len(raw_block_table) == 0:
                            continue
                        _, vis_summary = self._visibility_contract_summary(
                            seq, len(raw_block_table))
                        log_recovery_event(
                            "VISIBILITY_WINDOW_ENFORCED",
                            req_id=seq_group.request_id,
                            seq_id=seq.seq_id,
                            reason="preempt_visibility_contract",
                            detail={
                                "A_tokens": int(
                                    vis_summary.get("A_tokens", 0)),
                                "W_min_tokens": int(
                                    vis_summary.get("W_min_tokens", 0)),
                                "visible_tokens": int(
                                    vis_summary.get("visible_tokens", 0)),
                                "visible_blocks": int(
                                    vis_summary.get("visible_blocks", 0)),
                                "masked_blocks": 0,
                            },
                            block_manager=self.block_manager,
                        )
                can_swap_out = self.block_manager.can_swap_out(seq_group)
                if (not can_swap_out) and self._recovery_cfg.phase >= 3:
                    relax_fn = getattr(self.block_manager,
                                       "relax_mws_pin_for_swap", None)
                    relaxed_blocks = 0
                    if callable(relax_fn):
                        relaxed_blocks = int(
                            relax_fn(seq_group,
                                     min_unpinned=1,
                                     reason="preempt_swap_blocked"))
                    if relaxed_blocks > 0:
                        can_swap_out = self.block_manager.can_swap_out(
                            seq_group)
                        if can_swap_out:
                            log_recovery_event(
                                "MWS_PIN_SWAPOUT_RELAXED",
                                req_id=seq_group.request_id,
                                reason="swap_out_reenabled",
                                detail={"relaxed_blocks": relaxed_blocks},
                                block_manager=self.block_manager,
                            )
                if not can_swap_out:
                    try:
                        for seq in seq_group.seqs:
                            log_recovery_event(
                                "SWAP_OUT",
                                req_id=seq_group.request_id,
                                seq_id=seq.seq_id,
                                reason="swap_out_blocked",
                                detail={
                                    "blocks_count": 0,
                                    "bytes": None,
                                    "swapped_indices_sample": [],
                                    "pinned_indices_sample":
                                    seq.recovery_state.
                                    get_pinned_blocks_sorted()[:16],
                                },
                                block_manager=self.block_manager,
                            )
                    except Exception:
                        pass
                    if seq_group.get_max_num_running_seqs() == 1:
                        preemption_mode = PreemptionMode.RECOMPUTE
                        log_recovery_event(
                            "MWS_PIN_SWAPOUT_BLOCKED",
                            req_id=seq_group.request_id,
                            reason="fallback_to_recompute",
                            detail={
                                "preempt_mode_before": "SWAP",
                                "preempt_mode_after": "RECOMPUTE",
                            },
                            block_manager=self.block_manager,
                        )
            except Exception:
                pass
        try:
            for seq in seq_group.seqs:
                seq.recovery_obs.preempt_cnt += 1
        except Exception:
            pass
        req_id = str(getattr(seq_group, "request_id", "")).strip()
        if req_id:
            # Start fairness clock at preempt so freshly preempted requests can
            # get a guaranteed tiny recovery slice in the next tail window.
            prev = int(self._recovery_req_no_progress_cycles.get(req_id, 0))
            self._recovery_req_no_progress_cycles[req_id] = max(
                prev, self._recovery_force_progress_cycles)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        try:
            if preemption_mode == PreemptionMode.RECOMPUTE:
                for seq in seq_group.seqs:
                    n_gpu, n_host, n_missing = seq.recovery_state.count_hints()
                    log_recovery_event(
                        "RECOVERY_HINT_SNAPSHOT",
                        req_id=seq_group.request_id,
                        seq_id=seq.seq_id,
                        detail={
                            "n_gpu": n_gpu,
                            "n_host": n_host,
                            "n_missing": n_missing,
                        },
                        block_manager=self.block_manager,
                    )
        except Exception:
            pass
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
        k_swap_max: int,
    ) -> Tuple[int, bool]:
        mapping, k_done = self.block_manager.swap_in(seq_group, k_swap_max)
        blocks_to_swap_in.extend(mapping)
        fully_restored = True
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            # Phase1 correctness rule: "can run" and "recovery finished" are
            # separate states; keep polling recovery until ledger says done.
            if seq.recovery_state.needs_recovery():
                fully_restored = False
                break
        return k_done, fully_restored

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool,
                                 enable_chunking: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                # num_lookahead_slots was introduced in the context of decodes,
                # in Speculative Decoding.
                # When the num_scheduler_steps is 8, say, then the
                # num_lookahead_slots is 7. Meaning, we are doing a 1-step of
                # decode anyways and we wish to do 7 more.
                #
                # "lookaheads" for prefills, is introduced in support for
                # Chunked-Prefill in Multi-Step.
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_uncached_and_cached_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
        enable_chunking: bool,
        budget: SchedulingBudget,
    ) -> Tuple[int, int]:
        """
        Returns the number of new uncached and cached tokens to schedule for a
        given sequence group that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns (0, 0) if the new token cannot be computed due to token budget.

        The cached tokens's blocks are already computed, and the attention
        backend will reuse the cached blocks rather than recomputing them. So
        the scheduler could schedule these cached tokens "for free".

        Args:
            seq_group: The sequence group to get the number of new tokens to
                schedule.
            status: The status of the sequences to get the number of new tokens
                to schedule.
            enable_chunking: Whether to chunk the number of tokens to compute.
            budget: The budget to chunk the number of tokens to compute.


        Returns:
            A tuple of two ints. The first int is the number of new uncached
            tokens to schedule. The second int is the number of cached tokens.
            If no more new tokens can be scheduled, returns (0, 0).
        """
        num_cached_new_tokens = 0
        num_uncached_new_tokens = 0

        seqs = seq_group.get_seqs(status=status)
        # Compute the number of new uncached and cached tokens for
        # each sequence.
        for seq in seqs:
            if not seq.is_prefill():
                # Decode sequences should always just have 1 uncached token
                # TODO(rickyx): Actually is this still correct for multi-step?
                num_uncached_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                # If prefix caching is not enabled, all new tokens are uncached.
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue

            # NOTE: the cache token might be currently in a block that's in an
            # evictor meaning that it's not yet allocated. However, we don't
            # exclude such tokens in the cache count because it will be
            # guaranteed to be allocated later if the sequence can be allocated.
            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(
                seq)

            # Sanity check.
            if num_cached_tokens_seq < num_computed_tokens_seq:
                # This should only happen with chunked prefill, and
                # the seq is still in prefill. The `num_cached_tokens_seq`
                # is the value we calculated on scheduling the first prefill.
                # For subsequent continuous prefill steps, we cached the
                # number of cache tokens for the sequence so the cached token
                # count could be less than the number of computed tokens.
                # See comments on `ComputedBlocksTracker` for more details.
                assert (
                    seq.is_prefill() and seq.status == SequenceStatus.RUNNING
                    and self.scheduler_config.chunked_prefill_enabled
                ), ("Number of cached tokens should not be less than the "
                    "number of computed tokens for a sequence that's still "
                    f"in prefill. But there are {num_cached_tokens_seq} cached "
                    f"tokens and {num_computed_tokens_seq} computed tokens "
                    f"for sequence {seq.seq_id}.")

            num_cached_new_tokens_seq = max(
                0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = (all_num_new_tokens_seq -
                                           num_cached_new_tokens_seq)

            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq

        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            # For a fully cached hit sequence, we actually need to recompute the
            # last token. So we need at least 1 uncached token to schedule.
            # See ModelRunner._compute_for_prefix_cache_hit for more details.
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1

        if enable_chunking and len(seqs) == 1:
            # Chunk if a running request cannot fit in the given budget.
            # If number of seq > 1, it means it is doing beam search
            # in a decode phase. Do not chunk.
            num_uncached_new_tokens = self._chunk_new_tokens_to_schedule(
                self.scheduler_config,
                self.cache_config,
                budget,
                self._get_prompt_limit(seq_group),
                num_uncached_new_tokens,
            )

        return num_uncached_new_tokens, num_cached_new_tokens

    @staticmethod
    def _chunk_new_tokens_to_schedule(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        budget: SchedulingBudget,
        prompt_limit: int,
        num_new_tokens: int,
    ) -> int:
        """
        Chunks the number of new tokens to schedule based on the budget when
        chunked prefill is enabled.

        Args:
            scheduler_config: The scheduler config.
            cache_config: The cache config.
            budget: The budget to chunk the number of tokens to compute.
            prompt_limit: The maximum number of tokens allowed in a prompt.
            num_new_tokens: The number of new tokens to schedule.

        Returns:
            The number of new tokens to schedule after chunking.
        """
        remaining_token_budget = budget.remaining_token_budget()
        if scheduler_config.is_multi_step:
            # The current multi-step + chunked prefill capability does
            # not actually support chunking prompts.
            #
            # Therefore, `num_new_tokens` is computed in the same fashion
            # for both multi-step+chunked-prefill &
            # multi-step+chunked-prefill+APC
            #
            # Prompts with more tokens than the current remaining budget
            # are postponed to future scheduler steps
            if num_new_tokens > prompt_limit:
                # If the seq_group is in prompt-stage, pass the
                # num_new_tokens as-is so the caller can ignore
                # the sequence.
                return num_new_tokens

            return (0 if num_new_tokens > remaining_token_budget else
                    num_new_tokens)

        if cache_config.enable_prefix_caching:
            # Adjust the remaining token budget to be divisible by the block
            # size when prefix caching is enabled.

            # When prefix caching is enabled, we always allocate
            # the number of new tokens that is dividable by the block
            # size to avoid partial block matching.
            block_size = cache_config.block_size
            remainder = budget.token_budget % block_size
            if remainder != 0:
                raise ValueError("When enabling chunked prefill and "
                                 "prefix caching, max_num_batched_tokens "
                                 "(chunk size) must be dividable by "
                                 "block size, but got chunk_size "
                                 f"({budget.token_budget}) % block_size "
                                 f"({block_size}) = {remainder}")
            # Round down to block size.
            remaining_token_budget = (remaining_token_budget // block_size *
                                      block_size)

        num_new_tokens = min(num_new_tokens, remaining_token_budget)

        return num_new_tokens
