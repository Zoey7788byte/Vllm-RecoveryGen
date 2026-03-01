import enum
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
from vllm.recovery.observability import (RecoveryObservability, add_over_budget_drop,
                                         add_recompute, get_recovery_config,
                                         log_recovery_event)
from vllm.sequence import (RecoveryMode, Sequence, SequenceData, SequenceGroup,
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
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking),
                k_swap_max=(k_budget_remaining if needs_recovery else 0))
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

            k_done, fully_restored = self._swap_in(seq_group,
                                                   blocks_to_swap_in,
                                                   (k_budget_remaining
                                                    if needs_recovery else 0))
            k_budget_remaining = max(0, k_budget_remaining - max(0, k_done))
            if not fully_restored:
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
        online_first_phase2 = self._is_phase2_online_first_enabled()
        self._recovery_online_only_pass = online_first_phase2
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
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
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
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
        high_load = (len(self.waiting) > 0 or scheduler_outputs.preempted > 0)
        if online_first_phase2:
            stalled = self._has_stalled_recovery(time.time_ns())
        # In high-load cycles (queue pressure or fresh preemptions), treat
        # slack as unavailable to protect online path and suppress oscillation.
        if online_first_phase2 and high_load:
            s_ms = 0.0
            slack_forced_zero = 1
            # Escape hatch: when recovery is stalled, grant a tiny budget so
            # progress can continue and long swap-in gaps are capped.
            if stalled:
                s_ms = min(raw_s_ms, max(1e-3, self._recovery_cfg.budget_min_ms))
                if s_ms > 0.0:
                    slack_forced_zero = 0
                    slack_escape_hatch = 1
        else:
            s_ms = raw_s_ms
        recovery_overhead_ms = 0.0
        budget_target_ms = self._recovery_last_budget_target_ms
        budget_exec_ms = self._recovery_last_budget_ms
        if online_first_phase2:
            free_gpu_blocks = 0
            total_gpu_blocks = getattr(self.block_manager, "num_total_gpu_blocks", 0)
            try:
                free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
            except Exception:
                free_gpu_blocks = 0
            decision = self._recovery_budget_controller.update(
                BudgetSignals(
                    slack_ms=s_ms,
                    preempt_delta=scheduler_outputs.preempted,
                    free_kv_blocks=free_gpu_blocks,
                    total_kv_blocks=total_gpu_blocks,
                    waiting_len=len(self.waiting),
                    stalled=stalled,
                ))
            budget_target_ms = float(decision["B_target_ms"])
            budget_exec_ms = float(decision["B_ms"])
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
                        "preempt_delta": scheduler_outputs.preempted,
                        "waiting_len": len(self.waiting),
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
        stall_threshold_ns = int(max(1.0, float(self._recovery_cfg.stall_ms)) * 1e6)
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

                if (seq.status == SequenceStatus.SWAPPED
                        and seq.recovery_state.has_host_stored()):
                    remaining_host = 0
                    try:
                        _, remaining_host, _ = seq.recovery_state.count_hints()
                    except Exception:
                        remaining_host = 0
                    # Prefer finishing near-complete requests to reduce long
                    # recovery gaps on tail blocks.
                    finish_boost = 1.0 + (1.0 / max(1.0, float(remaining_host)))
                    new_task = _RecoveryTask(
                        kind="swap",
                        seq_group=seq_group,
                        seq=seq,
                        est_cost_ms=bar_t_swap,
                        gain_density=gain_swap,
                        priority=weight * gain_swap * finish_boost,
                        delta_tokens=0,
                        remaining_host_blocks=max(0, remaining_host),
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
                    est_cost_ms = max(1e-6, bar_t_rec * float(delta_tokens))
                    tasks.append(
                        _RecoveryTask(
                            kind="recompute",
                            seq_group=seq_group,
                            seq=seq,
                            est_cost_ms=est_cost_ms,
                            gain_density=gain_rec,
                            priority=weight * gain_rec,
                            delta_tokens=delta_tokens,
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

    def _run_recovery_tail_micro_tasks(
            self, budget_ms: float,
            blocks_to_swap_in: List[Tuple[int, int]]) -> Tuple[int, float]:
        # Phase2: allow tail-stage recompute even when swapped queue is empty.
        if budget_ms <= 0.0:
            return 0, 0.0
        start = time.perf_counter()
        remaining_ms = max(0.0, float(budget_ms))
        executed_tasks = 0

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
                    is_prefill = task.seq_group.is_prefill()
                    alloc_status = self.block_manager.can_swap_in(
                        task.seq_group,
                        self._get_num_lookahead_slots(
                            is_prefill, enable_chunking=False),
                        k_swap_max=1)
                    # For stalled recovery, allow a best-effort swap even if
                    # allocator reports LATER; this helps shrink long gaps on
                    # tail blocks without changing normal-path aggressiveness.
                    if alloc_status == AllocStatus.NEVER:
                        continue
                    if alloc_status == AllocStatus.LATER and (not task.stalled):
                        continue
                    k_done, _ = self._swap_in(task.seq_group, blocks_to_swap_in, 1)
                    if k_done <= 0:
                        continue
                    remaining_ms -= est_cost_ms
                    executed_tasks += 1
                    executed_one = True
                    break
                if task.kind == "recompute" and task.seq is not None:
                    delta_done, task_ms = self._execute_recompute_micro_task(
                        task.seq_group, task.seq, task.delta_tokens)
                    if delta_done <= 0:
                        continue
                    remaining_ms -= max(est_cost_ms, task_ms)
                    executed_tasks += 1
                    executed_one = True
                    break

            if not executed_one:
                # Count only budget-limited drops. If tasks are blocked by
                # allocator state rather than remaining budget, avoid inflating
                # the over_budget_drop metric.
                if budget_blocked > 0:
                    add_over_budget_drop(budget_blocked)
                break

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
                seq.recovery_state.mode = RecoveryMode.RECOVERY
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
        try:
            for seq in seq_group.seqs:
                seq.recovery_obs.preempt_cnt += 1
        except Exception:
            pass
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
