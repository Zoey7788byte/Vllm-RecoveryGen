"""A block manager that manages token blocks."""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm import envs
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.recovery.cost_model import observe_swap_task
from vllm.recovery.observability import (add_restore_commit, add_swap_in,
                                         add_swap_micro, add_swap_out,
                                         log_recovery_event)
from vllm.sequence import (RecoveryMode, RecoveryStorageHint, Sequence,
                           SequenceGroup, SequenceStatus)
from vllm.utils import Device

SeqId = int
EncoderSeqId = str


@dataclass
class _SwapPrefetchEntry:
    request_id: str
    seq_id: int
    indices: List[int]
    cpu_block_ids: List[int]
    created_ts_ns: int


class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)
        self._recovery_prefetch_max_blocks = max(
            0, int(envs.REC_PREFETCH_MAX_BLOCKS))
        self._swap_prefetch: Dict[int, _SwapPrefetchEntry] = {}
        self._swap_prefetch_blocks_total = 0

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_sequence(self, seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if seq.get_token_ids():
            # NOTE: If there are any factors affecting the block besides
            # token_ids, they should be added as input to extra_hash.
            extra_hash = seq.extra_hash()

            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(token_ids=seq.get_token_ids(),
                                 extra_hash=extra_hash)

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table
        try:
            seq.ensure_recovery_blocks()
            # Preserve recompute ledger for preempted requests.
            # For normal requests, bootstrap all current blocks as recovered.
            if not (seq.recovery_state.recovery_enabled
                    and seq.recovery_state.needs_recovery()):
                seq.recovery_state.mark_all(RecoveryStorageHint.GPU_RESIDENT)
                num_blocks = seq.recovery_state.num_blocks()
                if num_blocks > 0:
                    seq.recovery_state.commit_restored_range(0, num_blocks)
                    seq.recovery_state.swap_frontier = num_blocks
        except Exception:
            pass

        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)
            try:
                seq.ensure_recovery_blocks()
                if not (seq.recovery_state.recovery_enabled
                        and seq.recovery_state.needs_recovery()):
                    seq.recovery_state.mark_all(RecoveryStorageHint.GPU_RESIDENT)
                    num_blocks = seq.recovery_state.num_blocks()
                    if num_blocks > 0:
                        seq.recovery_state.commit_restored_range(0, num_blocks)
                        seq.recovery_state.swap_frontier = num_blocks
            except Exception:
                pass

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]
        prev_num_blocks = len(block_table.blocks)
        null_block = None
        if self.max_block_sliding_window is not None:
            try:
                null_block = self.block_allocator.allocate_or_get_null_block()
            except Exception:
                null_block = None

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
            extra_hash=seq.extra_hash(),
        )
        try:
            new_num_blocks = len(block_table.blocks)
            seq.ensure_recovery_blocks()
            if null_block is not None:
                dropped_blocks = 0
                for blk in block_table.blocks:
                    if blk is null_block:
                        dropped_blocks += 1
                    else:
                        break
                if dropped_blocks > 0:
                    seq.recovery_state.mark_range(
                        0,
                        dropped_blocks,
                        RecoveryStorageHint.MISSING_RECOMPUTE,
                    )
                    seq.recovery_state.swap_frontier = 0
                    for i in range(
                            min(dropped_blocks,
                                len(seq.recovery_state.restored_map))):
                        seq.recovery_state.restored_map[i] = False
                    seq.recovery_state.note_progress()
            if new_num_blocks > prev_num_blocks:
                seq.recovery_state.mark_range(
                    prev_num_blocks,
                    new_num_blocks,
                    RecoveryStorageHint.GPU_RESIDENT,
                )
                seq.recovery_state.swap_frontier = max(
                    seq.recovery_state.swap_frontier, new_num_blocks)
                seq.recovery_state.commit_restored_range(
                    prev_num_blocks, new_num_blocks)
        except Exception:
            pass
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            self._drop_prefetch_entry(seq_id)
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]
        self._drop_prefetch_entry(seq_id)

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = (
                self._computed_blocks_tracker.get_num_cached_tokens(seq))
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_block_ids = all_blocks[:num_cached_blocks]
            computed_seq_block_ids.append(computed_block_ids)

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def _drop_prefetch_entry(self, seq_id: int) -> None:
        entry = self._swap_prefetch.pop(seq_id, None)
        if entry is None:
            return
        self._swap_prefetch_blocks_total = max(
            0, self._swap_prefetch_blocks_total - len(entry.indices))

    def apply_mws_hard_pin(self,
                           seq_group: SequenceGroup,
                           prefix_tokens: int,
                           recent_tokens: int,
                           *,
                           reason: str = "enter_recovery") -> int:
        total_pinned = 0
        p_tokens = max(0, int(prefix_tokens))
        r_tokens = max(0, int(recent_tokens))
        for seq in seq_group.seqs:
            block_table = self.block_tables.get(seq.seq_id)
            num_blocks = len(
                block_table.blocks) if block_table is not None else seq.n_blocks
            try:
                seq.ensure_recovery_blocks()
                seq.recovery_state.align_to_num_blocks(num_blocks)
            except Exception:
                continue
            block_size = max(1, int(seq.block_size))
            p_blocks = ((p_tokens + block_size - 1) // block_size
                        if p_tokens > 0 else 0)
            r_blocks = ((r_tokens + block_size - 1) // block_size
                        if r_tokens > 0 else 0)
            pinned = seq.recovery_state.get_visible_indices_mws(
                p_blocks, r_blocks)
            seq.recovery_state.set_pinned_blocks(pinned)
            total_pinned += len(pinned)
            try:
                log_recovery_event(
                    "MWS_PIN_APPLIED",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    reason=reason,
                    detail={
                        "pinned_blocks": len(pinned),
                        "mws_prefix_blocks": p_blocks,
                        "mws_recent_blocks": r_blocks,
                        "num_blocks": num_blocks,
                        "pinned_sample": pinned[:16],
                    },
                    block_manager=self,
                )
            except Exception:
                pass
        return total_pinned

    def get_null_block_id(self) -> Optional[int]:
        try:
            null_block = self.block_allocator.allocate_or_get_null_block()
            return int(null_block.block_id)
        except Exception:
            return None

    def clear_mws_hard_pin(self,
                           seq_group: SequenceGroup,
                           *,
                           reason: str = "leave_recovery") -> int:
        total_cleared = 0
        for seq in seq_group.seqs:
            pinned = seq.recovery_state.get_pinned_blocks_sorted()
            if not pinned:
                continue
            total_cleared += len(pinned)
            seq.recovery_state.clear_pinned_blocks()
            try:
                log_recovery_event(
                    "MWS_PIN_RELEASED",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    reason=reason,
                    detail={
                        "cleared_blocks": len(pinned),
                        "cleared_sample": pinned[:16],
                    },
                    block_manager=self,
                )
            except Exception:
                pass
        return total_cleared

    def relax_mws_pin_for_swap(self,
                               seq_group: SequenceGroup,
                               *,
                               min_unpinned: int = 1,
                               reason: str = "swap_pressure") -> int:
        target_unpinned = max(1, int(min_unpinned))
        total_relaxed = 0
        for seq in seq_group.seqs:
            block_table = self.block_tables.get(seq.seq_id)
            num_blocks = len(
                block_table.blocks) if block_table is not None else seq.n_blocks
            if num_blocks <= 1:
                continue
            try:
                seq.ensure_recovery_blocks()
                seq.recovery_state.align_to_num_blocks(num_blocks)
            except Exception:
                continue
            pinned = seq.recovery_state.get_pinned_blocks_sorted()
            if not pinned:
                continue
            pinned_set = set(int(idx) for idx in pinned)
            need_relax = max(0, target_unpinned - (num_blocks - len(pinned_set)))
            if need_relax <= 0:
                continue

            block_size = max(1, int(seq.block_size))
            p_tokens = max(0, int(envs.REC_MWS_PREFIX_TOKENS))
            r_tokens = max(0, int(envs.REC_MWS_RECENT_TOKENS))
            p_blocks = ((p_tokens + block_size - 1) // block_size
                        if p_tokens > 0 else 0)
            r_blocks = ((r_tokens + block_size - 1) // block_size
                        if r_tokens > 0 else 0)
            mid_start = min(num_blocks, p_blocks)
            mid_end = max(mid_start, num_blocks - r_blocks)

            relaxed_indices: List[int] = []
            for idx in pinned:
                if need_relax <= 0:
                    break
                if idx < mid_start or idx >= mid_end:
                    continue
                if idx not in pinned_set:
                    continue
                pinned_set.remove(idx)
                relaxed_indices.append(int(idx))
                need_relax -= 1

            if need_relax > 0:
                center = num_blocks // 2
                fallback = sorted(pinned_set, key=lambda x: abs(int(x) - center))
                for idx in fallback:
                    if need_relax <= 0:
                        break
                    pinned_set.remove(idx)
                    relaxed_indices.append(int(idx))
                    need_relax -= 1

            if not relaxed_indices:
                continue

            seq.recovery_state.set_pinned_blocks(sorted(pinned_set))
            total_relaxed += len(relaxed_indices)
            try:
                log_recovery_event(
                    "MWS_PIN_RELAXED",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    reason=reason,
                    detail={
                        "num_blocks": num_blocks,
                        "pinned_before": len(pinned),
                        "pinned_after": len(pinned_set),
                        "relaxed_blocks": len(relaxed_indices),
                        "relaxed_sample": relaxed_indices[:16],
                    },
                    block_manager=self,
                )
            except Exception:
                pass
        return total_relaxed

    def _collect_swap_candidate_indices(
            self, seq: Sequence, blocks: List[Block],
            max_to_swap: int) -> Tuple[List[int], int]:
        if max_to_swap <= 0:
            return [], 0
        indices: List[int] = []
        seen_indices = set()
        skipped_already_restored = 0
        try:
            start = max(0, seq.recovery_state.swap_frontier)
            if int(envs.VLLM_RECOVERY_PHASE) >= 3 and (
                    seq.recovery_state.mode in
                (RecoveryMode.RECOVERY, RecoveryMode.FALLBACK)):
                block_size = max(1, int(seq.block_size))
                p_tokens = max(0, int(envs.REC_MWS_PREFIX_TOKENS))
                r_tokens = max(0, int(envs.REC_MWS_RECENT_TOKENS))
                p_blocks = ((p_tokens + block_size - 1) // block_size
                            if p_tokens > 0 else 0)
                r_blocks = ((r_tokens + block_size - 1) // block_size
                            if r_tokens > 0 else 0)
                preferred = seq.recovery_state.get_visible_indices_mws(
                    p_blocks, r_blocks)
                for idx in preferred:
                    if idx < start or idx >= len(blocks):
                        continue
                    if len(indices) >= max_to_swap:
                        break
                    if idx in seen_indices:
                        continue
                    if seq.recovery_state.storage_hint.get(
                            idx) != RecoveryStorageHint.HOST_STORED:
                        continue
                    if (idx < len(seq.recovery_state.restored_map)
                            and seq.recovery_state.restored_map[idx]):
                        skipped_already_restored += 1
                        continue
                    indices.append(idx)
                    seen_indices.add(idx)
            for idx in range(start, len(blocks)):
                if len(indices) >= max_to_swap:
                    break
                if idx in seen_indices:
                    continue
                if seq.recovery_state.storage_hint.get(
                        idx) == RecoveryStorageHint.HOST_STORED:
                    if (idx < len(seq.recovery_state.restored_map)
                            and seq.recovery_state.restored_map[idx]):
                        skipped_already_restored += 1
                        continue
                    indices.append(idx)
                    seen_indices.add(idx)
            if not indices and seq.recovery_state.storage_hint:
                for idx in range(0, len(blocks)):
                    if len(indices) >= max_to_swap:
                        break
                    if idx in seen_indices:
                        continue
                    if seq.recovery_state.storage_hint.get(
                            idx) == RecoveryStorageHint.HOST_STORED:
                        if (idx < len(seq.recovery_state.restored_map)
                                and seq.recovery_state.restored_map[idx]):
                            skipped_already_restored += 1
                            continue
                        indices.append(idx)
                        seen_indices.add(idx)
        except Exception:
            indices = list(range(0, min(len(blocks), max_to_swap)))
        return indices, skipped_already_restored

    def _consume_prefetched_indices(
            self, seq: Sequence, blocks: List[Block],
            max_to_swap: int) -> Tuple[List[int], int, Optional[float]]:
        entry = self._swap_prefetch.get(seq.seq_id)
        if entry is None:
            return [], 0, None
        now_ns = time.time_ns()
        prefetch_age_ms = max(0.0, (now_ns - entry.created_ts_ns) / 1e6)
        valid_pairs: List[Tuple[int, int]] = []
        skipped_already_restored = 0
        for idx, cpu_block_id in zip(entry.indices, entry.cpu_block_ids):
            if idx >= len(blocks):
                skipped_already_restored += 1
                continue
            if blocks[idx].block_id != cpu_block_id:
                skipped_already_restored += 1
                continue
            if seq.recovery_state.storage_hint.get(
                    idx) != RecoveryStorageHint.HOST_STORED:
                skipped_already_restored += 1
                continue
            if (idx < len(seq.recovery_state.restored_map)
                    and seq.recovery_state.restored_map[idx]):
                skipped_already_restored += 1
                continue
            valid_pairs.append((idx, cpu_block_id))

        take_pairs = valid_pairs[:max_to_swap]
        remain_pairs = valid_pairs[max_to_swap:]
        old_count = len(entry.indices)
        if remain_pairs:
            entry.indices = [idx for idx, _ in remain_pairs]
            entry.cpu_block_ids = [cpu_id for _, cpu_id in remain_pairs]
            self._swap_prefetch_blocks_total = max(
                0, self._swap_prefetch_blocks_total + len(remain_pairs) -
                old_count)
        else:
            self._drop_prefetch_entry(seq.seq_id)
        return [idx for idx, _ in take_pairs], skipped_already_restored, prefetch_age_ms

    def prefetch_swap_in(self,
                         seq_group: SequenceGroup,
                         k_prefetch_max: Optional[int] = None) -> int:
        if self._recovery_prefetch_max_blocks <= 0:
            return 0
        remaining_global = (self._recovery_prefetch_max_blocks -
                            self._swap_prefetch_blocks_total)
        if remaining_global <= 0:
            return 0
        limit = remaining_global
        if k_prefetch_max is not None:
            limit = min(limit, max(0, int(k_prefetch_max)))
        if limit <= 0:
            return 0

        total_ready = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if limit <= 0:
                break
            if seq.seq_id in self._swap_prefetch:
                continue
            block_table = self.block_tables.get(seq.seq_id)
            if block_table is None:
                continue
            blocks = block_table.blocks
            if len(blocks) == 0:
                continue
            try:
                seq.recovery_state.align_to_num_blocks(len(blocks))
            except Exception:
                pass
            max_to_prefetch = min(limit, len(blocks))
            indices, _ = self._collect_swap_candidate_indices(
                seq, blocks, max_to_prefetch)
            if not indices:
                continue
            cpu_block_ids = [blocks[idx].block_id for idx in indices]
            if not cpu_block_ids:
                continue
            created_ts_ns = time.time_ns()
            self._swap_prefetch[seq.seq_id] = _SwapPrefetchEntry(
                request_id=seq_group.request_id,
                seq_id=seq.seq_id,
                indices=list(indices),
                cpu_block_ids=list(cpu_block_ids),
                created_ts_ns=created_ts_ns,
            )
            ready = len(indices)
            self._swap_prefetch_blocks_total += ready
            total_ready += ready
            limit -= ready
            try:
                log_recovery_event(
                    "SWAP_PREFETCH_READY",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "k_ready": ready,
                        "prefetch_blocks_total": self._swap_prefetch_blocks_total,
                        "prefetch_max_blocks": self._recovery_prefetch_max_blocks,
                    },
                    block_manager=self,
                )
            except Exception:
                pass
        return total_ready

    def can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_lookahead_slots: int,
                    k_swap_max: Optional[int] = None,
                    *,
                    stalled: bool = False,
                    reserve_blocks: Optional[int] = None) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        if k_swap_max is not None:
            host_blocks = 0
            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                try:
                    seq.recovery_state.align_to_num_blocks(
                        len(self.block_tables[seq.seq_id].blocks))
                    host_blocks += sum(
                        1 for hint in seq.recovery_state.storage_hint.values()
                        if hint == RecoveryStorageHint.HOST_STORED)
                except Exception:
                    host_blocks += len(self.block_tables[seq.seq_id].blocks)
            num_blocks_touched = min(k_swap_max, host_blocks)
            if num_blocks_touched <= 0:
                return AllocStatus.OK
            if self.block_allocator.get_num_total_blocks(
                    Device.GPU) < num_blocks_touched:
                return AllocStatus.NEVER
            free_gpu_blocks = self.block_allocator.get_num_free_blocks(Device.GPU)
            watermark_required = self.watermark_blocks
            if stalled and int(k_swap_max) == 1:
                # Unlock one-block stalled recovery with a looser watermark.
                watermark_required = max(0, self.watermark_blocks - 1)
            reserve_required = (self.watermark_blocks if reserve_blocks is None
                                else max(0, int(reserve_blocks)))
            headroom_required = max(watermark_required, reserve_required)
            if free_gpu_blocks - num_blocks_touched >= headroom_required:
                return AllocStatus.OK
            return AllocStatus.LATER
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(
        self,
        seq_group: SequenceGroup,
        k_swap_max: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping: List[Tuple[int, int]] = []
        total_done = 0
        k_remaining = k_swap_max if k_swap_max is not None else None
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue
            try:
                seq.recovery_state.align_to_num_blocks(len(blocks))
            except Exception:
                pass
            max_to_swap = k_remaining
            if max_to_swap is None:
                max_to_swap = len(blocks)
            max_to_swap = min(max_to_swap,
                              self.block_allocator.get_num_free_blocks(
                                  Device.GPU))
            if max_to_swap <= 0:
                continue
            indices: List[int] = []
            skipped_already_restored = 0
            prefetch_hit = False
            prefetch_age_ms: Optional[float] = None
            prefetched_indices, prefetch_skipped, prefetch_age_ms = (
                self._consume_prefetched_indices(seq, blocks, max_to_swap))
            if prefetched_indices:
                indices = prefetched_indices
                skipped_already_restored += prefetch_skipped
                prefetch_hit = True
            else:
                indices, skipped_scan = self._collect_swap_candidate_indices(
                    seq, blocks, max_to_swap)
                skipped_already_restored += (prefetch_skipped + skipped_scan)
            if not indices:
                continue
            blocks_to_swap = [blocks[idx] for idx in indices]
            k_req = len(blocks_to_swap)
            requested_cpu_block_ids = [blk.block_id for blk in blocks_to_swap]
            swap_start = time.perf_counter()
            seq_swap_mapping = self.block_allocator.swap(
                blocks=blocks_to_swap,
                src_device=Device.CPU,
                dst_device=Device.GPU,
            )
            k_done = len(seq_swap_mapping)
            swap_task_ms = max(0.0, (time.perf_counter() - swap_start) * 1000.0)
            if k_done > 0:
                observe_swap_task(swap_task_ms, k_done)
            if k_done <= 0:
                continue
            # Swap-in progress should refresh starvation timestamp immediately.
            seq.recovery_state.note_progress()

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))
            total_done += k_done
            if k_remaining is not None:
                k_remaining -= k_done

            swapped_indices = [
                idx for idx, cpu_block_id in zip(indices,
                                                 requested_cpu_block_ids)
                if cpu_block_id in seq_swap_mapping
            ]
            if not swapped_indices and k_done > 0:
                swapped_indices = indices[:k_done]
            committed_blocks = len(swapped_indices)
            try:
                seq.recovery_obs.swapin_blocks_total += k_done
            except Exception:
                pass
            try:
                # Phase1 micro-task bookkeeping: only committed swapped indices
                # advance frontier/restored_map so interrupted work is resumable.
                seq.ensure_recovery_blocks()
                seq.recovery_state.recovery_enabled = True
                seq.recovery_state.set_mode(
                    RecoveryMode.RECOVERY,
                    reason="swap_in_progress",
                    force=True,
                )
                seq.recovery_state.align_to_num_blocks(len(blocks))
                if swapped_indices:
                    commit_start, commit_end, new_frontier = (
                        seq.recovery_state.commit_progress(
                            "swap",
                            0,
                            0,
                            indices=swapped_indices,
                        ))
                else:
                    commit_start = 0
                    commit_end = 0
                    new_frontier = seq.recovery_state.get_restored_frontier()
                seq.recovery_state.swapin_blocks_total += k_done
            except Exception:
                pass
            try:
                add_restore_commit(committed_blocks)
                log_recovery_event(
                    "RECOVERY_PROGRESS_COMMIT",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "kind": "swap",
                        "blocks_committed": committed_blocks,
                        "range": [commit_start, commit_end],
                        "new_frontier": new_frontier,
                    },
                    block_manager=self,
                )
            except Exception:
                pass
            try:
                add_swap_micro(1)
                log_recovery_event(
                    "SWAP_IN_MICRO",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "k_req": k_req,
                        "k_done": k_done,
                        "prefetch_hit": int(prefetch_hit),
                        "prefetch_age_ms": (
                            round(prefetch_age_ms, 3)
                            if (prefetch_hit and prefetch_age_ms is not None)
                            else None),
                    },
                    block_manager=self,
                )
            except Exception:
                pass
            if skipped_already_restored > 0:
                try:
                    log_recovery_event(
                        "SWAP_IN_SKIP_ALREADY_RESTORED",
                        req_id=seq_group.request_id,
                        seq_id=seq.seq_id,
                        detail={
                            "count": skipped_already_restored,
                        },
                        block_manager=self,
                    )
                except Exception:
                    pass
            try:
                log_recovery_event(
                    "RECOVERY_REMAINING_HOST_BLOCKS",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "n_host": seq.recovery_state.count_hints()[1],
                    },
                    block_manager=self,
                )
            except Exception:
                pass
            try:
                log_recovery_event(
                    "SWAP_IN",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "blocks_count": len(seq_swap_mapping),
                        "bytes": None,
                    },
                    block_manager=self,
                )
            except Exception:
                pass
            if k_remaining == 0:
                break

        add_swap_in(len(physical_block_id_mapping), None)

        return physical_block_id_mapping, total_done

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap out.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        blocks_to_swap: List[Block] = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables.get(seq.seq_id)
            if block_table is None:
                continue
            blocks = block_table.blocks
            if len(blocks) == 0:
                continue
            try:
                seq.recovery_state.align_to_num_blocks(len(blocks))
            except Exception:
                pass
            for idx, blk in enumerate(blocks):
                if seq.recovery_state.is_pinned(idx):
                    continue
                if (seq.recovery_state.storage_hint.get(idx) ==
                        RecoveryStorageHint.HOST_STORED):
                    continue
                blocks_to_swap.append(blk)
        if len(blocks_to_swap) == 0:
            return False
        num_blocks_touched = self.block_allocator.get_num_full_blocks_touched(
            blocks_to_swap, device=Device.CPU)
        if num_blocks_touched <= 0:
            return False
        if self.block_allocator.get_num_total_blocks(
                Device.CPU) < num_blocks_touched:
            return False
        return (self.block_allocator.get_num_free_blocks(Device.CPU) -
                num_blocks_touched) >= 0

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap out.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue
            try:
                seq.recovery_state.align_to_num_blocks(len(blocks))
            except Exception:
                pass
            swappable_indices = [
                idx for idx in range(len(blocks))
                if (not seq.recovery_state.is_pinned(idx))
                and (seq.recovery_state.storage_hint.get(idx) !=
                     RecoveryStorageHint.HOST_STORED)
            ]
            if len(swappable_indices) == 0:
                try:
                    log_recovery_event(
                        "SWAP_OUT_PINNED_SKIPPED",
                        req_id=seq_group.request_id,
                        seq_id=seq.seq_id,
                        reason="all_blocks_pinned",
                        detail={
                            "pinned_blocks": len(
                                seq.recovery_state.get_pinned_blocks_sorted()),
                        },
                        block_manager=self,
                    )
                except Exception:
                    pass
                try:
                    log_recovery_event(
                        "SWAP_OUT",
                        req_id=seq_group.request_id,
                        seq_id=seq.seq_id,
                        reason="all_blocks_pinned",
                        detail={
                            "blocks_count": 0,
                            "bytes": None,
                            "swapped_indices_sample": [],
                            "pinned_indices_sample": seq.recovery_state.
                            get_pinned_blocks_sorted()[:16],
                        },
                        block_manager=self,
                    )
                except Exception:
                    pass
                continue
            blocks_to_swap = [blocks[idx] for idx in swappable_indices]
            requested_gpu_block_ids = [blk.block_id for blk in blocks_to_swap]

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks_to_swap,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))
            swapped_indices = [
                idx for idx, gpu_block_id in zip(swappable_indices,
                                                 requested_gpu_block_ids)
                if gpu_block_id in seq_swap_mapping
            ]
            try:
                seq.recovery_obs.swapout_blocks_total += len(seq_swap_mapping)
            except Exception:
                pass
            try:
                seq.ensure_recovery_blocks()
                seq.recovery_state.recovery_enabled = True
                seq.recovery_state.set_mode(
                    RecoveryMode.RECOVERY,
                    reason="swap_out_triggered",
                    force=True,
                )
                seq.recovery_state.align_to_num_blocks(len(blocks))
                for idx in swapped_indices:
                    seq.recovery_state.storage_hint[
                        idx] = RecoveryStorageHint.HOST_STORED
                    if idx < len(seq.recovery_state.restored_map):
                        seq.recovery_state.restored_map[idx] = False
                for idx in seq.recovery_state.get_pinned_blocks_sorted():
                    seq.recovery_state.storage_hint[
                        idx] = RecoveryStorageHint.GPU_RESIDENT
                    if idx < len(seq.recovery_state.restored_map):
                        seq.recovery_state.restored_map[idx] = True
                first_host = seq.recovery_state.first_host_block()
                seq.recovery_state.swap_frontier = (first_host
                                                    if first_host is not None
                                                    else 0)
                seq.recovery_state.swapout_blocks_total += len(seq_swap_mapping)
                seq.recovery_state.note_progress()
            except Exception:
                pass
            try:
                log_recovery_event(
                    "RECOVERY_HINT_SNAPSHOT",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "n_gpu": seq.recovery_state.count_hints()[0],
                        "n_host": seq.recovery_state.count_hints()[1],
                        "n_missing": seq.recovery_state.count_hints()[2],
                    },
                    block_manager=self,
                )
            except Exception:
                pass
            try:
                log_recovery_event(
                    "SWAP_OUT",
                    req_id=seq_group.request_id,
                    seq_id=seq.seq_id,
                    detail={
                        "blocks_count": len(seq_swap_mapping),
                        "bytes": None,
                        "swapped_indices_sample": swapped_indices[:16],
                        "pinned_indices_sample": seq.recovery_state.
                        get_pinned_blocks_sorted()[:16],
                    },
                    block_manager=self,
                )
            except Exception:
                pass

        add_swap_out(len(physical_block_id_mapping), None)

        return physical_block_id_mapping

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Get the number of tokens in blocks that are already computed and
        cached in the block manager for the sequence.
        """
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)
