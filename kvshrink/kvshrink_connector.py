# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KVShrink external KV cache connector (hybrid GDN/Mamba aware).

ONE path for every model. vLLM already describes a model as a list of
KV cache groups, so a pure-attention model is the one-group case and a
GDN/Mamba model is the two-group case; the connector is written against
groups and needs no knowledge of which it is serving.

The scheduler role plans (hit detection, load/save ReqMeta) against a
read-only store; the worker role executes (canonical page views, engine
transfers) and owns this rank's writer lease. Both live on the
connector class, under the Scheduler/Worker Side banners below.

These structures mirror the vLLM v0.23.0 HMA (Hybrid Memory Allocator)
KV cache layout:

- ``KVCacheConfig.num_blocks`` is the GLOBAL shared block pool size. All
  KV cache groups share one block id space; each layer has its own block
  table.
- Physical page for (layer, block_id) = the layer pool's row at
  index block_id. Every pool is row-addressable: v0.23 lays out all
  groups page-contiguous with dim 0 = block index (mamba pages padded
  to a common width via as_strided).
- Mamba layers expose ``kv_caches[layer_name]`` as a LIST of tensors
  (conv_state, ssm_state) sharing one storage, page-wise concatenated
  (conv at [0, conv_bytes), ssm after). The connector hands both parts
  to the store separately; their union is one logical page.

GDN slot contract (v0.23.0): ``preprocess_mamba`` (the prev->curr slot
copy) runs in ``execute_model`` BEFORE the connector's
``bind_connector_metadata``/``start_load_kv``, so every external GDN
snapshot is written directly into the CURR slot during forward
(waited at ``start_load_kv``, before forward begins). There is no
"prev" write path and no vLLM patch.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_world_group,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from .async_load_config import AsyncLoadLayerConfig
    from vllm.config import ModelConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

from iaxl import KVStore, generate_block_hashs, setup_root_logger
from iaxl.kvflow.flow import Task

from .async_load_config import load_async_load_layer_config_from_env
setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


@dataclass
class ReqMeta:
    """All transfer instructions for one request in one step.

    The unit the worker iterates over: for each ReqMeta it executes
    every GroupTransferMeta (loads before forward, saves after).

    Fields:
    - group_ops: one GroupTransferMeta per KV cache group. Requests on
      hybrid models always have per-group plans: attention blocks and
      the mamba snapshot move independently.
    - external_hit_tokens: how many tokens the core accepted as
      externally backed for this request. Used for evidence and
      sanity checks (a LOAD plan with accepted external tokens but zero
      ops is the fail-closed case).
    - is_async: LOAD plans only. When set, vLLM parked this request in
      WAITING_FOR_REMOTE_KVS rather than giving it a forward step: the
      worker must keep the transfer alive ACROSS steps and name the
      request in get_finished() once enough of it has landed, otherwise
      the request never becomes runnable again.
    - async_load_layers: how many LEADING ATTENTION layers must land
      before the request is released; -1 means every layer. Counted in
      attention layers only: recurrent state is never partially
      released, because the whole snapshot is read at the very start of
      forward. The worker adds every mamba op to the release gate
      regardless of this number.
    """
    group_ops: tuple[GroupTransferMeta, ...] = ()
    external_hit_tokens: int = 0
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqGroupState:
    """Per-group mutable state for one request (scheduler side).

    - block_ids: our copy of vLLM's block table for this group --
      block ids are indices into the group's GPU block pool. Kept in
      sync via update_state_after_alloc (full replace, for new/resumed
      requests) and the scheduled_cached_reqs.new_block_ids append in
      build_connector_meta (for running requests). See the Scheduler
      Side Methods section for the two sync channels and their ordering.
    - next_stored_chunk_idx: incremental-save cursor. Block indices
      below it were already emitted in earlier save plans; on
      preemption resume (or any progress regression) it rolls back so
      blocks whose saves may never have landed are emitted again.
    """
    block_ids: list[int] = field(default_factory=list)
    next_stored_chunk_idx: int = 0


@dataclass
class ReqState:
    # Live, append-only list giving the request's block identity as it
    # GROWS: decode-completed blocks are hashed onto no scheduler
    # structure (CachedRequestData carries tokens, not hashes, and no
    # connector hook fires for running requests -- v0.23 calls the KV
    # connector's update_state_after_alloc only from the waiting path),
    # so the save path reads the live list itself. Which list depends on
    # the hash source: vLLM's own ``block_hashes`` ("vllm") or
    # ``all_token_ids`` ("legacy", re-hashed by us). v0.23 only ever
    # appends to either (Request.update_block_hashes / append_tokens),
    # never rebinds; same pattern as LMCache's
    # ConstantList(request.all_token_ids). None for requests registered
    # without a live Request (unit tests). Dropped in
    # on_request_finished together with the rest of the state.
    live_source: Any = None
    block_hashes: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    snapshot_boundary: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    # External tokens accepted by the core in the CURRENT scheduling
    # pass (recorded by update_state_after_alloc): tokens the core will
    # skip recompute for, so the worker MUST restore them before
    # forward. Consumed by build_resumed_load_meta's fail-closed guard:
    # a resumed request with pending external tokens but no restorable
    # pages must fail-stop, never enter forward reading unrestored KV.
    pending_load_tokens: int = 0
    # Last authoritative progress seen by the save path
    # (num_computed + scheduled of the last save plan). Used for
    # fail-closed regression detection: any drop below this value rolls
    # save cursors back even if the resumed flag is missing.
    last_known_progress: int = 0
    # Async load bookkeeping. When is_async is set, this request was
    # admitted with load_kv_async=True: vLLM parked it in
    # WAITING_FOR_REMOTE_KVS and runs OTHER requests while its pages
    # stream in, instead of stalling a forward step on us. The request
    # only becomes runnable again once the worker names it in
    # get_finished(); until then it consumes blocks but no compute.
    #
    # async_load_layers is how many LEADING attention layers must land
    # before we release it -- the rest are waited layer by layer during
    # forward. -1 means "every layer", i.e. no early release.
    is_async: bool = False
    async_load_layers: int = -1
    # Whether the async load plan was already handed to the worker.
    # vLLM calls update_state_after_alloc TWICE for an async request --
    # once when it allocates, once after the load completes -- so
    # without this the second call queues a SECOND transfer for a
    # request that is running by then, and reporting that one finished
    # trips vLLM's own assert (a finished-recving request must be
    # parked or done, never running).
    async_plan_emitted: bool = False


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker transfer plan.

    ``reqs_to_load`` are LOAD plans (executed before forward),
    ``reqs_to_save`` are SAVE plans (executed after forward). Both map
    request id to ``layout.ReqMeta``; the worker only ever sees this
    object, so each plan is fully self-describing.
    """
    reqs_to_load: RequestMetadata = field(default_factory=RequestMetadata)
    reqs_to_save: RequestMetadata = field(default_factory=RequestMetadata)


# ======================================================================
# lookup vocabulary (shared by scheduler, worker and the store)
# ======================================================================
# Cache hit policy for hybrid (Full Attention + GDN) models.
#
# Implements the hit-detection algorithm, verified against vLLM v0.21.0's
# HybridKVCacheCoordinator semantics:
#
# - Attention groups: left-to-right prefix scan; the prefix must exist
#   contiguously (downward-closed).
# - Mamba/GDN groups: right-to-left scan for the NEAREST committed snapshot;
#   earlier snapshots need not exist. Candidates are aligned down to
#   mamba_align_size and the final boundary recomputes exactly 1 token.
# - Multiple groups converge via fixed-point iteration (full attention
#   first, then mamba groups).
# - Store presence lookups return a bool; any error is a MISS
#   (fail-closed, see lookup_boundary).
#
# How a group maps onto the store
# -------------------------------
# The store keys data as ``(label, chunk_id, tensor_key)``:
#
# - ``tensor_key`` is the layer name, so layers never collide and a
#   caller can wait for one layer while the others stream.
# - ``chunk_id`` is the block's content hash.
# - ``label`` is per KV cache group (``g{idx}``): the same prefix hash
#   exists in every group, and the durability record is keyed by
#   ``(label, chunk_id)`` without the layer, so groups sharing a label
#   would be tracked as one unit despite having different lifetimes and
#   state kinds. Cross-rank isolation needs no key component: each rank
#   persists to its own ``{model}_rank{r}`` directory, and the
#   controller opens only the rank0 one.
#
# Why the whole group goes in one call
# ------------------------------------
# Not because the engine forbids mixed shapes -- it iterates the tensor
# dict and stores each entry independently, so it happily takes
# everything at once. The real reason is the label contract: an
# explicit-label call is finalized (Record-committed) as a whole, so a
# call must carry exactly one group's WHOLE layer set. Submitting all
# groups in one call would commit four labels' ledgers with data for
# only some of them. One group per call also makes the write atomic per
# group: a block is finalized once, with all of its layers, so presence
# IS the commit. There is no second phase to publish and therefore
# nothing that can dangle.
#
# (The canonical int8 page views are still what makes ANY of this
# expressible: conv/ssm over one storage become uniform rows the engine
# can chunk along dim 0, whatever shape the original tensors had.)


def group_label(group_idx: int) -> str:
    """Store label for one group's block space.

    Colon-free because the store splits full labels on ':'; see the
    section comment above for why one label per group is required.
    """
    return f"g{int(group_idx)}"


def lookup_boundary(store: "KVStore", key: "CacheKey") -> bool:
    """Is this boundary present in the store?

    The check runs under the key's own rank label: each rank keeps its
    own presence record, and the controller process shares one with the
    rank-0 worker only, so peer ledgers are not queryable here. TP
    ranks save in lockstep, so rank 0 present stands for all; a rank
    that diverged anyway fails loudly at load time (the native layer
    raises on a missing key).

    Any error is a MISS. A wrong hit silently corrupts output; a
    wrong miss costs one recompute.
    """
    try:
        present = store.has([key.hash_str],
                            label=group_label(key.group_idx))
        return bool(present and present[0])
    except Exception:  # pragma: no cover - fail closed to MISS
        logger.exception("lookup error; treating as MISS")
        return False


# ======================================================================
# hybrid layout vocabulary
# ======================================================================
@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage spec.

    vLLM buckets layers that share the same storage spec into "KV cache
    groups" (``KVCacheConfig.kv_cache_groups``), each with its own
    independent block pool. A typical hybrid model has two: full-
    attention layers (block-sliced pages, arbitrarily offsettable) and
    GDN/mamba layers (fixed-size recurrent state, whole-snapshot access
    only at segment boundaries).

    What we do with it:

    - Isolation: ``group_idx`` is part of every CacheKey / boundary
      identity, so the same prefix hash in the attention group and the
      mamba group can never alias each other.
    - Bookkeeping: the scheduler tracks per-request block_ids per group
      (each group's block pool is allocated independently).
    - Behavior dispatch: ``kind`` selects the access pattern --
      attention pages are sliceable per ``block_size`` tokens, mamba
      groups are stored/loaded whole at aligned boundaries.
    - Validation: the store fail-closed checks the group exists and
      page sizes match before any chunk move.
    """

    group_idx: int
    kind: str  # "attention" | "mamba"
    layer_names: tuple[str, ...]
    block_size: int  # tokens per block for this group
    mamba_align_size: Optional[int]  # offload chunk alignment for mamba
    # vLLM's own spec for this group, kept so the hit policy can hand it
    # back to vLLM's matching code instead of reimplementing it.
    spec: object = None


@dataclass(frozen=True)
class CacheKey:
    """Logical key for one page, or for a whole boundary.

    A boundary is the same key with ``layer_name == ""``; it names the
    group's blocks at that hash rather than one layer's page. Group,
    hash and layer are the whole identity: each rank persists to its
    own store directory, so no rank component is needed (and within
    one directory, one label per group keeps the two address spaces
    apart).
    """
    block_hash: object  # int (unit tests) or bytes/str (vLLM)
    group_idx: int
    layer_name: str  # "" addresses the boundary, not one layer

    @property
    def hash_str(self) -> str:
        """Stable string form for paths / JSON (bytes -> hex)."""
        h = self.block_hash
        if isinstance(h, bytes):
            return h.hex()
        return str(h)

    @property
    def boundary_key(self) -> tuple[str, int]:
        """Identity of a boundary: hash and group."""
        return (self.hash_str, self.group_idx)


def make_boundary_key(group_idx: int,
                      block_hash: object) -> "CacheKey":
    """A group's key at one block hash, with no layer: the address the
    hit policy asks about and the address the save/load builders expand
    into per-layer page keys."""
    return CacheKey(
        block_hash=block_hash, group_idx=group_idx, layer_name="")


@dataclass
class GroupTransferMeta:
    """Per-group transfer instructions for one request (one step).

    The worker receives ONLY this metadata -- it never sees the
    scheduler's bookkeeping. So each op must fully describe one data
    movement: WHICH group it belongs to, WHERE the data lives in the
    external store, and WHICH GPU blocks are involved.

    Fields:
    - group_idx: which KV cache group this op targets. Each group has
      its own independent GPU block pool and storage rules, so the
      worker must know the group to interpret gpu_block_ids and keys.
      The group's kind (attention/mamba) is deliberately NOT duplicated
      here: the worker derives it from its own registered GroupInfo
      (``self._groups[op.group_idx].kind``), keeping a single source of
      truth.
    - keys / gpu_block_ids: parallel tuples pairing store address with
      GPU destination -- keys[i] is the external-store identity (which
      chunks to read or write), gpu_block_ids[i] is the GPU block the
      page is loaded into (LOAD) or drained from (SAVE).

    GDN loads always target the CURR state slot (see module docstring);
    there is no slot field because there is no choice to make.
    """
    group_idx: int
    keys: tuple[CacheKey, ...] = ()
    gpu_block_ids: tuple[int, ...] = ()


# ======================================================================
# KVCacheConfig parsing
# ======================================================================
# Parse the real vLLM KVCacheConfig into KVShrink hybrid structures.
#
# Verified against vLLM v0.23.0 + Qwen3.5-4B TP2 (layout unchanged
# since v0.21:
#
# - 4 kv_cache_groups: 3 x MambaSpec(GDN) + 1 x FullAttentionSpec
# - 8 kv_cache_tensors, each shared by 4 consecutive layers (3 linear + 1 full)
# - vLLM pads the attention block size so ALL groups share one page size
# - layer names like "language_model.model.layers.0.linear_attn"
# - Mamba layers: kv_caches[layer] is a LIST [conv_tensor, ssm_tensor]
#   sharing one storage; page layout = conv bytes then ssm bytes.
#
# Fail-closed rules (never guess):
# - unknown spec types -> KVShrinkParseError
# - layers of one group disagreeing on page size or block size ->
#   KVShrinkParseError (one group is one engine call, which requires
#   identically shaped views)
# - KVCacheTensor is (size, shared_by) only, so every page view is
#   contiguous from storage offset 0 with stride == page_size.
class KVShrinkParseError(ValueError):
    """Raised when the vLLM cache config cannot be parsed safely
    (unknown spec or inconsistent layout). The parse never
    guesses (fail closed)."""
    pass


def validate_codec_env() -> None:
    """Warn when the codec is configured to lose bits.

    History: this used to be a hard startup refusal. It existed because
    mamba pages rode the same int8 pipeline as attention pages, so a
    lossy transform could not be kept away from the recurrent state.
    The entry-flags mechanism changed that -- KVStore now hard-clears
    the lossy bit for opaque (fused multi-dtype) layers no matter what,
    so ``IAXL_KV_LOSSY_TRUNC`` becomes a scoped knob: it applies to
    attention pools only and structurally cannot reach mamba state.

    Downgraded to a log line so operators get the knob back without
    losing the guard; nothing else reads this env on the hybrid path.

    ``IAXL_KV_DATA_SHUFFLE`` is byte reordering and fully reversible,
    and only fires on bf16 staging, so it stays untouched here.
    """
    lossy = os.getenv("IAXL_KV_LOSSY_TRUNC", "0").strip()
    if lossy in ("", "0"):
        return
    logger.warning(
        "IAXL_KV_LOSSY_TRUNC=%s is enabled. On hybrid models the flag "
        "gates per pool entry: attention layers honor it, while mamba "
        "state is exempted structurally (lossy bit hard-cleared by "
        "KVStore).", lossy)


def _spec_kind(spec: object) -> str:
    """Mamba or attention; unknown spec types raise KVShrinkParseError
    (fail closed). Sliding-window specs are AttentionSpec subclasses and
    are intentionally NOT distinguished: their block layout is the
    attention one, and the hit-rule difference lives in vLLM's spec
    registry, which is consulted directly."""
    if isinstance(spec, MambaSpec):
        return "mamba"
    if isinstance(spec, AttentionSpec):
        return "attention"
    raise KVShrinkParseError(
        f"Unsupported KV cache spec {type(spec).__name__}")


def _iter_layer_specs(group_spec: Any) -> Iterator[tuple[str, object]]:
    """Yield (layer_name, spec) pairs, expanding UniformTypeKVCacheSpecs."""
    spec = group_spec.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        per_layer = spec.kv_cache_specs
        for name in group_spec.layer_names:
            yield name, per_layer[name]
    else:
        for name in group_spec.layer_names:
            yield name, spec


def parse_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> tuple[list[GroupInfo], int]:
    """Return (groups, num_blocks).

    What we pull out of vLLM's KVCacheConfig, and where each piece goes:

    - ``kv_cache_groups`` -> ``groups`` (list[GroupInfo]): vLLM splits a
      hybrid model's KV cache into groups of layers with the same storage
      spec (e.g. group 0 = full-attention layers, group 1 = GDN/mamba
      layers). For each group we record its kind, layer names and
      block_size (tokens per block); mamba groups additionally get
      mamba_align_size. Consumed by:
      the scheduler (per-group block_ids bookkeeping and save/load
      planning), the store presence check (fail-closed, storage
      labels embed group_idx), and policy (attention pages are sliceable
      per block, mamba groups only at boundaries).
    - ``num_blocks`` -> int: global block-pool size.

    Per-layer page geometry is deliberately NOT parsed here anymore:
    pools are bound by KVStore directly from the live tensors, which
    carry their own layout. Raises KVShrinkParseError on unknown specs
    or any inconsistency (fail closed).
    """
    num_blocks = kv_cache_config.num_blocks
    groups: list[GroupInfo] = []

    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        kind = None
        block_size = None
        per_layer_specs: list[tuple[str, object]] = list(_iter_layer_specs(g))

        for name, spec in per_layer_specs:
            sk = _spec_kind(spec)
            if kind is None:
                kind = sk
            elif sk != kind:
                raise KVShrinkParseError(
                    f"Group {g_idx} mixes spec kinds {kind} and {sk}")
            bs = int(spec.block_size)
            if block_size is None:
                block_size = bs
            elif bs != block_size:
                raise KVShrinkParseError(
                    f"Group {g_idx} layers have differing block sizes")

        mamba_align = None
        if kind == "mamba":
            mamba_mode = per_layer_specs[0][1].mamba_cache_mode
            # Every GDN snapshot is addressed by an aligned boundary, and
            # the kernels only read the block-table column for the
            # current boundary in 'align' mode. In any other mode a
            # request keeps one max_model_len-sized block that is never
            # boundary-addressable, so there is nothing we could key a
            # snapshot on. vLLM silently rewrites the mode to 'none' when
            # prefix caching is off, and it defaults prefix caching off
            # for hybrid models, so this is the common misconfiguration.
            if mamba_mode != "align":
                raise KVShrinkParseError(
                    f"Group {g_idx} has mamba_cache_mode={mamba_mode!r}, "
                    "but the external cache requires 'align'. Start vLLM "
                    "with --enable-prefix-caching --mamba-cache-mode align "
                    "(vLLM forces the mode to 'none' when prefix caching "
                    "is disabled, and disables prefix caching by default "
                    "for hybrid models)")
            mamba_align = block_size
        group = GroupInfo(
            group_idx=g_idx,
            kind=kind,
            layer_names=tuple(g.layer_names),
            block_size=block_size,
            mamba_align_size=mamba_align,
            spec=per_layer_specs[0][1],
        )

        groups.append(group)

    # One block size for the whole model. vLLM aligns its groups onto a
    # common block size (a GDN model's attention groups take the mamba
    # size), and the request's block hashes are computed at exactly that
    # size -- so hash i names block i in EVERY group, which is what lets
    # one hash address a boundary across groups. If a model ever arrived
    # with mixed sizes, that correspondence would be silently wrong for
    # all but one group, so refuse it instead.
    sizes = {g.block_size for g in groups}
    if len(sizes) != 1:
        raise KVShrinkParseError(
            f"KV cache groups have different block sizes {sorted(sizes)}; "
            "the external cache addresses every group with one block hash "
            "and cannot express that")
    return groups, num_blocks


def save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")


# ======================================================================
# longest-hit policy
# ======================================================================
class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs that we must supply.

    ``find_longest_cache_hit`` asks a block pool "is this hash cached,
    for these groups?" and otherwise only compares the answers. Pointing
    that question at the external store is the whole adaptation; the
    matching rules stay upstream's.

    A hit returns a placeholder rather than a block: the callers only
    count and position the results, and the blocks the request will
    actually use are allocated by vLLM afterwards.
    """

    __slots__ = ("_present",)

    # Stands in for a skipped block. vLLM inserts it as padding and only
    # ever counts it, so it needs no identity beyond being a value.
    null_block = object()

    def __init__(self, present: Callable[[int, object], bool]):
        self._present: Callable[[int, object], bool] = present

    def get_cached_block(
        self, block_hash: object, kv_cache_group_ids: list[int],
    ) -> Optional[list[object]]:
        blocks = []
        for group_id in kv_cache_group_ids:
            if not self._present(group_id, block_hash):
                return None
            blocks.append(_StoreAsBlockPool.null_block)
        return blocks


class HybridHitPolicy:
    """Fixed-point multi-group hit detection (pure function, testable)."""

    def __init__(
        self,
        groups: list[GroupInfo],
        present: Callable[[int, object], bool],
        hash_block_size: int,
        num_computed_tokens: int,
    ):
        """Configure the policy for one request: the groups, a
        store-presence predicate ``present(group_idx, block_hash)`` and
        the request's computed tokens. Orders groups attention-first
        (tighter initial bound) and takes the global mamba alignment as
        the minimum across mamba groups."""
        self._groups: list[GroupInfo] = groups
        self._present: Callable[[int, object], bool] = present
        self._hash_block_size: int = hash_block_size
        self._num_computed: int = num_computed_tokens
        # full attention first (tighter initial bound)
        self._ordered: list[GroupInfo] = sorted(
            groups, key=lambda g: 0 if g.kind == "attention" else 1)
        self._mamba_align: Optional[int] = None
        for g in groups:
            if g.kind == "mamba":
                a = g.mamba_align_size
                self._mamba_align = a if self._mamba_align is None \
                    else min(self._mamba_align, a)

    # ------------------------------------------------------------------
    def _lookup(self, group: GroupInfo, block_hashes: list[int],
                candidate: int) -> int:
        """How far this group alone is restorable, in tokens.

        The matching rules are vLLM's, called here rather than copied:
        full attention is a downward-closed prefix scan, a recurrent
        group is a right-to-left search for the nearest snapshot that
        also sits on an alignment boundary, and each has its own
        handling of EAGLE and of a block size that differs from the hash
        granularity. Reimplementing that would mean our hit length
        silently drifting from vLLM's whenever upstream refines it.

        The one substitution is where "cached" is looked up: vLLM asks
        its GPU block pool, we ask the external store.
        """
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
        manager_cls = KVCacheSpecRegistry.get_manager_class(group.spec)
        # vLLM indexes the hash list directly out to max_length, so the
        # caller owes it a length its own hashes cover. Its scheduler
        # gets this for free (the bound comes from the same request);
        # ours can be a boundary the request has not reached.
        max_length = min(candidate,
                         len(block_hashes) * group.block_size)
        blocks = manager_cls.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_length,
            kv_cache_group_ids=[group.group_idx],
            block_pool=_StoreAsBlockPool(self._present),
            kv_cache_spec=group.spec,
            drop_eagle_block=False,
            alignment_tokens=(self._mamba_align or group.block_size),
        )
        return len(blocks[0]) * group.block_size

    # ------------------------------------------------------------------
    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> int:
        """Fixed-point convergence over all groups. Returns the
        restorable boundary in tokens; 0 = miss.

        Every group is looked up on its own: a hit on group A says
        nothing about group B, whose blocks live under a different
        label and may never have been written.
        """
        candidate = max_length
        if self._mamba_align is not None:
            # the last prompt token is always recomputed (logprobs + state)
            a = self._mamba_align
            candidate = min(candidate - 1, (candidate - 1) // a * a)

        while True:
            changed = False
            for group in self._ordered:
                hit = self._lookup(group, block_hashes, candidate)
                if hit < candidate:
                    candidate = hit
                    changed = True
                if candidate <= self._num_computed:
                    return 0
            if not changed:
                break
        return candidate if candidate > self._num_computed else 0


# ======================================================================
# worker bookkeeping
# ======================================================================
# Async load tracking uses main's four parallel dicts verbatim:
#   _pending_load_tasks                     -- in flight
#   _early_promoted_tasks                       -- released, draining
#   _active_promoted_tasks                      -- draining this forward
# One extra piece of information rides alongside: _gated_keys[req],
# the pool-key set that MUST land before release (every recurrent
# layer plus the configured first-N attention layers). A recurrent
# state is consumed whole at the start of forward, so releasing a
# request whose mamba pages are still in flight would read stale
# memory -- nothing downstream can tolerate half of it.


############################################################
# Connector
############################################################

class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    """KVShrink external KV cache connector.

    ONE path for every model. vLLM already describes a model as a list
    of KV cache groups, so a pure-attention model is the one-group case
    and a GDN/Mamba model is the two-group case; the scheduler and
    worker sections below are written against groups and need no
    knowledge of which they are serving.

    A recurrent group changes nothing about how bytes are stored: every
    group is a block space in one store, told apart by its label.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(
        cls, extra_config: dict[str, Any]
    ) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.vllm_config: VllmConfig = vllm_config
        self.model_config: ModelConfig = vllm_config.model_config
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers: int = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device: str = vllm_config.device_config.device_type
        self.rank: int = get_world_group().rank if model_parallel_is_initialized() else 0

        self._async_load_layer_config: AsyncLoadLayerConfig = (
            load_async_load_layer_config_from_env(
                num_layers=self.num_layers,
            )
        )

        self.kvstore: Optional[KVStore] = None

        # Scheduler-side planning state (harmless on the worker role,
        # which never receives scheduler hooks). Everything planned
        # against -- groups, labels, hash granularity -- is built
        # once in _init_kv_stack; mutating bookkeeping starts empty.
        self._req_states: dict[str, ReqState] = {}
        # Async requests whose load plan has not been emitted yet. See
        # update_state_after_alloc for why this cannot be derived from
        # the scheduler output.
        self._async_load_pending: set[str] = set()

        # In-flight ASYNC loads: req_id -> _AsyncLoad. Entries
        # deliberately OUTLIVE the step that submitted them -- the whole
        # point is that the request is not occupying a forward step
        # while its pages arrive. Entries leave in two stages: released
        # (reported through get_finished, so vLLM may schedule the
        # request again) and then drained (its remaining layers waited
        # by the per-layer hooks during that forward).
        # Async load tracking -- main's four parallel dicts verbatim,
        # plus one addition: _gated_keys[req], the pool keys that must
        # land before release (every recurrent layer + configured
        # first-N attention layers).
        self._pending_load_tasks: dict[str, dict[str, Task]] = {}
        self._gated_keys: dict[str, frozenset[str]] = {}
        self._early_promoted_tasks: dict[str, dict[str, Task]] = {}
        self._active_promoted_tasks: dict[str, dict[str, Task]] = {}
        # Async save lifecycle (same shape as main): put tasks per
        # contributing request, drained in get_finished; finished
        # requests are held until their loads AND saves have landed.
        self._current_put_tasks: dict[str, list[dict[str, Task]]] = {}
        self._deferred_finished_req_ids: set[str] = set()

        if kv_cache_config is not None:
            self._init_kv_stack(vllm_config, role, kv_cache_config)
        else:
            self._bind_cpu_affinity()
            self._bind_intel_accel()

    ############################################################
    # Construction
    ############################################################

    def _init_kv_stack(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Build the hybrid stack for this role.

        Scheduler role plans over a READ-ONLY store (presence checks
        only, no writer lease, no GPU pool). Worker role executes over a
        WRITER store holding this rank's single-writer lease -- but the
        store needs kv_caches, which arrive later, so the worker's
        KVStore is built in register_kv_caches.
        """
        pc = vllm_config.parallel_config
        tp_size = pc.tensor_parallel_size
        # Fail-closed: pipeline parallelism shards LAYERS across ranks,
        # so one rank holds half the model's KV and its pages alone name
        # only half a block. Every key would silently address a partial
        # state. Nothing here can degrade safely, so refuse at startup.
        if pc.pipeline_parallel_size != 1:
            raise RuntimeError(
                "kvshrink hybrid: pipeline parallelism is not supported "
                f"(pipeline_parallel_size={pc.pipeline_parallel_size}); "
                "each rank would persist only its own layers' pages. "
                "Set pipeline_parallel_size=1 or the KV connector.")
        if role == KVConnectorRole.WORKER:
            # parallel_config.rank is the authoritative identity and
            # carries no dependency on init order. (In this v0.23 engine
            # path the connector is actually built after distributed
            # init, so get_world_group() would also be right here -- the
            # config value stays correct either way.) Two ranks claiming
            # the same rank would land in the SAME persist dir and port
            # (KVStore derives both from this value) and clobber each
            # other's shards.
            rank = pc.rank
        else:
            # Scheduler keys carry no rank: each process talks to its
            # own per-rank store directory, and the controller opens
            # the rank0 one.
            rank = 0
        model_config = vllm_config.model_config

        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        self._hash_block_size = hash_block_size
        # Fail-closed: a lossy codec would corrupt GDN state (see the
        # function for why this path cannot tolerate what the
        # attention-only path is designed to).
        validate_codec_env()

        groups, num_blocks = parse_kv_cache_config(
            kv_cache_config)

        # Fail-closed: speculative decoding widens the GDN state gather.
        # v0.23.0's mamba_get_block_table_tensor returns
        # block_table[start : start + 1 + num_speculative_blocks] and the
        # decode path reads all of those columns, but an external
        # snapshot only ever restores column 0 (the block holding this
        # step's last scheduled token). Serving a hit would let the
        # kernel read unrestored speculative slots. The field exists
        # only on MambaSpec, so only mamba groups are checked.
        for g, parsed in zip(kv_cache_config.kv_cache_groups, groups):
            if parsed.kind != "mamba":
                continue
            spec_blocks = g.kv_cache_spec.num_speculative_blocks
            if spec_blocks:
                raise RuntimeError(
                    "kvshrink hybrid: speculative decoding is not "
                    f"supported (group has num_speculative_blocks="
                    f"{spec_blocks}); the external GDN snapshot only "
                    "restores the non-speculative state slot. Disable "
                    "speculative decoding or the KV connector.")
        self._groups = groups
        # Authoritative TP rank for addressing/labels. Kept separate from
        # self.rank (a world-group read guarded on init order) so
        # rank-sensitive code paths have one stable source.
        self._rank = rank
        # Attention layers in group order, used to size the
        # early-release prefix. Mamba layers are deliberately absent:
        # they are never partially released (see _decide_async).
        self._attention_layers = tuple(
            ln for g in groups if g.kind != "mamba"
            for ln in g.layer_names)
        # A recurrent group changes only which block hashes we ask
        # about (see _choose_block_hash_source); the storage below is
        # the same.
        recurrent = any(g.kind == "mamba" for g in groups)
        self._block_hash_source = self._choose_block_hash_source(recurrent)

        if role == KVConnectorRole.SCHEDULER:
            # Presence-only store: the scheduler asks whether boundaries
            # are readable and never moves bytes.
            self.kvstore = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(i) for i in range(self.num_layers)],
                tp_size=self.tp_size,
            )
        else:
            # One store label per group (see the lookup-vocabulary
            # section for why groups cannot share one).
            self._labels = [group_label(g.group_idx) for g in groups]
            # layer_name -> group idx, every cached layer.
            self._layer_group = {
                ln: g.group_idx for g in groups for ln in g.layer_names}
            # attention layer_name -> group idx (mamba layers map out).
            self._attn_layer_group = {
                ln: g.group_idx for g in groups if g.kind != "mamba"
                for ln in g.layer_names}

        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, "
            "hash_block_size=%d, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), hash_block_size, tp_size, rank)
        logger.info(
            "kvshrink groups: %s",
            [(g.group_idx, g.kind, g.block_size) for g in groups])

    @staticmethod
    def _choose_block_hash_source(recurrent: bool) -> str:
        """Which block-identity scheme to key the cache with.

        Defaults preserve what each layout already wrote, because
        switching schemes does not migrate data -- it renames it, and
        every existing entry becomes unreachable until it is written
        again. The block layout keeps its own token-derived hashes; the
        boundary layout keeps vLLM's, which is what it shipped with.

        This is a DATA COMPATIBILITY switch, not a behavioural one: the
        two sources produce different key values, so flipping it makes
        every previously written entry unreachable (a cold cache, not a
        corrupt one). Each layout therefore keeps the source it was
        written with unless an operator says otherwise.

        ``KVSHRINK_BLOCK_HASH_SOURCE=vllm|legacy`` overrides, for
        operators willing to trade one cold warm-up for a single scheme
        across both layouts.
        """
        choice = (os.getenv("KVSHRINK_BLOCK_HASH_SOURCE") or "auto").lower()
        if choice == "auto":
            return "vllm" if recurrent else "legacy"
        if choice not in ("vllm", "legacy"):
            raise ValueError(
                "KVSHRINK_BLOCK_HASH_SOURCE must be auto, vllm or "
                f"legacy; got {choice!r}")
        return choice

    def _bind_cpu_affinity(self) -> None:
        if self.vllm_device == "cpu":
            return

        omp_bind = envs.VLLM_CPU_OMP_THREADS_BIND
        if not omp_bind or omp_bind in ("all", "auto"):
            raise ValueError(
                "VLLM_CPU_OMP_THREADS_BIND must assign CPUs to each worker"
            )

        worker_cpu_specs = omp_bind.split("|")
        if len(worker_cpu_specs) < self.tp_size:
            raise ValueError(
                f"VLLM_CPU_OMP_THREADS_BIND has {len(worker_cpu_specs)} entries, "
                f"but tensor parallel size is {self.tp_size}"
            )

        cpu_ids: set[int] = set()
        for part in worker_cpu_specs[self.rank].split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-", maxsplit=1))
                if start > end:
                    raise ValueError(f"Invalid CPU range: {part}")
                cpu_ids.update(range(start, end + 1))
            else:
                cpu_ids.add(int(part))

        if not cpu_ids:
            raise ValueError(f"No CPUs configured for rank {self.rank}")
        os.sched_setaffinity(0, cpu_ids)
        logger.info("Bound rank %d to CPUs %s", self.rank, sorted(cpu_ids))

    def _bind_intel_accel(self) -> None:
        for source, target in (
            ("KVSHRINK_QAT_DEVICES", "IAXL_QAT_DEVICES"),
            ("KVSHRINK_DSA_DEVICES", "IAXL_DSA_WQS"),
        ):
            spec = os.getenv(source)
            if not spec:
                continue
            devices = spec.split("|")
            if len(devices) <= self.rank:
                raise ValueError(
                    f"{source} has {len(devices)} entries, but rank is {self.rank}"
                )
            os.environ[target] = devices[self.rank]
            logger.info("Bound rank %d: %s=%s", self.rank, target, devices[self.rank])

    ############################################################
    # Scheduler Side Methods
    ############################################################
    # Scheduler-side request state machine: hit detection + load/save
    # plan builder. The vLLM trigger map below says WHEN each entry
    # point fires and WHAT it is for; per-function docstrings describe
    # HOW.
    #
    # One scheduling pass looks like this:
    #
    # 1. NEW request arrives -> the core asks the connector
    #    ``get_num_new_matched_tokens(request, num_computed_tokens)``.
    #    Purpose: how many tokens beyond the local prefix-cache hit can be
    #    treated as already computed thanks to the external store.
    #    We run the hit policy over the request's block hashes (Record-
    #    gated, always synchronous), remember the authoritative restore
    #    point as ``snapshot_boundary``, and return the external token
    #    count. The core then skips recomputing those tokens.
    # 2. Block allocation succeeded (same pass) -> the core calls
    #    ``connector.update_state_after_alloc(request, blocks,
    #    num_external_computed_tokens)``.
    #    Purpose: tell us where the GPU blocks landed and how many
    #    external tokens the core accepted (i.e. will skip recompute for).
    #    We snapshot per-group block_ids and set ``pending_load_tokens`` --
    #    the external tokens the worker MUST restore before forward.
    # 3. End of the pass: the core calls
    #    ``connector.build_connector_meta(scheduler_output)``. Per-request
    #    plans ship to the worker inside the connector metadata. Four
    #    kinds of work:
    #
    #    a) LOAD plan for NEW requests -> build_load_meta.
    #       Restores pages up to the snapshot boundary recorded in step 1
    #       (never re-looks-up: after step 2 the progress counters already
    #       include external tokens, a fresh lookup would be polluted).
    #    b) LOAD plan for PREEMPTION-RESUMED requests ->
    #       build_resumed_load_meta. v1 carries resumed requests
    #       in ``scheduled_cached_reqs.resumed_req_ids``, NOT in
    #       ``scheduled_new_reqs``, so they need their own loop --
    #       missing them would yield garbage output after preemption.
    #    c) SAVE plan for EVERY request scheduled this pass ->
    #       build_save_meta. Incremental: only blocks/boundaries
    #       not previously emitted are saved. The worker executes it
    #       AFTER forward, when the GPU pages hold state up to
    #       computed+scheduled tokens.
    #    d) Bookkeeping for RUNNING cached requests ->
    #       on_cached_request, done before (c). Sync the
    #       authoritative progress and block tables from upstream. On
    #       resume (or any progress regression) roll the save cursor
    #       back, so boundaries emitted before a preemption but never
    #       provably persisted get re-emitted (safe: overwrite is
    #       idempotent; skipping them would lose data).
    # 4. Request teardown -> the core calls
    #    ``connector.request_finished(request)`` ->
    #    on_request_finished, which drops the ReqState.
    #
    # How per-group block tables (ReqGroupState.block_ids) stay current
    # ---------------------------------------------------------------
    # block_ids is our copy of vLLM's block table for the request, one
    # list per KV cache group. A block id is an index into that group's
    # GPU block pool (not a raw address; the worker multiplies it by the
    # page layout to locate the data).
    #
    # vLLM allocates new blocks inside ``kv_cache_manager.allocate_slots``
    # during every scheduling pass, whenever a request crosses a block
    # boundary (decode: a new block every block_size tokens; chunked
    # prefill: at each boundary crossing). Those new block ids reach us
    # through TWO channels, depending on which scheduling loop the
    # request is in:
    #
    # - Requests scheduled from the WAITING queue (new and
    #   preemption-resumed): immediately after allocate_slots succeeds,
    #   the core calls ``connector.update_state_after_alloc`` with the
    #   request's FULL current block table
    #   (``kv_cache_manager.get_blocks(request_id)``). We replace our
    #   copy wholesale.
    # - RUNNING requests: the core's running loop calls allocate_slots
    #   but does NOT notify the connector. Instead the newly allocated
    #   blocks travel inside the SchedulerOutput as
    #   ``scheduled_cached_reqs.new_block_ids`` (a parallel array to
    #   ``req_ids``). on_cached_request appends them to our copy
    #   (or replaces it for resumed requests, where upstream sends the
    #   full table again).
    #
    # Ordering inside build_connector_meta matters:
    # on_cached_request (table sync) runs BEFORE
    # build_save_meta, so the save plan already sees blocks
    # allocated in the SAME pass. The plan is executed by the worker
    # after forward (wait_for_save), at which point those blocks
    # actually contain the computed KV data.
    #
    # Save addressing: one logical block, two addresses
    # ---------------------------------------------------------------
    # ::
    #
    #   token stream     | block 0 | block 1 | ... | block i |
    #                    (block_size tokens each)
    #
    #   block_hashes[i] --> store key     (content-addressed: which
    #                                      chunks the page is written to
    #                                      / found under)
    #   block_ids[i]    --> GPU pool blk  (position-addressed: which
    #                                      physical block the worker
    #                                      reads the page from)
    #
    #   A save op pairs them: keys[k] <-> gpu_block_ids[k].

    def _track_new_request(
        self, req_id: str, block_hashes: list[int],
        num_computed_tokens: int,
        request: Optional["Request"] = None,
    ) -> None:
        """Register a fresh ReqState. Internal: called by us from
        get_num_new_matched_tokens / build_load_meta /
        update_state_after_alloc when a request first becomes visible
        (vLLM has no dedicated "new request" connector hook)."""
        live_source = None
        if request is not None:
            live_source = (request.block_hashes
                           if self._block_hash_source == "vllm"
                           else request.all_token_ids)
        self._req_states[req_id] = ReqState(
            live_source=live_source,
            block_hashes=list(block_hashes),
            num_computed_tokens=num_computed_tokens,
            groups=tuple(
                ReqGroupState() for _ in self._groups),
        )

    def take_async_load_plans(
        self, already_emitted: set[str]
    ) -> dict[str, ReqMeta]:
        """Load plans for requests vLLM parked, drained exactly once.

        Emitting twice would submit a second transfer for a request that
        already has one in flight; the worker refuses that outright,
        because the first submission's tasks would be left with nothing
        to drain them.

        ``already_emitted`` lets the caller skip requests whose plan was
        produced by the normal path in this same step (a request can be
        both newly scheduled and pending here if vLLM changed its mind
        between passes).
        """
        plans = {}
        for req_id in sorted(self._async_load_pending - already_emitted):
            state = self._req_states[req_id]
            meta = self._build_load_meta_from_state(
                req_id, state, scheduled_tokens=0)
            state.async_plan_emitted = True
            # Downgrade to synchronous from here on. Once released, vLLM
            # reschedules the request through the ordinary new-request
            # path, which builds a plan again. Leaving is_async set would
            # make the worker open a SECOND cross-step transfer and
            # report the request finished a second time -- by which point
            # it is RUNNING, and vLLM asserts that a finished-recving
            # request is either parked or done.
            state.is_async = False
            if meta is not None and meta.group_ops:
                plans[req_id] = meta
            else:
                logger.warning(
                    "async req=%s has no restorable pages; dropping the "
                    "plan so it is recomputed rather than left waiting",
                    req_id)
        self._async_load_pending -= (self._async_load_pending
                                     - already_emitted)
        return plans

    def _request_block_hashes(self, request: "Request") -> list[Any]:
        """This request's block identities, in block order.

        ``legacy`` recomputes them from the token ids the way the
        block-oriented path always has, so caches written by earlier
        versions stay readable. ``vllm`` adopts the engine's own
        prefix-cache hashes, which is what the boundary layout has
        always used and what lets a hit line up with vLLM's local
        prefix cache exactly.

        Both exclude the final token, matching vLLM: a block is only
        identified once its tokens are computed.
        """
        if self._block_hash_source == "vllm":
            return list(request.block_hashes)
        tokens = request.all_token_ids
        return [str(h) for h in generate_block_hashs(
            tokens[:-1], self._hash_block_size)]

    def on_request_finished(self, req_id: str) -> None:
        """vLLM trigger: core frees the request ->
        connector.request_finished -> here. Drop the ReqState;
        committed boundaries are content-addressed and stay."""
        self._req_states.pop(req_id, None)
        self._async_load_pending.discard(req_id)

    def on_cached_request(
        self, req_id: str, new_block_ids: tuple[list[int], ...],
        resumed: bool, num_computed_tokens: Optional[int],
    ) -> None:
        """vLLM trigger: every scheduling pass, for each running
        (cached) request, via connector.build_connector_meta.

        Track a scheduled CACHED request: sync the authoritative
        num_computed from upstream and extend block tables with newly
        allocated blocks so incremental save targets the right slots.
        ``resumed`` (preemption resume) REPLACES tables per upstream
        CachedRequestData semantics.

        Cursor rollback: the
        incremental save cursor means "proven not to need re-emission in
        THIS request lifecycle", NOT "metadata was once constructed".
        On resume (or any authoritative progress regression, even with a
        missing resumed flag -- fail-closed) every group's cursor rolls
        back to floor(N / block_size): boundaries emitted before a
        preemption but never provably persisted are re-emitted.
        Re-emission is safe (writing a block again is idempotent);
        NOT rolling back permanently skips un-persisted
        boundaries."""
        state = self._req_states[req_id]
        # Adopt block hashes vLLM has appended since we registered
        # (decode completes blocks too; without this, generated tokens
        # are never offloaded). Only ever extends -- hashes are
        # content-addressed and append-only.
        if state.live_source is not None:
            if self._block_hash_source == "vllm":
                live = state.live_source
            else:
                live = [str(h) for h in generate_block_hashs(
                    state.live_source[:-1], self._hash_block_size)]
            if len(live) > len(state.block_hashes):
                state.block_hashes.extend(live[len(state.block_hashes):])
        old_progress = max(state.num_computed_tokens,
                           state.last_known_progress)
        regression = (num_computed_tokens is not None
                      and num_computed_tokens < old_progress)
        if num_computed_tokens is not None:
            state.num_computed_tokens = num_computed_tokens
            state.last_known_progress = num_computed_tokens
        if resumed or regression:
            # fail-closed: a missing progress on resume is treated as
            # N=0 (roll everything back) rather than skipping the
            # rollback.
            safe_n = num_computed_tokens or 0
            for g_idx, group in enumerate(self._groups):
                gstate = state.groups[g_idx]
                safe = safe_n // group.block_size
                if gstate.next_stored_chunk_idx > safe:
                    if os.getenv("KVSHRINK_DEBUG_LOG"):
                        logger.info(
                            "cursor rollback req=%s g%d: %d -> %d "
                            "(progress %d -> %s, resumed=%s)",
                            req_id, g_idx, gstate.next_stored_chunk_idx,
                            safe, old_progress, num_computed_tokens,
                            resumed)
                    gstate.next_stored_chunk_idx = safe
        if new_block_ids:
            for gstate, ids in zip(state.groups, new_block_ids):
                if resumed:
                    # upstream semantics: for resumed requests
                    # new_block_ids IS the table (replace), per group --
                    # including an EMPTY list, which clears stale blocks
                    gstate.block_ids = list(ids) if ids else []
                elif ids:
                    gstate.block_ids.extend(ids)

    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """External lookup; returns (hit_tokens, has_async_load).

        vLLM trigger: the core calls connector.get_num_new_matched_tokens
        while scheduling a NEW request, BEFORE block allocation, to ask
        how many tokens the external store can vouch for. Our answer is
        added to num_computed_tokens by the core, so it must be backed
        by restorable pages.

        Always synchronous: the chunk-tier lookup is Record-gated and
        never defers (no PENDING/RETRY states exist anymore).
        """
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        block_hashes = self._request_block_hashes(request)
        self._track_new_request(
            request.request_id, block_hashes,
            num_computed_tokens, request=request)
        policy = HybridHitPolicy(
            self._groups, self._present, self._hash_block_size,
            num_computed_tokens)
        # Restorable boundary in tokens; 0 = miss. The policy already
        # gated on live chunk presence (engine Record), so a nonzero
        # boundary is complete by construction; only record it.
        boundary = policy.find_longest_cache_hit(
            block_hashes, request.num_tokens)
        state = self._req_states[request.request_id]
        if boundary and state.block_hashes:
            state.snapshot_boundary = boundary
        else:
            boundary = 0
        external = max(0, boundary - num_computed_tokens)
        use_async = self._decide_async(request.request_id, external)
        logger.debug(
            "req=%s external_hit=%d boundary=%d async=%s",
            request.request_id, external, boundary, use_async)
        return external, use_async

    # ------------------------------------------------------------------
    def _decide_async(self, req_id: str, external: int) -> bool:
        """Should this request's pages stream in while the GPU runs
        OTHER requests, instead of stalling a forward step on us?

        Answering True hands vLLM ``load_kv_async=True``: it parks the
        request in WAITING_FOR_REMOTE_KVS, allocates its blocks anyway
        (so we have somewhere to write), and only reschedules it once
        the worker names it in ``get_finished``. The alternative is what
        we did before -- enter forward immediately and block a layer
        hook until the bytes arrive, which idles the GPU for exactly as
        long as the storage is slow. That cost grows with concurrency
        and will grow again when the tier is remote rather than local
        disk.

        Concurrency is approximated by the number of live request
        states, matching the block-path policy so one knob means one
        thing everywhere.

        Returns False (stay synchronous) when there is nothing to load,
        when no policy is configured, or when the policy selects 0
        layers -- 0 means "synchronous", NOT "release before any layer".
        """
        if external <= 0 or self._async_load_layer_config is None:
            return False
        state = self._req_states[req_id]
        selected = self._async_load_layer_config.select(
            len(self._req_states))
        if selected == 0:
            return False
        state.is_async = True
        # Clamp: asking for more leading layers than exist would never
        # be satisfiable and would hang the request in
        # WAITING_FOR_REMOTE_KVS forever.
        if selected < 0 or selected > len(self._attention_layers):
            state.async_load_layers = -1  # require every layer
        else:
            state.async_load_layers = selected
        return True

    # ------------------------------------------------------------------
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record the allocated block tables per group (after alloc).

        vLLM trigger: the core calls connector.update_state_after_alloc
        right AFTER successful block allocation (same pass as the
        lookup), passing the num_external_tokens it accepted. Only on
        this path is pending_load_tokens set -- the alloc-failure path
        never calls us, so no pending load obligation can leak.

        Design note -- why this hook only RECORDS facts and defers plan
        building to build_connector_meta (unlike the legacy
        pure-attention connector, which computes the load range inline
        here):

        1. The restore point is fixed at lookup time, not by arithmetic.
           GDN state can only be restored at segment boundaries, so the
           hit policy pins ``snapshot_boundary`` during
           get_num_new_matched_tokens. Dividing
           num_external_tokens by block_size here would not necessarily
           land on a legal boundary; the plan must follow the recorded
           boundary, never a fresh computation.
        2. Multiple KV groups keep separate block tables. The attention
           group and the GDN group have independent block pools; the
           per-group ids recorded here are later assembled by different
           rules (attention per block, mamba = the last non-null
           snapshot slot). The legacy connector can hardcode
           ``get_block_ids()[0]``; we cannot.
        3. New and preemption-resumed requests share one plan builder.
           Resumed requests arrive via resumed_req_ids, not the new
           request path; assembling plans at alloc time would fork the
           logic. Recording facts here and building plans from the
           recorded state in build_connector_meta keeps both entries on
           the same code path (_build_load_meta_from_state).

        ``blocks`` is the result of kv_cache_manager.get_blocks(request_id):
        a tuple of per-group block sequences (KVCacheBlock objects).
        """
        req_id = request.request_id
        state = self._req_states.get(req_id)
        if state is None:
            self._track_new_request(
                req_id, self._request_block_hashes(request), 0,
                request=request)
            state = self._req_states[req_id]
        state.num_computed_tokens = (
            state.num_computed_tokens + num_external_tokens)
        state.pending_load_tokens = num_external_tokens
        all_block_ids = blocks.get_block_ids()
        for g_idx, ids in enumerate(all_block_ids):
            state.groups[g_idx].block_ids = list(ids)
        if (state.is_async and not state.async_plan_emitted
                and num_external_tokens > 0):
            # This is the ONLY moment we hear about an async request.
            # vLLM allocates its blocks, calls us here, and then parks it
            # in WAITING_FOR_REMOTE_KVS -- out of scheduled_new_reqs and
            # out of scheduled_cached_reqs. A plan builder that walks
            # those two lists would therefore never emit anything for
            # it, the worker would have nothing to transfer, nothing to
            # report finished, and the request would wait forever for a
            # release that cannot come. Record it here and emit from the
            # record instead.
            self._async_load_pending.add(req_id)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "update_state req=%s per-group block_ids: %s hashes=%d",
                req_id, [[b for b in g.block_ids] for g in state.groups],
                len(state.block_hashes))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # True = defer freeing to get_finished(): async puts (and an
        # in-flight async load) may still be reading these blocks, so
        # the worker names the request in finished_sending once they
        # land. Committed boundaries are content-addressed and outlive
        # the request; they are never deleted here.
        self.on_request_finished(request.request_id)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23.0 calls this instead of
        ``request_finished`` whenever the hybrid memory allocator is on,
        which is the default for every model).

        Both paths share one contract: block freeing is deferred to
        get_finished, which reports the request once every transfer
        reading its blocks has landed.
        """
        return self.request_finished(request, [])

    # ------------------------------------------------------------------
    def build_load_meta(
        self, new_req: "NewRequestData", scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build the LOAD ReqMeta for a NewRequestData entry.

        vLLM trigger: connector.build_connector_meta iterates
        ``scheduler_output.scheduled_new_reqs`` at the end of the pass.

        attention groups: all prefix blocks whose boundary hash is HIT;
        mamba groups: the single state snapshot block at the restore
        boundary (written into the CURR state block; see
           _build_load_meta_from_state).
        """
        req_id = new_req.req_id
        state = self._req_states[req_id]
        return self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build the LOAD ReqMeta for a PREEMPTION-RESUMED request.

        vLLM v1 carries resumed requests in
        ``scheduled_cached_reqs.resumed_req_ids`` (NOT
        ``scheduled_new_reqs``), so build_connector_meta must ask for
        their load meta explicitly. State (block tables, snapshot
        boundary, pending_load_tokens) was already refreshed this
        scheduling pass by
        get_num_new_matched_tokens + update_state_after_alloc.

        Fail-closed: if the core accepted external tokens
        (pending_load_tokens > 0) the meta MUST carry restorable pages;
        an empty load with pending external tokens means the forward
        would read
        unrestored KV while num_computed_tokens skips recompute. Raise
        instead of silently emitting wrong tokens.
        """
        state = self._req_states[req_id]
        meta = self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)
        if state.pending_load_tokens > 0:
            npages = sum(len(op.keys) for op in meta.group_ops)
            if npages == 0:
                raise RuntimeError(
                    "kvshrink resumed request has accepted external "
                    "tokens but no restorable pages (req="
                    f"{req_id} pending={state.pending_load_tokens} "
                    f"boundary={state.snapshot_boundary} "
                    f"sched={scheduled_tokens}): refusing to enter "
                    "forward with unrestored state")
        return meta

    def _build_load_meta_from_state(
        self, req_id: str, state: ReqState, scheduled_tokens: int,
    ) -> ReqMeta:
        """The snapshot_boundary recorded by get_num_new_matched_tokens
        is the AUTHORITATIVE restore boundary for this alloc/load. NEVER recompute here: after
        update_state_after_alloc the locally-computed counter already
        includes external tokens, and a fresh lookup would be polluted.
        Missing/expired boundary must
        FAIL CLOSED (boundary 0), not guess."""
        boundary = state.snapshot_boundary
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "TAIL req=%s snapshot_boundary=%d computed_before_fwd=%d "
                "external=%d",
                req_id, state.snapshot_boundary,
                state.num_computed_tokens,
                state.snapshot_boundary - state.num_computed_tokens)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            ids = state.groups[g_idx].block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            if group.kind == "attention":
                gran = group.block_size
                num_hash = boundary // gran
                for i in range(num_hash):
                    if i >= len(state.block_hashes):
                        break
                    blk_hash = state.block_hashes[i]
                    key = self._boundary_key(group, blk_hash)
                    if not lookup_boundary(self.kvstore, key):
                        break
                    # v0.21 hashes are per complete block: hash i == block i
                    if i < len(ids):
                        # one page key + gpu block per layer (full expansion)
                        for layer_name in group.layer_names:
                            keys.append(self._page_key(key, layer_name))
                            gpu_ids.append(ids[i])
            elif group.kind == "mamba":
                # Load the snapshot into the CURR state block ONLY
                # (v0.23.0 semantics, verified upstream): the align-mode
                # block table pins the GDN execution metadata to column 0
                # = the block holding this step's last scheduled token,
                # for BOTH the chunked-prefill and decode paths -- there
                # is no prev/curr distinction at execution time.
                # preprocess_mamba's prev -> curr copy runs BEFORE
                # start_load_kv (execute_model order), and our H2D write
                # lands before forward (waited at start_load_kv), i.e.
                # after the copy and before any GDN layer runs, so CURR
                # is the
                # one correct target. Writing PREV would be dead work:
                # the kernel never reads it that step.
                if state.block_hashes and boundary > 0:
                    # hash index of the snapshot AT boundary:
                    # hash[i] covers [i*bs, (i+1)*bs) -> snapshot at
                    # boundary lives at hash[boundary//bs - 1]
                    idx = boundary // group.block_size - 1
                    if 0 <= idx < len(state.block_hashes):
                        blk_hash = state.block_hashes[idx]
                        key = self._boundary_key(group, blk_hash)
                        if lookup_boundary(self.kvstore, key):
                            bs = group.block_size
                            # CURR running-state index for this step
                            # (upstream align-mode formula):
                            # (num_computed + num_scheduled - 1) // bs
                            # with num_computed == boundary here.
                            #
                            # Why exactly one slot, and why this one:
                            # in align mode the kernels do not scan the
                            # table, they gather a single column --
                            # mamba_get_block_table_tensor computes
                            # start = (seq_lens - 1) // block_size and
                            # mamba_attn then takes column 0 of the
                            # gathered result. So this index is the only
                            # location forward will ever read for this
                            # step. The previous generation wrote both a
                            # prev and a curr slot because the timing of
                            # vLLM's prev->curr copy was unclear; the
                            # v0.23 source settles it, making the second
                            # write dead weight. The flip side is that
                            # there is no fallback slot, so an invalid
                            # index below must fail-stop rather than
                            # degrade.
                            curr_idx = (boundary + scheduled_tokens -
                                        1) // bs

                            # Fail-closed contract: an external HIT has already
                            # committed num_computed_tokens=boundary via
                            # get_num_new_matched_tokens; silently skipping a
                            # required slot
                            # would let forward read unrestored state and
                            # emit wrong tokens. Fail-stop (EngineCore
                            # fatal, same semantics as the TOCTOU gate)
                            # instead of producing a partial mamba load.
                            if scheduled_tokens <= 0 and not state.is_async:
                                # SYNCHRONOUS restore with no scheduled
                                # tokens means no forward step, so
                                # start_load_kv never runs and the slot
                                # stays unrestored while the core has
                                # already credited the tokens.
                                #
                                # An ASYNC restore is a different thing
                                # and is correct here. vLLM gives a
                                # parked request zero scheduled tokens,
                                # so curr_idx collapses to
                                # (boundary - 1) // bs -- which is
                                # exactly the index preprocess_mamba
                                # will read as prev_state_idx when the
                                # request is finally scheduled
                                # (num_computed_tokens == boundary by
                                # then). Its own prev -> curr copy then
                                # carries the snapshot into the slot
                                # forward reads. No hook is needed
                                # because no kernel runs until the
                                # release gate has already waited for
                                # the transfer.
                                raise RuntimeError(
                                    "kvshrink mamba external HIT with "
                                    "scheduled_tokens=0 "
                                    f"(req={req_id} boundary={boundary}): "
                                    "production hits must schedule >= 1 "
                                    "token; refusing to build load meta")
                            if not (0 <= curr_idx < len(ids)
                                    and ids[curr_idx] != 0):
                                raise RuntimeError(
                                    "kvshrink mamba load curr slot "
                                    f"invalid (req={req_id} "
                                    f"boundary={boundary} "
                                    f"sched={scheduled_tokens} "
                                    f"table_idx={curr_idx} "
                                    f"table={ids}): refusing to enter "
                                    "forward with unrestored state")
                            gpu_block = ids[curr_idx]
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    key, layer_name))
                                gpu_ids.append(gpu_block)
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids)))
        return ReqMeta(
            external_hit_tokens=boundary - state.num_computed_tokens,
            group_ops=tuple(group_ops),
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def build_save_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Production save: INCREMENTAL per-group page persistence.

        vLLM trigger: connector.build_connector_meta asks for a save
        plan for EVERY request scheduled this pass (new + cached); the
        worker executes it after forward (wait_for_save), so the GPU
        pages then hold state up to ``computed + scheduled`` tokens;
        boundaries are computed against THAT progress.

        The save path for NEWLY COMPUTED KV, end to end
        (pass N, request advances by S scheduled tokens)
        ---------------------------------------------------------------
        ::

          [scheduler, pass N]
            progress P = num_computed_tokens + S   (predictive:
                                  the plan is built BEFORE forward but
                                  describes the state AFTER forward)
            per group, emit ops for work not previously emitted:
              attention: blocks [next_stored_chunk_idx, P//block_size)
                         -- every newly COMPLETED block, per layer
              mamba:     snapshot of the running state block, ONLY if
                         P lands exactly on a block boundary
            next_stored_chunk_idx advances at EMIT time (the worker
            save is fail-stop, so indices cannot silently diverge)
                    |
                    |  ReqMeta pickled inside connector metadata
                    v
          [worker, pass N]
            forward          -> KV for the S new tokens is now in the
                                GPU blocks (block_ids recorded earlier)
            wait_for_save    -> per (layer, block) in the plan:
                                read GPU block -> compress -> stage
                                chunks under the content-hash keys
                                a block becomes visible to later
                                lookups the moment its write lands,
                                because that write is the commit
                    |
                    v
          [any later pass / any later request]
            get_num_new_matched_tokens can now HIT these pages:
            same content hash -> same keys -> restore instead of
            recompute.

        Each group tracks ``next_stored_chunk_idx``; a step emits save
        ops only for blocks/boundaries not previously emitted.

        - attention: every completed block hash in
          [next_stored, progress//gran) is saved (per-block pages are
          valid as soon as the block completes).
        - mamba: the running state block (last NON-NULL table slot) is
          saved only when progress lands EXACTLY on a block boundary --
          a partial tail is not a valid restore point, and snapshots are
          only addressable by boundary hashes.
        """
        state = self._req_states[req_id]
        progress = state.num_computed_tokens + scheduled_tokens
        state.last_known_progress = max(state.last_known_progress,
                                        progress)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            gstate = state.groups[g_idx]
            ids = gstate.block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            if group.kind == "attention":
                num_hash = min(progress // group.block_size, len(ids),
                               len(state.block_hashes))
                start = gstate.next_stored_chunk_idx
                for i in range(start, num_hash):
                    blk_hash = state.block_hashes[i]
                    for layer_name in group.layer_names:
                        keys.append(self._page_key(
                            self._boundary_key(group, blk_hash),
                            layer_name))
                        gpu_ids.append(ids[i])
                if num_hash > start:
                    gstate.next_stored_chunk_idx = num_hash
            elif group.kind == "mamba":
                # Save the running state block: the last NON-NULL block in
                # the group's table. Block tables vary by token count:
                # single-element [X] (545-token req), null-prefixed
                # [0,0,X], or [null, X] -- block 0 is the reserved null
                # block. Do NOT assume len(ids) > 1.
                if state.block_hashes:
                    block_pos = None
                    for pos in range(len(ids) - 1, -1, -1):
                        if ids[pos] != 0:  # 0 = null block placeholder
                            block_pos = pos
                            break
                    if block_pos is None:
                        if os.getenv("KVSHRINK_DEBUG_LOG"):
                            logger.info(
                                "save mamba g%d: no non-null block in "
                                "ids=%s", g_idx, ids)
                    elif (progress > 0
                          and progress % group.block_size == 0):
                        idx = progress // group.block_size - 1
                        if (idx >= gstate.next_stored_chunk_idx
                                and idx < len(state.block_hashes)):
                            blk_hash = state.block_hashes[idx]
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    self._boundary_key(group, blk_hash),
                                    layer_name))
                                gpu_ids.append(ids[block_pos])
                            gstate.next_stored_chunk_idx = idx + 1
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids)))
        return ReqMeta(group_ops=tuple(group_ops))

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's hybrid plans.

        Order matters: for cached (running) requests the block-table
        sync (``on_cached_request``) MUST run before ``build_save_meta``,
        so the save plan already sees blocks allocated in the SAME pass.
        """
        meta = KVShrinkConnectorMetadata()
        debug = bool(os.getenv("KVSHRINK_DEBUG_LOG"))
        save_on = save_enabled()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            req_meta = self.build_load_meta(
                new_req, num_sched.get(new_req.req_id, 0))
            if debug:
                logger.info(
                    "LOADMETA req=%s ops=%d computed_before_fwd=%d "
                    "num_scheduled_tokens=%s",
                    new_req.req_id, len(req_meta.group_ops),
                    new_req.num_computed_tokens,
                    num_sched.get(new_req.req_id))
                for op in req_meta.group_ops:
                    logger.info(
                        "LOADMETA  g%d kind=%s keys=%d gpu_ids=%d",
                        op.group_idx, self._groups[op.group_idx].kind,
                        len(op.keys), len(op.gpu_block_ids))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.requests[new_req.req_id] = req_meta
            if save_on:
                save_meta = self.build_save_meta(
                    new_req.req_id, num_sched.get(new_req.req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[new_req.req_id] = save_meta

        # ASYNC requests ride NEITHER list: vLLM parks them in
        # WAITING_FOR_REMOTE_KVS, so they are absent from
        # scheduled_new_reqs and from scheduled_cached_reqs alike. Their
        # plan comes from what update_state_after_alloc recorded, and
        # without it the worker would have nothing to transfer and the
        # request would wait forever to be released.
        for req_id, req_meta in self.take_async_load_plans(
                set(meta.reqs_to_load.requests)).items():
            if debug:
                logger.info(
                    "LOADMETA(async) req=%s ops=%d layers=%s",
                    req_id, len(req_meta.group_ops),
                    req_meta.async_load_layers)
            meta.reqs_to_load.requests[req_id] = req_meta

        # PREEMPTION-RESUMED requests ride scheduled_cached_reqs.
        # resumed_req_ids, NOT scheduled_new_reqs. Their external-hit
        # tokens were accepted this same pass, so without a load plan
        # here the worker would never restore the pages while the core
        # already skips recompute -- silent garbage output.
        cr = scheduler_output.scheduled_cached_reqs
        for req_id in cr.resumed_req_ids:
            req_meta = self.build_resumed_load_meta(
                req_id, num_sched.get(req_id, 0))
            if debug:
                logger.info(
                    "LOADMETA(resumed) req=%s ops=%d",
                    req_id, len(req_meta.group_ops))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.requests[req_id] = req_meta

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings): sync their tables first,
        # then emit incremental saves.
        if save_on:
            resumed = cr.resumed_req_ids
            new_bids = cr.new_block_ids
            ncts = cr.num_computed_tokens
            for i, req_id in enumerate(cr.req_ids):
                self.on_cached_request(
                    req_id, new_bids[i], req_id in resumed, ncts[i])
                save_meta = self.build_save_meta(
                    req_id, num_sched.get(req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[req_id] = save_meta

        if debug:
            logger.info(
                "build_connector_meta: %d load reqs, %d save reqs",
                len(meta.reqs_to_load.requests),
                len(meta.reqs_to_save.requests))
        return meta

    def _boundary_key(self, group: GroupInfo, block_hash: object) -> CacheKey:
        """Boundary key for one group at one block hash."""
        return make_boundary_key(group.group_idx, block_hash)

    def _present(self, group_idx: int, block_hash: object) -> bool:
        """Store-presence predicate handed to the hit policy, which
        plans against boundary addresses without seeing store details."""
        return lookup_boundary(
            self.kvstore,
            make_boundary_key(group_idx, block_hash))

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same hash and
        group as the boundary, plus the layer name. This is the exact
        page address the worker must move."""
        return replace(boundary_key, layer_name=layer_name)

    ############################################################
    # Worker Side Methods
    ############################################################
    # Worker-side execution engine: canonical page views, load
    # submission, pipelined attention save, and the post-forward save
    # commit. The worker is the EXECUTE side; it owns the writer lease,
    # while the scheduler only plans against a read-only store.
    #
    # Load pipelining without any vLLM patch
    # --------------------------------------
    # vLLM calls ``wait_for_layer_load`` at every ATTENTION layer's entry
    # (piecewise cudagraph is forced) but never at GDN layers. So:
    #
    # - ``start_load``: submit ALL loads (attention pages + GDN snapshots)
    #   to the engine (async unzip+H2D on the engine's get_stream), then
    #   host-block ONLY on the LEADING GDN segment -- the GDN layers that
    #   execute before the first attention layer and therefore have no
    #   attention hook to ride on.
    # - ``wait_layer_load(attn_i)``: wait attention layer i's pages AND
    #   the GDN segment between attn_i and the next attention layer
    #   (those GDN layers execute after attn_i, so waiting at attn_i's
    #   entry is in time). Their transfers overlapped the preceding
    #   layers' compute -- this IS the layer pipeline.
    #
    # GDN snapshots are written into the CURR state slot: v0.23.0's GDN
    # execution metadata is pinned to the CURR block for both
    # chunked-prefill and decode, and preprocess_mamba's prev->curr copy
    # runs before start_load_kv, so a CURR write during forward is always
    # safe and a PREV write would be dead work.
    #
    # Save path (async lifecycle, same shape as main)
    # ------------------------------------------------
    # Puts are submitted as soon as the data is final and NEVER waited
    # inside the step: attention layers (and the mamba segment preceding
    # them) submit at the ``save_kv_layer`` hook, the trailing mamba
    # segment and any unhooked layers submit in ``wait_for_save``, and
    # every write drains in ``get_finished``. ``request_finished``
    # defers block freeing until the worker reports the request there,
    # so a block is never reused while a put is still reading it.
    # Correctness of reading mamba state mid-forward: the put is
    # self-gated on the compute stream, and in align mode a boundary
    # slot is never rewritten by its owner afterwards (the curr pointer
    # advances to the next slot).
    #
    # Fail-stop contract: any load/save anomaly raises (EngineCore
    # fatal). Silently dropping a save would lose a boundary permanently
    # (the scheduler already advanced its incremental indices); entering
    # forward with unrestored pages would emit wrong tokens (the core
    # already skipped recompute).

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        static_context = self.vllm_config.compilation_config.static_forward_context
        for layer in static_context.values():
            get_backend = getattr(layer, "get_attn_backend", None)
            if get_backend is not None:
                if "FLASHINFER" in get_backend().get_name().upper():
                    raise RuntimeError("FlashInfer is not supported")
                break

        from vllm.model_executor.models.utils import extract_layer_index

        # Execution order matters to the async release gate, which holds
        # a request until its FIRST N layers have landed -- a statement
        # about position, not about names. The ``kv_caches`` dict does
        # not carry it: v0.23.0 builds it group by group
        # (``_kv_cache_spec_attn_group_iterator``), so mamba and
        # attention layers arrive in separate runs. We recover the order
        # the way vLLM's own ``bind_kv_cache`` does, from the layer
        # index in the layer name.
        self.register(sorted(kv_caches, key=extract_layer_index))

        # The store binds the RAW kv_caches directly: single-tensor
        # attention pools pass through; mamba lists collapse into one
        # opaque int8 page view per layer INSIDE kvstore (see
        # KVStore._bind_pools). One bound pool = one layer name = one
        # store key; policies (lossy off for mamba) live there too.
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            kv_caches=kv_caches,
            # _rank: the authoritative rank captured in _init_kv_stack.
            # KVStore derives the per-rank persist dir and management
            # port from this value -- two ranks claiming one rank would
            # share both and clobber each other's shards.
            rank=self._rank,
            tp_size=self.tp_size,
        )
        logger.info("Registered %d KV cache layers",
                    len(self.kvstore.layer_names))

    def register(
        self,
        execution_order: list[str],
    ) -> None:
        """Record the model's execution order and which layers recur.

        Pool binding itself happens inside KVStore at construction;
        here we only derive the order-sensitive sets.

        ``execution_order``: all cached layer names in model execution
        order. Only the attention layers' order is used, by the async
        release gate -- "the first N layers" means nothing otherwise.
        """
        # Ordered layer names for the block store layout and the two
        # order-sensitive derived sets.
        self._layer_names = list(execution_order)
        # All GDN layers: waited as one barrier in start_load before any
        # attention layer runs.
        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        # Attention layers in model execution order, used by the async
        # release gate ("the first N layers" means nothing otherwise).
        self._attn_order = tuple(
            ln for ln in execution_order if ln in self._attn_layer_group)
        # Save pipelining segments: the mamba layers between attention
        # layer i-1 and attention layer i are final when i's save hook
        # fires, so they ride that hook (the trailing segment is
        # submitted by wait_for_save).
        segments: dict[str, tuple[str, ...]] = {}
        pending: list[str] = []
        for ln in execution_order:
            if ln in self._mamba_layers:
                pending.append(ln)
            elif ln in self._attn_layer_group and pending:
                segments[ln] = tuple(pending)
                pending = []
        self._mamba_save_segments = segments
        logger.info(
            "kvshrink hybrid worker registered: %d attention "
            "hook points, %d recurrent layers (tp=%d rank=%d)",
            len(self._attn_order),
            len(self._mamba_layers), self.tp_size, self._rank)

    def _metadata(self) -> KVShrinkConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkConnectorMetadata")
        return metadata

    # ------------------------------------------------------------------
    # store transfers
    # ------------------------------------------------------------------
    # One bound pool per layer; kvstore already normalized multi-part
    # mamba layers into single int8 page pools at bind time, so every
    # store call here carries plain layer names.

    def _wait_load(self, tasks: dict[str, Task]) -> None:
        """Host-block until these reads land; an incomplete transfer is
        fatal -- forward is about to read these blocks."""
        if tasks and not self.kvstore.get_wait(get_results=tasks,
                                               wait=True):
            raise RuntimeError(
                "kvshrink load failed: get_wait reported an incomplete "
                "transfer; forward would read unrestored blocks")

    # ------------------------------------------------------------------
    # load path
    # ------------------------------------------------------------------
    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        # Submits every load, then host-blocks on the recurrent ones;
        # attention layers are waited by their own hooks during forward.
        self.start_load(self._metadata())

    def start_load(self, metadata: KVShrinkConnectorMetadata) -> int:
        """Submit ALL of this step's loads, then host-block on the GDN
        ones. Attention layers stay pipelined: vLLM calls a hook on
        entry to each of them, so their pages are waited for exactly
        when they are about to be read.

        GDN gets no such hook, so it is waited for here, in one barrier.
        That costs the overlap for a request's recurrent state -- one
        block per layer, a few tens of MB in total, against a forward
        of a wholly different order -- and in exchange there is no
        machinery deciding which attention layer is responsible for
        which GDN layer, and no way for a GDN layer to reach forward
        unwaited. Returns the number of (layer, block) pages submitted.
        """
        # Per-step reset. _load_tasks holds "layer#part" -> engine Task
        # for this step's sync loads (waited per layer -- GDN as one
        # barrier here, attention at their forward-entry hooks);
        # _saved_layers/_step_save_pages track what the save hooks have
        # submitted.
        self._load_tasks = {}
        self._saved_layers = set()
        self._step_save_pages = 0
        npages = 0
        _t0 = time.monotonic()
        # Sync loads merge per group into ONE engine call per step;
        # duplicate labels across requests (two requests loading the
        # same external block into different GPU blocks) are fine --
        # the engine transfers each (label, index) pair independently.
        by_group: dict[int, tuple[list[tuple[int, str]], set[str]]] = {}
        for req_id, req_meta in metadata.reqs_to_load.requests.items():
            if req_meta.is_async:
                # Async tasks must survive this step and must not be
                # host-blocked by the GDN barrier below (that barrier
                # is for requests about to enter forward; an async one
                # is not): collect them into a private dict.
                tasks: dict[str, Task] = {}
                for op in req_meta.group_ops:
                    if not op.keys:
                        continue
                    t, n = self._submit_group_load(op)
                    tasks.update(t)
                    npages += n
                self._register_async_load(req_id, req_meta, tasks)
            else:
                for op in req_meta.group_ops:
                    if not op.keys:
                        continue
                    entries, layers = by_group.setdefault(
                        op.group_idx, ([], set()))
                    entries.extend(self._op_entries(op))
                    layers.update(key.layer_name for key in op.keys)
        for g_idx, (entries, layers) in by_group.items():
            tasks, n = self._submit_group_load_entries(
                g_idx, entries, layers)
            self._load_tasks.update(tasks)
            npages += n
        # Every GDN layer, waited before forward begins.
        recurrent = {k: self._load_tasks.pop(k)
                     for k in list(self._load_tasks)
                     if k.rsplit("#", 1)[0] in self._mamba_layers}
        if recurrent:
            self._wait_load(recurrent)
        if npages:
            logger.info(
                "start_load_kv: %d pages loaded "
                "elapsed_ms=%.3f (rank %d/%d)", npages,
                (time.monotonic() - _t0) * 1e3, self._rank, self.tp_size)
        return npages

    def _register_async_load(
        self, req_id: str, req_meta: ReqMeta,
        tasks: dict[str, Task],
    ) -> None:
        """Track one request's cross-step load and compute its release
        gate.

        The gate always includes every recurrent layer present in the
        plan, whatever the configured layer count says. A GDN/Mamba
        state is read whole at the start of forward, so releasing a
        request whose state is still in flight would let the model run
        on stale memory -- silently, with plausible output. Attention
        layers are safe to gate on a prefix because every one of them is
        waited immediately before its own kernels.
        """
        if not tasks:
            return
        layers = {k.rsplit("#", 1)[0] for k in tasks}
        recurrent = layers - self._attn_layer_group.keys()
        n = req_meta.async_load_layers
        if n is None or n < 0:
            gate = layers
        else:
            prefix = [ln for ln in self._attn_order if ln in layers][:n]
            gate = recurrent | set(prefix)
        # main's four-dict shape: tasks + per-request layer count.
        self._pending_load_tasks[req_id] = dict(tasks)
        self._gated_keys[req_id] = frozenset(
            k for k in tasks if k.rsplit("#", 1)[0] in gate)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "async load req=%s layers=%d gate=%d (recurrent=%d) "
                "requested_prefix=%s", req_id, len(layers), len(gate),
                len(recurrent), n)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # This attention layer's pages. GDN was already waited for
        # before forward began.
        self.wait_layer_load(layer_name)

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages.

        Drains EVERY promoted async entry (early-promoted = released),
        mirroring main's wait_for_layer_load.
        """
        keys = [k for k in self._load_tasks
                if k.rsplit("#", 1)[0] == layer_name]
        if keys:
            self._wait_load({k: self._load_tasks.pop(k) for k in keys})
        # Promoted leftovers drain too: early-promoted (released) and
        # active (bound to the current forward).
        books = [self._early_promoted_tasks, self._active_promoted_tasks]
        for book in books:
            for req_id, tasks in list(book.items()):
                sel = {k: t for k, t in tasks.items()
                       if k.rsplit("#", 1)[0] == layer_name}
                if sel:
                    self._wait_load(sel)
                    for k in sel:
                        del tasks[k]
                if not tasks:
                    book.pop(req_id, None)
        # The odd moment both books hold the request: fully drained.
        self._early_promoted_tasks = {
            r: t for r, t in self._early_promoted_tasks.items() if t}

    @staticmethod
    def _op_entries(op: GroupTransferMeta) -> list[tuple[int, str]]:
        """One (gpu_block_id, chunk_label) per block: keys expand per
        block x layer (scheduler invariant), so collapse by label."""
        seen: dict[str, int] = {}
        for key, gpu in zip(op.keys, op.gpu_block_ids):
            seen.setdefault(key.hash_str, gpu)
        return [(gpu, h) for h, gpu in seen.items()]

    def _submit_group_load(
        self, op: GroupTransferMeta
    ) -> tuple[dict[str, Task], int]:
        """Submit one GroupTransferMeta as one engine get; returns the
        flat tasks dict and the page count."""
        return self._submit_group_load_entries(
            op.group_idx, self._op_entries(op),
            {key.layer_name for key in op.keys})

    def _submit_group_load_entries(
        self, g_idx: int, entries: list[tuple[int, str]],
        layer_names: set[str],
    ) -> tuple[dict[str, Task], int]:
        """One engine get covering ``layer_names`` for the blocks in
        ``entries``; returns the tasks dict (layer name -> Task) and
        the page count."""
        tasks = self.kvstore.get(block_indices=[gpu for gpu, _ in entries],
                                 block_hashs=[h for _, h in entries],
                                 layer_names=list(layer_names),
                                 label=self._labels[g_idx])
        return tasks, len(entries) * len(layer_names)

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def _gather_save_candidates(
        self, metadata: KVShrinkConnectorMetadata
    ) -> dict[tuple[str, int], dict]:
        """Batch-level boundary candidates with cross-request dedup.

        Returns boundary_key -> {"group_idx", "pages": {layer: (key,
        gpu_block_id)}, "req_ids"} -- plain dicts, same shape main's
        inline bookkeeping used.
        """
        candidates: dict[tuple[str, int], dict] = {}
        for req_id, req_meta in metadata.reqs_to_save.requests.items():
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = {"group_idx": op.group_idx,
                                "pages": {}, "req_ids": set()}
                        candidates[key.boundary_key] = cand
                    cand["pages"][key.layer_name] = (key, gpu_block_id)
                    cand["req_ids"].add(req_id)
        return candidates

    def _submit_group_layers_save(
        self, g_idx: int, layer_names: list[str],
        entries: list[tuple[int, str]],
    ) -> dict[str, Task]:
        """Submit ONE async engine put covering ``layer_names`` for the
        blocks in ``entries`` (list of (gpu_block_id, chunk_label), same
        order for every layer -- scheduler invariant). Async D2H+zip on
        the engine's put_stream, self-gated on the compute stream so it
        reads final values. Returns the engine tasks dict."""
        return self.kvstore.put(block_indices=[gpu for gpu, _ in entries],
                                block_hashs=[h for _, h in entries],
                                layer_names=layer_names,
                                label=self._labels[g_idx])


    def _submit_layers_save(
        self, g_idx: int, layer_names: list[str],
        metadata: KVShrinkConnectorMetadata,
    ) -> None:
        """Submit ONE async engine put covering ``layer_names`` of one
        group for this step's complete boundary candidates. No waiting:
        the tasks join ``_current_put_tasks`` and drain in
        get_finished. Partial-boundary candidates are skipped here
        exactly as submit_saves skips them."""
        expected = sorted(self._groups[g_idx].layer_names)
        entries: list[tuple[int, str]] = []
        req_ids: set[str] = set()
        for cand in self._gather_save_candidates(metadata).values():
            if cand["group_idx"] != g_idx:
                continue
            if sorted(cand["pages"]) != expected:
                continue  # partial boundary: skipped at submit time too
            key, gpu_block_id = cand["pages"][layer_names[0]]
            entries.append((gpu_block_id, key.hash_str))
            req_ids |= cand["req_ids"]
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, layer_names,
                                               entries)
        for rid in req_ids:
            self._current_put_tasks.setdefault(rid, []).append(tasks)
        self._saved_layers.update(layer_names)
        self._step_save_pages += len(entries) * len(layer_names)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Pipelined save. vLLM calls this on exit of EVERY attention
        layer during forward (kv_transfer_utils decorator).

        An attention layer's page for this step's tokens is final the
        moment that layer returns, so its D2H+zip can overlap the
        remaining layers' compute instead of adding to the post-forward
        critical path. The mamba layers of the preceding segment are
        final too (their kernels already ran), so they ride the same
        hook -- the trailing segment after the last attention layer is
        submitted by wait_for_save.

        Submission only; the drain lives in get_finished (same
        lifecycle as main). KVSHRINK_SAVE_PIPELINED=0 disables this
        path (everything then submits in wait_for_save).
        """
        if self._connector_metadata is None:
            return
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not save_enabled():
            return
        metadata = self._metadata()
        segment = self._mamba_save_segments.get(layer_name)
        if segment:
            # A segment mixes groups (execution order interleaves the
            # three mamba groups); each group's layers go under their
            # own store label.
            by_group: dict[int, list[str]] = {}
            for ln in segment:
                by_group.setdefault(self._layer_group[ln], []).append(ln)
            for seg_g_idx, seg_layers in by_group.items():
                self._submit_layers_save(seg_g_idx, seg_layers, metadata)
        self._submit_layers_save(
            self._attn_layer_group[layer_name], [layer_name], metadata)

    def wait_for_save(self) -> None:
        if self.kvstore is None:
            return

        if not save_enabled():
            return
        metadata = self._metadata()
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("wait_for_save worker: reqs_to_save=%d",
                        len(metadata.reqs_to_save.requests))
        pages, boundaries = self.submit_saves(metadata)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("chunk_save: %d pages, %d boundaries",
                        pages, boundaries)
        self.debug_dump_state()

    def submit_saves(
        self, metadata: KVShrinkConnectorMetadata
    ) -> tuple[int, int]:
        """Post-forward save SUBMISSION: every layer not already
        pipelined through save_kv_layer (trailing mamba segment,
        attention layers whose hook never fired) goes out here, one
        engine put per group. Nothing is waited: the writes drain in
        get_finished, which is also where finished requests' blocks are
        released (the deferred-freeing contract, same as main).
        Returns (pages, boundaries) submitted this step."""
        candidates = self._gather_save_candidates(metadata)
        per_group: dict[int, dict[str, Any]] = {}
        nbound = 0
        for bkey, cand in candidates.items():
            blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand["pages"]) != expected:
                # A partial boundary is skipped (the scheduler's
                # incremental cursor has advanced, so it ages out as a
                # MISS, never wrong data). Log unconditionally: this is
                # the one save-side anomaly that does not fail-stop.
                logger.warning(
                    "chunk_save skip commit g%d h=%s: expected %d "
                    "layers, stored %d (%s)", g_idx, blk_hash,
                    len(expected), len(cand.pages),
                    set(expected) ^ set(cand.pages))
                continue
            nbound += 1
            blob = per_group.setdefault(g_idx, {"entries": [],
                                                "req_ids": set()})
            # entries are identical across a group's layers (same gpu
            # block and hash per boundary, expanded per layer)
            key, gpu_block_id = cand["pages"][expected[0]]
            blob["entries"].append((gpu_block_id, key.hash_str))
            blob["req_ids"] |= cand["req_ids"]
        for g_idx, blob in per_group.items():
            remaining = [ln for ln in self._groups[g_idx].layer_names
                         if ln not in self._saved_layers]
            if not remaining:
                continue
            tasks = self._submit_group_layers_save(
                g_idx, remaining, blob["entries"])
            for rid in blob["req_ids"]:
                self._current_put_tasks.setdefault(rid, []).append(tasks)
            self._saved_layers.update(remaining)
            self._step_save_pages += len(blob["entries"]) * len(remaining)
            # A block is finalized by its own write, with every layer of
            # the group accounted for in one step's submissions, so it
            # is committed the moment the writes land. There is no
            # second phase to publish and therefore nothing that can
            # outlive its data.
        if self._step_save_pages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            logger.info(
                "chunk_save: %d pages submitted, %d boundaries "
                "(rank %d/%d)", self._step_save_pages, nbound,
                self._rank, self.tp_size)
        return self._step_save_pages, nbound

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Report transfers that completed since the last step.

        Worker-side only; the scheduler role has nothing in flight.
        Loads report finished_recving (the async release gate); saves
        are drained per finished request so vLLM only reuses a block
        once every put reading it has landed (request_finished defers
        the freeing to here, same contract as main).
        """
        if self.kvstore is None:
            return None, None

        # Async release poll (main's early-promotion flow): a request
        # is promoted once its gated keys have landed; the remaining
        # layers drain through wait_layer_load during forward.
        finished_recving: set[str] = set()
        for req_id, tasks in list(self._pending_load_tasks.items()):
            gated = {k: t for k, t in tasks.items()
                     if k in self._gated_keys[req_id]}
            if gated and not self.kvstore.get_wait(get_results=gated,
                                                   wait=False):
                continue
            self._wait_load(gated)
            finished_recving.add(req_id)
            if not (tasks_remaining := {
                    k: t for k, t in tasks.items() if k not in gated}):
                del self._pending_load_tasks[req_id]
                del self._gated_keys[req_id]
                continue
            # Promote: released; the leftover keys drain at their layer
            # hooks during that request's forward.
            self._early_promoted_tasks[req_id] = tasks_remaining
            del self._pending_load_tasks[req_id]
            del self._gated_keys[req_id]

        self._deferred_finished_req_ids.update(finished_req_ids)
        completed: set[str] = set()
        for req_id in self._deferred_finished_req_ids:
            # Finished with an async load still in flight: drain it
            # here (its layer hooks will never fire again).
            load_tasks = (self._pending_load_tasks.pop(req_id, None)
                          or self._early_promoted_tasks.get(req_id)
                          or self._active_promoted_tasks.get(req_id))
            self._gated_keys.pop(req_id, None)
            if load_tasks is not None:
                if not self.kvstore.get_wait(get_results=load_tasks,
                                             wait=False):
                    continue
                self._wait_load(load_tasks)
                self._early_promoted_tasks.pop(req_id, None)
                self._active_promoted_tasks.pop(req_id, None)

            tasks = self._current_put_tasks.get(req_id)
            if tasks is None:
                completed.add(req_id)
                continue
            while tasks:
                if all(t.ctx is None for t in tasks[0].values()):
                    # A boundary shared with another finished request:
                    # that request's drain already finalized this put
                    # (put_wait sets ctx None on completion).
                    tasks.pop(0)
                    continue
                if not self.kvstore.put_wait(tasks[0], wait=False):
                    break
                tasks.pop(0)
            if not tasks:
                del self._current_put_tasks[req_id]
                completed.add(req_id)

        self._deferred_finished_req_ids.difference_update(completed)
        return (completed or None), (finished_recving or None)

    # ------------------------------------------------------------------
    # debug dump
    # ------------------------------------------------------------------
    def debug_dump_state(self) -> None:
        """KVSHRINK_DEBUG_DUMP=1: log sha256 of the first layer page of
        every mamba group at gpu blocks 0..9, so cold-vs-hot GPU states
        can be compared byte-exactly."""
        if not os.getenv("KVSHRINK_DEBUG_DUMP"):
            return
        for group in self._groups:
            if group.kind != "mamba":
                continue
            ln = group.layer_names[0]
            page_view = self.kvstore.kv_caches[ln]
            for blk in range(10):
                page = page_view[blk]
                h = hashlib.sha256(
                    page.cpu().numpy().tobytes()).hexdigest()
                logger.info("DUMP g%d block=%d sha=%s",
                            group.group_idx, blk, h[:16])
