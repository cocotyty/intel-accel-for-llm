# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KVShrink external KV cache connector (hybrid GDN/Mamba aware)."""

from __future__ import annotations

import hashlib
import logging
import math
import os
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

# Page-layout version: bump on any incompatible layout change so old
# pages are never read under a new layout (input to compute_namespace).
SCHEMA_VERSION = 4


@dataclass
class ReqMeta:
    """All transfer instructions for one request in one step."""
    group_ops: tuple[GroupTransferMeta, ...] = ()
    external_hit_tokens: int = 0
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqGroupState:
    """Per-group mutable state for one request (scheduler side)."""
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

    def add_request(
        self,
        req_id: ReqId,
        group_ops: tuple[GroupTransferMeta, ...] = (),
        external_hit_tokens: int = 0,
        is_async: bool = False,
        async_load_layers: int = -1,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            group_ops,
            external_hit_tokens,
            is_async,
            async_load_layers,
        )


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker transfer plan."""
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
# - ``label`` is the namespace, and it is where everything the store does
#   not otherwise know about must go. It carries the model namespace, the
#   KV cache group and the TP rank: the store's own ``rank`` argument
#   drives its management port and logs, NOT its keys, so two ranks
#   sharing a label would overwrite each other's shards. Two groups
#   sharing a label would be worse -- the same prefix hash exists in both
#   groups, and the durability record is keyed by ``(label, chunk_id)``
#   without the layer, so they would be tracked as one unit despite having
#   different lifetimes.
#
# Why the whole group goes in one call
# ------------------------------------
# The engine requires every tensor in a call to share shape and dtype, and
# a recurrent layer is two tensors of different shape (conv state, ssm
# state) over one storage. Passing canonical int8 page views satisfies
# that, and it also makes the call atomic: a block is finalized once, with
# all of its layers, so presence IS the commit. There is no second phase
# to publish and therefore nothing that can dangle.


def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank."""
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"


def lookup_boundary(store: "KVStore", key: "CacheKey") -> bool:
    """Is this boundary present in the store?"""
    try:
        present = store.has([key.hash_str],
                            label=group_label(key.namespace,
                                              key.group_idx, key.rank))
        return bool(present and present[0])
    except Exception:  # pragma: no cover - fail closed to MISS
        logger.exception("lookup error; treating as MISS")
        return False


# ======================================================================
# canonical page views
# ======================================================================
# Canonical page views over the vLLM GPU KV blocks.
#
# Layout verified on vLLM v0.23.0 + Qwen3.5-4B TP2:
#
# - Attention layer: single tensor; canonical view is
#   ``(num_blocks, page_size_bytes)`` int8 over its storage, contiguous
#   (stride == page size, offset 0 -- KVCacheTensor carries no layout
#   beyond size).
# - Mamba layer: ``kv_caches[layer]`` is a LIST of tensors (conv, ssm) sharing
#   one storage. The canonical page view is rebuilt from the first tensor's
#   storage (same approach as vLLM's offloading worker).
# - Physical page for block_id i = view[i].
# - The block pool is GLOBAL (HMA): block ids are shared across groups; each
#   layer simply indexes its own canonical view by block id.
#
# The worker connector exposes these views to the KVFlow chunk engine
# (GPU-direct put/get). Durability lives in iaxl.kvstore.KVStore,
# not here.
class Canonicalizer:
    """Builds canonical (num_blocks, page_size_bytes) int8 views per layer."""

    def __init__(self, layer_infos: dict[str, LayerPageInfo], num_blocks: int):
        """Record per-layer descriptors and the GLOBAL block-pool size
        (shared across groups); views are built by ``register``."""
        self._layer_infos: dict[str, LayerPageInfo] = layer_infos
        self._num_blocks: int = num_blocks
        self._views: dict[str, torch.Tensor] = {}

    def register(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        """Build the canonical (num_blocks, page_size_bytes) int8 views
        over the vLLM kv_caches, handling Mamba list-of-tensors storage
        and split-K/V attention layouts. Raises ValueError on any
        descriptor/storage mismatch (fail closed). Called once per layer
        set."""
        for layer_name, info in self._layer_infos.items():
            raw = kv_caches[layer_name]
            if isinstance(raw, (list, tuple)):
                # vLLM builds mamba state tensors by as_strided over one
                # raw storage with a running byte offset starting at 0,
                # so the first tensor's storage is the whole state pool.
                first = raw[0]
                tensor = torch.empty(
                    0, dtype=torch.int8, device=first.device
                ).set_(first.untyped_storage())
            else:
                tensor = torch.empty(
                    0, dtype=torch.int8, device=raw.device
                ).set_(raw.untyped_storage())

            # FlashAttention split-K/V layout fix: a pure
            # attention kv_cache is shaped [2, N, block_size, H, D] whose
            # physical storage is [K block0..N-1][V block0..N-1]. A logical
            # block's K and V are NOT contiguous, so a single flat
            # (num_blocks, page_size) view with stride=page_size strides
            # over TWO ADJACENT K blocks instead of block b's K+V. Detect
            # the physical dim that holds num_blocks (same algorithm as
            # vLLM's offloading worker) and, when K and V are split, keep a
            # (k_view, v_view) pair of (num_blocks, half_page) views; each
            # logical page is then K||V.
            if (not isinstance(raw, (list, tuple))
                    and self._is_split_kv_layout(raw, info)):
                N = info.num_blocks
                half = info.page_size_bytes // 2
                storage = raw.untyped_storage()
                base = torch.empty(
                    0, dtype=torch.int8, device=raw.device).set_(storage)
                flat = base.view(2, N, half)
                k_view, v_view = flat.unbind(0)  # each (num_blocks, half)
                self._views[layer_name] = (k_view, v_view)
                logger.info(
                    "Canonicalized split-K/V layer %s: N=%d half_page=%d",
                    layer_name, N, half)
                continue

            # Contiguous pages: last page end must fit in storage:
            # stride*(num_blocks-1) + page_size
            stride = info.page_size_bytes
            needed = stride * (info.num_blocks - 1) + info.page_size_bytes
            if needed > storage_size_bytes(raw):
                raise ValueError(
                    f"Layer {layer_name}: descriptor requires {needed} bytes "
                    f"but storage has {storage_size_bytes(raw)}")
            view = torch.as_strided(
                tensor,
                size=(info.num_blocks, info.page_size_bytes),
                stride=(stride, 1),
                storage_offset=0,
            )
            self._views[layer_name] = view
        logger.info(
            "Canonicalized %d layers, %d blocks, page=%d bytes",
            len(self._views), self._num_blocks,
            next(iter(self._layer_infos.values())).page_size_bytes)

    @staticmethod
    def _is_split_kv_layout(
        raw: torch.Tensor, info: LayerPageInfo
    ) -> bool:
        """True when num_blocks lives in a non-leading physical dim, i.e.
        the K/V-split [2, N, ...] FlashAttention layout. Mirrors the
        physical-to-logical stride mapping in vLLM's offloading worker."""
        if raw.dim() < 2 or raw.shape[0] != 2:
            return False
        # logical num_blocks dim: find which logical dim equals num_blocks
        strides = raw.stride()
        physical_to_logical = sorted(
            range(len(strides)), key=lambda i: strides[i], reverse=True)
        # the logical dim carrying num_blocks is dim 1 in [2, N, ...]
        try:
            logical_nb_dim = list(raw.shape).index(info.num_blocks)
        except ValueError:
            return False
        physical_pos = physical_to_logical.index(logical_nb_dim)
        return physical_pos != 0

    def _page_parts(
        self, layer_name: str, block_id: int
    ) -> tuple[torch.Tensor, ...]:
        """Physical tensor(s) holding logical block ``block_id``: a single
        view for contiguous layouts, or (K, V) halves for split layouts."""
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return (v[0][block_id], v[1][block_id])
        return (v[block_id],)

    def page_view_parts(self, layer_name: str) -> dict[str, torch.Tensor]:
        """Full-pool canonical page views for the KVFlow chunk engine."""
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return {"k": v[0], "v": v[1]}
        return {"page": v}

    def get_page(self, layer_name: str, block_id: int) -> torch.Tensor:
        """Single tensor for one logical page: the canonical view row,
        or a concatenated K||V view for split-K/V layers. Used by the
        read/zero paths."""
        parts = self._page_parts(layer_name, block_id)
        if len(parts) == 1:
            return parts[0]
        # split-K/V: return a concatenated K||V view for read/zero paths
        return torch.cat([p.reshape(-1) for p in parts])

def storage_size_bytes(t: torch.Tensor | list[torch.Tensor]) -> int:
    """Bytes of the underlying untyped storage for a kv_cache entry;
    Mamba entries (list/tuple of tensors sharing one storage) report
    the first tensor's storage. Used for descriptor bounds checks."""
    if isinstance(t, (list, tuple)):
        return t[0].untyped_storage().size()
    return t.untyped_storage().size()


# ======================================================================
# hybrid layout vocabulary
# ======================================================================
@dataclass(frozen=True)
class LayerPageInfo:
    """Canonical page info for one layer (as seen by the connector)."""
    num_blocks: int  # global block pool size for this layer's view
    page_size_bytes: int


@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage spec."""

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
    """Logical key for one page, or for a whole boundary."""
    namespace: str
    tp_size: int
    rank: int
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
    def boundary_key(self) -> tuple[str, int, int, str, int]:
        """Isolation-safe identity for a boundary (namespace/tp/rank/hash/group)."""
        return (self.namespace, self.tp_size, self.rank, self.hash_str,
                self.group_idx)


def make_boundary_key(namespace: str, tp_size: int, rank: int,
                      group_idx: int, block_hash: object) -> "CacheKey":
    """A group's key at one block hash, with no layer: the address the
    hit policy asks about and the address the save/load builders expand
    into per-layer page keys."""
    return CacheKey(
        namespace=namespace, tp_size=tp_size, rank=rank,
        block_hash=block_hash, group_idx=group_idx, layer_name="")


@dataclass
class GroupTransferMeta:
    """Per-group transfer instructions for one request (one step)."""
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
    """Refuse to start when the codec is configured to lose bits."""
    lossy = os.getenv("IAXL_KV_LOSSY_TRUNC", "0").strip()
    if lossy in ("", "0"):
        return
    raise KVShrinkParseError(
        f"IAXL_KV_LOSSY_TRUNC={lossy!r} is not supported for models with "
        "GDN/Mamba layers: state pages are stored as opaque bytes, so the "
        "truncation would corrupt the recurrent state itself and produce "
        "wrong output with no error. Set IAXL_KV_LOSSY_TRUNC=0.")


def compute_namespace(
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    cache_dtype: str,
    kv_schema_version: int,
    tp_size: int,
    pp_size: int,
) -> str:
    """Stable cache namespace: sha256 over model identity, cache dtype,
    schema version and tp/pp size, truncated to 16 hex chars. The schema
    version is an input so that changing the page layout renames the
    namespace, rather than reading old pages under the new layout."""
    raw = "|".join([
        model_id, model_revision, tokenizer_revision, cache_dtype,
        str(kv_schema_version), str(tp_size), str(pp_size),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
) -> tuple[list[GroupInfo], dict[str, LayerPageInfo], int]:
    """Return (groups, layer_infos, num_blocks)."""
    num_blocks = kv_cache_config.num_blocks
    groups: list[GroupInfo] = []
    layer_infos: dict[str, LayerPageInfo] = {}

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
        for name, spec in per_layer_specs:
            layer_infos[name] = LayerPageInfo(
                num_blocks=num_blocks,
                page_size_bytes=int(spec.page_size_bytes),
            )

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
    return groups, layer_infos, num_blocks


def save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")


def _now() -> float:
    """Monotonic clock for step-latency accounting: immune to NTP
    steps, so measured durations are never negative."""
    import time as _t
    return _t.monotonic()


# ======================================================================
# longest-hit policy
# ======================================================================
class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs that we must supply."""

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
        """How far this group alone is restorable, in tokens."""
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
class _AsyncLoad:
    """One request's cross-step load."""

    __slots__ = ("layer_tasks", "gate_layers", "released")

    def __init__(
        self, layer_tasks: dict[str, Task],
        gate_layers: set[str],
    ):
        self.layer_tasks: dict[str, Task] = layer_tasks
        self.gate_layers: set[str] = gate_layers
        self.released: bool = False


@dataclass
class _SaveCandidate:
    """One boundary's pages accumulated across this step's save plans
    (cross-request dedup by boundary key)."""
    group_idx: int
    pages: dict[str, tuple[CacheKey, int]] = field(default_factory=dict)


############################################################
# Connector
############################################################

class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    """KVShrink external KV cache connector."""

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

        # Ordered worker-side layer names (populated in
        # register_kv_caches) for the block store layout.
        self._layer_names: list[str] = []

        self._async_load_layer_config: AsyncLoadLayerConfig = (
            load_async_load_layer_config_from_env(
                num_layers=self.num_layers,
            )
        )

        self._groups: list[GroupInfo] = []
        self._layer_infos: dict[str, LayerPageInfo] = {}
        # Authoritative TP rank, set by _init_kv_stack. Distinct from
        # self.rank, which is read from the world group before
        # distributed init has run and is therefore 0 on every worker.
        self._rank: int = 0
        self.kvstore: Optional[KVStore] = None

        # Scheduler-side planning state (harmless on the worker role,
        # which never receives scheduler hooks).
        self._namespace: str = ""
        self._hash_block_size: int = 0
        # Where a block's cache identity comes from; see
        # _choose_block_hash_source.
        self._block_hash_source: str = "vllm"
        # Attention layers in group order, used to size the
        # early-release prefix. Mamba layers are deliberately absent:
        # they are never partially released (see _decide_async).
        self._attention_layers: tuple[str, ...] = ()
        self._req_states: dict[str, ReqState] = {}
        # Async requests whose load plan has not been emitted yet. See
        # update_state_after_alloc for why this cannot be derived from
        # the scheduler output.
        self._async_load_pending: set[str] = set()

        # Worker-side execution state.
        self._canon: Optional[Canonicalizer] = None
        # Store namespace per group (see the lookup-vocabulary section
        # for why group and rank must be part of it).
        self._labels: list[str] = []
        # Per-step load tasks, "layer#part" -> engine Task: populated
        # by start_load (one merged engine call per group per step),
        # waited and popped per layer -- GDN layers as one barrier in
        # start_load, attention layers at their forward-entry hooks.
        self._load_tasks: dict[str, Task] = {}
        # Pipelined attention saves: layer_name -> (group_idx, tasks).
        self._step_attn_saves: dict[str, tuple[int, dict[str, Task]]] = {}
        # In-flight ASYNC loads: req_id -> _AsyncLoad. Unlike
        # _load_tasks these deliberately OUTLIVE the step that submitted
        # them -- the whole point is that the request is not occupying a
        # forward step while its pages arrive. Entries leave in two
        # stages: released (reported through get_finished, so vLLM may
        # schedule the request again) and then drained (its remaining
        # layers waited by the per-layer hooks during that forward).
        self._async_loads: dict[str, "_AsyncLoad"] = {}
        # attention layer_name -> group idx (mamba layers map out).
        self._attn_layer_group: dict[str, int] = {}
        # All GDN layer names, waited as one barrier in start_load
        # before any attention layer runs (populated in
        # _register_layer_caches).
        self._mamba_layers: frozenset[str] = frozenset()
        # Attention layers in model execution order, used by the async
        # release gate ("the first N layers" means nothing otherwise).
        self._attn_order: tuple[str, ...] = ()

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
        """Build the hybrid stack for this role."""
        pc = vllm_config.parallel_config
        tp_size = pc.tensor_parallel_size
        if role == KVConnectorRole.WORKER:
            # parallel_config.rank, NOT get_world_group(): the connector
            # is constructed before distributed init in the worker
            # processes, so the world group would report rank 0 on every
            # TP rank and the ranks would overwrite each other's shards.
            rank = pc.rank
        else:
            # Scheduler-side keys are always rank 0; each worker verifies
            # its own shard through its own store.
            rank = 0
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        self._hash_block_size = hash_block_size
        self._namespace = compute_namespace(
            model_id=model_config.model,
            model_revision=model_config.revision or "",
            tokenizer_revision=str(
                model_config.tokenizer_revision or ""),
            cache_dtype=cache_config.cache_dtype,
            kv_schema_version=SCHEMA_VERSION,
            tp_size=tp_size,
            pp_size=1,
        )
        # Fail-closed: a lossy codec would corrupt GDN state (see the
        # function for why this path cannot tolerate what the
        # attention-only path is designed to).
        validate_codec_env()

        groups, layer_infos, num_blocks = parse_kv_cache_config(
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
        self._layer_infos = layer_infos
        self._rank = rank
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
            self._canon = Canonicalizer(layer_infos, num_blocks)
            self._labels = [
                group_label(self._namespace, g.group_idx, rank)
                for g in groups]
            self._attn_layer_group = {
                ln: g.group_idx for g in groups if g.kind != "mamba"
                for ln in g.layer_names}

        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, %d layers, "
            "hash_block_size=%d, namespace=%s, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), len(layer_infos), hash_block_size,
            self._namespace, tp_size, rank)
        logger.info(
            "kvshrink groups: %s",
            [(g.group_idx, g.kind, g.block_size) for g in groups])

    @staticmethod
    def _choose_block_hash_source(recurrent: bool) -> str:
        """Which block-identity scheme to key the cache with."""
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

    def on_new_request(
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
        """Load plans for requests vLLM parked, drained exactly once."""
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
        """This request's block identities, in block order."""
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
        """
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
        """External lookup; returns (hit_tokens, has_async_load)."""
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        block_hashes = self._request_block_hashes(request)
        self.on_new_request(
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
        """Record the allocated block tables per group (after alloc)."""
        req_id = request.request_id
        state = self._req_states.get(req_id)
        if state is None:
            self.on_new_request(
                req_id, self._request_block_hashes(request), 0,
                request=request)
            state = self._req_states[req_id]
        state.num_computed_tokens = (
            state.num_computed_tokens + num_external_tokens)
        state.pending_load_tokens = num_external_tokens
        all_block_ids = blocks.get_block_ids()
        for g_idx, group in enumerate(self._groups):
            if g_idx >= len(all_block_ids):
                continue
            ids = list(all_block_ids[g_idx])
            state.groups[g_idx].block_ids = ids
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
        # Saves are SYNCHRONOUS (wait_for_save writes every page
        # inside the step), so no in-flight job
        # can still reference these blocks and vLLM may free them
        # immediately. Returning True would promise a get_finished()
        # ack that never comes -- a deterministic block leak.
        # Committed boundaries are content-addressed and outlive the
        # request; they are never deleted here.
        self.on_request_finished(request.request_id)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23.0 calls this instead of
        ``request_finished`` whenever the hybrid memory allocator is on,
        which is the default for every model).
        """
        return self.request_finished(request, [])

    # ------------------------------------------------------------------
    def build_load_meta(
        self, new_req: "NewRequestData", scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build the LOAD ReqMeta for a NewRequestData entry."""
        req_id = new_req.req_id
        state = self._req_states[req_id]
        return self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build the LOAD ReqMeta for a PREEMPTION-RESUMED request."""
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
        """Production save: INCREMENTAL per-group page persistence."""
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
        """Assemble this pass's hybrid plans."""
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
        """This rank's boundary key for one group at one block hash."""
        return make_boundary_key(self._namespace, self.tp_size,
                                 self._rank, group.group_idx, block_hash)

    def _present(self, group_idx: int, block_hash: object) -> bool:
        """Store-presence predicate handed to the hit policy, which
        plans against boundary addresses without seeing store details."""
        return lookup_boundary(
            self.kvstore,
            make_boundary_key(self._namespace, self.tp_size, self._rank,
                              group_idx, block_hash))

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same
        namespace/tp/rank/hash/group as the boundary, plus the layer
        name. This is the exact page address the worker must move."""
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
    # Save path
    # ---------
    # Attention groups save PIPELINED: ``save_kv_layer`` submits each
    # layer's async D2H+zip at that layer's exit (the layer's pages for
    # this step are final then). GDN groups save in ``wait_save`` (their
    # state is final only post-forward). Waiting for every write happens
    # in ``wait_save``.
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

        if self._canon is not None:
            self._register_layer_caches(kv_caches)

            # The store is handed the canonical page views, not the raw
            # tensors: every view is a (num_blocks, page_bytes) int8
            # array whose dim-0 rows are blocks, which is the shape the
            # transfer path uses for both attention and GDN. Raw tensors
            # would not do -- a GDN layer is two tensors of different
            # shape and dtype, and the engine requires one shape per
            # call. Views are keyed per part because a split-K/V
            # attention layer contributes two.
            views = {
                f"{ln}#{part}": view
                for ln in self._layer_names
                for part, view in self._canon.page_view_parts(ln).items()
            }
            self.kvstore = KVStore(
                model_name=os.path.basename(self.model_config.model),
                block_dim=0,
                kv_caches=views,
                # _rank, not self.rank: the latter comes from the world
                # group, which reports 0 on every TP rank because the
                # connector is built before distributed init. Two ranks
                # claiming rank 0 collide on the management port and
                # would overwrite each other's shards.
                rank=self._rank,
                tp_size=self.tp_size,
            )
            logger.info("Registered %d KV cache layers (%d page views)",
                        len(kv_caches), len(views))

    def _register_layer_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        """Bind canonical page views, in model execution order."""
        from vllm.model_executor.models.utils import extract_layer_index

        self.register(kv_caches, sorted(kv_caches, key=extract_layer_index))

    def register(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        execution_order: list[str],
    ) -> None:
        """Bind canonical page views and record which layers recur."""
        self._layer_names = list(execution_order)
        self._canon.register(kv_caches)
        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        self._attn_order = tuple(
            ln for ln in execution_order if ln in self._attn_layer_group)
        logger.info(
            "kvshrink hybrid worker registered: %d layers, %d attention "
            "hook points, %d recurrent layers (namespace tp=%d rank=%d)",
            len(self._layer_infos), len(self._attn_order),
            len(self._mamba_layers), self.tp_size, self._rank)

    def _metadata(self) -> KVShrinkConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkConnectorMetadata")
        return metadata

    def _worker_key(self, key: CacheKey) -> CacheKey:
        """Remap a scheduler-built key (rank 0) to this worker's own
        rank: each TP rank persists and loads its OWN shard under its
        own rank path. Without this, TP>1 workers overwrite each
        other's pages under the shared rank-0 key."""
        if key.rank == self._rank:
            return key
        return replace(key, rank=self._rank)

    # ------------------------------------------------------------------
    # store transfers
    # ------------------------------------------------------------------
    # A layer contributes one page view, or two when its K and V are
    # separate tensors. The engine takes one flat tensor dict, so part
    # views are flattened under "layer#part" keys (loads regroup the
    # returned tasks by layer with an rsplit on "#").
    @staticmethod
    def _flat_views(
        layer_views: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {f"{ln}#{part}": view
                for ln, parts in layer_views.items()
                for part, view in parts.items()}

    def _wait_load(self, tasks: dict[str, Task]) -> None:
        """Host-block until these reads land; an incomplete transfer is
        fatal -- forward is about to read these blocks."""
        if tasks and not self.kvstore.get_wait(get_results=tasks,
                                               wait=True):
            raise RuntimeError(
                "kvshrink load failed: get_wait reported an incomplete "
                "transfer; forward would read unrestored blocks")

    def _wait_store(self, tasks: dict[str, Task]) -> None:
        """Host-block until these writes land. An incomplete write is
        fail-stop, same as an incomplete load: the scheduler's save
        cursor has already advanced past these blocks, so losing them
        silently would skip them for the rest of the process."""
        if not tasks:
            return
        if not self.kvstore.put_wait(put_results=tasks, wait=True):
            raise RuntimeError(
                "kvshrink save failed: put_wait reported an incomplete "
                "transfer; the save cursor has already advanced")

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
        """
        self._load_tasks = {}
        self._step_attn_saves = {}
        npages = 0
        _t0 = _now()
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
                (_now() - _t0) * 1e3, self._rank, self.tp_size)
        return npages

    def _register_async_load(
        self, req_id: str, req_meta: ReqMeta,
        tasks: dict[str, Task],
    ) -> None:
        """Track one request's cross-step load and compute its release
        gate.
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
        self._async_loads[req_id] = _AsyncLoad(tasks, gate)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "async load req=%s layers=%d gate=%d (recurrent=%d) "
                "requested_prefix=%s", req_id, len(layers), len(gate),
                len(recurrent), n)

    def poll_finished_loads(self) -> set[str]:
        """Report async requests whose gate layers have landed."""
        finished: set[str] = set()
        for req_id, entry in list(self._async_loads.items()):
            if entry.released:
                continue
            gate = {k: t for k, t in entry.layer_tasks.items()
                    if k.rsplit("#", 1)[0] in entry.gate_layers}
            if gate and not self.kvstore.get_wait(get_results=gate,
                                                  wait=False):
                continue  # not landed yet; ask again next step
            # Landed: finalize the gate layers and hand the rest to
            # the per-layer hooks.
            self._wait_load(gate)
            for k in gate:
                del entry.layer_tasks[k]
            entry.released = True
            finished.add(req_id)
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        return finished

    def wait_for_layer_load(self, layer_name: str) -> None:
        # This attention layer's pages. GDN was already waited for
        # before forward began.
        self.wait_layer_load(layer_name)

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages."""
        keys = [k for k in self._load_tasks
                if k.rsplit("#", 1)[0] == layer_name]
        if keys:
            self._wait_load({k: self._load_tasks.pop(k) for k in keys})
        for req_id, entry in list(self._async_loads.items()):
            if not entry.released:
                continue
            keys = [k for k in entry.layer_tasks
                    if k.rsplit("#", 1)[0] == layer_name]
            if keys:
                self._wait_load(
                    {k: entry.layer_tasks.pop(k) for k in keys})
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)

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
        ``entries``; returns the flat tasks dict ("layer#part" ->
        Task) and the page count."""
        tensors = self._flat_views(
            {ln: self._canon.page_view_parts(ln) for ln in layer_names})
        tasks = self.kvstore.get(block_indices=[gpu for gpu, _ in entries],
                                 block_hashs=[h for _, h in entries],
                                 layer_names=list(tensors),
                                 tensors=tensors,
                                 label=self._labels[g_idx])
        return tasks, len(entries) * len(tensors)

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def _gather_save_candidates(
        self, metadata: KVShrinkConnectorMetadata
    ) -> dict[tuple[str, int, int, str, int], _SaveCandidate]:
        """Batch-level boundary candidates with cross-request dedup.
        Returns boundary_key -> accumulated per-layer pages."""
        candidates: dict[tuple[str, int, int, str, int], _SaveCandidate] = {}
        for req_meta in metadata.reqs_to_save.requests.values():
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    key = self._worker_key(key)
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = _SaveCandidate(op.group_idx)
                        candidates[key.boundary_key] = cand
                    cand.pages[key.layer_name] = (key, gpu_block_id)
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
        tensors = self._flat_views(
            {ln: self._canon.page_view_parts(ln) for ln in layer_names})
        return self.kvstore.put(block_indices=[gpu for gpu, _ in entries],
                                block_hashs=[h for _, h in entries],
                                layer_names=list(tensors),
                                tensors=tensors,
                                label=self._labels[g_idx])

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Pipelined attention save. vLLM calls this on exit of EVERY
        attention layer during forward (kv_transfer_utils decorator).
        """
        if self._connector_metadata is None:
            return
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not save_enabled():
            return
        metadata = self._metadata()
        g_idx = self._attn_layer_group[layer_name]
        expected = sorted(self._groups[g_idx].layer_names)
        entries: list[tuple[int, str]] = []
        for _bkey, cand in self._gather_save_candidates(metadata).items():
            if cand.group_idx != g_idx:
                continue
            if sorted(cand.pages) != expected:
                continue  # partial boundary: skipped at commit time too
            if layer_name not in cand.pages:
                continue
            key, gpu_block_id = cand.pages[layer_name]
            entries.append((gpu_block_id, key.hash_str))
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, [layer_name],
                                               entries)
        self._step_attn_saves[layer_name] = (g_idx, tasks)

    def wait_for_save(self) -> None:
        if self.kvstore is None:
            return

        if not save_enabled():
            return
        metadata = self._metadata()
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("wait_for_save worker: reqs_to_save=%d",
                        len(metadata.reqs_to_save.requests))
        pages, boundaries = self.wait_save(metadata)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("chunk_save: %d pages, %d boundaries",
                        pages, boundaries)
        self.debug_dump_state()

    def wait_save(
        self, metadata: KVShrinkConnectorMetadata
    ) -> tuple[int, int]:
        """Post-forward save: GDN groups submit here; attention groups
        collect their pipelined tasks; then wait for the writes,
        write every page of every group.
        Fail-stop on any anomaly (the scheduler already advanced its
        incremental indices). Returns (pages, boundaries)."""
        _t0 = _now()
        candidates = self._gather_save_candidates(metadata)
        pipelined = os.getenv("KVSHRINK_SAVE_PIPELINED", "1") != "0"
        # group the complete candidates for one engine put per group
        per_group: dict[int, dict[str, list[tuple[int, str]]]] = {}
        nbound = 0
        for bkey, cand in candidates.items():
            namespace, tp_size, rank, blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand.pages) != expected:
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
            layers = per_group.setdefault(g_idx, {})
            for layer_name in expected:
                key, gpu_block_id = cand.pages[layer_name]
                layers.setdefault(layer_name, []).append(
                    (gpu_block_id, key.hash_str))
        npages = 0
        for g_idx, layers in per_group.items():
            entries = layers[next(iter(layers))]
            if self._groups[g_idx].kind != "mamba" and pipelined:
                # Attention: save_kv_layer submitted each layer during
                # forward; wait for them here. A layer the
                # decorator never fired for is submitted now -- the
                # plan must be fully covered either way.
                for layer_name in layers:
                    stashed = self._step_attn_saves.pop(layer_name, None)
                    tasks = (stashed[1] if stashed is not None
                             else self._submit_group_layers_save(
                                 g_idx, [layer_name], entries))
                    self._wait_store(tasks)
            else:
                tasks = self._submit_group_layers_save(
                    g_idx, list(layers), entries)
                self._wait_store(tasks)
            chunk_indices = [gpu for gpu, _ in entries]
            npages += len(chunk_indices) * len(layers)
            # A block is finalized by its own write, with every layer of
            # the group in one call, so it is committed the moment the
            # write lands. There is no second phase to publish and
            # therefore nothing that can outlive its data.
        if self._step_attn_saves:
            # Layers whose submit was never consumed by a group above
            # (plan changed mid-step): wait them so pinned staging is
            # released, then drop. Their data is identical to what the
            # group path stored (same labels), so nothing is lost.
            logger.warning(
                "chunk_save: %d stashed layer saves unconsumed (%s); "
                "draining", len(self._step_attn_saves),
                sorted(self._step_attn_saves))
            for _ln, (_g, tasks) in self._step_attn_saves.items():
                self._wait_store(tasks)
            self._step_attn_saves.clear()
        if npages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            logger.info(
                "chunk_save: %d pages stored, %d boundaries committed "
                "elapsed_ms=%.3f (rank %d/%d)", npages, nbound,
                (_now() - _t0) * 1e3, self._rank, self.tp_size)
        return npages, nbound

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Report transfers that completed since the last step."""
        if self.kvstore is None:
            return None, None
        # Saves complete within the step, so nothing is ever reported as
        # finished-sending. Loads may not: an async request stays parked
        # until we name it here.
        recving = self.poll_finished_loads()
        return None, (recving or None)

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
            for blk in range(10):
                page = self._canon.get_page(ln, blk)
                h = hashlib.sha256(
                    page.cpu().numpy().tobytes()).hexdigest()
                logger.info("DUMP g%d block=%d sha=%s",
                            group.group_idx, blk, h[:16])
