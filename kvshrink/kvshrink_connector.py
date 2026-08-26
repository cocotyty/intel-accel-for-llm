# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KVShrink hybrid (GDN) KV cache metadata structures."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

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
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, setup_root_logger

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
    # The live vLLM Request, when we were handed one. Held so the save
    # path can read AUTHORITATIVE block hashes as they grow: vLLM
    # appends to Request.block_hashes every time decode completes a
    # block, and recomputing them here instead would have to reproduce
    # vLLM's hashing byte for byte forever. Dropped in
    # on_request_finished together with the rest of the state.
    request: Any = None
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
# lookup vocabulary (shared by scheduler, worker and backends)
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
# - Backend lookups return HIT / MISS (chunk tier is Record-gated,
#   request, never allocates.
class LookupStatus(Enum):
    """Two-valued lookup verdict: HIT restores from external cache,
    MISS recomputes (the fail-closed default)."""
    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class LookupResult:
    """Frozen policy output: the verdict and the number of boundary
    tokens restorable from the external cache."""
    status: LookupStatus
    boundary_tokens: int = 0  # tokens that can be restored from external cache


def align_down(tokens: int, align: int) -> int:
    """Largest multiple of ``align`` not exceeding ``tokens``. Snaps
    mamba candidates to align boundaries: GDN state is only addressable
    on aligned positions."""
    return (tokens // align) * align


# ======================================================================
# canonical page views
# ======================================================================
# Canonical page views over the vLLM GPU KV blocks.
#
# Layout verified on vLLM v0.23.0 + Qwen3.5-4B TP2:
#
# - Attention layer: single tensor; canonical view is built per the
#   LayerPageInfo descriptor (block_stride_bytes / storage_offset_bytes).
# - Mamba layer: ``kv_caches[layer]`` is a LIST of tensors (conv, ssm) sharing
#   one storage. The canonical page view is rebuilt from the first tensor's
#   storage (same approach as vLLM's offloading worker).
# - Physical page for block_id i = view[i]; with
#   ``block_stride_bytes > page_size_bytes`` the layout is packed.
# - The block pool is GLOBAL (HMA): block ids are shared across groups; each
#   layer simply indexes its own canonical view by block id.
#
# The worker connector exposes these views to the KVFlow chunk engine
# (GPU-direct put/get). Durability lives in iaxl.kvstore.KVStore,
# not here.
class Canonicalizer:
    """Builds canonical (num_blocks, page_size_bytes) int8 views per layer."""

    def __init__(self, layer_infos: dict, num_blocks: int):
        """Record per-layer descriptors and the GLOBAL block-pool size
        (shared across groups); views are built by ``register``."""
        self._layer_infos = layer_infos
        self._num_blocks = num_blocks
        self._views: dict[str, torch.Tensor] = {}

    def register(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Build the canonical (num_blocks, page_size_bytes) int8 views
        over the vLLM kv_caches, handling Mamba list-of-tensors storage
        and split-K/V attention layouts. Raises ValueError on any
        descriptor/storage mismatch (fail closed). Called once per layer
        set."""
        for layer_name, info in self._layer_infos.items():
            raw = kv_caches[layer_name]
            if isinstance(raw, (list, tuple)):
                if len(raw) == 0:
                    raise ValueError(
                        f"Mamba layer {layer_name} has empty state list")
                first = raw[0]
                if first.storage_offset() != 0:
                    raise ValueError(
                        f"Mamba layer {layer_name} first state tensor has "
                        f"non-zero storage offset {first.storage_offset()}")
                storage = first.untyped_storage()
                tensor = torch.empty(
                    0, dtype=torch.int8, device=first.device
                ).set_(storage)
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

            stride = info.block_stride_bytes
            # last page end must fit in storage:
            # offset + stride*(num_blocks-1) + page_size
            needed = (info.storage_offset_bytes
                      + stride * (info.num_blocks - 1)
                      + info.page_size_bytes)
            if needed > storage_size_bytes(raw):
                raise ValueError(
                    f"Layer {layer_name}: descriptor requires {needed} bytes "
                    f"but storage has {storage_size_bytes(raw)}")
            view = torch.as_strided(
                tensor,
                size=(info.num_blocks, info.page_size_bytes),
                stride=(stride, 1),
                storage_offset=info.storage_offset_bytes,
            )
            self._views[layer_name] = view
        logger.info(
            "Canonicalized %d layers, %d blocks, page=%d bytes",
            len(self._views), self._num_blocks,
            next(iter(self._layer_infos.values())).page_size_bytes)

    @staticmethod
    def _is_split_kv_layout(raw: torch.Tensor, info) -> bool:
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

    def _page_parts(self, layer_name: str, block_id: int):
        """Physical tensor(s) holding logical block ``block_id``: a single
        view for contiguous layouts, or (K, V) halves for split layouts."""
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return (v[0][block_id], v[1][block_id])
        return (v[block_id],)

    def page_view_parts(self, layer_name: str):
        """Full-pool canonical page views for the KVFlow chunk engine."""
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return {"k": v[0], "v": v[1]}, 0
        return {"page": v}, 0

    def get_page(self, layer_name: str, block_id: int) -> torch.Tensor:
        """Single tensor for one logical page: the canonical view row,
        or a concatenated K||V view for split-K/V layers. Used by the
        read/zero paths."""
        parts = self._page_parts(layer_name, block_id)
        if len(parts) == 1:
            return parts[0]
        # split-K/V: return a concatenated K||V view for read/zero paths
        return torch.cat([p.reshape(-1) for p in parts])

def storage_size_bytes(t: torch.Tensor) -> int:
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
    block_stride_bytes: int
    storage_offset_bytes: int


@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage spec."""

    group_idx: int
    kind: str  # "attention" | "mamba" | "sliding_window" | "mla"
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
                      group_idx: int, block_hash) -> "CacheKey":
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
    snapshot_boundary_tokens: Optional[int] = None  # mamba restore point


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
# - unknown dtype -> KVShrinkParseError
# - packed / restride / non-zero storage offset layouts are parsed into
#   LayerPageInfo.block_stride_bytes / storage_offset_bytes; canonical views
#   are built from these descriptors, not from contiguity assumptions.
# - heterogeneous page sizes across layers are allowed (each layer carries
#   its own page_size_bytes).
class KVShrinkParseError(ValueError):
    """Raised when the vLLM cache config cannot be parsed safely
    (unknown spec, dtype or inconsistent layout). The parse never
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


def _dtype_size(dtype) -> int:
    """Byte size of a dtype. Fail closed on unknown dtypes."""
    s = str(dtype)
    if "bfloat16" in s or "float16" in s:
        return 2
    if "float32" in s or "float" in s:
        return 4
    if "float64" in s or "double" in s:
        return 8
    if "int8" in s or "uint8" in s:
        return 1
    if "int16" in s or "uint16" in s:
        return 2
    if "int32" in s or "uint32" in s or "int" in s:
        return 4
    if "int64" in s or "uint64" in s or "long" in s:
        return 8
    raise KVShrinkParseError(f"Unknown dtype: {dtype}")


def _spec_kind(spec) -> str:
    """Classify a vLLM cache spec as mamba / attention /
    sliding_window; unknown spec types raise KVShrinkParseError (fail
    closed)."""
    if isinstance(spec, MambaSpec):
        return "mamba"
    if isinstance(spec, AttentionSpec):
        if getattr(spec, "sliding_window", None) is not None:
            return "sliding_window"
        return "attention"
    raise KVShrinkParseError(
        f"Unsupported KV cache spec {type(spec).__name__}")


def _iter_layer_specs(group_spec):
    """Yield (layer_name, spec) pairs, expanding UniformTypeKVCacheSpecs."""
    spec = group_spec.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        per_layer = spec.kv_cache_specs
        for name in group_spec.layer_names:
            if name not in per_layer:
                raise KVShrinkParseError(
                    f"UniformTypeKVCacheSpecs missing spec for layer {name}")
            yield name, per_layer[name]
    else:
        for name in group_spec.layer_names:
            yield name, spec


def parse_kv_cache_config(
    kv_cache_config: KVCacheConfig,
    hash_block_size: int,
) -> tuple[list[GroupInfo], dict[str, LayerPageInfo], int]:
    """Return (groups, layer_infos, num_blocks)."""
    num_blocks = kv_cache_config.num_blocks
    groups: list[GroupInfo] = []
    layer_infos: dict[str, LayerPageInfo] = {}

    layer_to_tensor: dict[str, int] = {}
    for t_idx, t in enumerate(kv_cache_config.kv_cache_tensors):
        for name in t.shared_by:
            layer_to_tensor[name] = t_idx

    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        kind = None
        page_size = None
        block_size = None
        per_layer_specs: list[tuple[str, object]] = list(_iter_layer_specs(g))
        if not per_layer_specs:
            raise KVShrinkParseError(f"Group {g_idx} has no layers")

        for name, spec in per_layer_specs:
            sk = _spec_kind(spec)
            if kind is None:
                kind = sk
            elif sk != kind:
                raise KVShrinkParseError(
                    f"Group {g_idx} mixes spec kinds {kind} and {sk}")
            page = int(spec.page_size_bytes)
            if page_size is None:
                page_size = page
            elif page != page_size:
                # Unsatisfiable, not merely awkward: one group is
                # transferred in one engine call, and the engine
                # requires every tensor in a call to share shape and
                # dtype. Canonical views of differing page size cannot
                # satisfy that, so fail closed here rather than at the
                # first transfer.
                raise KVShrinkParseError(
                    f"Group {g_idx} layers have differing page sizes "
                    f"({page} vs {page_size}); unsupported within a group")
            bs = int(spec.block_size)
            if block_size is None:
                block_size = bs
            elif bs != block_size:
                raise KVShrinkParseError(
                    f"Group {g_idx} layers have differing block sizes")

        if kind == "mamba":
            mamba_spec = per_layer_specs[0][1]
            mamba_mode = mamba_spec.mamba_cache_mode
            align = block_size
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
            group = GroupInfo(
                group_idx=g_idx,
                kind=kind,
                layer_names=tuple(g.layer_names),
                block_size=block_size,
                mamba_align_size=align,
                spec=per_layer_specs[0][1],
            )
        elif kind in ("attention", "sliding_window"):
            group = GroupInfo(
                group_idx=g_idx,
                kind=kind,
                layer_names=tuple(g.layer_names),
                block_size=block_size,
                mamba_align_size=None,
                spec=per_layer_specs[0][1],
            )
        else:  # pragma: no cover - _spec_kind raises first
            raise KVShrinkParseError(f"Unsupported kind {kind}")

        groups.append(group)
        for name, spec in per_layer_specs:
            t_idx = layer_to_tensor.get(name)
            if t_idx is None:
                raise KVShrinkParseError(
                    f"Layer {name} not found in kv_cache_tensors")
            tensor = kv_cache_config.kv_cache_tensors[t_idx]
            # MambaSpec carries dtypes (list); AttentionSpec carries dtype.
            if isinstance(spec, MambaSpec):
                if not spec.dtypes:
                    raise KVShrinkParseError(
                        f"Layer {name} MambaSpec has empty dtypes")
                dtype = spec.dtypes[0]
            else:
                dtype = getattr(spec, "dtype", None)
                if dtype is None:
                    raise KVShrinkParseError(
                        f"Layer {name} has no dtype in spec")
            _dtype_size(dtype)  # fail closed on unknown dtype
            # KVCacheTensor is (size, shared_by) only; packed layouts
            # (block_stride/offset) are not expressible and are rejected
            # here rather than guessed.
            t_block_stride = int(getattr(tensor, "block_stride", None) or 0)
            if t_block_stride > 0:
                block_stride_bytes = t_block_stride
            else:
                block_stride_bytes = int(spec.page_size_bytes)
            storage_offset_bytes = int(getattr(tensor, "offset", None) or 0)
            layer_infos[name] = LayerPageInfo(
                num_blocks=num_blocks,
                page_size_bytes=int(spec.page_size_bytes),
                block_stride_bytes=block_stride_bytes,
                storage_offset_bytes=storage_offset_bytes,
            )

    if not groups:
        raise KVShrinkParseError("No kv cache groups parsed")

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


############################################################
# Connector
############################################################

def _save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")


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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device = vllm_config.device_config.device_type
        self.rank = get_world_group().rank if model_parallel_is_initialized() else 0

        # Ordered worker-side layer names (populated in
        # register_kv_caches) for the block store layout.
        self._layer_names: list[str] = []

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
        )

        # One stack for every model. vLLM already describes any model as
        # a list of KV cache groups, so a pure-attention model is simply
        # the one-group case; the only thing a GDN/Mamba model changes is
        # WHICH store layout is underneath, and that is an adapter
        # choice, not a second code path.
        self._sched = None
        self._worker = None
        self._backend = None
        self._canon = None
        self._groups: list = []
        # Authoritative TP rank, set by _init_kv_stack. Distinct from
        # self.rank, which is read from the world group before
        # distributed init has run and is therefore 0 on every worker.
        self._rank: Optional[int] = None
        self.kvstore: Optional[KVStore] = None

        if kv_cache_config is not None:
            self._init_kv_stack(vllm_config, role, kv_cache_config)
        else:
            self.kvstore = None
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
        from .backend import KVStoreBackend
        from .scheduler import HybridRequestScheduler
        from .worker import HybridWorker

        pc = vllm_config.parallel_config
        tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        if role == KVConnectorRole.WORKER:
            # parallel_config.rank, NOT get_world_group(): the connector
            # is constructed before distributed init in the worker
            # processes, so the world group would report rank 0 on every
            # TP rank and the ranks would overwrite each other's shards.
            rank = int(getattr(pc, "rank", 0) or 0)
        else:
            # Scheduler-side keys are always rank 0; each worker verifies
            # its own shard through its own backend.
            rank = 0
        self.kvstore = None
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        namespace = compute_namespace(
            model_id=str(getattr(model_config, "model", "model")),
            model_revision=str(getattr(model_config, "revision", None) or ""),
            tokenizer_revision=str(
                getattr(model_config, "tokenizer_revision", None) or ""),
            cache_dtype=str(getattr(cache_config, "cache_dtype", "auto")),
            kv_schema_version=SCHEMA_VERSION,
            tp_size=tp_size,
            pp_size=1,
        )
        # Fail-closed: a lossy codec would corrupt GDN state (see the
        # function for why this path cannot tolerate what the
        # attention-only path is designed to).
        validate_codec_env()

        groups, layer_infos, num_blocks = parse_kv_cache_config(
            kv_cache_config, hash_block_size=hash_block_size)

        # Fail-closed: speculative decoding widens the GDN state gather.
        # v0.23.0's mamba_get_block_table_tensor returns
        # block_table[start : start + 1 + num_speculative_blocks] and the
        # decode path reads all of those columns, but an external
        # snapshot only ever restores column 0 (the block holding this
        # step's last scheduled token). Serving a hit would let the
        # kernel read unrestored speculative slots.
        for g in kv_cache_config.kv_cache_groups:
            spec_blocks = int(
                getattr(g.kv_cache_spec, "num_speculative_blocks", 0) or 0)
            if spec_blocks:
                raise RuntimeError(
                    "kvshrink hybrid: speculative decoding is not "
                    f"supported (group has num_speculative_blocks="
                    f"{spec_blocks}); the external GDN snapshot only "
                    "restores the non-speculative state slot. Disable "
                    "speculative decoding or the KV connector.")
        self._groups = groups
        self._rank = rank
        # A recurrent group changes only which block hashes we ask
        # about (see _block_hash_source); the storage below is the same.
        recurrent = any(g.kind == "mamba" for g in groups)

        if role == KVConnectorRole.SCHEDULER:
            # Presence-only store: the scheduler asks whether boundaries
            # are readable and never moves bytes.
            self._backend = KVStoreBackend(KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(i) for i in range(self.num_layers)],
                tp_size=self.tp_size,
            ))
            self._backend.register_layout(namespace, tp_size, rank)
            self._sched = HybridRequestScheduler(
                groups, self._backend, hash_block_size, namespace,
                tp_size, rank,
                block_hash_source=self._block_hash_source(recurrent),
                async_load_config=self._async_load_layer_config)
        else:
            # The store needs kv_caches, which arrive later; bound in
            # register_kv_caches.
            self._backend = KVStoreBackend()
            self._backend.register_layout(namespace, tp_size, rank)
            self._canon = Canonicalizer(layer_infos, num_blocks)
            self._worker = HybridWorker(
                groups, layer_infos, self._backend,
                self._canon, rank, tp_size)

        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, %d layers, "
            "hash_block_size=%d, namespace=%s, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), len(layer_infos), hash_block_size, namespace,
            tp_size, rank)
        logger.info(
            "kvshrink groups: %s",
            [(g.group_idx, g.kind, g.block_size) for g in groups])

    @staticmethod
    def _block_hash_source(recurrent: bool) -> str:
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

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return self._sched.get_num_new_matched_tokens(
            request, num_computed_tokens)
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._sched.update_state_after_alloc(
            request, blocks, num_external_tokens)
        return
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
        self._sched.on_request_finished(request.request_id)
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
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        return self._build_connector_meta(scheduler_output)
    def _build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's hybrid plans."""
        meta = KVShrinkConnectorMetadata()
        sched = self._sched
        debug = bool(os.getenv("KVSHRINK_DEBUG_LOG"))
        save_enabled = _save_enabled()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            req_meta = sched.build_load_meta(
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
            if save_enabled:
                save_meta = sched.build_save_meta(
                    new_req.req_id, num_sched.get(new_req.req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[new_req.req_id] = save_meta

        # ASYNC requests ride NEITHER list: vLLM parks them in
        # WAITING_FOR_REMOTE_KVS, so they are absent from
        # scheduled_new_reqs and from scheduled_cached_reqs alike. Their
        # plan comes from what update_state_after_alloc recorded, and
        # without it the worker would have nothing to transfer and the
        # request would wait forever to be released.
        for req_id, req_meta in sched.take_async_load_plans(
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
        for req_id in (getattr(cr, "resumed_req_ids", None) or ()):
            req_meta = sched.build_resumed_load_meta(
                req_id, num_sched.get(req_id, 0))
            if req_meta is None:
                continue
            if debug:
                logger.info(
                    "LOADMETA(resumed) req=%s ops=%d",
                    req_id, len(req_meta.group_ops))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.requests[req_id] = req_meta

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings): sync their tables first,
        # then emit incremental saves.
        if save_enabled:
            resumed = getattr(cr, "resumed_req_ids", None) or set()
            new_bids = getattr(cr, "new_block_ids", None) or []
            ncts = getattr(cr, "num_computed_tokens", None) or []
            for i, req_id in enumerate(getattr(cr, "req_ids", []) or []):
                sched.on_cached_request(
                    req_id,
                    new_bids[i] if i < len(new_bids) else None,
                    req_id in resumed,
                    ncts[i] if i < len(ncts) else None)
                save_meta = sched.build_save_meta(
                    req_id, num_sched.get(req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[req_id] = save_meta

        if debug:
            logger.info(
                "build_connector_meta: %d load reqs, %d save reqs",
                len(meta.reqs_to_load.requests),
                len(meta.reqs_to_save.requests))
        return meta

    def _register_layer_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Bind canonical page views, in model execution order."""
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")
        from vllm.model_executor.models.utils import extract_layer_index

        missing = [ln for ln in self._worker._layer_infos
                   if ln not in kv_caches]
        if missing:
            raise RuntimeError(
                f"kvshrink hybrid: layers {sorted(missing)} are in the KV "
                "cache config but absent from kv_caches; refusing to start")

        indexed: dict[int, str] = {}
        for layer_name in kv_caches:
            try:
                idx = extract_layer_index(layer_name)
            except Exception as exc:
                raise RuntimeError(
                    "kvshrink hybrid: cannot derive the execution index of "
                    f"layer {layer_name!r} ({exc}); refusing to start"
                    ) from exc
            if idx in indexed:
                raise RuntimeError(
                    "kvshrink hybrid: layers "
                    f"{indexed[idx]!r} and {layer_name!r} share execution "
                    f"index {idx}; the execution order is ambiguous")
            indexed[idx] = layer_name
        order = [indexed[i] for i in sorted(indexed)]

        self._layer_names = order
        self._worker.register(kv_caches, order)

    def _metadata(self) -> KVShrinkConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkConnectorMetadata")
        return metadata

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        static_context = self.vllm_config.compilation_config.static_forward_context
        for layer in static_context.values():
            get_backend = getattr(layer, "get_attn_backend", None)
            if get_backend is not None:
                if "FLASHINFER" in get_backend().get_name().upper():
                    raise RuntimeError("FlashInfer is not supported")
                break

        if self._worker is not None:
            self._register_layer_caches(kv_caches)

        if self._backend is not None:
            # The store is handed the canonical page views, not the raw
            # tensors: every view is a (num_blocks, page_bytes) int8
            # array whose dim-0 rows are blocks, which is the shape the
            # transfer path uses for both attention and GDN. Raw tensors
            # would not do -- a GDN layer is two tensors of different
            # shape and dtype, and the engine requires one shape per
            # call. Views are keyed per part because a split-K/V
            # attention layer contributes two.
            views = {
                f"{ln}::{part}": view
                for ln in self._layer_names
                for part, view in self._canon.page_view_parts(ln)[0].items()
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
                rank=self._rank if self._rank is not None else self.rank,
                tp_size=self.tp_size,
            )
            self._backend.bind_store(self.kvstore)
            logger.info("Registered %d KV cache layers (%d page views)",
                        len(kv_caches), len(views))

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        # Submits every load, then host-blocks on the recurrent ones;
        # attention layers are waited by their own hooks during forward.
        self._worker.start_load(self._metadata())
        return
    def wait_for_layer_load(self, layer_name: str) -> None:
        # This attention layer's pages. GDN was already waited for
        # before forward began.
        self._worker.wait_layer_load(layer_name)
        return
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if self._connector_metadata is None:
            return
        # Pipelined attention save: submit this layer's D2H+zip now
        # so it overlaps the remaining layers' compute. GDN groups
        # never reach this hook; they save in wait_for_save.
        self._worker.save_kv_layer(layer_name, self._metadata())
        return
    def wait_for_save(self) -> None:
        if self._worker is None:
            return

        hw = self._worker
        # A sticky load poison and any un-waited load must surface here,
        # before anything is persisted: entering the save path after the
        # forward read unrestored state would commit wrong data.
        hw.raise_load_poison()
        hw.loads_drained_check()
        if not hw.save_enabled():
            return
        metadata = self._metadata()
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("wait_for_save worker: reqs_to_save=%d",
                        len(metadata.reqs_to_save.requests))
        pages, boundaries = hw.wait_save(metadata)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("chunk_save: %d pages, %d boundaries",
                        pages, boundaries)
        hw.debug_dump_state()

    def shutdown(self) -> None:
        """Release the store backend (Record flush, writer lease)."""
        if self._worker is not None:
            self._worker.shutdown()
        elif self._backend is not None:
            self._backend.close()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Report transfers that completed since the last step."""
        if self._worker is None:
            return None, None
        # A sticky poison surfaces at every hook entry and is never
        # swallowed by the finish protocol; raising here also stops a
        # poisoned request from being reported as successfully loaded
        # (vLLM aborts it instead, so it cannot hang).
        self._worker.raise_load_poison()
        # Saves complete within the step, so nothing is ever reported as
        # finished-sending. Loads may not: an async request stays parked
        # until we name it here.
        recving = self._worker.poll_finished_loads()
        return None, (recving or None)
