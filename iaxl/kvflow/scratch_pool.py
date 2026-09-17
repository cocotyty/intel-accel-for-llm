# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixed-size pinned scratch pool: one big [num_blocks, *block_shape] tensor
(IAXL_SCRATCH_POOL_SIZE_GB), blocks handed out as views. No growth: allocate()
raises once the pool is exhausted.

Each block carries its slot as `tensor.block_idx`, which is also its offset in
`pool` (register `pool` once for RDMA and use block_idx as the descriptor index).
allocate()/release() are not thread-safe: call them from one thread only.
"""

import logging
import math
from typing import List, Optional, Sequence, Tuple

import torch

from ..envs import envs

logger = logging.getLogger(__name__)


class ScratchPool:
    def __init__(self, block_shape: Sequence[int], dtype: torch.dtype, pin_memory: bool = True):
        self.block_shape = tuple(block_shape)
        self.dtype = dtype
        self.block_bytes = math.prod(self.block_shape) * torch.tensor([], dtype=dtype).element_size()
        num_blocks = int(envs.IAXL_SCRATCH_POOL_SIZE_GB * 1024**3) // self.block_bytes
        self.pool = torch.empty((num_blocks, *self.block_shape), dtype=dtype, device="cpu", pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        self._blocks: Tuple[torch.Tensor, ...] = self.pool.unbind(0)
        for i, t in enumerate(self._blocks):
            t.block_idx = i  # type: ignore[attr-defined]
        self._free: List[int] = list(range(num_blocks))  # LIFO: pop from the end
        self._allocate_count = 0
        self._release_count = 0
        logger.info("ScratchPool: %d x %s %s = %.2f MB pinned",
                    num_blocks, self.block_shape, dtype, self.pool.nbytes / 2**20)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def allocate(self, count: int, shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        shape = self.block_shape if shape is None else tuple(shape)
        dtype = self.dtype if dtype is None else dtype
        block_bytes = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        if block_bytes != self.block_bytes:
            raise ValueError(f"ScratchPool: block size {block_bytes} != {self.block_bytes}")
        free = self._free
        if count > len(free):
            raise RuntimeError(f"ScratchPool exhausted: need {count}, {len(free)}/{len(self._blocks)} free")
        cut = len(free) - count  # not free[-count:], which is the whole list for count == 0
        idx = free[cut:]
        del free[cut:]
        self._allocate_count += count
        blocks = self._blocks
        if shape == self.block_shape and dtype == self.dtype:
            return [blocks[i] for i in idx]
        # Hybrid attention and Mamba pages share a byte size, not a dtype/shape.
        # Keep the registered backing allocation and its RDMA slot indices.
        result = []
        for i in idx:
            block = blocks[i].view(dtype).view(shape)
            block.block_idx = i
            result.append(block)
        return result

    def release(self, tensors: List[torch.Tensor]):
        self._free.extend(t.block_idx for t in tensors)  # type: ignore[attr-defined]
        self._release_count += len(tensors)

    def available_count(self, shape: Optional[Sequence[int]] = None, dtype: Optional[torch.dtype] = None) -> int:
        return len(self._free)

    def total_count(self, shape: Optional[Sequence[int]] = None, dtype: Optional[torch.dtype] = None) -> int:
        return len(self._blocks)

    def total_bytes(self) -> int:
        return self.pool.nbytes

    def status(self) -> str:
        free, in_use = len(self._free), self._allocate_count - self._release_count
        return (f"ScratchPool: {len(self._blocks)} blocks {self.block_shape} {self.dtype}, "
                f"{free} available, {in_use} in-use, alloc/release={self._allocate_count}/{self._release_count}, "
                f"{self.pool.nbytes / 2**20:.2f} MB")
