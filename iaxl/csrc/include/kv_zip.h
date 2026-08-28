// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

namespace kv_zip {

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                                                     std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                                                     bool compress = true,
                                                     bool lossy_trunc_enabled = false);

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                                                         const std::vector<torch::Tensor> &tensors);

} // namespace kv_zip
