// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT and CPU workers share one task pool. Each worker claims another item when a request completes,
// so the faster backend naturally processes more of the batch. QAT workers keep multiple
// asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE request each.
// Both backends produce mutually compatible streams, so either backend can decompress any item.

#include <torch/extension.h>

#include <omp.h>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "cpu_zip.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

#define OMP_SCHEDULE dynamic

namespace kv_zip {

enum class ZipBackend { QAT, CPU };

static void ensure_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        IAXL_CHECK(envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE,
                   "kv_zip: QAT and CPU zip backends are both disabled");
        if (envs.IAXL_QAT_ZIP_ENABLE)
            IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed");
        if (envs.IAXL_CPU_ZIP_ENABLE)
            IAXL_CHECK(cpu_zip_init() == 0, "kv_zip: cpu_zip_init failed");
    });
}

template <class Submit, class Complete>
static void zip_pipeline(size_t n, Submit &&submit, Complete &&complete) {
    ensure_zip_init();
    const int qat_depth = envs.IAXL_QAT_ZIP_ENABLE ? qat_zip_queue_depth() : 1;
    const int qat_available = envs.IAXL_QAT_ZIP_ENABLE ? qat_zip_num_slots() / qat_depth : 0;
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + cpu_workers;
    IAXL_CHECK(qat_workers <= qat_available,
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(worker_count == envs.IAXL_OMP_THREAD_NUM,
               "kv_zip: compression workers do not match OMP_NUM_THREADS");

    std::atomic<size_t> next{0};
#pragma omp parallel num_threads(worker_count)
    {
        const int t = omp_get_thread_num();
        const bool use_qat = t < qat_workers;
        const ZipBackend backend = use_qat ? ZipBackend::QAT : ZipBackend::CPU;
        const int depth = use_qat ? qat_depth : 1;
        const int base = use_qat ? t * qat_depth : t - qat_workers;
        IAXL_CHECK(omp_get_num_threads() == worker_count,
                   "kv_zip: OpenMP did not create the configured worker team");

        int active_depth = 0;
        std::vector<size_t> slot_item(static_cast<size_t>(depth));
        for (int k = 0; k < depth; k++) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n)
                break;
            submit(backend, base + k, i);
            slot_item[k] = i;
            active_depth++;
        }

        int in_flight = active_depth;
        bool draining = false;
        for (int s = 0; in_flight > 0; s = (s + 1) % active_depth) {
            void *out;
            int out_len;
            const int status = use_qat ? qat_zip_wait(base + s, &out, &out_len)
                                       : cpu_zip_wait(base + s, &out, &out_len);
            IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
            complete(backend, slot_item[s], out, out_len);

            const size_t i = draining ? n : next.fetch_add(1, std::memory_order_relaxed);
            if (i < n) {
                submit(backend, base + s, i);
                slot_item[s] = i;
            } else {
                draining = true;
                in_flight--;
            }
        }
    }
}

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress, bool lossy_trunc_enabled) {
    const size_t n = tensors.size();

    if (!compress || !envs.IAXL_KV_COMPRESSION) {
#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
        for (size_t i = 0; i < n; i++) {
            const auto &tensor = tensors[i];
            IAXL_CHECK(tensor.is_contiguous() && tensor.device().type() == c10::DeviceType::CPU,
                       "kv_zip: tensor must be a contiguous CPU tensor");
            const size_t nbytes = tensor.numel() * tensor.element_size();
            char *buffer = static_cast<char *>(malloc(sizeof(int) * 2 + nbytes));
            IAXL_CHECK(buffer != nullptr, "kv_zip: raw cache buffer allocation failed");
            reinterpret_cast<int *>(buffer)[0] = 0;
            reinterpret_cast<int *>(buffer)[1] = 0;
            memcpy(buffer + sizeof(int) * 2, tensor.data_ptr(), nbytes);
            out_bufs[i] = buffer;
            out_sizes[i] = sizeof(int) * 2 + nbytes;
            orig_sizes[i] = nbytes;
        }
        return;
    }

    auto prep = [&](size_t i, char **data, size_t *nbytes) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        size_t nb = t.numel() * t.element_size();
        char *p = static_cast<char *>(t.data_ptr());
        if (lossy_trunc_enabled) {
            lossy_trunc(p, nb, t.element_size());
        }
        data_shuffle(p, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
        orig_sizes[i] = nb;
        *data = p;
        *nbytes = nb;
    };

    auto pack = [&](size_t i, const void *payload, int payload_len) {
        char *buf = static_cast<char *>(malloc(sizeof(int) * 2 + payload_len));
        IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
        reinterpret_cast<int *>(buf)[0] = payload_len;
        reinterpret_cast<int *>(buf)[1] = static_cast<int>(orig_sizes[i]);
        memcpy(buf + sizeof(int) * 2, payload, payload_len);
        out_bufs[i] = buf;
        out_sizes[i] = sizeof(int) * 2 + payload_len;
    };

    zip_pipeline(
        n,
        [&](ZipBackend backend, int slot, size_t i) {
            char *data;
            size_t nb;
            prep(i, &data, &nb);
            IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                       "kv_zip: tensor byte size exceeds zip integer length range");
            const int src_cap =
                backend == ZipBackend::QAT ? qat_zip_src_cap() : cpu_zip_src_cap();
            IAXL_CHECK(nb <= static_cast<size_t>(src_cap),
                       "kv_zip: tensor byte size exceeds zip source capacity");
            const int status = backend == ZipBackend::QAT
                                   ? qat_zip_compress(slot, data, static_cast<int>(nb))
                                   : cpu_zip_compress(slot, data, static_cast<int>(nb));
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend, size_t i, void *out, int out_len) { pack(i, out, out_len); });
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    const size_t n = tensors.size();
    IAXL_CHECK(data_ptrs.size() == n, "kv_zip: decompression inputs must have matching lengths");

    auto copy_raw = [&](size_t i) {
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        memcpy(t.data_ptr(), data_ptrs[i] + sizeof(int) * 2, nb);
    };

    std::vector<size_t> compressed_indices;
    compressed_indices.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] != 0)
            compressed_indices.push_back(i);
    }

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0)
            copy_raw(i);
    }

    if (compressed_indices.empty())
        return;

    auto finish = [&](size_t i, const void *out, int out_len) {
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        IAXL_CHECK(out_len >= 0 && static_cast<size_t>(out_len) == nb,
                   "kv_zip: decompressed size does not match tensor byte size");
        char *dst = static_cast<char *>(t.data_ptr());
        memcpy(dst, out, nb);
        data_shuffle(dst, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
    };

    const size_t compressed_count = compressed_indices.size();
    for (size_t item = 0; item < compressed_count; item++) {
        const size_t i = compressed_indices[item];
        const int encoded_len = reinterpret_cast<const int *>(data_ptrs[i])[0];
        IAXL_CHECK(encoded_len != 0 && encoded_len != INT_MIN,
                   "kv_zip: invalid compressed payload length");
    }

    zip_pipeline(
        compressed_count,
        [&](ZipBackend backend, int slot, size_t item) {
            const size_t i = compressed_indices[item];
            const int *hdr = reinterpret_cast<const int *>(data_ptrs[i]);
            const char *payload = data_ptrs[i] + sizeof(int) * 2;
            const int payload_len = hdr[0] < 0 ? -hdr[0] : hdr[0];
            const int status = backend == ZipBackend::QAT
                                   ? qat_zip_decompress(slot, const_cast<char *>(payload), payload_len)
                                   : cpu_zip_decompress(slot, const_cast<char *>(payload), payload_len);
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, size_t item, void *out, int out_len) {
            const size_t i = compressed_indices[item];
            finish(i, out, out_len);
        });
}

} // namespace kv_zip
