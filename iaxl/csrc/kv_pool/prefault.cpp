// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"
#include "env.h"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace kv_pool {

size_t prefault_heap(size_t bytes) {
    if (bytes == 0)
        return 0;
    // 1 MiB amortizes malloc overhead while leaving chunks for parallel warmup.
    // This is a tuning choice, not the OS page size (nor a THP request).
    const size_t chunk = size_t(env_int("IAXL_KV_PREFAULT_CHUNK_MB", 1)) << 20;
    const size_t count = 1 + (bytes - 1) / chunk;
    const size_t threads = std::min(count, size_t(env_int("IAXL_KV_PREFAULT_THREADS", 1)));
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        throw std::runtime_error("prefault_heap: cannot determine page size");

    // Keep every allocation alive until all writes finish, including on failure.
    struct Free {
        void operator()(char *p) const { std::free(p); }
    };
    std::vector<std::unique_ptr<char, Free>> blocks(count);
    auto touch = [&](size_t worker) {
        for (size_t i = worker; i < count; i += threads) {
            const size_t size = std::min(chunk, bytes - i * chunk);
            blocks[i].reset(static_cast<char *>(std::malloc(size)));
            volatile char *p = blocks[i].get();
            if (!p)
                return;
            // Volatile stores cannot be optimized away before free. Touch the last
            // byte too: malloc need not return a page-aligned address.
            for (size_t offset = 0; offset < size; offset += size_t(page_size))
                p[offset] = 1;
            p[size - 1] = 1;
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    try {
        for (size_t i = 1; i < threads; ++i)
            workers.emplace_back(touch, i); // Inherits the caller's CPU affinity on Linux.
        touch(0); // Default: the calling thread only. More threads reduce startup latency.
    } catch (...) {
        for (auto &worker : workers)
            worker.join();
        throw;
    }
    for (auto &worker : workers)
        worker.join();
    for (const auto &block : blocks)
        if (!block)
            throw std::bad_alloc();
    // free() must retain these pages in the allocator for prefaulting to be useful.
    return bytes;
}

} // namespace kv_pool
