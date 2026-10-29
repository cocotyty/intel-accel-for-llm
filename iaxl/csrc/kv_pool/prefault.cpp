// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"
#include "parallel.h"

#include <cstring>

// Optional symbols from the preloaded tcmalloc; no allocator link-time dependency.
extern "C" int MallocExtension_GetNumericProperty(const char *, size_t *) __attribute__((weak));
extern "C" void MallocExtension_ReleaseToSystem(size_t) __attribute__((weak));

namespace kv_pool {

bool prefault_heap(size_t bytes) {
    size_t unused = 0;
    if (!bytes || !MallocExtension_GetNumericProperty || !MallocExtension_ReleaseToSystem ||
        !MallocExtension_GetNumericProperty("tcmalloc.pageheap_free_bytes", &unused))
        return false;

    constexpr size_t chunk = 2 << 20; // Amortize malloc overhead; not a huge-page requirement.
    std::vector<void *> blocks(1 + (bytes - 1) / chunk);
    cpu_set_t caller_cpus;
    IAXL_CHECK(sched_getaffinity(0, sizeof(caller_cpus), &caller_cpus) == 0, "get prefault affinity");
    parallel::parallel_for(blocks.size(), [&](size_t i) {
        const size_t size = std::min(chunk, bytes - i * chunk);
        blocks[i] = malloc(size);
        IAXL_CHECK(blocks[i], "KV heap prefault allocation failed");
        memset(blocks[i], 1, size);
    });
    // parallel_for may bind its calling OpenMP thread to the IAXL worker CPUs.
    IAXL_CHECK(sched_setaffinity(0, sizeof(caller_cpus), &caller_cpus) == 0, "restore prefault affinity");
    // Keep every chunk until all writes finish, or malloc can reuse the same pages.
    for (void *p : blocks)
        free(p);
    fprintf(stderr, "[kv_pool] Prefaulted %.3f GiB of heap\n", double(bytes) / (size_t(1) << 30));
    return true;
}

void trim_heap(size_t keep_bytes, size_t slack_bytes) {
    size_t free_bytes = 0;
    if (MallocExtension_GetNumericProperty && MallocExtension_ReleaseToSystem &&
        MallocExtension_GetNumericProperty("tcmalloc.pageheap_free_bytes", &free_bytes) &&
        free_bytes > keep_bytes && free_bytes - keep_bytes > slack_bytes)
        MallocExtension_ReleaseToSystem(free_bytes - keep_bytes);
}

} // namespace kv_pool
