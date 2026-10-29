# Optional KV heap prefault

Enable `IAXL_KV_PREFAULT_GB` before sourcing `setvars.sh` and starting the process.
Each data rank prefaults at `KVStoreLocal` initialization, up to its DDR cache
budget. Has-only schedulers skip this step. Set per-rank budgets with enough
NUMA-local headroom for scratch buffers and other ranks.

| Environment variable | Default | Meaning |
|---|---:|---|
| `IAXL_KV_PREFAULT_GB` | `0` | Total heap budget per rank in GiB; zero disables prefaulting |
| `IAXL_KV_PREFAULT_THREADS` | `1` | Native concurrency, including the calling thread |
| `IAXL_KV_PREFAULT_CHUNK_MB` | `1` | Size of each malloc request in MiB; the last request can be smaller |

Thread and chunk settings use the native positive-integer environment parser;
nonpositive or invalid values fall back to their defaults.

## Threads only reduce initialization latency

The C++ helper allocates chunks, writes a nonzero byte to every OS page, then
frees all chunks after the writes finish. It does not change KV block allocation.
With one thread it runs directly on the caller. Additional native threads can
parallelize first-touch faults; they are optional and need not match the zip
worker count. Python makes one GIL-released call and creates no threads.

On Linux, new threads inherit the caller's CPU affinity. Placement still depends
on the process memory policy and available NUMA-local memory; affinity alone is
not a guarantee against fallback to another node.

## Why chunk the allocations?

The default **1 MiB** amortizes allocation overhead while providing independent
chunks for optional parallel warmup. It is a tuning starting point for tcmalloc,
not a correctness requirement, an OS page size, or a request for 2 MiB huge pages.
The actual page-touch stride comes from `sysconf(_SC_PAGESIZE)`.

A single allocation as large as the budget is allowed by setting the chunk size
accordingly. With release disabled, tcmalloc can retain and split the freed span
for later KV allocations: a large malloc is not inherently ineffective, and
`mmap` does not imply an unmap on `free`. Larger chunks reduce allocation and
bookkeeping work; smaller chunks provide more independent work for threads.
Concurrency is capped by the chunk count, so one chunk uses one thread even
when more are configured. Neither choice changes the total byte budget.

For comparison, mimalloc can directly unmap dedicated huge allocations on
`free`, even when normal purging is disabled (`MIMALLOC_PURGE_DELAY=-1`). Those
pages no longer benefit subsequent allocations. Tcmalloc with automatic release
disabled can instead retain and split a large free span. The huge-allocation
threshold is allocator/version dependent: a single large warmup allocation is
useful with tcmalloc but is not a portable assumption across allocators.

All chunks must remain allocated until touching finishes: freeing each chunk
immediately would allow the next allocation to reuse the same pages instead of
warming the requested budget.

## Retaining the pages

This feature targets the default tcmalloc preload. When enabled, `setvars.sh`
defaults `TCMALLOC_RELEASE_RATE=0` unless already set, to suppress automatic
release of free spans. Launchers that bypass `setvars.sh` must set this before
process startup. Other allocators are outside the scope of this feature.

This trades resident memory and startup work for fewer faults during cache fill.
Reuse after eviction depends on allocator behavior and the subsequent allocation
sizes. It is not a reservation or a guarantee of a fault-free steady state.
Allocation or thread-creation failure propagates after joining started workers
and releasing their allocations.
