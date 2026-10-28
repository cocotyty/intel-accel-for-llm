# Shared KV buffer regression

Hybrid KV cache groups can expose different layer tensors over the same GPU
buffer. There are two registration stages:

1. The client registers physical GPU regions with NIXL **before** exchanging
   metadata. Each shared region must be registered once (`kvstore_remote.py`).
2. The daemon receives every logical layer and prepares remote transfer lists.
   It also deduplicates by base address (`rpc.py`).

Daemon-side deduplication alone does not protect metadata import: the client
can still advertise a region multiple times and trigger
`loadRemoteMD ... NIXL_ERR_NOT_ALLOWED`. Conversely, deduplicating the logical
layer dictionary would lose layers needed by PUT/GET and their completions.

## Unit regression

Run from the repository root inside the dev container:

```bash
python3 -m unittest discover -s tests/unit -p test_remote_registration.py -v
```

The test uses real CPU tensor views and replaces the transport. It checks that
shared regions are registered once, all registrations precede metadata exchange,
all logical layers retain their identities, distinct attention buffers are not
dropped, and the scheduler's names-only path needs no GPU registration. No GPU
or live daemon is required for this part.

## GPU/RDMA regression

This test requires two RDMA-connected dev containers, a built IAFL extension on
both sides, and one client GPU per TP rank. Use the same source revision and
compatible NIXL versions on both sides. Configure CPU affinity and compression
as for the normal remote pool; the test exercises whichever daemon compression
backend is enabled.

On the daemon, start a fresh process with an empty test cache directory:

```bash
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=<daemon-rdma-ip>
export IAXL_RDMA_DAEMON_PORT=5655
export TP_SIZE=4 IAXL_RDMA_TP_SIZE=4
export IAXL_CACHE_DIR=<test-cache-directory>
python3 -X faulthandler -m iaxl.remote_pool.daemon
```

On the client, using matching daemon endpoint and TP size:

```bash
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=<daemon-rdma-ip>
export IAXL_RDMA_CLIENT_NIC_IPS=<client-rdma-ip>
export IAXL_RDMA_DAEMON_PORT=5655
export IAXL_RDMA_TP_SIZE=4
python3 tests/remote/shared_kv_buffers.py --tp-size 4 --layout shared
```

Each rank exposes five logical layers over three GPU allocations. The test
checks real registration/handshake, RDMA PUT/GET, distinct attention and Mamba
values under the same block hashes, loading into different block indices,
untouched destination blocks, and scheduler namespace/prefix lookups. A run
must exit zero and print `RESULT: ALL PASS`; inspect the daemon log for metadata
import errors as well.

Repeat with a **fresh daemon and empty test cache** for each run. Use
`--layout disjoint` for the attention-only control (three layers, three
allocations). This avoids conflating shared-buffer registration with reconnecting
a new process under an existing NIXL agent name.

For a negative control, the shared-view unit test must fail with the client
registration loop from before #44. The real shared-layout test can also expose
the metadata import error on that revision; initialization failures can depend
on the NIXL/backend environment. A single successful startup is not sufficient
evidence that repeated registration is safe.

This small-buffer test isolates the transport. Model-level validation should
add a hybrid model run: compute a prompt, wait for external saves, reset only
vLLM's local prefix cache, then verify external hits and restored outputs. Keep
the hybrid cache mode aligned and HMA enabled.
