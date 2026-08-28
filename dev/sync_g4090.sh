#!/usr/bin/env bash
# Sync this repo to the g4090 regression box and (optionally) start the
# full regression there.
#
#   dev/sync_g4090.sh           # sync only
#   dev/sync_g4090.sh --regress # sync, then run dev/regress.sh on g4090
#
# CRITICAL: '*.so' must stay excluded. The local kvshrink-test container
# is torch cu130 while the g4090 vLLM image is cu129; syncing the local
# torch_ext build over the g4090 one loads fine and then SEGFAULTS in
# cudaMemcpy3DBatchAsync. Rebuilds on g4090 must happen inside the cu129
# image (see dev/regress.sh header).
set -euo pipefail

FULL="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
    --exclude '*.so' --exclude '_lib' --exclude '_build' --exclude 'build' \
    --exclude '*.egg-info' --exclude 'log.*' --exclude '.claude' \
    "$FULL/" g4090:~/iaxl-hybrid/

if [[ "${1:-}" == "--regress" ]]; then
    ssh g4090 'nohup bash ~/iaxl-hybrid/dev/regress.sh \
        > ~/iaxl-hybrid/regress.log 2>&1 & echo "regress started, pid $!"'
fi
