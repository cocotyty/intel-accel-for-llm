#!/usr/bin/env bash
# Regenerate the slim PR tree from this (full) branch.
#
# The slim branch is the same code as this branch, minus what does not
# ship: tests, the hybrid design document, the long-form docstrings and
# this dev/ tooling. Files that upstream (main) already owns and the
# hybrid work only touches for a cross-reference (README.md,
# doc/design/kvshrink.md) are reset to their main content.
#
# Usage:
#   dev/gen_slim.sh [target-dir]
#
# The target must be a git checkout of the slim branch. Nothing is
# committed by this script; review and commit in the target yourself.
set -euo pipefail

FULL="$(cd "$(dirname "$0")/.." && pwd)"
SLIM="${1:-/data/tangyang/intel-accel-for-llm-slim}"

if [[ ! -d "$SLIM/.git" ]]; then
    echo "error: $SLIM is not a git checkout" >&2
    exit 1
fi

rsync -a --delete \
    --exclude '.git' \
    --exclude 'dev' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
    --exclude '*.so' --exclude '_lib' --exclude 'build' --exclude '*.egg-info' \
    --exclude 'log.*' \
    "$FULL/" "$SLIM/"

cd "$SLIM"
rm -rf tests/unit tests/gpu tests/README.md doc/design/kvshrink-hybrid.md

# Files the hybrid change must leave exactly as main has them.
git -C "$FULL" show main:README.md > README.md
git -C "$FULL" show main:doc/design/kvshrink.md > doc/design/kvshrink.md

python3 "$FULL/dev/trim_docstrings.py" \
    kvshrink/kvshrink_connector.py

echo "slim tree regenerated at $SLIM"
