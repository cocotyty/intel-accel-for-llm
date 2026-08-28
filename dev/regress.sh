#!/bin/bash
# Full regression: unit tests + every GPU gate on both models.
# Usage: ./regress.sh
set -euo pipefail
MODELS=/home/tangyang/models
HYB=$MODELS/qwen3.5-4b
ATT=$MODELS/Qwen3-14B
IMG=vllm/vllm-openai:v0.23.0-cu129-ubuntu2404

docker rm -f iaxl-regress >/dev/null 2>&1 || true
docker run --rm --name iaxl-regress \
  --runtime nvidia --gpus all --ipc=host --shm-size=16g --network host \
  -v "$HOME/iaxl-hybrid:/w" -w /w \
  -v "$MODELS:$MODELS:ro" \
  -e http_proxy=http://proxy-dmz.intel.com:912 \
  -e https_proxy=http://proxy-dmz.intel.com:912 \
  -e no_proxy=127.0.0.1,localhost,.intel.com \
  -e PYTHONHASHSEED=0 \
  -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_DSA_GD_ENABLE=0 \
  -e IAXL_PREALLOC_LIMIT=2048 \
  -e GATE_MODEL_HYBRID="$HYB" -e GATE_MODEL_ATTENTION="$ATT" \
  -e TP_SIZE=2 \
  --entrypoint bash "$IMG" -c '
set -e
pip install -q xxhash 2>&1 | tail -1
export PYTHONPATH=/w

echo "################ UNIT TESTS ################"
python3 -m pytest tests/unit -q 2>&1 | tail -3

echo
echo "################ GPU GATES ################"
bash tests/gpu/run_gates.sh
'
