#!/bin/bash -e
#
# NOTE: When running multiple benchmark rounds, KV Cache offloaded to DDR
# accumulates across runs and can cause CPU memory OOM. Evict cache between
# runs:
#
#   curl -X POST http://localhost:18700/v1/cache/evict \
#     -d '{"count": 999999}'
#
# Or persist valuable caches to disk first, then evict:
#
#   curl -X POST http://localhost:18700/v1/cache/persist -d '{"count": 999999}'
#   curl -X POST http://localhost:18700/v1/cache/evict   -d '{"count": 999999}'
#

export MODEL=${MODEL:-Qwen/Qwen3-32B}
export HOST=${HOST:-localhost}
export PORT=${PORT:-8000}
export ENDPOINT=http://$HOST:$PORT

input_len=8000
output_len=128
num_prompts=10
concurrency=4
hit_rate=80
random_prefix_len=$((input_len * hit_rate / 100))
random_input_len=$((input_len - random_prefix_len))
seed=$(date +%s)

ARGS=(
    --backend vllm
    --model "$MODEL"
    --tokenizer "$MODEL"
    --dataset-name "random"
    --host $HOST
    --port $PORT
    --random-input-len $random_input_len
    --random-prefix-len $random_prefix_len
    --random-output-len $output_len
    --ignore-eos
    --percentile-metrics "ttft,tpot"
    --metric-percentiles "50,95"
    --seed $seed
    --trust-remote-code
    --request-rate inf
)

echo "=== Hit_rate=${hit_rate} Input=${input_len} (${random_prefix_len}+${random_input_len}), Output=${output_len} Concurrency=${concurrency}, num_prompts=${num_prompts} === "
vllm bench serve "${ARGS[@]}" \
    --num-warmups 5 \
    --num-prompts $num_prompts \
    --max-concurrency $concurrency
