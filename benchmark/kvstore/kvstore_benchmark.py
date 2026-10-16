#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import collections
import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from iaxl.utils.logger import setup_root_logger

setup_root_logger(show_pid_tid=False)


_env_parser = argparse.ArgumentParser(add_help=False)
_env_parser.add_argument(
    "--quant", type=str, default=os.environ.get("IAXL_KV_LOSSY_TRUNC", "0")
)
_env_parser.add_argument(
    "--data-shuffle",
    type=int,
    choices=(0, 1),
    default=int(os.environ.get("IAXL_KV_DATA_SHUFFLE", "0")),
)
_env_args, _ = _env_parser.parse_known_args()
os.environ["IAXL_KV_LOSSY_TRUNC"] = str(_env_args.quant)
os.environ["IAXL_KV_DATA_SHUFFLE"] = str(_env_args.data_shuffle)

from iaxl.kvstore import KVStore, get_accelerator_device
from iaxl.envs import envs as iaxl_envs

DEFAULT_KV_CACHE_SHAPE = (2, 1024, 16, 4, 128)


if iaxl_envs.IAXL_RDMA_ENABLE:
    DEFAULT_METRICS_URL = (
        f"http://{iaxl_envs.IAXL_RDMA_DAEMON_NIC_IPS.split(',')[0].strip()}:18800/v1/cache/metrics")
else:
    DEFAULT_METRICS_URL = "http://127.0.0.1:18800/v1/cache/metrics"
DEFAULT_KV_DATA_DIR = "/_data/kvstore_benchmark"
DEFAULT_MODEL_SEQ_LEN = 16384
SEED = 42
BLOCK_DIM = 1
FP8_MAX = 448.0
INT4_QMAX = 7.0
INT4_GROUP_SIZE = 128

# int4 is stored as two 4-bit values packed into one uint8.
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "int4": torch.uint8,
}


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quant",
        type=str,
        default=os.environ.get("IAXL_KV_LOSSY_TRUNC", "0"),
        help="Quantization: 'auto', 0 (off), or N LSB bits to truncate "
        "(env IAXL_KV_LOSSY_TRUNC).",
    )
    parser.add_argument(
        "--data-shuffle",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("IAXL_KV_DATA_SHUFFLE", "0")),
        help="Enable data shuffle before compression (env IAXL_KV_DATA_SHUFFLE).",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KVStore PUT/GET with generated or file-backed data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kv-cache-shape",
        "--shape",
        type=int,
        nargs=5,
        default=DEFAULT_KV_CACHE_SHAPE,
        metavar=("KV", "CACHE_BLOCKS", "TOKENS", "KV_HEADS", "HEAD_DIM"),
        help="Shape of one KV cache layer. KV_HEADS is 8 for TP=1, 1 for TP=8.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="bf16",
        help="KV cache tensor dtype.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of KV cache layers to allocate, mimicking a real model. "
        "PUT/GET and their waits are issued layer by layer like vLLM. "
        "Each layer holds a full --kv-cache-shape tensor, so GPU memory "
        "scales with this value.",
    )
    parser.add_argument(
        "--data-source",
        choices=("mock", "model", "file"),
        default=None,
        help="Where the KV cache content comes from: a real transformer prefill, "
        "synthetic mock data, or raw file bytes. Defaults to 'model', or to "
        "'file' when --data-file is set.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional test-data file whose bytes fill the KV cache (cycled "
        "to fit). If omitted, deterministic mock data is generated.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL"),
        help="Hugging Face model ID or local path used by --data-source model "
        "(env MODEL).",
    )
    parser.add_argument(
        "--model-seq-len",
        type=int,
        default=DEFAULT_MODEL_SEQ_LEN,
        help="Prefill length of one forward pass when generating model data. "
        "Must be a multiple of the block TOKENS dimension.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Text file tokenized (and cycled) into the prefill prompt. "
        "If omitted, deterministic random token IDs are used.",
    )
    parser.add_argument(
        "--kv-data-dir",
        default=DEFAULT_KV_DATA_DIR,
        help="Directory holding generated model KV cache files, named "
        "<model>_<dtype>.pt and reused whenever they match the requested shape.",
    )
    parser.add_argument(
        "--metrics-url",
        default=DEFAULT_METRICS_URL,
        help="KVStore REST endpoint for compression throughput metrics.",
    )
    parser.add_argument(
        "--mix",
        default=None,
        metavar="C:D",
        help="Mixed mode: after the preload PUT, issue PUTs (compress, new keys) and GETs "
        "(decompress, preloaded keys) interleaved at a C:D byte ratio, e.g. 12:40, and "
        "report the compress/decompress throughput reached together.",
    )
    parser.add_argument(
        "--mix-ops",
        type=int,
        default=60,
        help="Mixed mode: total number of PUT+GET ops "
        "(one op = one layer x --mix-op-blocks blocks).",
    )
    parser.add_argument(
        "--mix-op-blocks",
        type=int,
        default=0,
        help="Mixed mode: blocks per op (0 = all tested blocks of a layer).",
    )
    parser.add_argument(
        "--mix-inflight",
        type=int,
        default=4,
        help="Mixed mode: max ops in flight; the oldest is waited for before issuing more.",
    )
    parser.add_argument(
        "--mix-put-keep",
        type=int,
        default=0,
        help="Mixed mode: cycle PUT keys over this many rounds (overwriting older ones) so long "
        "runs keep memory bounded; 0 = every round uses new keys.",
    )
    add_env_arguments(parser)
    args = parser.parse_args(argv)
    if args.mix is not None:
        try:
            c, d = (float(x) for x in args.mix.split(":"))
        except ValueError:
            parser.error("--mix must look like C:D, e.g. 12:40")
        if c < 0 or d < 0 or c + d == 0:
            parser.error("--mix ratios must be >= 0 and not both 0")
        args.mix = (c, d)
        if (args.mix_ops <= 0 or args.mix_inflight <= 0 or args.mix_op_blocks < 0
                or args.mix_put_keep < 0):
            parser.error("--mix-ops / --mix-inflight must be > 0, --mix-op-blocks >= 0")
    shape = tuple(args.kv_cache_shape)
    if shape[0] != 2:
        parser.error("the first --kv-cache-shape dimension must be 2 (key/value)")
    if any(dim <= 0 for dim in shape):
        parser.error("all --kv-cache-shape dimensions must be greater than 0")
    if args.num_layers <= 0:
        parser.error("--num-layers must be greater than 0")
    if args.dtype == "int4" and shape[-1] % 2:
        parser.error("--dtype int4 requires an even HEAD_DIM")
    if args.data_source is None:
        use_file = args.data_file is not None and args.data_file != "random"
        args.data_source = "file" if use_file else "model"
    if args.data_source == "file" and not args.data_file:
        parser.error("--data-source file requires --data-file")
    if args.data_source == "model":
        if not args.model:
            parser.error("--data-source model requires --model or the MODEL env var")
        if args.model_seq_len <= 0 or args.model_seq_len % shape[2]:
            parser.error("--model-seq-len must be a positive multiple of TOKENS")
    args.kv_cache_shape = shape
    return args


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def make_block_hashes(num_blocks: int) -> List[str]:
    return [str(i) for i in range(num_blocks)]


def storage_shape(shape: Tuple[int, ...], dtype_name: str) -> Tuple[int, ...]:
    if dtype_name == "int4":
        return shape[:-1] + (shape[-1] // 2,)
    return shape


def quantize_kv(values: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Quantize a float KV tensor [2, ...] the way vLLM does."""
    if dtype_name == "bf16":
        return values.to(torch.bfloat16).contiguous()

    values = values.float()
    if dtype_name == "fp8_e4m3":
        # vLLM stores one per-tensor k_scale/v_scale per layer: scale = amax / 448.
        reduce_dims = tuple(range(1, values.ndim))
        scale = values.abs().amax(dim=reduce_dims, keepdim=True) / FP8_MAX
        scaled = (values / scale.clamp_min(1e-12)).clamp_(-FP8_MAX, FP8_MAX)
        return scaled.to(torch.float8_e4m3fn).contiguous()

    # vLLM int4: symmetric group-wise uint4b8, scale = amax / 7, low nibble first.
    head_dim = values.shape[-1]
    group = INT4_GROUP_SIZE if head_dim % INT4_GROUP_SIZE == 0 else head_dim
    grouped = values.unflatten(-1, (-1, group))
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / INT4_QMAX).clamp_min(1e-6)
    codes = (grouped / scale).round_().clamp_(-8, 7).add_(8)
    codes = codes.to(torch.uint8).flatten(-2)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def generate_mock_kv_cache(shape: Tuple[int, ...], dtype_name: str) -> torch.Tensor:
    kv_count, cache_blocks, block_tokens, kv_heads, head_dim = shape
    latent_dim = min(32, max(8, head_dim // 8))
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    block_context = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    token_innovations = torch.randn(
        (cache_blocks, block_tokens, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    block_drift = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    token_positions = torch.linspace(-1.0, 1.0, block_tokens, device="cuda").view(
        1, block_tokens, 1, 1
    )
    hidden_states = (
        0.65 * block_context
        + 0.70 * token_innovations
        + 0.15 * token_positions * block_drift
    )

    projections = torch.randn(
        (kv_count, kv_heads, latent_dim, head_dim),
        device="cuda",
        generator=generator,
    ) / math.sqrt(latent_dim)
    kv_cache = torch.einsum("bthr,khrd->kbthd", hidden_states, projections)

    channel_scales = (
        torch.randn(
            (kv_count, 1, 1, kv_heads, head_dim),
            device="cuda",
            generator=generator,
        )
        .mul_(0.25)
        .exp_()
    )
    kv_cache.mul_(channel_scales)
    return quantize_kv(kv_cache, dtype_name)


def _layer_keys_values(past_key_values, index: int):
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return layers[index].keys, layers[index].values
    return past_key_values.key_cache[index], past_key_values.value_cache[index]


def _to_blocks(
    tensor: torch.Tensor, block_tokens: int, kv_heads: int, head_dim: int
) -> torch.Tensor:
    """[1, heads, seq, head_dim] -> [blocks, block_tokens, kv_heads, head_dim]."""
    tokens = tensor[0].transpose(0, 1)
    if tokens.shape[1] < kv_heads:
        repeats = -(-kv_heads // tokens.shape[1])
        tokens = tokens.repeat(1, repeats, 1)
    return tokens[:, :kv_heads, :head_dim].reshape(-1, block_tokens, kv_heads, head_dim)


def _make_token_ids(
    tokenizer, vocab_size: int, total_tokens: int, prompt_file
) -> torch.Tensor:
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8", errors="replace") as fp:
            text = fp.read()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"prompt file {prompt_file!r} produced no tokens")
        ids = (ids * (-(-total_tokens // len(ids))))[:total_tokens]
        print(f"[model] tokenized {prompt_file}, cycled to {total_tokens} tokens")
        return torch.tensor(ids, dtype=torch.long)

    generator = torch.Generator().manual_seed(SEED)
    print(f"[model] using {total_tokens} deterministic random token IDs")
    return torch.randint(
        0, vocab_size, (total_tokens,), generator=generator, dtype=torch.long
    )


def generate_model_kv_cache(
    shape: Tuple[int, ...], dtype_name: str, args: argparse.Namespace
) -> List[torch.Tensor]:
    """Prefill a real transformer and return one CPU KV tensor per model layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, cache_blocks, block_tokens, kv_heads, head_dim = shape
    total_tokens = cache_blocks * block_tokens

    print(f"[model] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    token_ids = _make_token_ids(
        tokenizer, model.config.vocab_size, total_tokens, args.prompt_file
    )
    num_model_layers = min(model.config.num_hidden_layers, args.num_layers)
    parts: List[List[torch.Tensor]] = [[] for _ in range(num_model_layers)]
    backbone = getattr(model, "model", model)  # skip the LM head, only KV is needed

    for start in range(0, total_tokens, args.model_seq_len):
        chunk = token_ids[start : start + args.model_seq_len].unsqueeze(0).to("cuda")
        with torch.no_grad():
            past_key_values = backbone(input_ids=chunk, use_cache=True).past_key_values
        for layer in range(num_model_layers):
            keys, values = _layer_keys_values(past_key_values, layer)
            if keys.shape[-1] < head_dim:
                raise ValueError(
                    f"model head_dim {keys.shape[-1]} is smaller than the "
                    f"requested HEAD_DIM {head_dim}"
                )
            kv = torch.stack(
                (
                    _to_blocks(keys, block_tokens, kv_heads, head_dim),
                    _to_blocks(values, block_tokens, kv_heads, head_dim),
                ),
                dim=0,
            )
            parts[layer].append(kv.cpu())
        del past_key_values
        torch.cuda.empty_cache()
        done = min(start + args.model_seq_len, total_tokens)
        print(f"[model] prefilled {done}/{total_tokens} tokens")

    del model
    torch.cuda.empty_cache()

    # Quantize whole layers so fp8 gets one k_scale/v_scale per layer, like vLLM.
    layers = []
    for part in parts:
        merged = torch.cat(part, dim=1)[:, :cache_blocks].to("cuda")
        part.clear()
        layers.append(quantize_kv(merged, dtype_name).cpu())
        del merged
    torch.cuda.empty_cache()
    return layers


def model_cache_path(args: argparse.Namespace, dtype_name: str) -> str:
    model_tag = os.path.basename(args.model.rstrip("/"))
    return os.path.join(args.kv_data_dir, f"{model_tag}_{dtype_name}.pt")


def load_or_generate_model_kv_cache(
    args: argparse.Namespace, shape: Tuple[int, ...], dtype_name: str
) -> List[torch.Tensor]:
    path = model_cache_path(args, dtype_name)
    expected = storage_shape(shape, dtype_name)
    if os.path.exists(path):
        layers = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(layers[0].shape) == expected:
            print(f"[model] reusing {len(layers)} cached KV layers from {path}")
            return layers
        print(
            f"[model] {path} holds shape {tuple(layers[0].shape)}, "
            f"regenerating for {expected}"
        )
        del layers

    layers = generate_model_kv_cache(shape, dtype_name, args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(layers, path)
    print(f"[model] saved {len(layers)} KV layers to {path}")
    return layers


def load_kv_cache_from_file(
    path: str, shape: Tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    with open(path, "rb") as fp:
        data = fp.read()
    if not data:
        raise ValueError(f"test-data file {path!r} is empty")

    total_bytes = math.prod(shape) * dtype.itemsize
    repeats = -(-total_bytes // len(data))
    raw = bytearray(data) * repeats
    del raw[total_bytes:]

    flat = torch.frombuffer(raw, dtype=torch.uint8).view(dtype)
    print(f"[data] loaded {path} ({len(data)} bytes, cycled to fill cache)")
    return flat.reshape(shape).to("cuda")


def print_result(name: str, elapsed: float, num_blocks: int, num_bytes: int) -> None:
    print(
        f"{name:>4}: {elapsed:.3f} s, "
        f"{num_blocks / elapsed:,.2f} blocks/s, "
        f"{num_bytes / elapsed / (1024**3):.3f} GiB/s"
    )


def _rest_get_json(url: str, query: str = "") -> dict:
    if query:
        url = f"{url}?{query}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cache_metrics_rest(url: str, query: str = "") -> dict:
    return _rest_get_json(url, query)


def cache_status_rest(metrics_url: str) -> dict:
    status_url = metrics_url.rsplit("/", 1)[0] + "/status"
    return _rest_get_json(status_url)


def build_kv_caches(
    args: argparse.Namespace, layer_names: List[str]
) -> Dict[str, torch.Tensor]:
    shape = args.kv_cache_shape
    if args.data_source == "model":
        print(f"\nBuilding KV cache from {args.model}...")
        layers = load_or_generate_model_kv_cache(args, shape, args.dtype)
        return {
            name: layers[i % len(layers)].to("cuda")
            for i, name in enumerate(layer_names)
        }
    if args.data_source == "file":
        print(f"\nLoading KV cache from {args.data_file}...")
        base = load_kv_cache_from_file(
            args.data_file, storage_shape(shape, args.dtype), DTYPE_MAP[args.dtype]
        )
    else:
        print("\nGenerating mock KV cache data...")
        base = generate_mock_kv_cache(shape, args.dtype)
    return {name: base.clone() for name in layer_names}


class Benchmark:
    """One rank's KV caches plus KVStore; the steps mirror vLLM's layer-wise PUT/GET."""

    def __init__(self, args: argparse.Namespace, rank: int = 0, tp_size: int = 1):
        self.args = args
        self.rank = rank
        self.tp_size = tp_size
        self.shape: Tuple[int, ...] = args.kv_cache_shape
        self.store_shape = storage_shape(self.shape, args.dtype)
        self.dtype = DTYPE_MAP[args.dtype]
        cache_blocks = self.shape[BLOCK_DIM]
        self.block_indices = list(range(0, cache_blocks, 2))
        self.num_blocks = len(self.block_indices)
        bytes_per_block = math.prod(self.store_shape) // cache_blocks * self.dtype.itemsize
        self.total_blocks = self.num_blocks * args.num_layers
        self.total_bytes = bytes_per_block * self.total_blocks
        self.layer_names = [str(i) for i in range(args.num_layers)]
        self.block_hashes = make_block_hashes(self.num_blocks)

    def print_config(self) -> None:
        print("=" * 80)
        print("KVStore Benchmark")
        print("=" * 80)
        print(f"KV cache shape: {list(self.shape)} (one layer)")
        print(f"Layers:         {self.args.num_layers}")
        print(
            f"Blocks tested:  {self.num_blocks} (even indices, strided) "
            f"x {self.args.num_layers} layers"
        )
        print(f"Dtype:          {self.args.dtype}")
        if self.store_shape != self.shape:
            print(f"Stored shape:   {list(self.store_shape)} ({self.dtype}, two int4 per byte)")
        print(f"Data source:    {self.args.data_source}")
        print(f"Transfer size:  {format_bytes(self.total_bytes)}")

    def setup(self) -> None:
        self.kv_caches = build_kv_caches(self.args, self.layer_names)
        self.index = torch.tensor(
            [
                self.block_indices[0],
                self.block_indices[self.num_blocks // 2],
                self.block_indices[-1],
            ],
            device="cuda",
        )
        self.expected = {
            name: tensor.index_select(BLOCK_DIM, self.index).clone()
            for name, tensor in self.kv_caches.items()
        }
        self.kvstore = KVStore(
            model_name="kvstore_benchmark",
            kv_caches=self.kv_caches,
            block_dim=BLOCK_DIM,
            rank=self.rank,
            tp_size=self.tp_size,
        )

    def metrics(self, **params) -> dict:
        return cache_metrics_rest(self.args.metrics_url, urllib.parse.urlencode(params))

    def status(self) -> dict:
        return cache_status_rest(self.args.metrics_url)

    def warmup(self) -> None:
        self.metrics(enable=1, reset=1)
        print("\nWarming up...")
        warmup_hashes = [f"warmup_{h}" for h in self.block_hashes]
        put = self.kvstore.put(self.block_indices, warmup_hashes, description="warmup PUT")
        if not self.kvstore.put_wait(put):
            raise RuntimeError("warm-up PUT did not complete")
        get = self.kvstore.get(self.block_indices, warmup_hashes, description="warmup GET")
        if not self.kvstore.get_wait(get):
            raise RuntimeError("warm-up GET did not complete")
        torch.cuda.synchronize()
        self.metrics(reset=1)

    def put(self) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.nvtx.range("benchmark PUT"):
            put_tasks = {}
            for name in self.layer_names:
                put_tasks.update(
                    self.kvstore.put(
                        self.block_indices,
                        self.block_hashes,
                        layer_names=[name],
                        description="benchmark PUT",
                    )
                )
            if not self.kvstore.put_wait(put_tasks):
                raise RuntimeError("PUT did not complete")
        return time.perf_counter() - start

    def get(self) -> float:
        for tensor in self.kv_caches.values():
            tensor.zero_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.nvtx.range("benchmark GET"):
            get_tasks = self.kvstore.get(
                self.block_indices,
                self.block_hashes,
                layer_names=None,
                description="benchmark GET",
            )
            for name in self.layer_names:
                if not self.kvstore.get_wait(get_tasks, layer_names=[name]):
                    raise RuntimeError("GET did not complete")
        return time.perf_counter() - start

    def verify(self) -> bool:
        return all(
            torch.equal(tensor.index_select(BLOCK_DIM, self.index), self.expected[name])
            for name, tensor in self.kv_caches.items()
        )

    # ---- mixed compress/decompress mode (--mix C:D)
    # PUT ops read the even (tested) blocks and store them under new keys, GET ops fetch the
    # keys stored by the preload PUT into the odd blocks, so the two never touch the same GPU
    # memory. Ops are issued in one stream that keeps the C:D byte ratio at every point, with
    # at most --mix-inflight ops outstanding.
    def mix_plan(self) -> List[Tuple[str, str, List[int], List[str], int]]:
        c, d = self.args.mix
        op_blocks = self.args.mix_op_blocks or self.num_blocks
        slices = [(s, min(s + op_blocks, self.num_blocks))
                  for s in range(0, self.num_blocks, op_blocks)]
        n_layers = len(self.layer_names)
        plan, n_put, n_get = [], 0, 0
        put_blocks = get_blocks = 0
        keep = self.args.mix_put_keep
        put_keys, get_keys, even_idx, odd_idx = {}, {}, {}, {}  # shared lists, built once
        for _ in range(self.args.mix_ops):
            # Pick the side that is furthest behind its share of the bytes.
            if d == 0 or (c > 0 and put_blocks * d <= get_blocks * c):
                s0, s1 = slices[n_put % len(slices)]
                layer = self.layer_names[(n_put // len(slices)) % n_layers]
                # Like vLLM, every layer of a block is stored under the same block hash; a new
                # round of hashes starts once all layers have been put.
                put_round = n_put // (len(slices) * n_layers)
                if keep:
                    put_round %= keep
                if (put_round, s0) not in put_keys:
                    put_keys[(put_round, s0)] = [f"mix{put_round}_{i}" for i in range(s0, s1)]
                if s0 not in even_idx:
                    even_idx[s0] = self.block_indices[s0:s1]
                plan.append(("put", layer, even_idx[s0], put_keys[(put_round, s0)], s0))
                n_put += 1
                put_blocks += s1 - s0
            else:
                s0, s1 = slices[n_get % len(slices)]
                layer = self.layer_names[(n_get // len(slices)) % n_layers]
                if s0 not in odd_idx:
                    odd_idx[s0] = [i + 1 for i in self.block_indices[s0:s1]]
                    get_keys[s0] = self.block_hashes[s0:s1]
                plan.append(("get", layer, odd_idx[s0], get_keys[s0], s0))
                n_get += 1
                get_blocks += s1 - s0
        return plan

    def mix(self) -> dict:
        plan = self.mix_plan()
        bytes_per_block = self.total_bytes // self.total_blocks
        for tensor in self.kv_caches.values():
            tensor[:, 1::2].zero_()  # GET targets; even blocks keep the PUT source data
        self.mix_fetched = set()  # (layer, first, end) slices of tested blocks fetched by GET
        torch.cuda.synchronize()
        self.metrics(reset=1)
        inflight = collections.deque()

        def wait(entry):
            kind, tasks = entry
            ok = self.kvstore.put_wait(tasks) if kind == "put" else self.kvstore.get_wait(tasks)
            if not ok:
                raise RuntimeError(f"mixed {kind} did not complete")

        put_blocks = get_blocks = n_put = n_get = 0
        issue_s = wait_s = 0.0
        start = time.perf_counter()
        last_report = start
        with torch.cuda.nvtx.range("benchmark MIX"):
            for kind, layer, idx, hashes, s0 in plan:
                t0 = time.perf_counter()
                if self.rank == 0 and t0 - last_report >= 10:
                    last_report = t0
                    el = t0 - start
                    zip_gbps = put_blocks * bytes_per_block / el / 1e9
                    unzip_gbps = get_blocks * bytes_per_block / el / 1e9
                    print(f"MIXPROG rank0 t={el:.0f}s ops={n_put + n_get}/{len(plan)} "
                          f"zip {zip_gbps:.2f} GB/s unzip {unzip_gbps:.2f} GB/s "
                          f"(x{self.tp_size} ranks ~ {self.tp_size * zip_gbps:.1f} / "
                          f"{self.tp_size * unzip_gbps:.1f})", flush=True)
                while len(inflight) >= self.args.mix_inflight:
                    wait(inflight.popleft())
                t1 = time.perf_counter()
                wait_s += t1 - t0
                if kind == "put":
                    tasks = self.kvstore.put(
                        idx, hashes, layer_names=[layer], description="mix PUT")
                    put_blocks += len(idx)
                    n_put += 1
                else:
                    tasks = self.kvstore.get(
                        idx, hashes, layer_names=[layer], description="mix GET")
                    get_blocks += len(idx)
                    n_get += 1
                    self.mix_fetched.add((layer, s0, s0 + len(idx)))
                issue_s += time.perf_counter() - t1
                inflight.append((kind, tasks))
            t0 = time.perf_counter()
            while inflight:
                wait(inflight.popleft())
            torch.cuda.synchronize()
            wait_s += time.perf_counter() - t0
        elapsed = time.perf_counter() - start
        m = self.metrics()
        return {
            "elapsed": elapsed,
            "issue_s": issue_s,
            "wait_s": wait_s,
            "put_ops": n_put,
            "get_ops": n_get,
            "put_bytes": put_blocks * bytes_per_block,
            "get_bytes": get_blocks * bytes_per_block,
            "compress_bytes": m["compress_bytes"],
            "compress_ns": m["compress_ns"],
            "decompress_bytes": m["decompress_bytes"],
            "decompress_ns": m["decompress_ns"],
        }

    def mix_verify(self) -> bool:
        # Each sampled tested block covered by a GET op must now hold its data in block + 1.
        positions = [0, self.num_blocks // 2, self.num_blocks - 1]
        checked = 0
        for layer, s0, s1 in self.mix_fetched:
            tensor = self.kv_caches[layer]
            for k, pos in enumerate(positions):
                if s0 <= pos < s1:
                    got = tensor.select(BLOCK_DIM, self.block_indices[pos] + 1)
                    if not torch.equal(got, self.expected[layer].select(BLOCK_DIM, k)):
                        return False
                    checked += 1
        return checked > 0 or not self.mix_fetched


def print_report(
    put_time: float,
    get_time: float,
    verified: bool,
    status: dict,
    metrics: dict,
    total_blocks: int,
    total_bytes: int,
    title: str = "Results",
) -> None:
    print(f"\n{title}")
    print("-" * 80)
    print_result("PUT", put_time, total_blocks, total_bytes)
    print_result("GET", get_time, total_blocks, total_bytes)
    print(f"Verification: {'passed' if verified else 'FAILED'}")
    ratio = status["compression_ratio (unzip/zip, higher=better)"]
    saved_pct = (1 - 1 / ratio) * 100 if ratio else 0.0
    print(f"Compression ratio: {ratio:.3f}x ({saved_pct:.1f}%)")
    print(
        f"Native cache size: {format_bytes(status['current_bytes'])} "
        f"({status['cache_entries']} chunks in {status['group_count']} groups)"
    )
    print(
        f"Cache hits/misses/puts: {status['hits']}/{status['misses']}/{status['puts']}"
    )
    print(
        f"Compressed/uncompressed bytes: {format_bytes(status['total_zip_bytes'])} / "
        f"{format_bytes(status['total_unzip_bytes'])}"
    )
    print(
        f"Compress throughput:   {metrics['compress_gbps']:.3f} GB/s "
        f"({format_bytes(metrics['compress_bytes'])} in {metrics['compress_ns'] / 1e6:.1f} ms)"
    )
    print(
        f"Decompress throughput: {metrics['decompress_gbps']:.3f} GB/s "
        f"({format_bytes(metrics['decompress_bytes'])} in {metrics['decompress_ns'] / 1e6:.1f} ms)"
    )


def print_mix_report(results: List[dict], verified: List[bool], target) -> None:
    c, d = target
    print(f"\nMixed compress:decompress = {c:g}:{d:g} (bytes), {results[0]['put_ops']} PUT + "
          f"{results[0]['get_ops']} GET ops per rank")
    print("-" * 80)
    print(f"{'Rank':>4} {'Wall s':>8} {'Issue s':>8} {'Wait s':>7} "
          f"{'Zip GB/s':>9} {'Unzip GB/s':>10} "
          f"{'ZipBusy%':>9} {'UnzipBusy%':>10} {'Zip-only':>9} {'Unzip-only':>10} {'Verify':>6}")
    for rank, r in enumerate(results):
        wall_ns = r["elapsed"] * 1e9
        zip_only = r["compress_bytes"] / r["compress_ns"] if r["compress_ns"] else 0.0
        unzip_only = r["decompress_bytes"] / r["decompress_ns"] if r["decompress_ns"] else 0.0
        print(f"{rank:>4} {r['elapsed']:8.3f} {r['issue_s']:8.3f} {r['wait_s']:7.3f} "
              f"{r['put_bytes'] / wall_ns:9.3f} {r['get_bytes'] / wall_ns:10.3f} "
              f"{100 * r['compress_ns'] / wall_ns:9.1f} {100 * r['decompress_ns'] / wall_ns:10.1f} "
              f"{zip_only:9.3f} {unzip_only:10.3f} {'pass' if verified[rank] else 'FAIL':>6}")
    zip_sum = sum(r["put_bytes"] / (r["elapsed"] * 1e9) for r in results)
    unzip_sum = sum(r["get_bytes"] / (r["elapsed"] * 1e9) for r in results)
    print(f"{'Sum':>4} {'':>8} {'':>8} {'':>7} {zip_sum:9.3f} {unzip_sum:10.3f}")
    wall_ns = max(r["elapsed"] for r in results) * 1e9
    print(f"Mixed total (bytes / slowest wall): "
          f"compress {sum(r['put_bytes'] for r in results) / wall_ns:.3f} GB/s, "
          f"decompress {sum(r['get_bytes'] for r in results) / wall_ns:.3f} GB/s")
    print(f"Mixed verification: {'passed' if all(verified) else 'FAILED'}")


def run_benchmark(args: argparse.Namespace) -> bool:
    if get_accelerator_device() != "cuda":
        print("No CUDA device available; KVStore benchmark cannot run.")
        return False

    bench = Benchmark(args)
    bench.print_config()
    bench.setup()
    bench.warmup()
    if args.mix is not None:
        put_time = bench.put()
        print_result("PUT", put_time, bench.total_blocks, bench.total_bytes)
        result = bench.mix()
        verified = bench.mix_verify()
        print_mix_report([result], [verified], args.mix)
        return verified
    put_time = bench.put()
    get_time = bench.get()
    verified = bench.verify()
    print_report(
        put_time,
        get_time,
        verified,
        bench.status(),
        bench.metrics(),
        bench.total_blocks,
        bench.total_bytes,
    )
    return verified


def main() -> int:
    args = parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    raise SystemExit(main())
