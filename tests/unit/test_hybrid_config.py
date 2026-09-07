"""Config parser tests against the real Qwen3.5-4B TP2 KVCacheConfig dump."""
import json
import os


from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KVCacheTensor, KVCacheGroupSpec,
    MambaSpec, FullAttentionSpec, MambaAttentionBackendEnum,
)

from kvshrink.kvshrink_connector import (
    parse_kv_cache_config, project_block_hashes)

FIXTURE = os.path.join(os.path.dirname(__file__),
                       "fixture_kvconfig_4b_tp2.json")


def _mamba_spec():
    import torch
    return MambaSpec(
        block_size=528,
        shapes=((3, 4096), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=1081344,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )


def _attn_spec():
    import torch
    return FullAttentionSpec(
        block_size=528,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        page_size_padded=1081344,
    )


def _real_config():
    """Rebuild KVCacheConfig from the M0 dump of Qwen3.5-4B TP2."""
    with open(FIXTURE) as f:
        d = json.load(f)
    tensors = [KVCacheTensor(size=t["size"], shared_by=t["shared_by"])
               for t in d["kv_cache_tensors"]]
    groups = []
    for g in d["kv_cache_groups"]:
        spec = g["spec"]
        if spec["type"] == "MambaSpec":
            s = _mamba_spec()
        else:
            s = _attn_spec()
        groups.append(KVCacheGroupSpec(
            layer_names=g["layer_names"], kv_cache_spec=s))
    return KVCacheConfig(
        num_blocks=d["num_blocks"],
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def test_fixture_shape():
    cfg = _real_config()
    assert cfg.num_blocks == 1843
    assert len(cfg.kv_cache_tensors) == 8
    assert len(cfg.kv_cache_groups) == 4


def test_parse_real_config():
    cfg = _real_config()
    groups, block_size = parse_kv_cache_config(cfg)
    assert block_size == 528
    assert len(groups) == 4
    kinds = [g.kind for g in groups]
    assert kinds == ["mamba", "mamba", "mamba", "attention"]
    # 32 layers, all mapped
    for g in groups:
        assert len(g.layer_names) == 8

def test_recurrent_page_spec_declares_both_states():
    """A GDN page is the conv state and the ssm state back to back, and
    the two have different shapes AND different dtypes. That is why the
    page travels as opaque bytes (KVStore fuses the parts at bind)."""
    cfg = _real_config()
    groups, _ = parse_kv_cache_config(cfg)
    lin = groups[0].spec
    conv_bytes = 3 * 4096 * 2              # bf16
    ssm_bytes = 16 * 128 * 128 * 4         # fp32
    # vLLM pads the page, so the size is not the bare sum; what matters
    # is that one page holds both states, which is why it is moved as
    # opaque bytes rather than as tensors.
    assert lin.page_size_bytes >= conv_bytes + ssm_bytes


def test_block_hashes_project_onto_our_blocks():
    """vLLM computes Request.block_hashes at hash_block_size, which can
    be FINER than the block size our plans address (config/cache.py
    documents it as a knob for computing prefix-caching keys at the
    finest common granularity, to be merged for larger physical
    blocks). Merging is the consumer's job; we do it with the same
    stride vLLM's own offloading connector uses.
    """
    # factor 1: the usual case, an identity projection.
    keys = []
    project_block_hashes([10, 11, 12], keys, 1)
    assert keys == ["10", "11", "12"]

    # factor 4: one key per block, taken at the block's LAST hash --
    # a prefix hash only names the whole block's content at its end.
    keys = []
    project_block_hashes(list(range(8)), keys, 4)
    assert keys == ["3", "7"]

    # A partial block contributes nothing until its last hash lands.
    keys = []
    project_block_hashes([0, 1, 2], keys, 4)
    assert keys == []

    # Incremental: re-running after the engine appended in place picks
    # up exactly the newly completed blocks, never re-emitting.
    live = list(range(8))
    keys = []
    project_block_hashes(live, keys, 4)
    assert keys == ["3", "7"]
    live.extend(range(8, 16))
    project_block_hashes(live, keys, 4)
    assert keys == ["3", "7", "11", "15"]
