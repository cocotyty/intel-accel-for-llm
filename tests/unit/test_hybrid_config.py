"""Config parser tests against the real Qwen3.5-4B TP2 KVCacheConfig dump."""
import json
import os


from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KVCacheTensor, KVCacheGroupSpec,
    MambaSpec, FullAttentionSpec, MambaAttentionBackendEnum,
)

from kvshrink.kvshrink_connector import (
    parse_kv_cache_config)

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
    groups, num_blocks = parse_kv_cache_config(
        cfg)
    assert num_blocks == 1843
    assert len(groups) == 4
    kinds = [g.kind for g in groups]
    assert kinds == ["mamba", "mamba", "mamba", "attention"]
    # 32 layers, all mapped
    for g in groups:
        assert len(g.layer_names) == 8
    mamba = groups[0]
    assert mamba.mamba_align_size == 528


def test_recurrent_page_spec_declares_both_states():
    """A GDN page is the conv state and the ssm state back to back, and
    the two have different shapes AND different dtypes. That is why the
    page travels as opaque bytes (KVStore fuses the parts at bind)."""
    cfg = _real_config()
    groups, _num_blocks = parse_kv_cache_config(
        cfg)
    lin = groups[0].spec
    conv_bytes = 3 * 4096 * 2              # bf16
    ssm_bytes = 16 * 128 * 128 * 4         # fp32
    # vLLM pads the page, so the size is not the bare sum; what matters
    # is that one page holds both states, which is why it is moved as
    # opaque bytes rather than as tensors.
    assert lin.page_size_bytes >= conv_bytes + ssm_bytes
