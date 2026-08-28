"""Config parser tests against the real Qwen3.5-4B TP2 KVCacheConfig dump."""
import dataclasses
import json
import os

import pytest


from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KVCacheTensor, KVCacheGroupSpec,
    MambaSpec, FullAttentionSpec, MambaAttentionBackendEnum,
)

from kvshrink.kvshrink_connector import (
    parse_kv_cache_config, KVShrinkParseError)

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
    import torch
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


def test_fail_closed_unknown_spec():
    cfg = _real_config()
    # swap one group's spec for an unsupported type
    class Weird:
        block_size = 1
        page_size_bytes = 1

    bad_groups = list(cfg.kv_cache_groups)
    bad_groups[1] = KVCacheGroupSpec(
        layer_names=bad_groups[1].layer_names, kv_cache_spec=Weird())
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError:
        pass


def test_groups_must_share_one_block_size():
    """vLLM aligns every group onto a common block size -- a GDN model's
    attention groups take the mamba size -- and a request's block hashes
    are computed at that size, so hash i names block i in EVERY group.
    That correspondence is what lets one hash address a boundary across
    groups; mixed sizes would make it wrong for all but one of them.
    """
    cfg = _real_config()
    g = cfg.kv_cache_groups[0]
    cfg.kv_cache_groups[0] = KVCacheGroupSpec(
        layer_names=g.layer_names,
        kv_cache_spec=dataclasses.replace(
            g.kv_cache_spec, block_size=g.kv_cache_spec.block_size * 2))
    with pytest.raises(KVShrinkParseError, match="different block sizes"):
        parse_kv_cache_config(cfg)


def test_fail_closed_mamba_cache_mode_not_align():
    """A non-'align' mamba cache mode must be refused at startup.

    vLLM defaults prefix caching OFF for hybrid models and then silently
    rewrites --mamba-cache-mode to 'none'. In that mode a request keeps a
    single max_model_len block that no boundary can address, so the
    connector would quietly cache nothing. Refuse loudly instead.
    """
    import dataclasses
    cfg = _real_config()
    bad_groups = []
    for g in cfg.kv_cache_groups:
        spec = g.kv_cache_spec
        if type(spec).__name__ == "MambaSpec":
            spec = dataclasses.replace(spec, mamba_cache_mode="none")
        bad_groups.append(KVCacheGroupSpec(
            layer_names=g.layer_names, kv_cache_spec=spec))
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError as e:
        assert "align" in str(e), e


def test_lossy_truncation_downgraded_to_warning(monkeypatch):
    """The lossy env no longer refuses startup: per-entry flags keep
    mamba pools structurally exact, so the request is only logged.
    """
    from kvshrink.kvshrink_connector import validate_codec_env

    for value in ("1", "4", "8", "auto"):
        monkeypatch.setenv("IAXL_KV_LOSSY_TRUNC", value)
        # Downgraded to a scoped warning: entry flags exempt opaque
        # mamba pools structurally, so the knob can stay enabled.
        validate_codec_env()


def test_lossless_settings_are_allowed(monkeypatch):
    """Off, unset and byte shuffling must all pass.

    Byte shuffling is reversible, so rejecting it would guard more than
    correctness requires and push operators to disable the connector to
    keep a feature they are entitled to.
    """
    from kvshrink.kvshrink_connector import validate_codec_env

    monkeypatch.delenv("IAXL_KV_LOSSY_TRUNC", raising=False)
    validate_codec_env()

    for value in ("0", " 0 ", ""):
        monkeypatch.setenv("IAXL_KV_LOSSY_TRUNC", value)
        validate_codec_env()

    monkeypatch.setenv("IAXL_KV_DATA_SHUFFLE", "1")
    validate_codec_env()
