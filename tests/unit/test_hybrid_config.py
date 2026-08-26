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
    parse_kv_cache_config, compute_namespace, KVShrinkParseError)
from kvshrink.kvshrink_connector import SCHEMA_VERSION

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
    groups, layer_infos, num_blocks = parse_kv_cache_config(
        cfg)
    assert num_blocks == 1843
    assert len(groups) == 4
    kinds = [g.kind for g in groups]
    assert kinds == ["mamba", "mamba", "mamba", "attention"]
    # 32 layers, all mapped
    assert len(layer_infos) == 32
    for g in groups:
        assert len(g.layer_names) == 8
    for name, info in layer_infos.items():
        assert info.page_size_bytes == 1081344, name
    mamba = groups[0]
    assert mamba.mamba_align_size == 528
    # real layer name format
    assert "language_model.model.layers.3.self_attn.attn" in layer_infos
    assert "language_model.model.layers.0.linear_attn" in layer_infos


def test_recurrent_page_holds_both_states():
    """A GDN page is the conv state and the ssm state back to back, and
    the two have different shapes AND different dtypes. That is why the
    page is moved as opaque bytes rather than as tensors."""
    cfg = _real_config()
    _, layer_infos, _ = parse_kv_cache_config(
        cfg)
    lin = layer_infos["language_model.model.layers.0.linear_attn"]
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


def test_fail_closed_uniform_missing_layer():
    """UniformTypeKVCacheSpecs missing a layer must raise."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    cfg = _real_config()
    spec = UniformTypeKVCacheSpecs(  # only 7 of 8 layers registered
        block_size=528,
        kv_cache_specs={
            n: _attn_spec()
            for n in cfg.kv_cache_groups[3].layer_names[:-1]
        })
    bad_groups = list(cfg.kv_cache_groups)
    bad_groups[3] = KVCacheGroupSpec(
        layer_names=cfg.kv_cache_groups[3].layer_names, kv_cache_spec=spec)
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




def test_fail_closed_heterogeneous_pages_in_group():
    """Layers within one group with different page sizes must raise.

    Uses REAL FullAttentionSpec instances with differing page sizes so the
    heterogeneous-page check (not the spec-kind check) fires.
    """
    import torch
    cfg = _real_config()
    layers = list(cfg.kv_cache_groups[3].layer_names)
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    per_layer = {}
    for i, n in enumerate(layers):
        per_layer[n] = FullAttentionSpec(
            block_size=528, num_kv_heads=2, head_size=256,
            dtype=torch.bfloat16,
            page_size_padded=1114112 if i == 0 else 1081344)
    bad_groups = list(cfg.kv_cache_groups)
    bad_groups[3] = KVCacheGroupSpec(
        layer_names=layers,
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=528, kv_cache_specs=per_layer))
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError as e:
        assert "differing page sizes" in str(e), str(e)


def test_parse_real_config_layout_descriptors():
    """Real 4B TP2 config: contiguous pages, zero offsets (v0.21 semantics)."""
    cfg = _real_config()
    _, layer_infos, _ = parse_kv_cache_config(
        cfg)
    info = layer_infos["language_model.model.layers.0.linear_attn"]
    assert info.block_stride_bytes == info.page_size_bytes == 1081344
    assert info.storage_offset_bytes == 0
    assert "language_model.model.layers.3.self_attn.attn" in layer_infos


def test_namespace_stability():
    a = compute_namespace("m", "r", "t", "auto", SCHEMA_VERSION, 2, 1)
    b = compute_namespace("m", "r", "t", "auto", SCHEMA_VERSION, 2, 1)
    c = compute_namespace("m", "r", "t", "auto", SCHEMA_VERSION, 4, 1)
    assert a == b
    assert a != c


def test_fail_closed_lossy_truncation_rejected(monkeypatch):
    """A lossy codec must be refused on the hybrid path.

    IAXL_KV_LOSSY_TRUNC masks the low bits of every element in place,
    and iaxl's lossy_trunc has an element_size == 1 branch. Hybrid
    pages are opaque int8 canonical views, so the mask hits every byte
    and destroys the exponent bits of the bf16 values inside, not just
    their precision. An approximate attention block still decodes --
    that is what the knob is for -- but a corrupted GDN recurrent state
    is fed back into the next step and yields wrong tokens with no
    error, so startup must refuse.
    """
    from kvshrink.kvshrink_connector import validate_codec_env

    for value in ("1", "4", "8", "auto"):
        monkeypatch.setenv("IAXL_KV_LOSSY_TRUNC", value)
        try:
            validate_codec_env()
            raise AssertionError(
                f"expected KVShrinkParseError for {value!r}")
        except KVShrinkParseError as e:
            assert "IAXL_KV_LOSSY_TRUNC" in str(e), e


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
