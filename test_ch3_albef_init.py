"""Tests for the ALBEF -> Ch3Model parameter mapping."""

from __future__ import annotations

import pytest
import torch

from ch3_mlaff.albef_init import remap_albef, target_names


def _albef_like() -> dict:
    """The key shapes the released ALBEF.pth actually contains."""
    sd = {
        "temp": torch.tensor(0.07),
        "image_queue": torch.zeros(256, 65536),
        "text_queue": torch.zeros(256, 65536),
        "queue_ptr": torch.zeros(1),
        "vision_proj.weight": torch.zeros(256, 768),
        "vision_proj.bias": torch.zeros(256),
        "text_proj.weight": torch.zeros(256, 768),
        "text_proj.bias": torch.zeros(256),
        "itm_head.weight": torch.zeros(2, 768),
        "itm_head.bias": torch.zeros(2),
        "visual_encoder.cls_token": torch.zeros(1, 1, 768),
        "visual_encoder.pos_embed": torch.zeros(1, 257, 768),
        "visual_encoder.patch_embed.proj.weight": torch.zeros(768, 3, 16, 16),
        "visual_encoder.norm.weight": torch.zeros(768),
        "text_encoder.bert.embeddings.position_ids": torch.zeros(1, 512),
        "text_encoder.bert.embeddings.word_embeddings.weight": torch.zeros(30522, 768),
        "text_encoder.bert.embeddings.LayerNorm.weight": torch.zeros(768),
        "text_encoder.cls.predictions.bias": torch.zeros(30522),
        "text_encoder.cls.predictions.transform.dense.weight": torch.zeros(768, 768),
        "text_encoder.cls.predictions.decoder.weight": torch.zeros(30522, 768),
        # momentum copies, which must all be dropped
        "visual_encoder_m.cls_token": torch.zeros(1, 1, 768),
        "text_encoder_m.bert.embeddings.word_embeddings.weight": torch.zeros(30522, 768),
        "vision_proj_m.weight": torch.zeros(256, 768),
    }
    for layer in range(12):
        p = f"text_encoder.bert.encoder.layer.{layer}."
        sd[p + "attention.self.query.weight"] = torch.zeros(768, 768)
        sd[p + "attention.output.dense.weight"] = torch.zeros(768, 768)
        sd[p + "attention.output.LayerNorm.weight"] = torch.zeros(768)
        sd[p + "intermediate.dense.weight"] = torch.zeros(3072, 768)
        sd[p + "output.dense.weight"] = torch.zeros(768, 3072)
        sd[p + "output.LayerNorm.weight"] = torch.zeros(768)
        if layer >= 6:
            sd[p + "crossattention.self.key.weight"] = torch.zeros(768, 768)
            sd[p + "crossattention.output.dense.weight"] = torch.zeros(768, 768)
    return sd


def test_vit_keys_land_under_the_timm_submodule():
    out = remap_albef(_albef_like())
    assert "visual_encoder.vit.pos_embed" in out
    assert out["visual_encoder.vit.pos_embed"].shape == (1, 257, 768)
    assert "visual_encoder.vit.patch_embed.proj.weight" in out
    assert "visual_encoder.vit.norm.weight" in out


def test_first_six_bert_layers_become_the_text_encoder():
    out = remap_albef(_albef_like(), text_layers=6)
    for layer in range(6):
        assert f"text_encoder.text_blocks.{layer}.attention.query.weight" in out
        assert f"text_encoder.text_blocks.{layer}.intermediate.weight" in out


def test_last_six_bert_layers_become_the_fusion_encoder_with_cross_attention():
    out = remap_albef(_albef_like(), text_layers=6)
    for layer in range(6):
        assert f"text_encoder.fusion_blocks.{layer}.attention.query.weight" in out
        assert f"text_encoder.fusion_blocks.{layer}.crossattention.key.weight" in out
        assert f"text_encoder.fusion_blocks.{layer}.crossattention_output.dense.weight" in out
    assert not any("text_blocks" in k and "crossattention" in k for k in out)


def test_attention_output_is_not_confused_with_the_ffn_output():
    """'attention.output.dense' and 'output.dense' share a suffix."""
    out = remap_albef(_albef_like())
    assert out["text_encoder.text_blocks.0.attention_output.dense.weight"].shape == (768, 768)
    assert out["text_encoder.text_blocks.0.output.dense.weight"].shape == (768, 3072)
    assert out["text_encoder.text_blocks.0.intermediate.weight"].shape == (3072, 768)


def test_embeddings_layernorm_becomes_emb_ln():
    out = remap_albef(_albef_like())
    assert "text_encoder.emb_ln.weight" in out
    assert "text_encoder.word_embeddings.weight" in out
    assert not any("position_ids" in k for k in out)


def test_momentum_copies_and_queues_are_dropped():
    out = remap_albef(_albef_like())
    assert not any("_m." in k or k.endswith("_m") for k in out)
    assert not any(k in out for k in ("image_queue", "text_queue", "queue_ptr"))


def test_retrieval_and_projection_heads_are_carried_over():
    out = remap_albef(_albef_like())
    for k in ("vision_proj.weight", "text_proj.bias", "itm_head.weight", "temp"):
        assert k in out


def test_mlm_head_is_kept_for_stage_one_and_dropped_for_stage_two():
    out = remap_albef(_albef_like())
    assert "mlm_head.decoder.weight" in out
    assert "mlm_head.dense.weight" in out

    retrieval = target_names(out, has_mlm_head=False)
    assert not any(k.startswith("mlm_head.") for k in retrieval)
    assert "visual_encoder.vit.pos_embed" in retrieval

    pretrain = target_names(out, has_mlm_head=True)
    assert "mlm_head.decoder.weight" in pretrain
    assert "base.visual_encoder.vit.pos_embed" in pretrain
    assert not any(k.startswith("base.mlm_head") for k in pretrain)
