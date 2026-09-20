"""Encoders for the chapter 3 model.

Two wrappers are provided:

``MultiLayerViT``
    a timm ViT-B/16 that can return the hidden states of several blocks in one
    forward pass, which is what section 3.2.1 needs for layers 4 / 8 / 12.

``SplitBert``
    BERT-base cut into a 6-layer text encoder (section 3.3.1) and a 6-layer
    image-grounded fusion encoder used by the ITM head.  The transformer blocks
    are re-implemented here rather than imported from ``transformers`` so that
    the cross-attention wiring does not depend on a particular library version;
    the weights are still the pretrained ``bert-base-uncased`` ones.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# vision
# --------------------------------------------------------------------------


class MultiLayerViT(nn.Module):
    """ViT-B/16 exposing intermediate block outputs.

    ``forward_layers(image, [4, 8, 12])`` returns ``{4: h4, 8: h8, 12: h12}``
    where each tensor is ``[B, 1 + P, D]``.  Layer numbering is 1-based, so 12
    is the last block of ViT-B/16 and its output is the one the final norm is
    applied to.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
                 img_size: int = 256, pretrained: bool = True, drop_path_rate: float = 0.1):
        super().__init__()
        import timm

        self.vit = timm.create_model(
            model_name, pretrained=pretrained, img_size=img_size,
            num_classes=0, drop_path_rate=drop_path_rate,
        )
        self.num_layers = len(self.vit.blocks)
        self.width = self.vit.embed_dim

    def forward_layers(self, image: torch.Tensor, layers: Iterable[int]) -> Dict[int, torch.Tensor]:
        wanted = sorted(set(int(l) for l in layers))
        if wanted and (wanted[0] < 1 or wanted[-1] > self.num_layers):
            raise ValueError(f"layers must be within 1..{self.num_layers}, got {wanted}")
        x = self.vit.patch_embed(image)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        out: Dict[int, torch.Tensor] = {}
        last = wanted[-1] if wanted else 0
        for i, blk in enumerate(self.vit.blocks, start=1):
            x = blk(x)
            if i in wanted:
                # the final block's output feeds the retrieval head, so it gets
                # the encoder's output norm; intermediate taps stay pre-norm.
                out[i] = self.vit.norm(x) if i == self.num_layers else x
            if i == last:
                break
        return out

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_layers(image, [self.num_layers])[self.num_layers]


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


class BertSelfAttention(nn.Module):
    def __init__(self, width: int, num_heads: int, dropout: float, kv_width: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        kv_width = kv_width or width
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(kv_width, width)
        self.value = nn.Linear(kv_width, width)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, kv: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv = hidden if kv is None else kv
        q, k, v = self._shape(self.query(hidden)), self._shape(self.key(kv)), self._shape(self.value(kv))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        probs = self.dropout(scores.softmax(dim=-1))
        ctx = torch.matmul(probs, v).transpose(1, 2)
        return ctx.reshape(hidden.shape[0], hidden.shape[1], -1)


class BertOutput(nn.Module):
    def __init__(self, in_width: int, out_width: int, dropout: float, eps: float):
        super().__init__()
        self.dense = nn.Linear(in_width, out_width)
        self.LayerNorm = nn.LayerNorm(out_width, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout(self.dense(hidden)) + residual)


class BertLayer(nn.Module):
    """One BERT block, optionally with a cross-attention sub-layer."""

    def __init__(self, width: int, num_heads: int, intermediate: int, dropout: float,
                 eps: float, cross_width: Optional[int] = None):
        super().__init__()
        self.attention = BertSelfAttention(width, num_heads, dropout)
        self.attention_output = BertOutput(width, width, dropout, eps)
        self.has_cross = cross_width is not None
        if self.has_cross:
            self.crossattention = BertSelfAttention(width, num_heads, dropout, kv_width=cross_width)
            self.crossattention_output = BertOutput(width, width, dropout, eps)
        self.intermediate = nn.Linear(width, intermediate)
        self.output = BertOutput(intermediate, width, dropout, eps)

    def forward(self, hidden, attn_mask=None, enc_hidden=None, enc_mask=None):
        hidden = self.attention_output(self.attention(hidden, mask=attn_mask), hidden)
        if self.has_cross:
            if enc_hidden is None:
                raise ValueError("fusion layer called without encoder_hidden_states")
            hidden = self.crossattention_output(
                self.crossattention(hidden, kv=enc_hidden, mask=enc_mask), hidden
            )
        return self.output(F.gelu(self.intermediate(hidden)), hidden)


class SplitBert(nn.Module):
    """BERT-base split into a text encoder and an image-grounded fusion encoder."""

    def __init__(self, model_name: str = "bert-base-uncased", text_layers: int = 6,
                 cross_width: int = 768, pretrained: bool = True):
        super().__init__()
        from transformers import BertConfig, BertModel

        cfg = BertConfig.from_pretrained(model_name)
        self.width = cfg.hidden_size
        self.text_layers = text_layers
        self.fusion_layers = cfg.num_hidden_layers - text_layers
        eps = cfg.layer_norm_eps
        drop = cfg.hidden_dropout_prob

        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.emb_ln = nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.emb_drop = nn.Dropout(drop)

        def mk(cross: Optional[int]) -> BertLayer:
            return BertLayer(cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size,
                             cfg.attention_probs_dropout_prob, eps, cross_width=cross)

        self.text_blocks = nn.ModuleList(mk(None) for _ in range(text_layers))
        self.fusion_blocks = nn.ModuleList(mk(cross_width) for _ in range(self.fusion_layers))
        self._emb_drop_p = drop

        if pretrained:
            self._load_pretrained(BertModel.from_pretrained(model_name))

    @torch.no_grad()
    def _load_pretrained(self, bert) -> None:
        """Copy bert-base weights; cross-attention stays randomly initialised."""
        e = bert.embeddings
        self.word_embeddings.weight.copy_(e.word_embeddings.weight)
        self.position_embeddings.weight.copy_(e.position_embeddings.weight)
        self.token_type_embeddings.weight.copy_(e.token_type_embeddings.weight)
        self.emb_ln.weight.copy_(e.LayerNorm.weight)
        self.emb_ln.bias.copy_(e.LayerNorm.bias)

        blocks = list(self.text_blocks) + list(self.fusion_blocks)
        for dst, src in zip(blocks, bert.encoder.layer):
            dst.attention.query.load_state_dict(src.attention.self.query.state_dict())
            dst.attention.key.load_state_dict(src.attention.self.key.state_dict())
            dst.attention.value.load_state_dict(src.attention.self.value.state_dict())
            dst.attention_output.dense.load_state_dict(src.attention.output.dense.state_dict())
            dst.attention_output.LayerNorm.load_state_dict(src.attention.output.LayerNorm.state_dict())
            dst.intermediate.load_state_dict(src.intermediate.dense.state_dict())
            dst.output.dense.load_state_dict(src.output.dense.state_dict())
            dst.output.LayerNorm.load_state_dict(src.output.LayerNorm.state_dict())

    @staticmethod
    def _extend(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        m = mask[:, None, None, :].to(dtype)
        return (1.0 - m) * torch.finfo(dtype).min

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = (self.word_embeddings(input_ids) + self.position_embeddings(pos)
             + self.token_type_embeddings(torch.zeros_like(input_ids)))
        return self.emb_drop(self.emb_ln(x))

    def forward(self, input_ids=None, attention_mask=None, encoder_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None, mode="text"):
        if mode == "text":
            x = self.embed(input_ids)
            m = self._extend(attention_mask, x.dtype)
            for blk in self.text_blocks:
                x = blk(x, attn_mask=m)
            return x
        if mode == "fusion":
            x = encoder_embeds
            m = self._extend(attention_mask, x.dtype)
            em = self._extend(encoder_attention_mask, x.dtype)
            for blk in self.fusion_blocks:
                x = blk(x, attn_mask=m, enc_hidden=encoder_hidden_states, enc_mask=em)
            return x
        raise ValueError(f"unknown mode: {mode}")
