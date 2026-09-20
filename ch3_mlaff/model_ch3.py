"""Model for thesis chapter 3 (section 3.2).

The backbone follows the ALBEF / FashionSAP two-tower + fusion layout that the
thesis baseline reproduces: a ViT-B/16 image encoder, the first 6 BERT-base
layers as the text encoder, and the remaining 6 BERT layers (with cross
attention) as the image-grounded fusion encoder used by the ITM head.

On top of that baseline the chapter adds three modules, each implemented here
exactly as written in section 3.2 and switchable through ``Ch3Config`` so that
the table 3.5 ablation ladder is one code path with five configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class Ch3Config:
    """Configuration for the chapter 3 model.

    The five rows of table 3.5 correspond to:

    ==========  ===============  ==============  ==============  =========
    model       vis_layers       cross_attn      fusion          rea
    ==========  ===============  ==============  ==============  =========
    1           (12,)            False           ``"none"``      False
    2           (12,)            True            ``"none"``      False
    3           (4, 8, 12)       True            ``"direct"``    False
    4           (4, 8, 12)       True            ``"gated"``     False
    5 (full)    (4, 8, 12)       True            ``"gated"``     True
    ==========  ===============  ==============  ==============  =========
    """

    vis_layers: Tuple[int, ...] = (4, 8, 12)
    use_cross_attn: bool = True
    fusion: str = "gated"  # "none" | "direct" | "gated"
    use_rea: bool = True

    image_res: int = 256
    patch_size: int = 16
    embed_dim: int = 256          # shared retrieval space (table 3.1)
    width: int = 768              # ViT / BERT hidden size
    max_word_num: int = 180       # table 3.1: L fixed at 180
    num_heads: int = 8            # section 3.3.6.2 default
    text_layers: int = 6          # section 3.3.1: first 6 BERT layers

    topk_rerank: int = 128        # Full-protocol shortlist depth before reranking

    temp_init: float = 0.07
    rea_temp: float = 0.07        # tau_r in Eq. (3.11)
    rea_weight: float = 0.5       # lambda in Eq. (3.13)
    queue_size: int = 16384
    momentum: float = 0.995
    alpha: float = 0.4            # ALBEF distillation weight on ITC

    vit_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k"
    bert_name: str = "bert-base-uncased"

    def validate(self) -> None:
        if self.fusion not in ("none", "direct", "gated"):
            raise ValueError(f"unknown fusion mode: {self.fusion}")
        if not self.vis_layers:
            raise ValueError("vis_layers must not be empty")
        if self.fusion == "none" and len(self.vis_layers) != 1:
            raise ValueError("fusion='none' requires exactly one visual layer")
        if self.fusion != "none" and not self.use_cross_attn:
            raise ValueError("multi-level fusion requires cross attention")
        if self.use_rea and not self.use_cross_attn:
            raise ValueError("the region-enhanced loss needs the cross-attention output")
        if self.width % self.num_heads != 0:
            raise ValueError("width must be divisible by num_heads")


# --------------------------------------------------------------------------
# section 3.2.2 -- hierarchical cross-modal cross attention
# --------------------------------------------------------------------------


class CrossModalBlock(nn.Module):
    """One level of Eq. (3.1)-(3.7).

    Text tokens provide the Query; the visual patch tokens of one ViT level
    provide Key and Value.  The output projection, residual connection, layer
    norm and feed-forward network of Eq. (3.5)-(3.7) follow the attention.
    """

    def __init__(self, width: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(d) in Eq. (3.4)

        self.w_q = nn.Linear(width, width)           # W_Q, Eq. (3.1)
        self.w_k = nn.Linear(width, width)           # W_K, Eq. (3.2)
        self.w_u = nn.Linear(width, width)           # W_U, Eq. (3.3)
        self.w_o = nn.Linear(width, width)           # W_O, Eq. (3.5)
        self.drop = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(width)               # Eq. (3.6)
        self.ln2 = nn.LayerNorm(width)               # Eq. (3.7)
        hidden = int(width * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width)
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        text_tok: torch.Tensor,          # T_tok, [B, M, D]
        vis_patch: torch.Tensor,         # I_patch^l, [B, P, D]
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split(self.w_q(text_tok))          # [B, h, M, d]
        k = self._split(self.w_k(vis_patch))         # [B, h, P, d]
        u = self._split(self.w_u(vis_patch))         # [B, h, P, d]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # A^l, Eq. (3.4)
        attn = attn.softmax(dim=-1)
        ctx = torch.matmul(self.drop(attn), u)                     # C^l = A^l U^l
        ctx = ctx.transpose(1, 2).reshape(text_tok.shape[0], text_tok.shape[1], self.width)

        h1 = self.ln1(text_tok + self.drop(self.w_o(ctx)))         # Eq. (3.5)+(3.6)
        out = self.ln2(h1 + self.drop(self.ffn(h1)))               # Eq. (3.7)
        if text_mask is not None:
            out = out * text_mask.unsqueeze(-1).to(out.dtype)
        return out


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Token-dimension average pooling of Eq. (3.8), ignoring padding."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


# --------------------------------------------------------------------------
# section 3.2.3 -- text-guided gated fusion
# --------------------------------------------------------------------------


class TextGuidedGate(nn.Module):
    """Eq. (3.9): beta = softmax(W_g t_cls + b_g), one weight per visual level."""

    def __init__(self, width: int, num_levels: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, num_levels))

    def forward(self, text_cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(text_cls).softmax(dim=-1)      # [B, num_levels]


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------


class Ch3Model(nn.Module):
    """Multi-level adaptive feature fusion retrieval model of section 3.2."""

    def __init__(
        self,
        cfg: Ch3Config,
        visual_encoder: nn.Module,
        text_encoder: nn.Module,
        visual_encoder_m: Optional[nn.Module] = None,
        text_encoder_m: Optional[nn.Module] = None,
    ):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

        w, e = cfg.width, cfg.embed_dim
        self.vision_proj = nn.Linear(w, e)
        self.text_proj = nn.Linear(w, e)
        self.itm_head = nn.Linear(w, 2)
        self.temp = nn.Parameter(torch.tensor(cfg.temp_init))

        n_lvl = len(cfg.vis_layers)
        if cfg.use_cross_attn:
            self.cross_blocks = nn.ModuleList(
                CrossModalBlock(w, cfg.num_heads) for _ in range(n_lvl)
            )
        else:
            self.cross_blocks = None

        if cfg.fusion == "gated":
            self.gate = TextGuidedGate(w, n_lvl)
        else:
            self.gate = None

        # F_final is scored against the text CLS embedding (Eq. 3.11), so it
        # needs its own projection into the shared 256-d space.
        self.region_proj = nn.Linear(w, e) if cfg.use_cross_attn else None

        # momentum branch + queues (ALBEF-style ITC, section 3.2.4 note)
        self.visual_encoder_m = visual_encoder_m
        self.text_encoder_m = text_encoder_m
        self.use_momentum = visual_encoder_m is not None and text_encoder_m is not None
        if self.use_momentum:
            self.vision_proj_m = nn.Linear(w, e)
            self.text_proj_m = nn.Linear(w, e)
            self.model_pairs = [
                (self.visual_encoder, self.visual_encoder_m),
                (self.vision_proj, self.vision_proj_m),
                (self.text_encoder, self.text_encoder_m),
                (self.text_proj, self.text_proj_m),
            ]
            if cfg.use_cross_attn:
                self.cross_blocks_m = nn.ModuleList(
                    CrossModalBlock(w, cfg.num_heads) for _ in range(n_lvl)
                )
                self.region_proj_m = nn.Linear(w, e)
                self.model_pairs += [(self.cross_blocks, self.cross_blocks_m),
                                     (self.region_proj, self.region_proj_m)]
                if cfg.fusion == "gated":
                    self.gate_m = TextGuidedGate(w, n_lvl)
                    self.model_pairs.append((self.gate, self.gate_m))
            self.copy_params()
            for name in ("image_queue", "text_queue") + (("region_queue",) if cfg.use_cross_attn else ()):
                self.register_buffer(name, F.normalize(torch.randn(e, cfg.queue_size), dim=0))
            self.register_buffer("idx_queue", torch.full((1, cfg.queue_size), -100, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # -- momentum bookkeeping ------------------------------------------------

    @torch.no_grad()
    def copy_params(self) -> None:
        for model, model_m in self.model_pairs:
            for p, p_m in zip(model.parameters(), model_m.parameters()):
                p_m.data.copy_(p.data)
                p_m.requires_grad = False

    @torch.no_grad()
    def momentum_update(self) -> None:
        m = self.cfg.momentum
        for model, model_m in self.model_pairs:
            for p, p_m in zip(model.parameters(), model_m.parameters()):
                p_m.data.mul_(m).add_(p.data, alpha=1.0 - m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, feats: Dict[str, torch.Tensor], idx: torch.Tensor) -> None:
        """Append one batch of momentum features to the ring buffers."""
        bs = idx.shape[0]
        qs = self.cfg.queue_size
        ptr = int(self.queue_ptr)
        first = min(bs, qs - ptr)
        for name, value in feats.items():
            queue = getattr(self, name)
            queue[:, ptr:ptr + first] = value[:first].T
            queue[:, : bs - first] = value[first:].T
        self.idx_queue[:, ptr:ptr + first] = idx[:first].view(1, -1)
        self.idx_queue[:, : bs - first] = idx[first:].view(1, -1)
        self.queue_ptr[0] = (ptr + bs) % qs

    # -- encoders ------------------------------------------------------------

    def encode_image(self, image: torch.Tensor, encoder=None) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Return the layer-12 sequence (for ITC/ITM) and the selected patch levels.

        ``visual_encoder`` must expose ``forward_layers(image, layers)`` returning
        ``{layer: [B, 1 + P, D]}`` with the CLS token first.  Section 3.2.1 drops
        the CLS token from the multi-level patch features.
        """
        encoder = encoder or self.visual_encoder
        wanted = sorted(set(self.cfg.vis_layers) | {12})
        outs = encoder.forward_layers(image, wanted)
        return outs[12], {l: outs[l][:, 1:, :] for l in self.cfg.vis_layers}

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    encoder=None) -> torch.Tensor:
        encoder = encoder or self.text_encoder
        return encoder(input_ids=input_ids, attention_mask=attention_mask, mode="text")

    # -- section 3.2.2 + 3.2.3 ----------------------------------------------

    def region_representation(
        self,
        text_seq: torch.Tensor,           # [B, M+1, D] with CLS at index 0
        attention_mask: torch.Tensor,     # [B, M+1]
        patches: Dict[int, torch.Tensor],
        blocks: Optional[nn.ModuleList] = None,
        gate: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce ``(F_final, beta)`` of Eq. (3.8)-(3.10)."""
        blocks = blocks if blocks is not None else self.cross_blocks
        if blocks is None:
            raise RuntimeError("model configured without cross attention")
        gate = gate if gate is not None else self.gate

        text_tok = text_seq[:, 1:, :]                   # T_tok, CLS excluded (3.2.1)
        tok_mask = attention_mask[:, 1:]

        pooled = [masked_mean(blk(text_tok, patches[layer], tok_mask), tok_mask)  # Eq. (3.7)-(3.8)
                  for blk, layer in zip(blocks, self.cfg.vis_layers)]
        stacked = torch.stack(pooled, dim=1)                     # [B, L, D]

        if gate is not None:
            beta = gate(text_seq[:, 0, :])                       # Eq. (3.9)
        else:
            beta = stacked.new_full(stacked.shape[:2], 1.0 / stacked.shape[1])
        return (stacked * beta.unsqueeze(-1)).sum(dim=1), beta   # Eq. (3.10)

    # -- retrieval features --------------------------------------------------

    def retrieval_features(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Normalised embeddings used for the losses and the similarity matrix.

        ``cls_feat`` is the text-independent image embedding: it is what the Full
        protocol shortlists with and what the ITM head samples hard negatives
        from.  ``region_feat`` is the projected F_final of Eq. (3.10) and, when
        the chapter-3 modules are present, it is the image side of the retrieval
        score.
        """
        last, patches = self.encode_image(image)
        text_seq = self.encode_text(input_ids, attention_mask)

        out = {
            "image_embeds": last,
            "text_embeds": text_seq,
            "patches": patches,
            "cls_feat": F.normalize(self.vision_proj(last[:, 0, :]), dim=-1),
            "text_feat": F.normalize(self.text_proj(text_seq[:, 0, :]), dim=-1),
        }
        if self.cross_blocks is not None:
            f_final, beta = self.region_representation(text_seq, attention_mask, patches)
            out["region_feat"] = F.normalize(self.region_proj(f_final), dim=-1)
            out["gate"] = beta
        return out

    # -- section 3.2.4 -- training objective --------------------------------

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        idx: torch.Tensor,
        alpha: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Return ``{"loss", "loss_itc", "loss_itm", "loss_rea"}`` for Eq. (3.13)."""
        cfg = self.cfg
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        feats = self.retrieval_features(image, input_ids, attention_mask)
        text_feat, cls_feat = feats["text_feat"], feats["cls_feat"]
        region_feat = feats.get("region_feat")
        idx = idx.view(-1, 1)

        loss_itc = self._itc_loss(image, input_ids, attention_mask, cls_feat, region_feat,
                                  text_feat, idx, alpha)
        loss_itm = self._itm_loss(feats["image_embeds"], feats["text_embeds"], attention_mask,
                                  cls_feat, text_feat, idx)
        loss_rea = (self._rea_loss(region_feat, text_feat, idx) if cfg.use_rea
                    else cls_feat.new_zeros(()))

        loss = loss_itc + loss_itm + cfg.rea_weight * loss_rea
        return {"loss": loss, "loss_itc": loss_itc.detach(), "loss_itm": loss_itm.detach(),
                "loss_rea": loss_rea.detach()}

    def _contrastive(self, query, gallery_all, targets):
        return -torch.sum(F.log_softmax(query @ gallery_all / self.temp, dim=1) * targets, dim=1).mean()

    def _itc_loss(self, image, input_ids, attention_mask, cls_feat, region_feat, text_feat,
                  idx, alpha):
        """Image-text contrastive loss (Eq. 3.13's ``L_itc``).

        The CLS embedding is always contrasted, because the Full protocol needs a
        text-independent image embedding to shortlist with.  When the chapter-3
        modules are present, ``F_final`` is contrasted as well, which is what
        gives the cross-attention and gating blocks their training signal.
        """
        if not self.use_momentum:
            targets = torch.eq(idx, idx.t()).float()
            targets = targets / targets.sum(1, keepdim=True)
            terms = [(self._contrastive(cls_feat, text_feat.t(), targets)
                      + self._contrastive(text_feat, cls_feat.t(), targets)) / 2]
            if region_feat is not None:
                terms.append((self._contrastive(region_feat, text_feat.t(), targets)
                              + self._contrastive(text_feat, region_feat.t(), targets)) / 2)
            return sum(terms) / len(terms)

        with torch.no_grad():
            self.momentum_update()
            last_m, patches_m = self.encode_image(image, self.visual_encoder_m)
            cls_feat_m = F.normalize(self.vision_proj_m(last_m[:, 0, :]), dim=-1)
            text_m = self.encode_text(input_ids, attention_mask, self.text_encoder_m)
            text_feat_m = F.normalize(self.text_proj_m(text_m[:, 0, :]), dim=-1)
            enqueue = {"image_queue": cls_feat_m, "text_queue": text_feat_m}
            if region_feat is not None:
                f_final_m, _ = self.region_representation(
                    text_m, attention_mask, patches_m, self.cross_blocks_m,
                    getattr(self, "gate_m", None),
                )
                enqueue["region_queue"] = F.normalize(self.region_proj_m(f_final_m), dim=-1)

            in_batch = torch.eq(idx, idx.t()).float()
            in_queue = torch.eq(idx, self.idx_queue.clone().detach()).float()
            targets = torch.cat([in_batch, in_queue], dim=1)
            targets = targets / targets.sum(1, keepdim=True).clamp(min=1.0)

        text_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
        terms = []
        for feat, feat_m, queue in (
            (cls_feat, cls_feat_m, self.image_queue),
            *(((region_feat, enqueue["region_queue"], self.region_queue),) if region_feat is not None else ()),
        ):
            image_all = torch.cat([feat_m.t(), queue.clone().detach()], dim=1)
            with torch.no_grad():
                soft_i2t = alpha * F.softmax(feat_m @ text_all / self.temp, dim=1) + (1 - alpha) * targets
                soft_t2i = alpha * F.softmax(text_feat_m @ image_all / self.temp, dim=1) + (1 - alpha) * targets
            terms.append((self._contrastive(feat, text_all, soft_i2t)
                          + self._contrastive(text_feat, image_all, soft_t2i)) / 2)

        self.dequeue_and_enqueue(enqueue, idx)
        return sum(terms) / len(terms)

    def _itm_loss(self, image_embeds, text_embeds, attention_mask, cls_feat, text_feat, idx):
        """Image-text matching loss with in-batch hard negatives."""
        bs = image_embeds.shape[0]
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        pos = self.text_encoder(
            encoder_embeds=text_embeds, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, mode="fusion",
        )[:, 0, :]

        with torch.no_grad():
            mask = torch.eq(idx, idx.t())
            w_i2t = F.softmax(cls_feat @ text_feat.t() / self.temp, dim=1) + 1e-5
            w_t2i = F.softmax(text_feat @ cls_feat.t() / self.temp, dim=1) + 1e-5
            w_i2t.masked_fill_(mask, 0.0)
            w_t2i.masked_fill_(mask, 0.0)
            if (w_i2t.sum(1) == 0).any() or (w_t2i.sum(1) == 0).any():
                neg_t = neg_i = torch.arange(bs, device=idx.device)
            else:
                neg_t = torch.multinomial(w_i2t, 1).squeeze(1)
                neg_i = torch.multinomial(w_t2i, 1).squeeze(1)

        text_all = torch.cat([text_embeds, text_embeds[neg_t]], dim=0)
        att_all = torch.cat([attention_mask, attention_mask[neg_t]], dim=0)
        img_all = torch.cat([image_embeds[neg_i], image_embeds], dim=0)
        img_att_all = torch.cat([image_atts[neg_i], image_atts], dim=0)
        neg = self.text_encoder(
            encoder_embeds=text_all, attention_mask=att_all,
            encoder_hidden_states=img_all, encoder_attention_mask=img_att_all, mode="fusion",
        )[:, 0, :]

        logits = self.itm_head(torch.cat([pos, neg], dim=0))
        labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)]
        ).to(logits.device)
        return F.cross_entropy(logits, labels)

    def _rea_loss(self, region_feat: torch.Tensor, text_feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Eq. (3.11)-(3.12): in-batch InfoNCE between F_final and t_cls.

        Unlike ``L_itc`` this term uses neither the momentum queue nor soft
        distillation targets, and it has its own temperature tau_r, so it is the
        sharper local constraint that section 3.2.4 adds on top of the base loss.
        """
        sim = region_feat @ text_feat.t() / self.cfg.rea_temp      # s^rea_ij, Eq. (3.11)
        targets = torch.eq(idx, idx.t()).float()
        targets = targets / targets.sum(1, keepdim=True).clamp(min=1.0)
        l_r2t = -torch.sum(F.log_softmax(sim, dim=1) * targets, dim=1).mean()
        l_t2r = -torch.sum(F.log_softmax(sim.t(), dim=1) * targets, dim=1).mean()
        return (l_r2t + l_t2r) / 2

    # -- pairwise scoring for the Sample protocol and the Full rerank ---------

    @torch.no_grad()
    def score_pairs(self, patches: Dict[int, torch.Tensor], text_seq: torch.Tensor,
                    attention_mask: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Retrieval score of Eq. (3.10)+(3.11) for a batch of aligned pairs.

        ``patches[l]``, ``text_seq``, ``attention_mask`` and ``text_feat`` all
        carry the same leading dimension: row ``n`` is one (image, text)
        candidate pair.
        """
        f_final, _ = self.region_representation(text_seq, attention_mask, patches)
        region_feat = F.normalize(self.region_proj(f_final), dim=-1)
        return (region_feat * text_feat).sum(dim=-1)
