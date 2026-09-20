"""Stage-1 domain pre-training wrapper (section 3.3.1).

Section 3.3.1 specifies the schedule of the pre-training stage but not its
objective beyond "domain pre-training of the image/text encoders".  We use the
ALBEF/FashionSAP pre-training objective that the chapter's baseline inherits:
image-text contrastive + image-text matching + masked language modelling on the
FashionGen product descriptions.  The chapter-3 modules are *not* present in
this stage; they are added when the retrieval stage loads these weights.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_ch3 import Ch3Config, Ch3Model


class MLMHead(nn.Module):
    """BERT masked-LM head on top of the fusion encoder output."""

    def __init__(self, width: int, vocab_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(width, width)
        self.LayerNorm = nn.LayerNorm(width, eps=eps)
        self.decoder = nn.Linear(width, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.LayerNorm(F.gelu(self.dense(hidden))))


class Ch3PretrainModel(nn.Module):
    """Ch3Model (baseline configuration) + MLM head."""

    def __init__(self, base: Ch3Model, vocab_size: int):
        super().__init__()
        if base.cfg.use_cross_attn or base.cfg.use_rea:
            raise ValueError("stage 1 pre-trains the plain two-tower baseline")
        self.base = base
        self.mlm_head = MLMHead(base.cfg.width, vocab_size)

    @property
    def cfg(self) -> Ch3Config:
        return self.base.cfg

    def forward(self, image, input_ids, attention_mask, mlm_ids, mlm_labels, idx, alpha=0.0):
        out = self.base(image, input_ids, attention_mask, idx, alpha=alpha)

        image_embeds = self.base.encode_image(image)[0]
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image.device)
        masked_embeds = self.base.encode_text(mlm_ids, attention_mask)
        fused = self.base.text_encoder(
            encoder_embeds=masked_embeds, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, mode="fusion",
        )
        logits = self.mlm_head(fused)
        loss_mlm = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), mlm_labels.view(-1), ignore_index=-100
        )
        if torch.isnan(loss_mlm):           # a batch where nothing was masked
            loss_mlm = logits.new_zeros(())

        out["loss"] = out["loss"] + loss_mlm
        out["loss_mlm"] = loss_mlm.detach()
        return out

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Weights handed to stage 2: encoders, projections, ITM head, temperature."""
        keep = ("visual_encoder.", "text_encoder.", "vision_proj.", "text_proj.",
                "itm_head.", "temp")
        return {k: v for k, v in self.base.state_dict().items() if k.startswith(keep)}
