"""SATE: Segment-Aware Target Encoder for composed image retrieval.

Motivation
----------
MCoT-MVS composes the query from reference patches, LLM retained/deleted cues
and SAM segment features, but represents every gallery image with a bare CLIP
CLS vector (`extract_target` is a single `encode_image` call). Matching therefore
happens across an asymmetric pair of representations, and our CrossPath probes
showed the gallery side is the weaker one: swapping in another model's gallery
embeddings improved R@10 while swapping the query side did not.

SATE injects segment-level structure into the gallery representation. The fusion
block is zero-initialised, so at step 0 the model is bit-identical to the frozen
baseline: the `no-head` ablation row is an identity, not a separate run.

The gallery stays a single L2-normalised vector, so retrieval remains a dot
product and gallery storage is unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetSegmentHead(nn.Module):
    """Pools per-image SAM segment features and residually injects them.

    The gallery side has no modification text, so segment pooling cannot be
    conditioned on retained/deleted phrases the way the query side is. We use a
    learned attention query instead, which lets the head decide which segments
    carry retrieval-relevant structure.
    """

    def __init__(self, dim: int = 1024, hidden: int = 2048):
        super().__init__()
        self.dim = dim
        self.seg_proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.seg_norm = nn.LayerNorm(dim)
        self.attn_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        # Zero-init the last fusion layer: forward() returns cls exactly at init.
        nn.init.zeros_(self.fuse[2].weight)
        nn.init.zeros_(self.fuse[2].bias)

    def pool_segments(self, seg_feature_list):
        pooled = []
        for seg in seg_feature_list:
            if seg is None or seg.numel() == 0:
                pooled.append(torch.zeros(self.dim, device=self.attn_query.device))
                continue
            seg = seg.to(self.attn_query.device).float()
            projected = self.seg_norm(self.seg_proj(seg))
            scores = projected @ self.attn_query / math.sqrt(self.dim)
            weights = torch.softmax(scores, dim=0).unsqueeze(1)
            pooled.append((projected * weights).sum(dim=0))
        return torch.stack(pooled)

    def forward(self, cls_feature, seg_feature_list):
        pooled = self.pool_segments(seg_feature_list).to(cls_feature.dtype)
        delta = self.fuse(torch.cat([cls_feature, pooled], dim=-1))
        return cls_feature + delta


def build_sate(cir_model, dim: int = 1024, hidden: int = 2048):
    """Attach a SATE head to an already-loaded CIRModel instance."""
    head = TargetSegmentHead(dim=dim, hidden=hidden)
    cir_model.target_segment_head = head
    return cir_model


def extract_target_sate(cir_model, target_image, target_seg_feature_list):
    """Gallery encoding with the SATE head; mirrors CIRModel.extract_target."""
    cls_feature = cir_model.encode_image(target_image)
    fused = cir_model.target_segment_head(cls_feature, target_seg_feature_list)
    return F.normalize(fused, dim=-1)
