"""Query-conditioned segment re-ranker over frozen MCoT-MVS rankings.

Why this and not SATE
---------------------
SATE pooled a candidate's segments with a *fixed* learned attention query, so it
could only ever produce a better static gallery vector -- still a dual encoder,
still compressing the candidate before it has seen the query. Across three
training regimes it moved global R@1 by at most +0.45 and never improved subset
R@1 at all, which is consistent with that structural limit: subset ranking is a
six-way discrimination among near-identical images, and a query-independent
vector cannot express "which part of this candidate matters for *this* edit".

Here the attention query is the actual query embedding, so the same candidate
pools differently depending on what the modification text asks for. That is the
capability a dual encoder cannot have, and the measured headroom is large: the
frozen model puts the target in its top-10 for 91.9% of val queries but ranks it
first only 55.0% of the time.

The output layer is zero-initialised, so at step 0 the re-ranked order is exactly
the frozen baseline order and the "no re-ranker" ablation row is an identity.
"""

import math

import torch
import torch.nn as nn


class SegmentReranker(nn.Module):
    def __init__(self, dim: int = 1024, head_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, head_dim)
        self.to_k = nn.Linear(dim, head_dim)
        self.to_v = nn.Linear(dim, head_dim)
        self.score = nn.Sequential(
            nn.Linear(3 * head_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.score[2].weight)
        nn.init.zeros_(self.score[2].bias)

    def forward(self, query, seg_padded, seg_mask, base_score):
        """query (B,D); seg_padded (B,C,S,D); seg_mask (B,C,S); base (B,C)."""
        batch, cand, segs, _ = seg_padded.shape
        q = self.to_q(query)                                   # (B,H)
        k = self.to_k(seg_padded)                              # (B,C,S,H)
        v = self.to_v(seg_padded)

        logits = (k @ q[:, None, :, None]).squeeze(-1)         # (B,C,S)
        logits = logits / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~seg_mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * v).sum(dim=2)        # (B,C,H)

        q_exp = q[:, None, :].expand(batch, cand, self.head_dim)
        features = torch.cat([q_exp * pooled, q_exp, pooled], dim=-1)
        delta = self.score(features).squeeze(-1)               # (B,C)
        return base_score + delta


def pad_segments(seg_list, device, max_segments=8):
    """seg_list: list (B) of list (C) of (S_i, D) tensors -> padded batch."""
    batch = len(seg_list)
    cand = len(seg_list[0])
    dim = seg_list[0][0].shape[-1]
    padded = torch.zeros(batch, cand, max_segments, dim, device=device)
    mask = torch.zeros(batch, cand, max_segments, dtype=torch.bool, device=device)
    for b in range(batch):
        for c in range(cand):
            seg = seg_list[b][c]
            n = min(seg.shape[0], max_segments)
            if n > 0:
                padded[b, c, :n] = seg[:n].to(device)
                mask[b, c, :n] = True
            else:
                mask[b, c, 0] = True
    return padded, mask
