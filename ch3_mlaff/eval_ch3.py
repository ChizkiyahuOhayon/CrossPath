"""Evaluation for chapter 3 (section 3.3.2).

The retrieval score of a candidate pair is the cosine similarity between the
projected region-enhanced representation ``F_final`` of Eq. (3.10) and the
global text embedding.  ``F_final`` depends on both the image and the text, so
the score is pairwise and the two protocols reach it differently:

* **Sample** - 1,000 queries against 101 candidates each; every pair is scored
  directly.
* **Full** - the text-independent CLS embeddings shortlist ``topk_rerank``
  candidates per query, and only those are scored pairwise.  Candidates outside
  the shortlist keep their CLS score, shifted below every reranked one, so a
  positive that the shortlist misses is counted as a miss rather than dropped.

For the plain two-tower baseline (model 1, no cross attention) there is no
``F_final``; the CLS cosine matrix is the score and both protocols read off it
directly, which is the FashionSAP evaluation this chapter's baseline reproduces.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------
# recall computation
# --------------------------------------------------------------------------


def _recall_at(ranks: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
    return {f"R@{k}": float(100.0 * (ranks < k).mean()) for k in ks}


def _summarise(ranks_i2t: np.ndarray, ranks_t2i: np.ndarray,
               ks: Sequence[int]) -> Dict[str, object]:
    i2t, t2i = _recall_at(ranks_i2t, ks), _recall_at(ranks_t2i, ks)
    return {"I2T": i2t, "T2I": t2i, "Mean R@1": (i2t["R@1"] + t2i["R@1"]) / 2}


def full_recall(sim_i2t: np.ndarray, sim_t2i: np.ndarray,
                img2txt: Dict[int, int], txt2img: Dict[int, List[int]],
                ks: Sequence[int] = (1, 5, 10)) -> Dict[str, object]:
    """Recall over the whole validation gallery."""
    ranks_i2t = np.empty(sim_i2t.shape[0])
    for i, row in enumerate(sim_i2t):
        order = np.argsort(-row, kind="stable")
        ranks_i2t[i] = int(np.where(order == img2txt[i])[0][0])

    ranks_t2i = np.empty(sim_t2i.shape[0])
    for i, row in enumerate(sim_t2i):
        order = np.argsort(-row, kind="stable")
        ranks_t2i[i] = int(np.isin(order, np.asarray(txt2img[i])).nonzero()[0][0])

    return _summarise(ranks_i2t, ranks_t2i, ks)


def sample_recall(scores_i2t: np.ndarray, scores_t2i: np.ndarray,
                  ks: Sequence[int] = (1, 5, 10)) -> Dict[str, object]:
    """Recall over the 101-candidate Sample protocol.

    Both arrays are ``[num_queries, 1 + num_negatives]`` candidate scores whose
    last column is the positive, matching the layout that
    :func:`ch3_mlaff.data_fashiongen.sample_protocol_indices` produces.
    """
    def ranks(scores: np.ndarray) -> np.ndarray:
        positive = scores[:, -1][:, None]
        return (scores[:, :-1] > positive).sum(axis=1).astype(float)

    return _summarise(ranks(scores_i2t), ranks(scores_t2i), ks)


def gather_candidate_scores(sim: np.ndarray, queries: np.ndarray,
                            candidates: np.ndarray) -> np.ndarray:
    """Read a candidate matrix out of a dense similarity matrix."""
    return np.take_along_axis(sim[queries], candidates, axis=1)


# --------------------------------------------------------------------------
# corpus encoding
# --------------------------------------------------------------------------


@torch.no_grad()
def encode_corpus(model, image_loader: DataLoader, texts: Sequence[str], tokenizer,
                  device, max_words: int = 180, text_batch: int = 256,
                  amp_dtype=None) -> Dict[str, torch.Tensor]:
    """Encode every gallery image and text once.

    Returns the L2-normalised CLS embeddings plus the text token sequences,
    which the pairwise scorer needs and which are small enough to keep on the
    CPU (7.5k texts x 181 x 768 in bf16 is about 2 GB).
    """
    model.eval()
    autocast = torch.autocast("cuda", dtype=amp_dtype,
                              enabled=amp_dtype is not None and device.type == "cuda")

    cls_feats = []
    for images, _ in image_loader:
        with autocast:
            last = model.visual_encoder.forward_layers(images.to(device, non_blocking=True), [12])[12]
            feat = model.vision_proj(last[:, 0, :])
        cls_feats.append(F.normalize(feat.float(), dim=-1).cpu())

    text_feats, text_seqs, text_masks = [], [], []
    for i in range(0, len(texts), text_batch):
        enc = tokenizer(list(texts[i:i + text_batch]), padding="max_length", truncation=True,
                        max_length=max_words, return_tensors="pt").to(device)
        with autocast:
            seq = model.encode_text(enc.input_ids, enc.attention_mask)
            feat = model.text_proj(seq[:, 0, :])
        text_feats.append(F.normalize(feat.float(), dim=-1).cpu())
        text_seqs.append(seq.to(torch.bfloat16).cpu())
        text_masks.append(enc.attention_mask.cpu())

    return {
        "cls_feat": torch.cat(cls_feats),
        "text_feat": torch.cat(text_feats),
        "text_seq": torch.cat(text_seqs),
        "text_mask": torch.cat(text_masks),
    }


def cls_similarity(corpus: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """``(sim_i2t, sim_t2i)``; sim_i2t is the transpose of sim_t2i (table 3.1)."""
    sim_t2i = (corpus["text_feat"] @ corpus["cls_feat"].T).numpy()
    return sim_t2i.T, sim_t2i


# --------------------------------------------------------------------------
# pairwise scoring
# --------------------------------------------------------------------------


@torch.no_grad()
def score_pairs(model, dataset, corpus: Dict[str, torch.Tensor], pairs: np.ndarray,
                device, pair_batch: int = 256, amp_dtype=None) -> np.ndarray:
    """Score ``pairs`` of ``(image_row, text_col)`` with Eq. (3.10)+(3.11).

    Pairs are visited in image order, so a batch spans only a handful of
    distinct images and the ViT runs about once per image however many
    candidates reference it.  The returned array follows the order of ``pairs``.
    """
    model.eval()
    autocast = torch.autocast("cuda", dtype=amp_dtype,
                              enabled=amp_dtype is not None and device.type == "cuda")

    order = np.argsort(pairs[:, 0], kind="stable")
    out = np.empty(len(pairs), dtype=np.float32)

    for start in range(0, len(order), pair_batch):
        slot = order[start:start + pair_batch]
        uniq, inverse = np.unique(pairs[slot, 0], return_inverse=True)
        images = torch.stack([dataset.image_at(int(r)) for r in uniq]).to(device)
        cols = torch.as_tensor(pairs[slot, 1], dtype=torch.long)
        spread = torch.as_tensor(inverse, device=device)

        with autocast:
            _, patches = model.encode_image(images)
            scores = model.score_pairs(
                {l: v[spread] for l, v in patches.items()},
                corpus["text_seq"][cols].to(device).float(),
                corpus["text_mask"][cols].to(device),
                corpus["text_feat"][cols].to(device),
            )
        out[slot] = scores.float().cpu().numpy()
    return out


def shortlist(sim: np.ndarray, topk: int) -> np.ndarray:
    """Per-query indices of the ``topk`` highest CLS scores, best first."""
    topk = min(topk, sim.shape[1])
    part = np.argpartition(-sim, topk - 1, axis=1)[:, :topk]
    ordered = np.take_along_axis(sim, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, ordered, axis=1)


def merge_reranked(sim: np.ndarray, candidates: np.ndarray,
                   scores: np.ndarray) -> np.ndarray:
    """Put reranked candidates above every candidate the shortlist dropped.

    The shortlist is a recall ceiling: a positive it misses can never be
    retrieved.  Keeping the dropped candidates at their (shifted) CLS score
    rather than discarding them makes that ceiling visible in the metric
    instead of silently changing the gallery size.
    """
    floor = float(scores.min()) - 1.0          # every dropped candidate lands at or below this
    merged = sim - float(sim.max()) + floor
    np.put_along_axis(merged, candidates, scores, axis=1)
    return merged


# --------------------------------------------------------------------------
# the two protocols of section 3.3.2
# --------------------------------------------------------------------------


def _pairs_for(query_rows: np.ndarray, candidates: np.ndarray,
               queries_are_images: bool) -> np.ndarray:
    """Flatten ``[num_queries, k]`` candidates into ``(image_row, text_col)`` pairs."""
    k = candidates.shape[1]
    repeated = np.repeat(query_rows, k)
    flat = candidates.ravel()
    return (np.stack([repeated, flat], axis=1) if queries_are_images
            else np.stack([flat, repeated], axis=1))


def evaluate_sample(model, dataset, corpus, idxs: Dict[str, np.ndarray], device,
                    amp_dtype=None, pair_batch: int = 256) -> Dict[str, object]:
    """Recall under the 101-candidate Sample protocol."""
    sim_i2t, sim_t2i = cls_similarity(corpus)
    if model.cross_blocks is None:
        return sample_recall(
            gather_candidate_scores(sim_i2t, idxs["i2t_img"], idxs["i2t_txt"]),
            gather_candidate_scores(sim_t2i, idxs["t2i_txt"], idxs["t2i_img"]),
        )

    def run(query_rows, candidates, queries_are_images):
        pairs = _pairs_for(query_rows, candidates, queries_are_images)
        scores = score_pairs(model, dataset, corpus, pairs, device, pair_batch, amp_dtype)
        return scores.reshape(candidates.shape)

    return sample_recall(
        run(idxs["i2t_img"], idxs["i2t_txt"], queries_are_images=True),
        run(idxs["t2i_txt"], idxs["t2i_img"], queries_are_images=False),
    )


def evaluate_full(model, dataset, corpus, img2txt: Dict[int, int],
                  txt2img: Dict[int, List[int]], device, amp_dtype=None,
                  topk: int = 128, pair_batch: int = 256) -> Dict[str, object]:
    """Recall over the whole gallery, reranking the CLS shortlist."""
    sim_i2t, sim_t2i = cls_similarity(corpus)
    if model.cross_blocks is None:
        return full_recall(sim_i2t, sim_t2i, img2txt, txt2img)

    def run(sim, queries_are_images):
        candidates = shortlist(sim, topk)
        pairs = _pairs_for(np.arange(sim.shape[0]), candidates, queries_are_images)
        scores = score_pairs(model, dataset, corpus, pairs, device, pair_batch, amp_dtype)
        return merge_reranked(sim, candidates, scores.reshape(candidates.shape))

    return full_recall(run(sim_i2t, True), run(sim_t2i, False), img2txt, txt2img)
