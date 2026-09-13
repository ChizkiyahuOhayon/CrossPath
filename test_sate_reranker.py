import torch

from sate_reranker import SegmentReranker, pad_segments


def _batch(batch=2, cand=4, segs=3, dim=16, device="cpu"):
    query = torch.randn(batch, dim)
    seg_list = [[torch.randn(torch.randint(1, segs + 1, (1,)).item(), dim)
                 for _ in range(cand)] for _ in range(batch)]
    padded, mask = pad_segments(seg_list, device, max_segments=segs)
    base = torch.randn(batch, cand)
    return query, padded, mask, base


def test_zero_init_reproduces_baseline_scores_exactly():
    model = SegmentReranker(dim=16, head_dim=8, hidden=16)
    query, padded, mask, base = _batch(dim=16)
    out = model(query, padded, mask, base)
    assert torch.allclose(out, base, atol=1e-6), "must start as the frozen ranking"


def test_padding_mask_excludes_absent_segments():
    """A candidate's score must not depend on padded slots."""
    torch.manual_seed(0)
    model = SegmentReranker(dim=16, head_dim=8, hidden=16)
    torch.nn.init.normal_(model.score[2].weight, std=0.2)
    query, padded, mask, base = _batch(dim=16, segs=5)
    out_a = model(query, padded, mask, base)
    noised = padded.clone()
    noised[~mask] = torch.randn_like(noised[~mask]) * 50.0
    out_b = model(query, noised, mask, base)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_pooling_is_query_conditioned():
    """The same candidate must pool differently under different queries."""
    torch.manual_seed(0)
    model = SegmentReranker(dim=16, head_dim=8, hidden=16)
    torch.nn.init.normal_(model.score[2].weight, std=0.2)
    _, padded, mask, base = _batch(batch=1, cand=1, segs=4, dim=16)
    q1 = torch.randn(1, 16)
    q2 = torch.randn(1, 16)
    s1 = model(q1, padded, mask, torch.zeros(1, 1))
    s2 = model(q2, padded, mask, torch.zeros(1, 1))
    assert not torch.allclose(s1, s2, atol=1e-5), "score must depend on the query"


def test_gradient_unblocks_after_first_step():
    model = SegmentReranker(dim=16, head_dim=8, hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    query, padded, mask, base = _batch(dim=16)
    grads = []
    for _ in range(3):
        out = model(query, padded, mask, base)
        loss = torch.nn.functional.cross_entropy(out, torch.zeros(len(out), dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        grads.append(model.to_k.weight.grad.abs().sum().item())
        opt.step()
    assert grads[0] == 0.0
    assert grads[1] > 0.0
