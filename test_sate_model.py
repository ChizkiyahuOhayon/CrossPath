"""Unit tests for the SATE target head.

The identity property at initialisation is the load-bearing claim: it makes the
"no head" ablation exact rather than a separate training run.
"""

import torch

from sate_model import TargetSegmentHead


def test_head_is_identity_at_initialisation():
    head = TargetSegmentHead(dim=8, hidden=16)
    cls = torch.randn(3, 8)
    segs = [torch.randn(5, 8), torch.randn(2, 8), torch.randn(9, 8)]
    out = head(cls, segs)
    assert torch.allclose(out, cls, atol=1e-6), "head must start as identity"


def test_head_departs_from_identity_once_trained():
    head = TargetSegmentHead(dim=8, hidden=16)
    torch.nn.init.normal_(head.fuse[2].weight, std=0.1)
    cls = torch.randn(3, 8)
    segs = [torch.randn(5, 8), torch.randn(2, 8), torch.randn(9, 8)]
    out = head(cls, segs)
    assert not torch.allclose(out, cls, atol=1e-6)


def test_variable_segment_counts_and_empty_segments():
    head = TargetSegmentHead(dim=8, hidden=16)
    cls = torch.randn(3, 8)
    segs = [torch.randn(1, 8), torch.empty(0, 8), torch.randn(12, 8)]
    out = head(cls, segs)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_attention_pooling_is_permutation_invariant():
    head = TargetSegmentHead(dim=8, hidden=16)
    torch.nn.init.normal_(head.fuse[2].weight, std=0.1)
    cls = torch.randn(1, 8)
    seg = torch.randn(6, 8)
    a = head(cls, [seg])
    b = head(cls, [seg[torch.randperm(6)]])
    assert torch.allclose(a, b, atol=1e-5), "segment order must not matter"


def test_zero_init_blocks_upstream_gradient_only_on_first_step():
    """Zero-init means step 0 cannot reach seg_proj; step 1 onward must."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    head = TargetSegmentHead(dim=16, hidden=32)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    q = F.normalize(torch.randn(8, 16), dim=-1)
    cls = F.normalize(torch.randn(8, 16), dim=-1)
    segs = [torch.randn(3, 16) for _ in range(8)]

    grads = []
    for _ in range(3):
        t = F.normalize(head(cls, segs), dim=-1)
        loss = F.cross_entropy(100.0 * q @ t.T, torch.arange(8))
        opt.zero_grad()
        loss.backward()
        grads.append(head.seg_proj[0].weight.grad.abs().sum().item())
        opt.step()

    assert grads[0] == 0.0, "step 0 must not reach the segment projection"
    assert grads[1] > 0.0, "gradient must unblock once fuse[2] leaves zero"
    assert grads[2] > 0.0
