import torch

from weave_train_relational_crosspath import RelationalCrossPath


def test_relational_crosspath_starts_as_identity():
    model = RelationalCrossPath(dim=4, hidden_dim=3, max_step=1.0)
    query = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    source = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    fused, relation, step = model(query, source)
    assert torch.allclose(step, torch.zeros_like(step))
    assert torch.allclose(relation, query)
    assert torch.allclose(fused, query)


def test_step_is_bounded():
    model = RelationalCrossPath(dim=4, hidden_dim=3, max_step=0.25)
    torch.nn.init.constant_(model.step_head[-1].bias, 100.0)
    query = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    source = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    _, _, step = model(query, source)
    assert torch.all(step <= 0.25)
    assert torch.all(step >= -0.25)
