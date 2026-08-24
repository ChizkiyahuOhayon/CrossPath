import numpy as np
import torch

from weave_train_composition_crosspath import CompositionCrossPath, original_query, parse_args


def test_composition_head_starts_at_original_lambda():
    model = CompositionCrossPath(dim=4, hidden_dim=3)
    text = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    visual = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    mixing = torch.rand(5, 1)
    query, output_mixing = model(text, visual, mixing)
    expected = torch.nn.functional.normalize(mixing * text + (1 - mixing) * visual, dim=-1)
    assert torch.allclose(query, expected)
    assert torch.allclose(output_mixing, mixing.squeeze(-1))


def test_original_query_uses_saved_scalar():
    arrays = {
        "text": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "visual": np.asarray([[0.0, 1.0]], dtype=np.float32),
        "original_lambda": np.asarray([[0.25]], dtype=np.float32),
    }
    expected = torch.nn.functional.normalize(torch.tensor([[0.25, 0.75]]), dim=-1)
    assert torch.allclose(original_query(arrays), expected)


def test_parser_allows_full_train_refit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--train-features",
            str(tmp_path),
            "--official-features",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--val-percent",
            "0",
        ],
    )
    assert parse_args().val_percent == 0
