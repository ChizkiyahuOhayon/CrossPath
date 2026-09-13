import json
from unittest.mock import patch

import numpy as np
import torch

from sate_eval_crosspath_reranker import make_submission


def test_make_submission_excludes_source_and_formats_both_rankings(tmp_path):
    gallery_ids = ["source", "a", "b", "c", "d"]
    (tmp_path / "gallery_ids.json").write_text(json.dumps(gallery_ids))
    row = {
        "pair_id": 7,
        "source_id": "source",
        "group_members": ["source", "a", "b", "c"],
    }
    (tmp_path / "queries.jsonl").write_text(json.dumps(row) + "\n")
    np.save(tmp_path / "queries.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    gallery = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        dtype=np.float32,
    )
    np.save(tmp_path / "gallery.npy", gallery)
    dqu_gallery = tmp_path / "dqu_gallery.npy"
    np.save(dqu_gallery, gallery)

    with patch(
        "sate_eval_crosspath_reranker.rerank_scores",
        side_effect=lambda _model, _queries, _indices, _features, base, *_args: torch.as_tensor(base),
    ):
        general, subset = make_submission(
            object(), tmp_path, dqu_gallery, tmp_path / "segments", 0.25,
            torch.device("cpu"), 4, 1, 8,
        )

    assert general["version"] == "rc2"
    assert general["7"] == ["a", "b", "c", "d"]
    assert subset["version"] == "rc2"
    assert subset["7"] == ["a", "b", "c"]
