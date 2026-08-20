import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from weave_extract_crosspath_dqu import select_gallery_ids, validate_alignment


class DQUCrossPathExtractionTest(unittest.TestCase):
    def test_original_split_uses_complete_validation_gallery(self):
        specs = [{"source_id": "a", "target_id": "b"}]
        with tempfile.TemporaryDirectory() as directory:
            split_dir = Path(directory) / "image_splits"
            split_dir.mkdir()
            (split_dir / "split.dress.val.json").write_text(
                json.dumps(["c", "a", "b"])
            )
            gallery = select_gallery_ids(
                specs, directory, "dress", "val", "original-split"
            )
        self.assertEqual(gallery, ["a", "b", "c"])

    def test_endpoint_alignment_contract(self):
        endpoint = (
            ["a", "b"],
            np.zeros((2, 4), dtype=np.float32),
            [{"dataset": "d", "source_id": "a", "target_id": "b"}],
            np.zeros((1, 4), dtype=np.float32),
        )
        validate_alignment(endpoint, endpoint)

    def test_mismatched_metadata_fails(self):
        base = (
            ["a", "b"],
            np.zeros((2, 4)),
            [{"source_id": "a", "target_id": "b"}],
            np.zeros((1, 4)),
        )
        correction = (
            ["a", "b"],
            np.zeros((2, 4)),
            [{"source_id": "b", "target_id": "a"}],
            np.zeros((1, 4)),
        )
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_alignment(base, correction)


if __name__ == "__main__":
    unittest.main()
