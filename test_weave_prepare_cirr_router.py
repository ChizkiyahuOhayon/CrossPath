import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from weave_prepare_cirr_router import main


class PrepareCIRRRouterTest(unittest.TestCase):
    def test_small_cpu_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings"
            output = root / "cache"
            embeddings.mkdir()
            gallery_ids = [f"g{index}" for index in range(12)]
            (embeddings / "gallery_ids.json").write_text(json.dumps(gallery_ids))
            rows = [
                {"source_id": "g0", "target_id": "g1", "group_members": ["g0", "g1", "g2"]},
                {"source_id": "g3", "target_id": "g4", "group_members": ["g3", "g4", "g5"]},
            ]
            (embeddings / "queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            rng = np.random.default_rng(7)
            for name, shape in (
                ("base_gallery", (12, 3)),
                ("correction_gallery", (12, 3)),
                ("base_queries", (2, 3)),
                ("correction_queries", (2, 3)),
            ):
                np.save(embeddings / f"{name}.npy", rng.normal(size=shape).astype(np.float32))
            main([
                "--embedding-dir", str(embeddings),
                "--output-dir", str(output),
                "--alphas", "0.25",
                "--batch-size", "2",
                "--device", "cpu",
            ])
            features = np.load(output / "features.npy")
            global_ranks = np.load(output / "global_ranks.npy")
        self.assertEqual(features.shape, (2, 44))
        self.assertEqual(global_ranks.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
