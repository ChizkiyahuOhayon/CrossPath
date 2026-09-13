import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from weave_train_cirr_router import calibrate, main, ranking_metrics


class TrainCIRRRouterTest(unittest.TestCase):
    def test_calibration_can_choose_conservative_fallback(self):
        logits = np.asarray([[0.0, 1.0], [0.0, 0.1]])
        global_ranks = np.asarray([[6, 5], [5, 6]])
        subset_ranks = np.asarray([[2, 1], [1, 2]])
        result = calibrate(logits, global_ranks, subset_ranks)
        actions = result[3]
        np.testing.assert_array_equal(actions, [1, 0])
        self.assertEqual(result[0], 100.0)

    def test_ranking_metrics_matches_cirr_average(self):
        global_ranks = np.asarray([[1, 6], [6, 5]])
        subset_ranks = np.asarray([[1, 2], [2, 1]])
        result = ranking_metrics(global_ranks, subset_ranks, np.asarray([0, 1]))
        self.assertEqual(result["R@5"], 100.0)
        self.assertEqual(result["subset_R@1"], 100.0)
        self.assertEqual(result["Avg"], 100.0)

    def test_one_epoch_cpu_run_writes_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "output"
            cache.mkdir()
            rng = np.random.default_rng(3)
            np.save(cache / "features.npy", rng.normal(size=(12, 4)).astype(np.float32))
            global_ranks = np.tile(np.asarray([[6, 5]], dtype=np.uint16), (12, 1))
            subset_ranks = np.tile(np.asarray([[2, 1]], dtype=np.uint16), (12, 1))
            np.save(cache / "global_ranks.npy", global_ranks)
            np.save(cache / "subset_ranks.npy", subset_ranks)
            np.save(cache / "partitions.npy", np.asarray([0] * 6 + [1] * 3 + [2] * 3, dtype=np.uint8))
            (cache / "manifest.json").write_text(json.dumps({"actions": ["mcot", "alt"]}))
            main([
                "--train-cache", str(cache),
                "--output-dir", str(output),
                "--epochs", "1",
                "--batch-size", "3",
                "--hidden-width", "4",
                "--device", "cpu",
            ])
            report = json.loads((output / "report.json").read_text())
            checkpoint_exists = (output / "router.pt").exists()
        self.assertTrue(checkpoint_exists)
        self.assertEqual(report["internal_test"]["queries"], 3)


if __name__ == "__main__":
    unittest.main()
