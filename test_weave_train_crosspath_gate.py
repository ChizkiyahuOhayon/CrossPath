import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from weave_build_crosspath_cache import compute_query_cache, pack_query_caches
from weave_train_crosspath_gate import (
    RecordSource,
    evaluate_nll,
    infer_split,
    pair_partition,
    policy_metrics,
    policy_report,
    train_width,
)


class TrainCrossPathGateTest(unittest.TestCase):
    def test_pair_partition_keeps_reverse_pair_together(self):
        forward = {"dataset": "d", "source_id": "a", "target_id": "b"}
        reverse = {"dataset": "d", "source_id": "b", "target_id": "a"}
        self.assertEqual(pair_partition(forward, 7), pair_partition(reverse, 7))

    def test_policy_metrics_uses_one_action_for_all_cutoffs(self):
        target_ranks = np.asarray([[11, 1, 6], [1, 8, 12]])
        selected_actions = np.asarray([2, 0])
        metrics = policy_metrics(target_ranks, selected_actions, cutoffs=(1, 5, 10))
        self.assertEqual(metrics, {"R@1": 50.0, "R@5": 50.0, "R@10": 100.0})

    def test_policy_report_exposes_path_ceiling_without_changing_policy(self):
        bundle = {
            "target_ranks": np.asarray([[11, 1], [1, 12]]),
            "realized_utilities": np.asarray([[0.0, 1.0], [0.0, -2.0]]),
        }
        report = policy_report(bundle, np.asarray([0, 0]), regression_cost=2.0)

        self.assertEqual(report["metrics"], report["base_metrics"])
        self.assertEqual(report["objective_oracle_metrics"]["R@1"], 100.0)
        self.assertEqual(report["per_cutoff_path_ceiling"]["R@1"], 100.0)
        self.assertEqual(report["objective_oracle_mean_realized_utility"], 0.5)

    def test_small_cpu_training_pipeline(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir = root / "embeddings"
            cache_dir = root / "cache"
            embedding_dir.mkdir()
            cache_dir.mkdir()
            query_count, gallery_size, dimension = 60, 12, 4
            gallery_ids = [f"p{index:02d}" for index in range(gallery_size)]
            base_gallery = rng.normal(size=(gallery_size, dimension)).astype(np.float32)
            correction_gallery = rng.normal(size=(gallery_size, dimension)).astype(np.float32)
            base_queries = rng.normal(size=(query_count, dimension)).astype(np.float32)
            correction_queries = rng.normal(size=(query_count, dimension)).astype(np.float32)
            for name, array in (
                ("base_gallery", base_gallery),
                ("correction_gallery", correction_gallery),
                ("base_queries", base_queries),
                ("correction_queries", correction_queries),
            ):
                np.save(embedding_dir / f"{name}.npy", array)
            (embedding_dir / "gallery_ids.json").write_text(json.dumps(gallery_ids))
            with (embedding_dir / "queries.jsonl").open("w") as handle:
                for index in range(query_count):
                    handle.write(
                        json.dumps(
                            {
                                "query_index": index,
                                "dataset": "d",
                                "source_id": f"s{index:03d}",
                                "target_id": gallery_ids[index % gallery_size],
                            }
                        )
                        + "\n"
                    )

            base_gallery_norm = base_gallery / np.linalg.norm(base_gallery, axis=1, keepdims=True)
            correction_gallery_norm = correction_gallery / np.linalg.norm(
                correction_gallery, axis=1, keepdims=True
            )
            caches = []
            for index in range(query_count):
                base_query = base_queries[index] / np.linalg.norm(base_queries[index])
                correction_query = correction_queries[index] / np.linalg.norm(
                    correction_queries[index]
                )
                caches.append(
                    compute_query_cache(
                        np.asarray(gallery_ids),
                        base_query @ base_gallery_norm.T,
                        correction_query @ correction_gallery_norm.T,
                        target_index=index % gallery_size,
                    )
                )
            np.savez_compressed(
                cache_dir / "shard_00000.npz",
                **pack_query_caches(range(query_count), caches, (1, 5, 10)),
            )
            (cache_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "queries": query_count,
                        "alphas": np.linspace(0.0, 1.0, 9).tolist(),
                        "cutoffs": [1, 5, 10],
                        "shards": [{"path": "shard_00000.npz"}],
                    }
                )
            )

            args = Namespace(
                seed=13,
                learning_rate=1e-3,
                weight_decay=1e-4,
                epochs=1,
                batch_records=16,
                grad_clip=1.0,
            )
            source = RecordSource(embedding_dir, cache_dir, "cpu", args.seed)
            gate, history = train_width(source, width=4, args=args)
            self.assertTrue(np.isfinite(history[-1]["calibration_nll"]))
            self.assertTrue(np.isfinite(evaluate_nll(gate, source, "test", 16)))
            bundle = infer_split(gate, source, "test", regression_cost=2.0)
            self.assertEqual(bundle["target_ranks"].shape[1], 9)

            zero_source = RecordSource(
                embedding_dir, cache_dir, "cpu", args.seed, zero_trace=True
            )
            shard_path = zero_source.shard_paths[0]
            with np.load(shard_path) as shard:
                _, features, _, membership = zero_source.features(shard, 0, 5)
            trace_start = 4 * zero_source.embedding_dim + 2
            trace_stop = trace_start + len(zero_source.alphas)
            self.assertTrue((features[:, trace_start:trace_stop] == 0).all())
            self.assertGreater(float(membership.sum()), 0.0)

            query_only = RecordSource(
                embedding_dir,
                cache_dir,
                "cpu",
                args.seed,
                feature_mode="query_only",
            )
            with np.load(query_only.shard_paths[0]) as shard:
                _, features, _, _ = query_only.features(shard, 0, 5)
            d = query_only.embedding_dim
            self.assertTrue((features[:, d : 4 * d + 2 + len(query_only.alphas)] == 0).all())

            margin_only = RecordSource(
                embedding_dir,
                cache_dir,
                "cpu",
                args.seed,
                feature_mode="margin_only",
            )
            with np.load(margin_only.shard_paths[0]) as shard:
                _, features, _, membership = margin_only.features(shard, 0, 5)
            self.assertTrue((features[:, : 4 * d] == 0).all())
            self.assertGreater(float(features[:, 4 * d : 4 * d + 2].abs().sum()), 0.0)
            self.assertTrue(
                (features[:, 4 * d + 2 : 4 * d + 2 + len(margin_only.alphas)] == 0).all()
            )
            self.assertGreater(float(membership.sum()), 0.0)

    def test_manifest_cutoffs_control_training_records(self):
        self.assertEqual(policy_metrics(
            np.asarray([[60, 40], [8, 12]]),
            np.asarray([1, 0]),
            cutoffs=(1, 10, 50),
        ), {"R@1": 0.0, "R@10": 50.0, "R@50": 100.0})


if __name__ == "__main__":
    unittest.main()
