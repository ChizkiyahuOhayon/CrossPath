import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from weave_extract_crosspath_embeddings import (
    FashionGenTrainProductDataset,
    FashionGenTrainQueryDataset,
    align_endpoint_metadata,
    build_gallery,
)


class CrossPathEmbeddingExportTest(unittest.TestCase):
    def test_fashiongen_train_datasets_preserve_order_and_filter_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            image_root = root / "images"
            data_dir.mkdir()
            rows = [
                {
                    "dataset": "fashiongen_train",
                    "source_id": "source-b",
                    "target_id": "target-b",
                    "modification_text_short": "make it blue",
                },
                {
                    "dataset": "fashiongen_train",
                    "source_id": "missing",
                    "target_id": "target-b",
                    "modification_text_short": "missing source",
                },
                {
                    "dataset": "fashiongen_train",
                    "source_id": "source-a",
                    "target_id": "target-a",
                    "modification_text_short": "make it red",
                },
            ]
            with (data_dir / "train_triplets.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            for product_id in ("source-a", "source-b", "target-a", "target-b"):
                product_dir = image_root / "fashiongen_train" / product_id
                product_dir.mkdir(parents=True)
                Image.new("RGB", (2, 2), color="white").save(product_dir / "0.jpg")

            queries = FashionGenTrainQueryDataset(data_dir, image_root)
            self.assertEqual([row["source_id"] for row in queries.samples], ["source-b", "source-a"])
            self.assertEqual(queries[0]["modification_text_short"], "make it blue")
            self.assertEqual(queries[0]["source_images"][0].size, (2, 2))

            products = FashionGenTrainProductDataset(queries.samples)
            self.assertEqual(
                [row["product_id"] for row in products.samples],
                ["source-a", "source-b", "target-a", "target-b"],
            )
            self.assertEqual(products[0]["images"][0].size, (2, 2))

    def test_gallery_is_deduplicated_and_sorted_by_id(self):
        product_ids = ["b", "a", "b"]
        product_embeddings = [
            np.asarray([2.0]),
            np.asarray([1.0]),
            np.asarray([20.0]),
        ]
        metadata = [
            {"source_id": "c", "target_id": "a", "dataset": "fashiongen_val"},
            {"source_id": "a", "target_id": "b", "dataset": "fashiongen_val"},
        ]
        source_embeddings = [np.asarray([3.0]), np.asarray([10.0])]

        ids, embeddings = build_gallery(
            product_ids, product_embeddings, metadata, source_embeddings
        )

        self.assertEqual(ids, ["a", "b", "c"])
        np.testing.assert_array_equal(embeddings[:, 0], [1.0, 2.0, 3.0])

    def test_endpoint_metadata_must_match_exactly(self):
        row = {"dataset": "fashiongen_val", "source_id": "s", "target_id": "t"}
        align_endpoint_metadata([row], [dict(row)])
        with self.assertRaises(ValueError):
            align_endpoint_metadata([row], [{**row, "target_id": "other"}])


if __name__ == "__main__":
    unittest.main()
