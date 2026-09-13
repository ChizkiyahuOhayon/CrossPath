import json
import tempfile
import unittest
from pathlib import Path

from weave_extract_crosspath_dqu_cirr import official_metadata, parse_args, train_dataset


class DQUCIRRExtractionTest(unittest.TestCase):
    def test_train_metadata_contains_target_and_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "captions" / "captions").mkdir(parents=True)
            (root / "captions" / "image_splits").mkdir(parents=True)
            captions = [{
                "reference": "source",
                "target_hard": "target",
                "caption": "make it blue",
                "img_set": {"members": ["source", "target"]},
            }]
            (root / "captions" / "captions" / "cap.rc2.train.json").write_text(
                json.dumps(captions)
            )
            (root / "captions" / "image_splits" / "split.rc2.train.json").write_text(
                json.dumps({"source": "s.png", "target": "t.png"})
            )
            gallery, rows = official_metadata(root, "train")
        self.assertEqual(gallery, ["source", "target"])
        self.assertEqual(rows[0]["target_id"], "target")
        self.assertEqual(rows[0]["group_members"], ["source", "target"])

    def test_parser_accepts_train_split(self):
        args = parse_args([
            "--repo-src", ".",
            "--cirr-path", ".",
            "--checkpoint", "model.pt",
            "--split", "train",
            "--output-dir", "out",
        ])
        self.assertEqual(args.split, "train")

    def test_train_loader_does_not_call_eager_constructor(self):
        class Dataset:
            def __init__(self):
                raise AssertionError("eager constructor must not be called")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "captions" / "captions").mkdir(parents=True)
            (root / "captions" / "image_splits").mkdir(parents=True)
            files = {
                root / "captions" / "captions" / "cap.rc2.train.json": [],
                root / "captions" / "image_splits" / "split.rc2.train.json": {"a": "a.png"},
                root / "image_captions_cirr_train.json": {"a": "caption"},
                root / "keywords_in_mods_cirr_train.json": {"a+b": ["blue"]},
            }
            for path, value in files.items():
                path.write_text(json.dumps(value))
            dataset = train_dataset(Dataset, root, ["train", "val"])
        self.assertEqual(dataset.train_image_name, ["a"])
        self.assertEqual(dataset.transform, ["train", "val"])


if __name__ == "__main__":
    unittest.main()
