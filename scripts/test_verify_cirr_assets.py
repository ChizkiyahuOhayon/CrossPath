import json
import tempfile
import unittest
from pathlib import Path

from verify_cirr_assets import verify_split


class VerifyCIRRAssetsTest(unittest.TestCase):
    def test_missing_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cirr" / "captions").mkdir(parents=True)
            (root / "cirr" / "image_splits").mkdir(parents=True)
            captions = [
                {
                    "reference": "ref",
                    "target_hard": "target",
                    "img_set": {"members": ["ref", "target"]},
                }
            ]
            images = {"ref": "dev/ref.png", "target": "dev/target.png"}
            (root / "cirr" / "captions" / "cap.rc2.val.json").write_text(json.dumps(captions))
            (root / "cirr" / "image_splits" / "split.rc2.val.json").write_text(json.dumps(images))
            segment_root = root / "segments"
            for name in images:
                path = segment_root / name / "seg_feature.pt"
                path.parent.mkdir(parents=True)
                path.touch()
            with self.assertRaisesRegex(ValueError, "image files are missing"):
                verify_split(root, segment_root, "val")


if __name__ == "__main__":
    unittest.main()
