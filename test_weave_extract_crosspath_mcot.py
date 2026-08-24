import unittest

from weave_extract_crosspath_mcot import validate_query_alignment


class MCoTCrossPathExtractionTest(unittest.TestCase):
    def test_query_alignment_contract(self):
        rows = [
            {"source_id": "a", "target_id": "b"},
            {"source_id": "c", "target_id": "d"},
        ]
        validate_query_alignment(rows, ["c"], ["d"], offset=1)

    def test_mismatched_query_order_fails(self):
        rows = [{"source_id": "a", "target_id": "b"}]
        with self.assertRaisesRegex(ValueError, "mismatch at 0"):
            validate_query_alignment(rows, ["b"], ["a"])


if __name__ == "__main__":
    unittest.main()
