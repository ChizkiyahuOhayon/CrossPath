import unittest

import torch

from weave_eval_cirr_procrustes import orthogonal_map


class CIRRProcrustesTest(unittest.TestCase):
    def test_recovers_orthogonal_alignment(self):
        torch.manual_seed(5)
        source = torch.nn.functional.normalize(torch.randn(30, 4), dim=1)
        rotation, _ = torch.linalg.qr(torch.randn(4, 4))
        target = source @ rotation
        fitted = orthogonal_map(source, target)
        torch.testing.assert_close(source @ fitted, target, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fitted.T @ fitted, torch.eye(4), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
