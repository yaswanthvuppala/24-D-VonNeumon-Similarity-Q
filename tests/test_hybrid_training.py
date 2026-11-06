import unittest

try:
    import torch
    from src.feature.hybrid_training import split_hpatches_sequences, detector_supervision_loss
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional outside the Hybrid training environment")
class HybridTrainingTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_reproducible(self):
        sequences = [
            {"name": f"i_{index}", "type": "illumination", "path": str(index)}
            for index in range(57)
        ] + [
            {"name": f"v_{index}", "type": "viewpoint", "path": str(index)}
            for index in range(59)
        ]
        first = split_hpatches_sequences(sequences, seed=7)
        second = split_hpatches_sequences(sequences, seed=7)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (81, 18, 17))
        self.assertEqual(len(set(sum(first.as_dict().values(), []))), 116)

    def test_detector_loss_is_finite_and_has_gradients(self):
        logits = torch.zeros((1, 8, 8), requires_grad=True)
        targets = torch.zeros((1, 8, 8))
        targets[:, 3, 3] = 1.0
        loss = detector_supervision_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
