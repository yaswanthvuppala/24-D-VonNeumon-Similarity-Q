import unittest

import numpy as np

from src.feature.entropy_descriptor import (
    compute_image_similarity_entropy_descriptor,
    extract_all_entropy_descriptors,
    extract_entropy_descriptor,
)


class ConfigurableDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(121, dtype=np.float64).reshape(11, 11) + 1

    def test_five_by_five_matches_legacy_24d(self):
        legacy = extract_entropy_descriptor(self.image, (5, 5), dimension=24)
        generic = extract_entropy_descriptor(self.image, (5, 5), n=5, y=3)
        np.testing.assert_allclose(generic, legacy)
        self.assertEqual(generic.shape, (24,))

    def test_seven_by_seven_produces_48_values(self):
        descriptor = extract_entropy_descriptor(self.image, (5, 5), dimension=9, n=7, y=3)
        self.assertEqual(descriptor.shape, (48,))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_generic_unit_entropy_uses_the_inner_window_size(self):
        descriptor = extract_entropy_descriptor(self.image, (5, 5), n=7, y=5, norm="unit_entropy")
        self.assertTrue(np.all((descriptor >= 0) & (descriptor <= 1)))

    def test_batch_shape(self):
        descriptors = extract_all_entropy_descriptors(self.image, [(3, 3), (7, 7)], n=7, y=3)
        self.assertEqual(descriptors.shape, (2, 48))

    def test_image_similarity_reports_generic_dimension(self):
        result = compute_image_similarity_entropy_descriptor(
            self.image, self.image, keypoints_a=[(5, 5)], keypoints_b=[(5, 5)], n=7, y=3
        )
        self.assertEqual(result["dimension"], 48)

    def test_invalid_window_parameters(self):
        for n, y in ((5, None), (4, 3), (5, 4), (3, 5)):
            with self.subTest(n=n, y=y):
                with self.assertRaises(ValueError):
                    extract_entropy_descriptor(self.image, (5, 5), n=n, y=y)

    def test_legacy_dimensions_unchanged(self):
        self.assertEqual(extract_entropy_descriptor(self.image, (5, 5), dimension=9).shape, (9,))
        self.assertEqual(extract_entropy_descriptor(self.image, (5, 5), dimension=24).shape, (24,))


if __name__ == "__main__":
    unittest.main()
