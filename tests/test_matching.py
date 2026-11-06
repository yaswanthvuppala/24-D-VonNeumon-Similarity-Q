import unittest
import numpy as np
import cv2

# Import matching and ransac utilities
from src.feature.hpatches_adapter import (
    compute_corner_error,
    compute_homography_auc,
    format_ripe_comparison_table,
    scale_homography,
)


def mnn_match(desc_a: np.ndarray, desc_b: np.ndarray, ratio=None):
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    a2 = np.sum(desc_a ** 2, axis=1, keepdims=True)
    b2 = np.sum(desc_b ** 2, axis=1, keepdims=True)
    d2 = np.maximum(a2 + b2.T - 2.0 * np.dot(desc_a, desc_b.T), 0.0)
    a_to_b, b_to_a = d2.argmin(1), d2.argmin(0)
    ia = np.arange(len(desc_a))
    keep = b_to_a[a_to_b] == ia
    if ratio is not None and len(desc_b) >= 2:
        two = np.partition(d2, 1, axis=1)[:, :2]
        keep &= two[:, 0] / np.maximum(two[:, 1], 1e-12) < (ratio * ratio)
    pairs = np.column_stack((ia[keep], a_to_b[keep])).astype(np.int32)
    dists = np.sqrt(d2[pairs[:, 0], pairs[:, 1]]).astype(np.float32) if len(pairs) else np.empty(0, np.float32)
    return pairs, dists


def homography_ransac(xy_a: np.ndarray, xy_b: np.ndarray, matches: np.ndarray, threshold=3.0):
    if len(matches) < 4:
        return None, np.zeros(len(matches), bool)
    pts_a = xy_a[matches[:, 0]].astype(np.float32)
    pts_b = xy_b[matches[:, 1]].astype(np.float32)
    try:
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, threshold)
    except cv2.error:
        return None, np.zeros(len(matches), bool)
    return H, np.zeros(len(matches), bool) if mask is None else mask.ravel().astype(bool)


def repeatability(xy_a: np.ndarray, xy_b: np.ndarray, H: np.ndarray, threshold=3.0):
    if len(xy_a) == 0 or len(xy_b) == 0:
        return 0.0
    homogeneous = np.c_[xy_a, np.ones(len(xy_a))] @ H.T
    denom = np.where(np.abs(homogeneous[:, 2:3]) < 1e-12, 1e-12, homogeneous[:, 2:3])
    projected = homogeneous[:, :2] / denom
    distance = np.linalg.norm(projected[:, None] - xy_b[None], axis=2)
    return float((distance.min(axis=1) <= threshold).sum() / max(1, min(len(xy_a), len(xy_b))))


class MatchingAndEvaluationTests(unittest.TestCase):
    def test_mnn_match_empty(self):
        desc = np.ones((5, 24), dtype=np.float32)
        pairs, dists = mnn_match(np.empty((0, 24), dtype=np.float32), desc)
        self.assertEqual(pairs.shape, (0, 2))
        self.assertEqual(len(dists), 0)

        pairs, dists = mnn_match(desc, np.empty((0, 24), dtype=np.float32))
        self.assertEqual(pairs.shape, (0, 2))
        self.assertEqual(len(dists), 0)

    def test_mnn_match_exact(self):
        desc_a = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        desc_b = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        pairs, dists = mnn_match(desc_a, desc_b)
        self.assertEqual(len(pairs), 3)
        # Expected pairs: 0 -> 1, 1 -> 0, 2 -> 2
        expected_map = {0: 1, 1: 0, 2: 2}
        for ia, ib in pairs:
            self.assertEqual(expected_map[ia], ib)
        np.testing.assert_allclose(dists, [0.0, 0.0, 0.0], atol=1e-5)

    def test_homography_ransac_insufficient_matches(self):
        xy = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32)
        matches = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        H, mask = homography_ransac(xy, xy, matches)
        self.assertIsNone(H)
        self.assertEqual(len(mask), 3)
        self.assertFalse(mask.any())

    def test_homography_ransac_identity(self):
        xy_a = np.array([[10, 10], [50, 10], [50, 50], [10, 50], [30, 30]], dtype=np.int32)
        xy_b = xy_a.copy()
        matches = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32)
        H, mask = homography_ransac(xy_a, xy_b, matches)
        self.assertIsNotNone(H)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-3)
        self.assertTrue(mask.all())

    def test_repeatability_identity(self):
        xy = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
        rep = repeatability(xy, xy, np.eye(3), threshold=1.0)
        self.assertEqual(rep, 1.0)

        rep_empty = repeatability(np.empty((0, 2)), xy, np.eye(3))
        self.assertEqual(rep_empty, 0.0)

    def test_compute_corner_error_identity(self):
        err = compute_corner_error(np.eye(3), np.eye(3), (480, 640))
        self.assertAlmostEqual(err, 0.0, places=5)

    def test_compute_homography_auc_empty(self):
        auc = compute_homography_auc([], [1, 3, 5])
        self.assertEqual(auc["AUC@1px"], 0.0)
        self.assertEqual(auc["AUC@3px"], 0.0)
        self.assertEqual(auc["AUC@5px"], 0.0)

    def test_format_ripe_comparison_table(self):
        tbl = format_ripe_comparison_table(35.0, 55.0, 65.0, model_name="TestModel", dim=24)
        self.assertIn("TestModel", tbl)
        self.assertIn("35.00%", tbl)
        self.assertIn("55.00%", tbl)
        self.assertIn("65.00%", tbl)


if __name__ == "__main__":
    unittest.main()
