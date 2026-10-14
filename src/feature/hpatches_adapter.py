"""
HPatches Benchmark Adapter for QSED 24-D Von Neumann Entropy Descriptor.

Provides utility functions to bridge the QSED descriptor pipeline with the
HPatches dataset format:
  - Image loading (.ppm format, grayscale, aspect-preserving resize)
  - Ground-truth homography loading and scaling for resized images
  - RANSAC homography estimation from matched keypoints
  - Corner reprojection error computation
  - AUC metric computation at multiple pixel thresholds
  - Dataset directory scanning and sequence/pair enumeration

HPatches Dataset Structure:
    hpatches-sequences-release/
    ├── i_ajuntament/          # illumination change sequence
    │   ├── 1.ppm              # reference image
    │   ├── 2.ppm ... 6.ppm   # target images
    │   ├── H_1_2 ... H_1_6   # 3×3 ground-truth homographies (text files)
    ├── v_adam/                # viewpoint change sequence
    └── ...                    # 116 sequences total
"""

import os
import glob
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any


# =============================================================================
# 1. IMAGE LOADING
# =============================================================================

def load_hpatches_image(
    path: str,
    max_side: int = 480
) -> Tuple[np.ndarray, float]:
    """
    Load an HPatches .ppm image as grayscale and resize preserving aspect ratio.

    The longest side is capped at `max_side` pixels. If the image is already
    smaller, no resizing is applied and scale=1.0 is returned.

    Args:
        path: Absolute or relative path to a .ppm image file.
        max_side: Maximum allowed length of the longest side (default: 480).

    Returns:
        Tuple of:
            - image: 2D float64 grayscale array, values in [0, 255].
            - scale: Scale factor applied (1.0 if no resize was needed).

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")

    h, w = img.shape
    scale = max_side / max(h, w)

    if scale < 1.0:
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    return img.astype(np.float64), scale


# =============================================================================
# 2. HOMOGRAPHY LOADING & SCALING
# =============================================================================

def load_homography(path: str) -> np.ndarray:
    """
    Load a 3×3 ground-truth homography matrix from an HPatches text file.

    HPatches stores each H as 9 whitespace-separated floats (row-major).

    Args:
        path: Path to the homography file (e.g., 'H_1_2').

    Returns:
        np.ndarray: 3×3 float64 homography matrix.
    """
    H = np.loadtxt(path).reshape(3, 3)
    return H.astype(np.float64)


def scale_homography(
    H: np.ndarray,
    scale_ref: float,
    scale_tgt: float
) -> np.ndarray:
    """
    Adjust a homography matrix for images that have been resized.

    If the reference image was scaled by `scale_ref` and the target image
    by `scale_tgt`, the adjusted homography maps points in the resized
    reference to points in the resized target:

        H_scaled = S_tgt @ H @ inv(S_ref)

    where S = diag([scale, scale, 1]).

    Args:
        H: Original 3×3 homography (maps original ref → original tgt).
        scale_ref: Scale factor applied to the reference image.
        scale_tgt: Scale factor applied to the target image.

    Returns:
        np.ndarray: Adjusted 3×3 homography for the resized images.
    """
    S_ref = np.diag([scale_ref, scale_ref, 1.0])
    S_tgt = np.diag([scale_tgt, scale_tgt, 1.0])
    return S_tgt @ H @ np.linalg.inv(S_ref)


# =============================================================================
# 3. HOMOGRAPHY ESTIMATION (RANSAC)
# =============================================================================

def estimate_homography_ransac(
    kps_ref: List[Any],
    kps_tgt: List[Any],
    matches: List[Tuple[int, int, float]],
    ransac_thresh: float = 3.0
) -> Optional[np.ndarray]:
    """
    Estimate a homography from matched keypoints using RANSAC.

    Args:
        kps_ref: List of reference keypoints (objects with .x, .y attributes).
        kps_tgt: List of target keypoints (objects with .x, .y attributes).
        matches: List of (idx_ref, idx_tgt, distance) tuples from descriptor matching.
        ransac_thresh: RANSAC reprojection threshold in pixels (default: 3.0).

    Returns:
        3×3 estimated homography matrix, or None if fewer than 4 matches.
    """
    if len(matches) < 4:
        return None

    pts_ref = np.float32([[kps_ref[m[0]].x, kps_ref[m[0]].y] for m in matches])
    pts_tgt = np.float32([[kps_tgt[m[1]].x, kps_tgt[m[1]].y] for m in matches])

    H_est, mask = cv2.findHomography(pts_ref, pts_tgt, cv2.RANSAC, ransac_thresh)

    if H_est is None or H_est.shape != (3, 3):
        return None

    return H_est


# =============================================================================
# 4. EVALUATION METRICS
# =============================================================================

def compute_corner_error(
    H_gt: np.ndarray,
    H_est: np.ndarray,
    img_shape: Tuple[int, int]
) -> float:
    """
    Compute average corner reprojection error between ground-truth and estimated homographies.

    Transforms the four corners of the reference image with both H_gt and H_est,
    then computes the mean Euclidean distance between the two sets of warped corners.

    Args:
        H_gt: 3×3 ground-truth homography.
        H_est: 3×3 estimated homography.
        img_shape: (height, width) of the reference image.

    Returns:
        Average corner reprojection error in pixels. Returns float('inf') if
        the perspective transform fails.
    """
    h, w = img_shape
    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ]).reshape(-1, 1, 2)

    try:
        corners_gt = cv2.perspectiveTransform(corners, H_gt)
        corners_est = cv2.perspectiveTransform(corners, H_est)
        error = float(np.mean(np.linalg.norm(corners_gt - corners_est, axis=2)))
    except cv2.error:
        error = float('inf')

    return error


def compute_homography_auc(
    errors: List[float],
    thresholds: List[int] = None
) -> Dict[str, float]:
    """
    Compute homography estimation accuracy (AUC) at multiple pixel thresholds.

    For each threshold t, the metric is the percentage of image pairs whose
    average corner error is below t pixels.

    Args:
        errors: List of corner reprojection errors (one per image pair).
        thresholds: List of pixel thresholds (default: [1, 3, 5, 10]).

    Returns:
        Dict mapping threshold labels to accuracy percentages, e.g.:
        {'AUC@1px': 23.5, 'AUC@3px': 55.2, 'AUC@5px': 68.1, 'AUC@10px': 82.3}
    """
    if thresholds is None:
        thresholds = [1, 3, 5, 10]

    if len(errors) == 0:
        return {f"AUC@{t}px": 0.0 for t in thresholds}

    errors_arr = np.array(errors)
    results = {}
    for t in thresholds:
        correct = np.sum(errors_arr < t)
        results[f"AUC@{t}px"] = float(correct / len(errors_arr) * 100.0)

    return results


# =============================================================================
# 5. DATASET ENUMERATION
# =============================================================================

def get_hpatches_sequences(dataset_dir: str) -> List[Dict[str, str]]:
    """
    Scan the HPatches dataset directory for all sequence folders.

    Args:
        dataset_dir: Path to the extracted hpatches-sequences-release directory.

    Returns:
        Sorted list of dicts, each with keys:
            - 'name': Sequence folder name (e.g., 'i_ajuntament')
            - 'path': Absolute path to the sequence folder
            - 'type': 'illumination' (i_*) or 'viewpoint' (v_*)

    Raises:
        FileNotFoundError: If dataset_dir does not exist.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"HPatches dataset directory not found: {dataset_dir}\n"
            f"Download from: https://github.com/hpatches/hpatches-sequences\n"
            f"Or run: git clone https://github.com/hpatches/hpatches-sequences.git {dataset_dir}"
        )

    sequences = []
    for entry in sorted(os.listdir(dataset_dir)):
        seq_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(seq_path):
            continue

        # Only include i_* (illumination) and v_* (viewpoint) sequences
        if entry.startswith("i_"):
            seq_type = "illumination"
        elif entry.startswith("v_"):
            seq_type = "viewpoint"
        else:
            continue

        # Verify it contains the expected files
        ref_img = os.path.join(seq_path, "1.ppm")
        if not os.path.isfile(ref_img):
            continue

        sequences.append({
            "name": entry,
            "path": os.path.abspath(seq_path),
            "type": seq_type,
        })

    if len(sequences) == 0:
        raise FileNotFoundError(
            f"No valid HPatches sequences found in: {dataset_dir}\n"
            f"Expected folders like i_ajuntament/, v_adam/ containing 1.ppm etc."
        )

    return sequences


def get_sequence_pairs(seq_path: str) -> List[Dict[str, str]]:
    """
    Get the 5 image pairs for a single HPatches sequence.

    Each sequence has a reference image (1.ppm) paired with 5 target images
    (2.ppm through 6.ppm), with corresponding ground-truth homographies.

    Args:
        seq_path: Absolute path to a sequence folder (e.g., .../i_ajuntament/).

    Returns:
        List of 5 dicts, each with keys:
            - 'ref_path': Path to reference image (1.ppm)
            - 'tgt_path': Path to target image (k.ppm)
            - 'H_path': Path to ground-truth homography file (H_1_k)
            - 'pair_name': Descriptive name (e.g., '1-3')
    """
    pairs = []
    ref_path = os.path.join(seq_path, "1.ppm")

    for k in range(2, 7):
        tgt_path = os.path.join(seq_path, f"{k}.ppm")
        h_path = os.path.join(seq_path, f"H_1_{k}")

        if os.path.isfile(tgt_path) and os.path.isfile(h_path):
            pairs.append({
                "ref_path": ref_path,
                "tgt_path": tgt_path,
                "H_path": h_path,
                "pair_name": f"1-{k}",
            })

    return pairs
