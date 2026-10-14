"""
Quantum 24-Dimensional Local Von Neumann Entropy Feature Descriptor.

This module implements a local image-similarity feature descriptor based on
Von Neumann entropy computed from a 5x5 neighborhood surrounding detected keypoints.

Architecture & Formulation:
1. For every detected keypoint KP at (x, y):
   A 5x5 window centered at KP contains 25 pixels:
   - 1 center pixel = KP (excluded from the descriptor)
   - 24 surrounding pixels P1, P2, ..., P24 in a fixed, deterministic ordering.
2. For EACH pixel Pi in {P1, ..., P24}:
   Pi serves as the center of its own 3x3 local neighborhood (9 pixels).
   For example, for P7 (offset dy=-1, dx=-1 relative to KP):
       P1   P2   P3
       P6   P7   P8
       P11  P12  KP
3. Reflection Padding:
   Boundary keypoints are preserved using NumPy-style reflection padding:
       np.pad(image, pad_width=3, mode="reflect")
   For a 1D sequence [P1, P2, P3, P4], reflection padding produces:
       [P3, P2 | P1, P2, P3, P4 | P3, P2]
   The boundary pixel itself is NOT duplicated.
4. Quantum Density Matrix & Von Neumann Entropy:
   Each 3x3 patch (9 pixels) is converted into a density matrix rho using the
   existing QSED method (build_intensity_density_matrix):
       q = (v + epsilon) / sum(v + epsilon)
       rho = diag(q)
   which is Hermitian, positive semi-definite, and has Tr(rho) = 1.
   Von Neumann entropy is computed using the existing QSED function:
       S(rho) = -Tr(rho log2 rho) = -sum(lambda_i log2(lambda_i))
   yielding entropy Si in [0, log2(9)] bits.
5. 24-D Feature Vector:
   F(KP) = [S1, S2, ..., S24] in R^24.
6. Matching & Image Similarity:
   Mutual Nearest Neighbor matching on 24-D descriptors between Image A and Image B.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse existing QSED density matrix and Von Neumann entropy routines
from .density_matrix import build_intensity_density_matrix, verify_density_matrix
from .von_neumann import von_neumann_entropy
from .keypoint_detection import Keypoint


# =============================================================================
# 1. FIXED 5x5 NEIGHBORHOOD POSITION ORDERING
# =============================================================================

# Offsets (dy, dx) relative to KP for the 24 surrounding positions in row-major order:
#
#   (-2,-2)=P1   (-2,-1)=P2   (-2, 0)=P3   (-2,+1)=P4   (-2,+2)=P5
#   (-1,-2)=P6   (-1,-1)=P7   (-1, 0)=P8   (-1,+1)=P9   (-1,+2)=P10
#   ( 0,-2)=P11  ( 0,-1)=P12   CENTER=KP   ( 0,+1)=P13  ( 0,+2)=P14
#   (+1,-2)=P15  (+1,-1)=P16  (+1, 0)=P17  (+1,+1)=P18  (+1,+2)=P19
#   (+2,-2)=P20  (+2,-1)=P21  (+2, 0)=P22  (+2,+1)=P23  (+2,+2)=P24

P_LABELS_RASTER: List[str] = [f"P{i}" for i in range(1, 25)]

P_OFFSETS_DICT: Dict[str, Tuple[int, int]] = {
    "P1":  (-2, -2), "P2":  (-2, -1), "P3":  (-2,  0), "P4":  (-2,  1), "P5":  (-2,  2),
    "P6":  (-1, -2), "P7":  (-1, -1), "P8":  (-1,  0), "P9":  (-1,  1), "P10": (-1,  2),
    "P11": ( 0, -2), "P12": ( 0, -1),                   "P13": ( 0,  1), "P14": ( 0,  2),
    "P15": ( 1, -2), "P16": ( 1, -1), "P17": ( 1,  0), "P18": ( 1,  1), "P19": ( 1,  2),
    "P20": ( 2, -2), "P21": ( 2, -1), "P22": ( 2,  0), "P23": ( 2,  1), "P24": ( 2,  2),
}

# Reverse lookup: (dy, dx) -> label
OFFSET_TO_LABEL: Dict[Tuple[int, int], str] = {v: k for k, v in P_OFFSETS_DICT.items()}
OFFSET_TO_LABEL[(0, 0)] = "KP"

# Canonical list of offsets in raster order (length 24)
P_OFFSETS_RASTER: List[Tuple[int, int]] = [P_OFFSETS_DICT[name] for name in P_LABELS_RASTER]

# Optional Clockwise Perimeter Ordering:
# Outer perimeter (16 pixels) followed by Inner perimeter (8 pixels), starting at top-left:
# Outer: P1 -> P2 -> P3 -> P4 -> P5 -> P10 -> P14 -> P19 -> P24 -> P23 -> P22 -> P21 -> P20 -> P15 -> P11 -> P6
# Inner: P7 -> P8 -> P9 -> P13 -> P18 -> P17 -> P16 -> P12
P_LABELS_CLOCKWISE: List[str] = [
    # Outer ring (clockwise from top-left)
    "P1", "P2", "P3", "P4", "P5",
    "P10", "P14", "P19", "P24",
    "P23", "P22", "P21", "P20",
    "P15", "P11", "P6",
    # Inner ring (clockwise from top-left)
    "P7", "P8", "P9",
    "P13", "P18",
    "P17", "P16", "P12"
]
P_OFFSETS_CLOCKWISE: List[Tuple[int, int]] = [P_OFFSETS_DICT[name] for name in P_LABELS_CLOCKWISE]

# 3x3 local neighborhood relative offsets (row, col)
LOCAL_3X3_OFFSETS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 0), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

# 9-D Sliding Window: 9 valid positions of a 3x3 window sliding across a 5x5 matrix (stride 1)
# Center coordinates relative to KP:
#   Top-Left=(-1,-1) [P7], Top-Center=(-1,0) [P8], Top-Right=(-1,1) [P9]
#   Mid-Left=(0,-1) [P12], Center=(0,0) [KP],      Mid-Right=(0,1) [P13]
#   Bot-Left=(1,-1) [P16], Bot-Center=(1,0) [P17], Bot-Right=(1,1) [P18]
SLIDING_9_LABELS: List[str] = [
    "W1_TL", "W2_TC", "W3_TR",
    "W4_ML", "W5_CENTER", "W6_MR",
    "W7_BL", "W8_BC", "W9_BR"
]
SLIDING_9_OFFSETS: List[Tuple[int, int]] = [
    (-1, -1), (-1,  0), (-1,  1),
    ( 0, -1), ( 0,  0), ( 0,  1),
    ( 1, -1), ( 1,  0), ( 1,  1)
]


# =============================================================================
# 2. NEIGHBORHOOD VERIFICATION & HELPER FUNCTIONS
# =============================================================================

def get_p_neighborhood_labels(p_name: str) -> List[List[str]]:
    """
    Return the 3x3 matrix of position labels centered at Pi.

    For example, for P7, returns:
    [['P1', 'P2', 'P3'],
     ['P6', 'P7', 'P8'],
     ['P11', 'P12', 'KP']]

    Args:
        p_name: Position name, e.g., 'P7'.

    Returns:
        3x3 nested list of string labels for each pixel in Pi's 3x3 neighborhood.
    """
    if p_name not in P_OFFSETS_DICT:
        raise ValueError(f"Unknown position name: {p_name}. Expected P1..P24.")

    dy_pi, dx_pi = P_OFFSETS_DICT[p_name]
    grid_labels = []

    for d_row in (-1, 0, 1):
        row_labels = []
        for d_col in (-1, 0, 1):
            target_offset = (dy_pi + d_row, dx_pi + d_col)
            label = OFFSET_TO_LABEL.get(target_offset, f"Ext({target_offset[0]},{target_offset[1]})")
            row_labels.append(label)
        grid_labels.append(row_labels)

    return grid_labels


def apply_reflection_padding(image: np.ndarray, pad_width: int = 3) -> np.ndarray:
    """
    Apply reflection padding to image boundaries using NumPy-style reflection.

    Design rationale:
    For [P1, P2, P3, P4], reflection padding produces:
        [P3, P2 | P1, P2, P3, P4 | P3, P2]
    The boundary pixel itself is NOT duplicated.
    This preserves continuity at image boundaries and allows complete 3x3 neighborhoods
    for all 24 positions around boundary keypoints without introducing artificial zero or
    constant step-discontinuities.

    Args:
        image: 2D grayscale image array of shape (H, W).
        pad_width: Number of padding pixels on each side (default: 3).

    Returns:
        Padded 2D image array of shape (H + 2*pad_width, W + 2*pad_width).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    return np.pad(image, pad_width=pad_width, mode="reflect")


# =============================================================================
# 3. 24-D VON NEUMANN ENTROPY DESCRIPTOR EXTRACTION
# =============================================================================

def extract_single_p_entropy(
    padded_image: np.ndarray,
    kp_y_pad: int,
    kp_x_pad: int,
    dy_pi: int,
    dx_pi: int,
    epsilon: float = 1e-10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Extract the 3x3 neighborhood centered at Pi = (kp_y + dy_pi, kp_x + dx_pi),
    construct the quantum density matrix rho, and calculate Von Neumann entropy S(rho).

    Args:
        padded_image: Reflection-padded 2D image.
        kp_y_pad: Keypoint row coordinate in padded image.
        kp_x_pad: Keypoint col coordinate in padded image.
        dy_pi: Row offset of Pi relative to KP.
        dx_pi: Col offset of Pi relative to KP.
        epsilon: Small numerical stability constant for density matrix construction.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            - entropy: Von Neumann entropy S(rho) in bits.
            - rho: 9x9 quantum density matrix.
            - patch_3x3: 3x3 pixel intensity patch.
    """
    # Center of Pi in padded coordinate system
    pi_y = kp_y_pad + dy_pi
    pi_x = kp_x_pad + dx_pi

    # Extract complete 3x3 neighborhood centered at Pi
    patch_3x3 = padded_image[pi_y - 1 : pi_y + 2, pi_x - 1 : pi_x + 2]

    # Build density matrix using QSED's canonical intensity-based formulation
    rho = build_intensity_density_matrix(patch_3x3, epsilon=epsilon)

    # Calculate Von Neumann entropy S(rho) = -Tr(rho log2 rho)
    entropy = von_neumann_entropy(rho)

    return float(entropy), rho, patch_3x3


def extract_entropy_descriptor(
    image: np.ndarray,
    keypoint: Union[Keypoint, Tuple[int, int], List[int]],
    ordering: str = "raster",
    norm: str = "none",
    pad_width: int = 3,
    epsilon: float = 1e-10,
    dimension: int = 24
) -> np.ndarray:
    """
    Calculate the Von Neumann entropy descriptor for a single keypoint (9-D or 24-D).

    Architectures:
    - dimension = 9: Sliding 3x3 window moved across the 5x5 matrix with stride 1 (9 valid sub-windows).
      Centers: [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)] relative to KP.
      Captures the complete 5x5 neighborhood including the center KP patch.
    - dimension = 24: 24 probe positions P1..P24 surrounding KP (center KP excluded).
      Each Pi is the center of its own 3x3 patch (extends into 7x7 receptive field).

    Args:
        image: 2D input grayscale image.
        keypoint: Keypoint object (with .x, .y attributes) or (x, y) tuple/list.
        ordering: Position ordering: 'raster' (default) or 'clockwise' (for 24-D).
        norm: Normalization strategy ('none', 'unit_entropy', 'l2').
        pad_width: Padding width for boundary reflection padding (default: 3).
        epsilon: Stability constant for density matrix construction.
        dimension: Descriptor dimension (9 or 24, default: 24).

    Returns:
        np.ndarray: Descriptor vector of shape (dimension,).
    """
    # Extract coordinates (x = col, y = row)
    if hasattr(keypoint, 'x') and hasattr(keypoint, 'y'):
        x, y = int(keypoint.x), int(keypoint.y)
    else:
        x, y = int(keypoint[0]), int(keypoint[1])

    padded = apply_reflection_padding(image, pad_width=pad_width)
    kp_y_pad = y + pad_width
    kp_x_pad = x + pad_width

    # Select offsets based on dimension
    if dimension == 9:
        offsets = SLIDING_9_OFFSETS
    elif dimension == 24:
        if ordering == "raster":
            offsets = P_OFFSETS_RASTER
        elif ordering == "clockwise":
            offsets = P_OFFSETS_CLOCKWISE
        else:
            raise ValueError(f"Unknown ordering '{ordering}'. Expected 'raster' or 'clockwise'.")
    else:
        raise ValueError(f"Unsupported descriptor dimension: {dimension}. Choose 9 or 24.")

    descriptor = np.zeros(dimension, dtype=np.float64)
    for i, (dy, dx) in enumerate(offsets):
        s, _, _ = extract_single_p_entropy(padded, kp_y_pad, kp_x_pad, dy, dx, epsilon=epsilon)
        descriptor[i] = s

    descriptor = apply_descriptor_normalization(descriptor, norm=norm)
    return descriptor


def extract_all_entropy_descriptors(
    image: np.ndarray,
    keypoints: List[Union[Keypoint, Tuple[int, int]]],
    ordering: str = "raster",
    norm: str = "none",
    pad_width: int = 3,
    epsilon: float = 1e-10,
    dimension: int = 24
) -> np.ndarray:
    """
    Extract Von Neumann entropy descriptors (9-D or 24-D) for a list of N keypoints.

    Args:
        image: 2D grayscale image.
        keypoints: List of Keypoint objects or (x, y) coordinates.
        ordering: 'raster' (default) or 'clockwise'.
        norm: 'none' (default), 'unit_entropy', or 'l2'.
        pad_width: Padding width (default: 3).
        epsilon: Stability constant.
        dimension: Descriptor dimension (9 or 24, default: 24).

    Returns:
        np.ndarray: Feature matrix of shape (N, dimension).
    """
    if len(keypoints) == 0:
        return np.empty((0, dimension), dtype=np.float64)

    padded = apply_reflection_padding(image, pad_width=pad_width)

    if dimension == 9:
        offsets = SLIDING_9_OFFSETS
    elif dimension == 24:
        if ordering == "raster":
            offsets = P_OFFSETS_RASTER
        elif ordering == "clockwise":
            offsets = P_OFFSETS_CLOCKWISE
        else:
            raise ValueError(f"Unknown ordering '{ordering}'. Expected 'raster' or 'clockwise'.")
    else:
        raise ValueError(f"Unsupported descriptor dimension: {dimension}. Choose 9 or 24.")

    n_kps = len(keypoints)
    descriptors = np.zeros((n_kps, dimension), dtype=np.float64)

    for idx, kp in enumerate(keypoints):
        if hasattr(kp, 'x') and hasattr(kp, 'y'):
            x, y = int(kp.x), int(kp.y)
        else:
            x, y = int(kp[0]), int(kp[1])

        kp_y_pad = y + pad_width
        kp_x_pad = x + pad_width

        for i, (dy, dx) in enumerate(offsets):
            s, _, _ = extract_single_p_entropy(padded, kp_y_pad, kp_x_pad, dy, dx, epsilon=epsilon)
            descriptors[idx, i] = s

        if norm != "none":
            descriptors[idx] = apply_descriptor_normalization(descriptors[idx], norm=norm)

    return descriptors


def apply_descriptor_normalization(descriptor: np.ndarray, norm: str = "none") -> np.ndarray:
    """
    Apply explicit, documented normalization to a 24-D entropy descriptor.

    Design rationale:
    - 'none' (default): Entropy values for 9x9 density matrices are already naturally bounded
      in [0, log2(9)] = [0, 3.1699] bits. They are directly comparable across all 24 positions
      and across images without arbitrary scaling.
    - 'unit_entropy': Normalizes each entropy value by S_max = log2(9), placing all values in [0, 1].
    - 'l2': Normalizes the 24-D vector to unit L2 length (||v||_2 = 1).

    Args:
        descriptor: Array of shape (24,).
        norm: Normalization mode.

    Returns:
        Normalized array of shape (24,).
    """
    if norm == "none":
        return descriptor
    elif norm == "unit_entropy":
        s_max = np.log2(9.0)
        return descriptor / s_max
    elif norm == "l2":
        l2_norm = np.linalg.norm(descriptor)
        if l2_norm > 1e-12:
            return descriptor / l2_norm
        return descriptor
    else:
        raise ValueError(f"Unknown normalization mode: '{norm}'. Choose 'none', 'unit_entropy', or 'l2'.")


# =============================================================================
# 4. DESCRIPTOR MATCHING & IMAGE-LEVEL SIMILARITY
# =============================================================================

def compute_pairwise_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix between descriptors from Image A and Image B.

    Args:
        desc_a: Descriptors of Image A of shape (N_A, 24).
        desc_b: Descriptors of Image B of shape (N_B, 24).

    Returns:
        np.ndarray: Distance matrix of shape (N_A, N_B).
    """
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return np.empty((desc_a.shape[0], desc_b.shape[0]), dtype=np.float64)

    # Vectorized Euclidean distance computation
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b
    diff = desc_a[:, np.newaxis, :] - desc_b[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def match_entropy_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    method: str = "mutual_nn",
    max_distance: Optional[float] = None
) -> Dict[str, Any]:
    """
    Match 24-dimensional descriptors between two images.
    Does NOT assume Image A and Image B have the same number of keypoints.

    Matching strategies:
    - 'mutual_nn': Mutual Nearest Neighbor (cross-check). Keypoint i in A matches j in B
      if and only if j is the NN of i AND i is the NN of j.
    - 'nn_a_to_b': For every keypoint in A, find its nearest neighbor in B.
    - 'nn_b_to_a': For every keypoint in B, find its nearest neighbor in A.

    Args:
        desc_a: Feature matrix for Image A of shape (N_A, 24).
        desc_b: Feature matrix for Image B of shape (N_B, 24).
        method: Matching method ('mutual_nn', 'nn_a_to_b', 'nn_b_to_a').
        max_distance: Optional distance threshold to filter out low-confidence matches.

    Returns:
        Dict containing:
            - 'matches': List of tuples (idx_a, idx_b, distance)
            - 'distances': Array of matched distances
            - 'mean_distance': Average distance among matches
            - 'num_matches': Total number of matches found
            - 'distance_matrix': Full pairwise distance matrix of shape (N_A, N_B)
    """
    n_a = desc_a.shape[0]
    n_b = desc_b.shape[0]

    if n_a == 0 or n_b == 0:
        return {
            "matches": [],
            "distances": np.array([], dtype=np.float64),
            "mean_distance": float('inf'),
            "num_matches": 0,
            "distance_matrix": np.empty((n_a, n_b), dtype=np.float64)
        }

    dist_matrix = compute_pairwise_distances(desc_a, desc_b)

    matches = []
    if method == "greedy":
        # Greedy one-to-one matching: pairs with lowest distance matched first
        flat_indices = np.argsort(dist_matrix, axis=None)
        used_a = set()
        used_b = set()
        for idx in flat_indices:
            i = int(idx // n_b)
            j = int(idx % n_b)
            if i not in used_a and j not in used_b:
                dist = float(dist_matrix[i, j])
                if max_distance is not None and dist > max_distance:
                    break
                matches.append((i, j, dist))
                used_a.add(i)
                used_b.add(j)
                if len(used_a) == n_a or len(used_b) == n_b:
                    break

    elif method == "mutual_nn":
        # Nearest neighbor of each keypoint in A -> index in B
        nn_a_to_b = np.argmin(dist_matrix, axis=1)
        # Nearest neighbor of each keypoint in B -> index in A
        nn_b_to_a = np.argmin(dist_matrix, axis=0)

        for i in range(n_a):
            j = nn_a_to_b[i]
            # Match if mutual NN or if identical zero-distance pair
            if nn_b_to_a[j] == i or (dist_matrix[i, j] < 1e-12 and i == j):
                dist = float(dist_matrix[i, j])
                if max_distance is None or dist <= max_distance:
                    matches.append((i, int(j), dist))

    elif method == "nn_a_to_b":
        nn_a_to_b = np.argmin(dist_matrix, axis=1)
        for i in range(n_a):
            j = nn_a_to_b[i]
            dist = float(dist_matrix[i, j])
            if max_distance is None or dist <= max_distance:
                matches.append((i, int(j), dist))

    elif method == "nn_b_to_a":
        nn_b_to_a = np.argmin(dist_matrix, axis=0)
        for j in range(n_b):
            i = nn_b_to_a[j]
            dist = float(dist_matrix[i, j])
            if max_distance is None or dist <= max_distance:
                matches.append((int(i), j, dist))

    else:
        raise ValueError(f"Unknown matching method: {method}")

    matched_distances = np.array([m[2] for m in matches], dtype=np.float64)
    mean_dist = float(np.mean(matched_distances)) if len(matched_distances) > 0 else float('inf')

    return {
        "matches": matches,
        "distances": matched_distances,
        "mean_distance": mean_dist,
        "num_matches": len(matches),
        "distance_matrix": dist_matrix
    }


def compute_image_similarity_entropy_descriptor(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a: Optional[List[Any]] = None,
    keypoints_b: Optional[List[Any]] = None,
    matching_method: str = "greedy",
    norm: str = "none",
    ordering: str = "raster",
    dimension: int = 24
) -> Dict[str, Any]:
    """
    Full end-to-end comparison between Image A and Image B using the Von Neumann descriptor (9-D or 24-D).

    Pipeline:
    Image A -> detect/use keypoints -> calculate entropy descriptors (9-D or 24-D)
    Image B -> detect/use keypoints -> calculate entropy descriptors (9-D or 24-D)
    Match descriptors between Image A and Image B (handles differing keypoint counts).
    Compute distance metrics and image-level similarity score.

    Args:
        image_a: 2D grayscale array of Image A.
        image_b: 2D grayscale array of Image B.
        keypoints_a: Optional pre-detected keypoints for Image A.
                     If None, detects keypoints using QSED edge/score pipeline.
        keypoints_b: Optional pre-detected keypoints for Image B.
        matching_method: 'mutual_nn' or 'greedy' (default).
        norm: Normalization strategy ('none', 'unit_entropy', 'l2').
        ordering: Position ordering ('raster' or 'clockwise').
        dimension: Descriptor dimension (9 or 24, default: 24).

    Returns:
        Dict containing:
            - 'similarity_score': Overall similarity in [0, 100]%
            - 'mean_descriptor_distance': Average Euclidean distance of matched keypoints
            - 'dimension': 9 or 24
            - 'num_keypoints_a': Count of keypoints in Image A
            - 'num_keypoints_b': Count of keypoints in Image B
            - 'num_matches': Number of corresponding keypoints
            - 'match_ratio': Matches / min(num_keypoints_a, num_keypoints_b)
            - 'raw_matches': List of (idx_a, idx_b, distance)
            - 'distance_matrix': Full pairwise distance matrix (N_A, N_B)
            - 'descriptors_a': (N_A, dimension) array
            - 'descriptors_b': (N_B, dimension) array
            - 'keypoints_a': Keypoints used for Image A
            - 'keypoints_b': Keypoints used for Image B
    """
    # 1. Detect keypoints if not provided
    if keypoints_a is None:
        keypoints_a = detect_default_keypoints(image_a)
    if keypoints_b is None:
        keypoints_b = detect_default_keypoints(image_b)

    # 2. Extract descriptors for both images
    desc_a = extract_all_entropy_descriptors(image_a, keypoints_a, ordering=ordering, norm=norm, dimension=dimension)
    desc_b = extract_all_entropy_descriptors(image_b, keypoints_b, ordering=ordering, norm=norm, dimension=dimension)

    n_a = len(keypoints_a)
    n_b = len(keypoints_b)

    if n_a == 0 or n_b == 0:
        return {
            "similarity_score": 0.0,
            "mean_descriptor_distance": float('inf'),
            "dimension": dimension,
            "num_keypoints_a": n_a,
            "num_keypoints_b": n_b,
            "num_matches": 0,
            "match_ratio": 0.0,
            "raw_matches": [],
            "distance_matrix": np.empty((n_a, n_b), dtype=np.float64),
            "descriptors_a": desc_a,
            "descriptors_b": desc_b,
            "keypoints_a": keypoints_a,
            "keypoints_b": keypoints_b
        }

    # 3. Match descriptors
    match_result = match_entropy_descriptors(desc_a, desc_b, method=matching_method)
    num_matches = match_result["num_matches"]
    mean_dist = match_result["mean_distance"]
    min_kps = min(n_a, n_b)
    match_ratio = num_matches / min_kps if min_kps > 0 else 0.0

    # 4. Image-level similarity score:
    max_theoretical_dist = np.sqrt(float(dimension)) * np.log2(9.0) if norm == "none" else np.sqrt(float(dimension))

    if num_matches > 0:
        sigma = 1.0 if dimension == 9 else 1.5
        dist_quality = float(np.mean(np.exp(-match_result["distances"] / sigma)))
        similarity_score = (0.70 * dist_quality + 0.30 * match_ratio) * 100.0
        similarity_score = float(np.clip(similarity_score, 0.0, 100.0))
    else:
        similarity_score = 0.0

    return {
        "similarity_score": similarity_score,
        "mean_descriptor_distance": mean_dist,
        "dimension": dimension,
        "num_keypoints_a": n_a,
        "num_keypoints_b": n_b,
        "num_matches": num_matches,
        "match_ratio": match_ratio,
        "raw_matches": match_result["matches"],
        "distance_matrix": match_result["distance_matrix"],
        "descriptors_a": desc_a,
        "descriptors_b": desc_b,
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b
    }


def detect_default_keypoints(
    image: np.ndarray,
    max_keypoints: int = 100,
    grad_threshold_ratio: float = 0.02,
    variation_threshold: float = 0.02
) -> List[Keypoint]:
    """
    Detect salient keypoints using QSED directional variation & keypoint score pipeline.
    Falls back gracefully if edge response is sparse.

    Args:
        image: 2D grayscale image.
        max_keypoints: Maximum number of keypoints to return.
        grad_threshold_ratio: Minimum gradient magnitude ratio.
        variation_threshold: Minimum directional variation ratio.

    Returns:
        List of Keypoint objects.
    """
    from .directional_variation import compute_directional_variation, compute_keypoint_score
    from .keypoint_detection import detect_keypoints
    from ..classical.sobel_operator import apply_sobel_masks
    from ..classical.gradient import compute_gradient_magnitude

    # Compute 8-direction Sobel responses
    dir_grads = apply_sobel_masks(image)
    grad_mag, dom_dir = compute_gradient_magnitude(dir_grads)

    # Edge mask
    m_max = np.max(grad_mag)
    edge_mask = (grad_mag >= 0.02 * m_max).astype(np.uint8)

    # Directional variation & Keypoint score
    dir_var = compute_directional_variation(dom_dir, grad_mag, neighborhood_size=3)
    kp_score = compute_keypoint_score(grad_mag, dir_var)

    kps = detect_keypoints(
        grad_mag, dom_dir, edge_mask, dir_var, kp_score,
        grad_threshold_ratio=grad_threshold_ratio,
        variation_threshold=variation_threshold,
        min_distance=3.0,
        max_keypoints=max_keypoints
    )

    kps.sort(key=lambda k: k.keypoint_score, reverse=True)
    if len(kps) > max_keypoints:
        kps = kps[:max_keypoints]

    # If very few keypoints detected, add grid corner points to guarantee coverage
    if len(kps) < 5:
        h, w = image.shape
        step = max(8, min(h, w) // 8)
        for y in range(4, h - 4, step):
            for x in range(4, w - 4, step):
                kps.append(Keypoint(
                    x=int(x), y=int(y),
                    dominant_direction=float(dom_dir[y, x] * 22.5),
                    gradient_magnitude=float(grad_mag[y, x]),
                    directional_variation=float(dir_var[y, x]),
                    keypoint_score=float(kp_score[y, x])
                ))
                if len(kps) >= max_keypoints:
                    break
            if len(kps) >= max_keypoints:
                break

    return kps


# =============================================================================
# 5. VISUALIZATION & DEBUGGING TOOLS
# =============================================================================

def visualize_keypoint_neighborhood(
    image: np.ndarray,
    keypoint: Union[Keypoint, Tuple[int, int]],
    p_selected: str = "P7",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualization tool demonstrating:
    1. Keypoint position in the original image with 5x5 neighborhood overlaid.
    2. Grid layout of the 24 surrounding positions P1..P24 with KP at center.
    3. The 3x3 local neighborhood centered at the selected Pi (e.g. P7).
       For P7, shows:
           P1   P2   P3
           P6   P7   P8
           P11  P12  KP
    4. The 24-dimensional entropy descriptor bar chart F(KP) = [S1, ..., S24].

    Args:
        image: 2D grayscale image.
        keypoint: Keypoint or (x, y) coordinates.
        p_selected: Name of Pi to highlight (default: 'P7').
        save_path: Optional path to save the generated PNG figure.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    if hasattr(keypoint, 'x') and hasattr(keypoint, 'y'):
        x, y = int(keypoint.x), int(keypoint.y)
    else:
        x, y = int(keypoint[0]), int(keypoint[1])

    # Compute descriptor
    desc = extract_entropy_descriptor(image, (x, y), ordering="raster", norm="none")

    # Get padded image and 5x5 patch
    padded = apply_reflection_padding(image, pad_width=3)
    kp_y_pad = y + 3
    kp_x_pad = x + 3
    patch_5x5 = padded[kp_y_pad - 2 : kp_y_pad + 3, kp_x_pad - 2 : kp_x_pad + 3]

    # Selected Pi
    dy_pi, dx_pi = P_OFFSETS_DICT[p_selected]
    s_pi, rho_pi, patch_3x3_pi = extract_single_p_entropy(padded, kp_y_pad, kp_x_pad, dy_pi, dx_pi)
    pi_labels = get_p_neighborhood_labels(p_selected)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"24-D Local Von Neumann Entropy Feature Descriptor\nKeypoint KP at (x={x}, y={y})",
        fontsize=15, fontweight='bold'
    )

    # Subplot 1: 5x5 Neighborhood with labels
    ax0 = axes[0, 0]
    ax0.imshow(patch_5x5, cmap='gray', interpolation='nearest')
    ax0.set_title("5x5 Neighborhood Layout (KP & P1..P24)", fontsize=12, fontweight='bold')
    ax0.set_xticks(range(5))
    ax0.set_yticks(range(5))
    ax0.set_xticklabels([-2, -1, 0, 1, 2])
    ax0.set_yticklabels([-2, -1, 0, 1, 2])
    ax0.set_xlabel("Col Offset (dx)", fontsize=10)
    ax0.set_ylabel("Row Offset (dy)", fontsize=10)

    # Label every cell
    for r in range(5):
        for c in range(5):
            dy = r - 2
            dx = c - 2
            label = OFFSET_TO_LABEL[(dy, dx)]
            is_kp = (dy == 0 and dx == 0)
            is_sel = (label == p_selected)

            color = 'red' if is_kp else ('gold' if is_sel else 'white')
            weight = 'bold' if (is_kp or is_sel) else 'normal'

            val = patch_5x5[r, c]
            text = f"{label}\n({val:.0f})"
            ax0.text(c, r, text, ha='center', va='center', color=color,
                     fontsize=9, fontweight=weight,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor=color))

    # Subplot 2: Selected Pi 3x3 local neighborhood
    ax1 = axes[0, 1]
    ax1.imshow(patch_3x3_pi, cmap='magma', interpolation='nearest')
    ax1.set_title(
        f"Local 3x3 Neighborhood Centered at {p_selected}\nEntropy S({p_selected}) = {s_pi:.4f} bits",
        fontsize=12, fontweight='bold'
    )
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels([-1, 0, 1])
    ax1.set_yticklabels([-1, 0, 1])
    ax1.set_xlabel("Local Col Offset", fontsize=10)
    ax1.set_ylabel("Local Row Offset", fontsize=10)

    for r in range(3):
        for c in range(3):
            cell_label = pi_labels[r][c]
            val = patch_3x3_pi[r, c]
            is_center = (r == 1 and c == 1)
            is_kp = (cell_label == "KP")
            color = 'gold' if is_center else ('cyan' if is_kp else 'white')
            ax1.text(c, r, f"{cell_label}\n{val:.0f}", ha='center', va='center', color=color,
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor=color))

    # Subplot 3: 24-D Entropy Descriptor Vector
    ax2 = axes[1, 0]
    p_indices = range(1, 25)
    colors = ['#FFC107' if f"P{i}" == p_selected else '#2196F3' for i in p_indices]
    bars = ax2.bar(p_indices, desc, color=colors, edgecolor='black', alpha=0.85)
    ax2.axhline(np.log2(9.0), color='red', linestyle='--', linewidth=1.5,
                label=f"Max Entropy log2(9) = {np.log2(9.0):.4f}")
    ax2.set_xlabel("Position (P1 ... P24)", fontsize=11)
    ax2.set_ylabel("Von Neumann Entropy S(rho) [bits]", fontsize=11)
    ax2.set_title(f"24-D Feature Vector F(KP): Shape (24,)", fontsize=12, fontweight='bold')
    ax2.set_xticks(p_indices)
    ax2.set_xticklabels([f"P{i}" for i in p_indices], rotation=60, fontsize=8)
    ax2.set_ylim([0, 3.5])
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right')

    # Subplot 4: Image context with keypoint marker
    ax3 = axes[1, 1]
    ax3.imshow(image, cmap='gray')
    ax3.plot(x, y, 'r+', markersize=14, markeredgewidth=2)
    # Draw 5x5 bounding box
    rect = plt.Rectangle((x - 2.5, y - 2.5), 5, 5, fill=False, edgecolor='gold', linewidth=2)
    ax3.add_patch(rect)
    ax3.set_title(f"Keypoint Location on Input Image (x={x}, y={y})", fontsize=12, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_matches(
    image_a: np.ndarray,
    keypoints_a: List[Any],
    image_b: np.ndarray,
    keypoints_b: List[Any],
    matches: List[Tuple[int, int, float]],
    max_display: int = 30,
    save_path: Optional[str] = None,
    similarity_score: Optional[float] = None,
    mean_distance: Optional[float] = None,
    dimension: int = 9
) -> plt.Figure:
    """
    Visualize keypoint correspondences between Image A and Image B based on entropy descriptors.

    Args:
        image_a: 2D array of Image A.
        keypoints_a: Keypoints in Image A.
        image_b: 2D array of Image B.
        keypoints_b: Keypoints in Image B.
        matches: List of (idx_a, idx_b, distance) tuples.
        max_display: Maximum number of match lines to draw.
        save_path: Optional path to save the generated image.
        similarity_score: Overall similarity score in [0, 100]%.
        mean_distance: Average Euclidean descriptor distance.
        dimension: 9 or 24 descriptor dimension.

    Returns:
        matplotlib.figure.Figure
    """
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape
    h_max = max(h_a, h_b)

    # Side-by-side composite canvas
    composite = np.zeros((h_max, w_a + w_b), dtype=np.uint8)
    composite[:h_a, :w_a] = image_a
    composite[:h_b, w_a:w_a + w_b] = image_b

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(composite, cmap='gray')

    title = f"Quantum {dimension}-D Local Entropy Keypoint Matches ({len(matches)} pairs)"
    if similarity_score is not None:
        title += f"  |  Similarity Score: {similarity_score:.2f}%"
    if mean_distance is not None:
        title += f"  |  Mean Dist: {mean_distance:.4f}"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=14)
    ax.axis('off')

    # HUD Result Banner in top-left corner
    if similarity_score is not None:
        hud_text = (
            f" SIMILARITY: {similarity_score:.2f}% \n"
            f" Matches: {len(matches)} corresponding pairs\n"
            f" Mean Distance: {mean_distance:.4f}\n"
            f" Architecture: {dimension}-D Sliding Window"
        )
        ax.text(
            0.02, 0.96, hud_text,
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            color='yellow',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='black', alpha=0.85, edgecolor='cyan', linewidth=2)
        )

    # Sort matches by distance (lowest distance / highest confidence first)
    sorted_matches = sorted(matches, key=lambda m: m[2])[:max_display]

    cmap = plt.get_cmap('spring')
    for idx, (ia, ib, dist) in enumerate(sorted_matches):
        kpa = keypoints_a[ia]
        kpb = keypoints_b[ib]

        xa = kpa.x if hasattr(kpa, 'x') else kpa[0]
        ya = kpa.y if hasattr(kpa, 'y') else kpa[1]
        xb = (kpb.x if hasattr(kpb, 'x') else kpb[0]) + w_a
        yb = kpb.y if hasattr(kpb, 'y') else kpb[1]

        color = cmap(idx / max(1, len(sorted_matches)))
        ax.plot([xa, xb], [ya, yb], '-', color=color, linewidth=1.5, alpha=0.75)
        ax.plot(xa, ya, 'o', color='cyan', markersize=4)
        ax.plot(xb, yb, 'o', color='yellow', markersize=4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
