"""
Directional Variation Module for Von Neumann Entropy Keypoint Detection.

Computes local directional variation from the 8-direction Sobel gradient outputs.
This is a CLASSICAL post-processing stage applied to the quantum-pipeline outputs.

Mathematical Foundation:
    For every pixel (x,y), the existing QSED pipeline provides:
        M(x,y) = max_theta |G_theta(x,y)|     (gradient magnitude)
        d(x,y) = argmax_theta |G_theta(x,y)|  (dominant direction index 0..7)

    Angular difference with 180° periodicity (edge orientations):
        Δθ(a, b) = min(|a - b|, 180° - |a - b|)

    Local directional variation over neighborhood N(x,y):
        D(x,y) = (1/|N|) * Σ_{(u,v) in N} w(u,v) * Δθ(d(x,y), d(u,v))

    where w(u,v) = min(M(u,v), M(x,y)) / (M(x,y) + ε) ∈ [0, 1]

    Weight rationale: Neighbors with weak gradients have unreliable directions,
    so they contribute less. The min() clip ensures weights never exceed 1.0,
    preventing flat-region explosion when center magnitude is near zero.

    Keypoint score:
        K(x,y) = M_norm(x,y) × D_norm(x,y)

    where M_norm and D_norm are each normalized to [0, 1].
    The product ensures BOTH strong gradient AND high directional variation are needed.

Paper Reference: Extension of Section 2.2 (8-direction Sobel) and Section 3.2 (gradient computation).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def angular_difference(a: float, b: float) -> float:
    """
    Compute angular difference with 180° periodicity.

    Edge orientations are symmetric (0° and 180° represent the same edge direction),
    so the maximum meaningful difference is 90°.

    Args:
        a: First angle in degrees.
        b: Second angle in degrees.

    Returns:
        float: Angular difference in [0, 90].

    Examples:
        angular_difference(0, 90) → 90.0
        angular_difference(0, 157.5) → 22.5  (since 180 - 157.5 = 22.5)
        angular_difference(45, 45) → 0.0
    """
    diff = abs(a - b)
    return min(diff, 180.0 - diff)


def compute_directional_variation(
    dominant_direction: np.ndarray,
    gradient_magnitude: np.ndarray,
    neighborhood_size: int = 3
) -> np.ndarray:
    """
    Compute full-image H×W directional variation map D(x,y).

    For each pixel, measures how much the dominant edge direction varies
    across its local neighborhood, weighted by gradient strength.

    High D → directions change rapidly → corner/junction/high-curvature
    Low D → directions are uniform → straight edge or flat region

    This is a CLASSICAL computation on the quantum-pipeline outputs.

    Args:
        dominant_direction: H×W int array with values in {0,1,...,7}.
            Index i maps to angle i × 22.5°.
        gradient_magnitude: H×W float64 array of max gradient magnitudes.
        neighborhood_size: Size of the local window (3 or 5).

    Returns:
        np.ndarray: H×W float64 directional variation map D(x,y) in [0, 90].
            Border pixels (within pad of edges) have D=0.

    Time Complexity: O(H × W × neighborhood_size²)
    Space Complexity: O(H × W)
    """
    h, w = dominant_direction.shape
    D = np.zeros((h, w), dtype=np.float64)

    if h < neighborhood_size or w < neighborhood_size:
        return D

    # Convert direction indices (0..7) to angles in degrees (0, 22.5, ..., 157.5)
    angles = dominant_direction.astype(np.float64) * 22.5
    epsilon = 1e-10
    pad = neighborhood_size // 2
    n_elements = neighborhood_size * neighborhood_size

    # Create sliding window views for vectorized computation
    angles_win = sliding_window_view(angles, (neighborhood_size, neighborhood_size))
    mag_win = sliding_window_view(gradient_magnitude, (neighborhood_size, neighborhood_size))

    # Center pixel values (broadcast-ready)
    center_angles = angles[pad:h-pad, pad:w-pad][..., np.newaxis, np.newaxis]
    center_mags = gradient_magnitude[pad:h-pad, pad:w-pad][..., np.newaxis, np.newaxis]

    # Gradient-strength weights: w(u,v) = min(M(u,v), M(x,y)) / (M(x,y) + eps)
    # Clipped to [0, 1] — neighbors weaker than center get proportional weight,
    # neighbors stronger than center get weight ≈ 1.0
    w_uv = np.minimum(mag_win, center_mags) / (center_mags + epsilon)

    # Angular differences with 180° periodicity
    diffs = np.abs(center_angles - angles_win)
    diffs = np.minimum(diffs, 180.0 - diffs)

    # Weighted average directional variation
    D_inner = np.sum(w_uv * diffs, axis=(2, 3)) / n_elements
    D[pad:h-pad, pad:w-pad] = D_inner

    return D


def compute_keypoint_score(
    gradient_magnitude: np.ndarray,
    directional_variation: np.ndarray
) -> np.ndarray:
    """
    Compute keypoint score K(x,y) = M_norm(x,y) × D_norm(x,y).

    The product ensures that a pixel must have BOTH:
    1. Strong edge/gradient magnitude (M_norm high)
    2. High local directional variation (D_norm high)

    A straight edge has high M but low D → low K.
    A flat region has low M → low K.
    Only corners/junctions/high-curvature points have both → high K.

    This is a CLASSICAL computation.

    Args:
        gradient_magnitude: H×W float64 array.
        directional_variation: H×W float64 array.

    Returns:
        np.ndarray: H×W float64 keypoint score map, values in [0, 1].
    """
    M_max = np.max(gradient_magnitude)
    D_max = np.max(directional_variation)

    M_norm = gradient_magnitude / M_max if M_max > 0 else np.zeros_like(gradient_magnitude)
    D_norm = directional_variation / D_max if D_max > 0 else np.zeros_like(directional_variation)

    return M_norm * D_norm
