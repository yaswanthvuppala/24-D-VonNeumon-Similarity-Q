"""
Eight-Direction Sobel Operator Masks Module for Quantum Image Edge Detection (QSED).

This module defines the 5x5 neighborhood Sobel masks for 8 orientations:
0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°.

Paper Reference:
- Section 2.2: Classical Sobel edge detection (Equations 3 & 4, Figures 2 & 3).
"""

from typing import Dict
import numpy as np


def get_eight_sobel_masks() -> Dict[str, np.ndarray]:
    """
    Construct and return the eight 5x5 Sobel directional masks according to Equations (3) and (4).

    Purpose:
        The paper extends traditional 3x3 two-direction (0°, 90°) or four-direction (0°, 45°, 90°, 135°)
        Sobel masks to a 5x5 neighborhood with eight orientations (0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°).
        This captures subtle diagonal and sub-diagonal edge details that smaller 3x3 kernels miss.

    Outputs:
        Dict[str, np.ndarray]: Dictionary mapping direction name (e.g. '0', '22.5', ..., '157.5')
        to its 5x5 integer kernel matrix.

    Time Complexity:
        O(1) constant mask construction.

    Space Complexity:
        O(1) 8 matrices of 5x5 floating/integer values.

    Reference:
        Paper Section 2.2, Equation (3), Equation (4), Figure 3.
    """
    # 0° direction kernel (Horizontal gradient measuring difference between X+1 and X-1 columns)
    # G0 = p(Y-2, X+1) + 2*p(Y-1, X+1) + 4*p(Y, X+1) + 2*p(Y+1, X+1) + p(Y+2, X+1)
    #    - p(Y-2, X-1) - 2*p(Y-1, X-1) - 4*p(Y, X-1) - 2*p(Y+1, X-1) - p(Y+2, X-1)
    g0 = np.array([
        [ 0, -1, 0,  1, 0],
        [ 0, -2, 0,  2, 0],
        [ 0, -4, 0,  4, 0],
        [ 0, -2, 0,  2, 0],
        [ 0, -1, 0,  1, 0]
    ], dtype=np.float64)

    # 22.5° direction kernel
    # G22.5 = p(Y+2, X) + 2*p(Y+1, X+1) + 2*p(Y-1, X+1) + 4*p(Y, X+1) + 4*p(Y+1, X)
    #       - p(Y-2, X) - 2*p(Y+1, X-1) - 2*p(Y-1, X-1) - 4*p(Y, X-1) - 4*p(Y-1, X)
    g22_5 = np.array([
        [ 0, -2, -1,  0, 0],
        [ 0, -4, -2,  2, 0],
        [ 0, -4,  0,  4, 0],
        [ 0, -2,  2,  4, 0],
        [ 0,  0,  1,  2, 0]
    ], dtype=np.float64)

    # 45° direction kernel
    # G45 = p(Y+2, X-1) + p(Y-1, X+2) + 2*p(Y+1, X+1) + 4*p(Y+1, X) + 4*p(Y, X+1)
    #     - p(Y+1, X-2) - p(Y-2, X+1) - 2*p(Y-1, X-1) - 4*p(Y-1, X) - 4*p(Y, X-1)
    g45 = np.array([
        [ 0, -1, -1,  0, 0],
        [-1, -2, -4,  2, 1],
        [ 0, -4,  0,  4, 0],
        [-1, -2,  4,  2, 1],
        [ 0,  0,  1,  1, 0]
    ], dtype=np.float64)

    # 67.5° direction kernel
    g67_5 = np.array([
        [ 0,  0, -1, -2, 0],
        [ 0, -2, -4,  0, 0],
        [ 0, -2,  0,  2, 0],
        [ 0,  0,  4,  2, 0],
        [ 0,  2,  1,  0, 0]
    ], dtype=np.float64)

    # 90° direction kernel (Vertical gradient measuring difference between Y+1 and Y-1 rows)
    g90 = np.array([
        [ 0,  0,  0,  0, 0],
        [-1, -2, -4, -2, -1],
        [ 0,  0,  0,  0, 0],
        [ 1,  2,  4,  2, 1],
        [ 0,  0,  0,  0, 0]
    ], dtype=np.float64)

    # 112.5° direction kernel
    g112_5 = np.array([
        [ 0, -2, -1,  0, 0],
        [ 0,  0, -4, -2, 0],
        [ 0, -2,  0,  2, 0],
        [ 0,  2,  4,  0, 0],
        [ 0,  0,  1,  2, 0]
    ], dtype=np.float64)

    # 135° direction kernel
    g135 = np.array([
        [ 0,  0, -1, -1, 0],
        [ 1,  2, -4, -2, -1],
        [ 0, -4,  0,  4, 0],
        [ 1,  2,  4, -2, -1],
        [ 0,  1,  1,  0, 0]
    ], dtype=np.float64)

    # 157.5° direction kernel
    g157_5 = np.array([
        [ 0,  0,  1,  2, 0],
        [ 0, -2,  2,  4, 0],
        [ 0, -4,  0,  4, 0],
        [ 0, -4, -2,  2, 0],
        [ 0, -2, -1,  0, 0]
    ], dtype=np.float64)

    return {
        "0": g0,
        "22.5": g22_5,
        "45": g45,
        "67.5": g67_5,
        "90": g90,
        "112.5": g112_5,
        "135": g135,
        "157.5": g157_5
    }
