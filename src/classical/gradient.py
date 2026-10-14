"""
Gradient Computation Module for Quantum Image Edge Detection (QSED).

Calculates absolute gradient values across 8 directions and extracts the maximum gradient magnitude
and dominant edge direction per pixel.

Paper Reference:
- Section 2.2: Equation (5) G = max( |G0|, |G22.5|, |G45|, |G67.5|, |G90|, |G112.5|, |G135|, |G157.5| ).
- Section 3.2: Equation (11) & (12) Quantum max gradient selection state |G>.
"""

from typing import Dict, Tuple
import numpy as np


def compute_gradient_magnitude(
    directional_gradients: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute maximum gradient magnitude and dominant direction index for each pixel.

    Purpose:
        Evaluate Eq. (5) & Eq. (11) across the 8 directional response matrices.
        In the quantum pipeline, this max operation is implemented using absolute value (AV) gates
        and quantum comparators (QC) in a binary tree structure.

    Inputs:
        directional_gradients (Dict[str, np.ndarray]): Dictionary of 8 directional gradient matrices.

    Outputs:
        Tuple[np.ndarray, np.ndarray]:
            - gradient_magnitude (np.ndarray): 2D array of shape (H, W) containing G(Y,X) = max_d |G_d(Y,X)|.
            - dominant_direction (np.ndarray): 2D array of shape (H, W) containing direction index (0..7)
              corresponding to ['0', '22.5', '45', '67.5', '90', '112.5', '135', '157.5'].

    Time Complexity:
        O(8 * H * W) = O(H * W) to iterate over 8 directional matrices.

    Space Complexity:
        O(H * W) to store magnitude and direction matrices.

    Reference:
        Paper Section 2.2, Equation (5); Section 3.2, Equations (11)-(12).
    """
    directions = ["0", "22.5", "45", "67.5", "90", "112.5", "135", "157.5"]
    h, w = next(iter(directional_gradients.values())).shape

    # Stack absolute values of all 8 direction matrices
    abs_grads = np.zeros((8, h, w), dtype=np.float64)
    for idx, dir_key in enumerate(directions):
        abs_grads[idx] = np.abs(directional_gradients[dir_key])

    # Find maximum absolute gradient and its corresponding direction index
    max_gradient = np.max(abs_grads, axis=0)
    dominant_direction = np.argmax(abs_grads, axis=0)

    return max_gradient, dominant_direction
