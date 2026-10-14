"""
Sobel Operator Module for Quantum Image Edge Detection (QSED).

Performs 2D convolution of the input image with the eight 5x5 Sobel directional masks.

Paper Reference:
- Section 2.2: Classical 8-direction Sobel calculation (Eq. 3 & Eq. 4).
- Section 3.2: Step 3 Gradients calculation (Eq. 10).
"""

from typing import Dict
import numpy as np
from scipy.signal import convolve2d
from .sobel_masks import get_eight_sobel_masks


def apply_sobel_masks(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convolve input grayscale image with all eight 5x5 Sobel directional masks.

    Purpose:
        Calculate directional gradient components G_0, G_22.5, ..., G_157.5 across the image.
        In the quantum circuit, this is performed in parallel across all pixels using CT (cyclic shift),
        PA (parallel adder), DO (double operation), and CA (complement) operations.

    Inputs:
        image (np.ndarray): Grayscale matrix of shape (H, W) with dtype np.uint8 or float.

    Outputs:
        Dict[str, np.ndarray]: Dictionary mapping direction string ('0', '22.5', ..., '157.5')
        to gradient matrix G_d of shape (H, W).

    Time Complexity:
        O(8 * 5^2 * H * W) = O(H * W) linear in image size.

    Space Complexity:
        O(8 * H * W) space to store 8 directional gradient matrices.

    Reference:
        Paper Section 2.2, Equation (4); Section 3.2, Equation (10).
    """
    masks = get_eight_sobel_masks()
    gradients = {}
    img_float = image.astype(np.float64)

    for direction, mask in masks.items():
        # Perform 2D spatial convolution with boundary reflection padding (boundary handling)
        # Note: Scipy convolve2d flips the kernel; to match correlation math in paper, we use flip or correlate
        grad = convolve2d(img_float, np.flipud(np.fliplr(mask)), mode='same', boundary='symm')
        gradients[direction] = grad

    return gradients
