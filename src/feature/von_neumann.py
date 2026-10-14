import numpy as np
import os
from typing import Tuple

def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute Von Neumann entropy S(ρ) = -Σ λi log2(λi)
    
    Args:
        rho: Density matrix.
        
    Returns:
        float: Computed entropy.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 0]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def normalized_entropy(entropy: float, dim: int = 9) -> float:
    """
    Normalize entropy by log2(dim).
    
    Args:
        entropy: Computed entropy.
        dim: Dimensionality of the density matrix.
        
    Returns:
        float: Normalized entropy in [0, 1].
    """
    return entropy / np.log2(dim)

def compute_entropy_map(image: np.ndarray, neighborhood_size: int = 3, epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FULL HxW entropy matrix.
    
    Args:
        image: 2D image.
        neighborhood_size: Size of neighborhood window.
        epsilon: Stability constant.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (entropy_map, normalized_entropy_map)
    """
    h, w = image.shape
    entropy_map = np.zeros((h, w), dtype=np.float64)
    pad = neighborhood_size // 2
    
    from numpy.lib.stride_tricks import sliding_window_view
    
    if h < neighborhood_size or w < neighborhood_size:
        return entropy_map, entropy_map
        
    windows = sliding_window_view(image, (neighborhood_size, neighborhood_size))
    v = windows.reshape(windows.shape[0], windows.shape[1], -1)
    
    q = (v + epsilon) / np.sum(v + epsilon, axis=2, keepdims=True)
    q_safe = np.where(q > 0, q, 1) 
    
    H = -np.sum(q * np.log2(q_safe), axis=2)
    
    entropy_map[pad:h-pad, pad:w-pad] = H
    
    max_entropy = np.log2(neighborhood_size * neighborhood_size)
    entropy_map_normalized = entropy_map / max_entropy
    
    return entropy_map, entropy_map_normalized

def save_entropy_outputs(entropy_map: np.ndarray, entropy_map_normalized: np.ndarray, keypoints: list, output_dir: str) -> None:
    """
    Save entropy matrices and keypoint tables.
    
    Args:
        entropy_map: Full entropy map.
        entropy_map_normalized: Normalized entropy map.
        keypoints: List of keypoint objects.
        output_dir: Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "entropy_matrix_full.npy"), entropy_map)
    np.savetxt(os.path.join(output_dir, "entropy_matrix_full.csv"), entropy_map, delimiter=",")
    np.save(os.path.join(output_dir, "entropy_matrix_full_normalized.npy"), entropy_map_normalized)
    
    import csv
    with open(os.path.join(output_dir, "keypoint_entropy_table.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "direction", "magnitude", "variation", "score", "entropy", "entropy_norm", "combined_score"])
        for kp in keypoints:
            writer.writerow([kp.x, kp.y, kp.dominant_direction, kp.gradient_magnitude, kp.directional_variation, kp.keypoint_score, kp.entropy, kp.entropy_normalized, kp.combined_score])
