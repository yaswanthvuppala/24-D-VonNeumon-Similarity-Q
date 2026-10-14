import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Keypoint:
    x: int
    y: int
    dominant_direction: float      # degrees (0, 22.5, ..., 157.5)
    gradient_magnitude: float      # M(x,y)
    directional_variation: float   # D(x,y)
    keypoint_score: float          # K(x,y)
    entropy: float = 0.0           # Von Neumann entropy S(ρ)
    entropy_normalized: float = 0.0  # S/log2(9)
    combined_score: float = 0.0    # Q = α·K_norm + (1-α)·S_norm

def detect_keypoints(
    gradient_magnitude: np.ndarray,
    dominant_direction: np.ndarray,
    edge_mask: np.ndarray,
    directional_variation_map: np.ndarray,
    keypoint_score_map: np.ndarray,
    grad_threshold_ratio: float = 0.1,
    variation_threshold: float = 0.15,
    nms_size: int = 3,
    min_distance: float = 5.0,
    border: int = 3,
    max_keypoints: int = None
) -> List[Keypoint]:
    """
    Detect keypoints applying thresholds and Non-Maximum Suppression.
    """
    h, w = gradient_magnitude.shape
    
    M_max = np.max(gradient_magnitude)
    D_max = np.max(directional_variation_map)
    
    if D_max > 0:
        D_norm = directional_variation_map / D_max
    else:
        D_norm = np.zeros_like(directional_variation_map)
        
    mask = edge_mask > 0
    mask &= (gradient_magnitude >= grad_threshold_ratio * M_max)
    mask &= (D_norm >= variation_threshold)
    
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    
    pad = nms_size // 2
    
    try:
        import cv2
        kernel = np.ones((nms_size, nms_size), np.uint8)
        dilated = cv2.dilate(keypoint_score_map, kernel)
        peaks = (keypoint_score_map == dilated) & mask & (keypoint_score_map > 0)
        y_idx, x_idx = np.where(peaks)
        scores = keypoint_score_map[y_idx, x_idx]
        order = np.argsort(scores)[::-1]
        pts = [(int(y_idx[i]), int(x_idx[i]), float(scores[i])) for i in order]
    except Exception:
        y_idx, x_idx = np.where(mask)
        valid_pts = (y_idx >= border) & (y_idx < h - border) & (x_idx >= border) & (x_idx < w - border)
        y_idx = y_idx[valid_pts]
        x_idx = x_idx[valid_pts]
        pts = []
        for y, x in zip(y_idx, x_idx):
            win = keypoint_score_map[y-pad:y+pad+1, x-pad:x+pad+1]
            if keypoint_score_map[y, x] >= np.max(win):
                pts.append((y, x, keypoint_score_map[y, x]))
        pts.sort(key=lambda p: p[2], reverse=True)
    
    min_dist_sq = min_distance ** 2
    final_pts = []
    candidate_limit = len(pts) if max_keypoints is None else min(len(pts), max(2000, max_keypoints * 15))
    
    for y, x, score in pts[:candidate_limit]:
        keep = True
        for fy, fx in final_pts:
            if (fy - y) ** 2 + (fx - x) ** 2 < min_dist_sq:
                keep = False
                break
        if keep:
            final_pts.append((y, x))
            if max_keypoints is not None and len(final_pts) >= max_keypoints:
                break
            
    keypoints = []
    for y, x in final_pts:
        kp = Keypoint(
            x=int(x),
            y=int(y),
            dominant_direction=float(dominant_direction[y, x] * 22.5),
            gradient_magnitude=float(gradient_magnitude[y, x]),
            directional_variation=float(directional_variation_map[y, x]),
            keypoint_score=float(keypoint_score_map[y, x])
        )
        keypoints.append(kp)
        
    return keypoints

def keypoints_to_csv(keypoints: List[Keypoint], filepath: str) -> None:
    """Export to CSV."""
    import csv
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "dominant_direction", "gradient_magnitude", "directional_variation", "keypoint_score", "entropy", "entropy_normalized", "combined_score"])
        for kp in keypoints:
            writer.writerow([kp.x, kp.y, kp.dominant_direction, kp.gradient_magnitude, kp.directional_variation, kp.keypoint_score, kp.entropy, kp.entropy_normalized, kp.combined_score])

def keypoints_to_array(keypoints: List[Keypoint]) -> np.ndarray:
    """Convert to structured numpy array."""
    dtype = [('x', 'i4'), ('y', 'i4'), ('dominant_direction', 'f8'), 
             ('gradient_magnitude', 'f8'), ('directional_variation', 'f8'),
             ('keypoint_score', 'f8'), ('entropy', 'f8'), 
             ('entropy_normalized', 'f8'), ('combined_score', 'f8')]
             
    arr = np.zeros(len(keypoints), dtype=dtype)
    for i, kp in enumerate(keypoints):
        arr[i] = (kp.x, kp.y, kp.dominant_direction, kp.gradient_magnitude, 
                  kp.directional_variation, kp.keypoint_score, kp.entropy, 
                  kp.entropy_normalized, kp.combined_score)
    return arr
