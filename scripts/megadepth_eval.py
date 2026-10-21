"""
MegaDepth-1500 Relative Camera Pose Benchmark for QSED 24-D Descriptor.

Evaluates the QSED quantum descriptor on the MegaDepth-1500 benchmark
(Brandenburger Tor and St. Peter's Square), exactly matching the evaluation protocol
and metrics of the RIPE paper (Table 2 Left - arXiv:2507.04839):
  - Metric: AUC of relative camera pose error max(e_R, e_t) at 5°, 10°, and 20°
  - Matching: Mutual Nearest Neighbor (MNN)
  - Solver: Essential Matrix with RANSAC (5-point algorithm / PoseLib)
  - Comparison: Side-by-side with RIPE, SIFT, SuperPoint, DISK, ALIKED, DeDoDe, LoFTR

Usage:
  python scripts/megadepth_eval.py --data-dir data/megadepth_test_1500
  python scripts/megadepth_eval.py --num-pairs 100 --dimension 24
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm

from src.feature.entropy_descriptor import (
    extract_all_entropy_descriptors,
    match_entropy_descriptors,
    detect_default_keypoints,
)


# =============================================================================
# RIPE PAPER (TABLE 2 LEFT) MEGADEPTH-1500 BASELINES
# =============================================================================

RIPE_MEGADEPTH_RESULTS = {
    # Method: (Category, Dim, Supervision Label, AUC@5°, AUC@10°, AUC@20°)
    "Eff. LoFTR (CVPR'24)":      ("(Semi-)Dense", "Dense", "D",    64.63, 78.03, 87.21),
    "Mast3r (ECCV'24)":          ("(Semi-)Dense", "Dense", "D",    22.84, 35.29, 47.62),
    "RoMa (CVPR'24)":            ("(Semi-)Dense", "Dense", "D",    68.28, 80.53, 88.77),
    "SIFT (IJCV'04)":            ("Sparse",       "128",   "—",    37.29, 50.61, 61.89),
    "SuperPoint (CVPRW'18)":     ("Sparse",       "256",   "H",    49.50, 62.51, 72.12),
    "DISK (NeurIPS'20)":         ("Sparse",       "128",   "P/D",  51.83, 64.49, 74.30),
    "ALIKED (TIM'23)":           ("Sparse",       "128",   "P+H",  56.71, 69.86, 79.54),
    "SiLK (ICCV'23)":            ("Sparse",       "128",   "H",    29.83, 40.59, 49.98),
    "DeDoDe-B (3DV'24)":         ("Sparse",       "256",   "P",    55.02, 67.99, 77.31),
    "RIPE (Künzel et al. 2025)": ("Sparse",       "256",   "Pair", 55.11, 68.34, 78.03),
}


# =============================================================================
# POSE METRICS & AUC
# =============================================================================

def compute_relative_pose_error(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R_est: np.ndarray,
    t_est: np.ndarray
) -> float:
    """
    Compute relative pose error max(e_R, e_t) in degrees, exactly matching
    LoFTR, LightGlue, and RIPE paper Section 4.4.
    """
    # Rotation error
    tr = np.clip((np.trace(R_gt.T @ R_est) - 1.0) / 2.0, -1.0, 1.0)
    e_R = np.degrees(np.arccos(tr))

    # Translation error (angular error of unit direction vector)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-10)
    t_est_norm = t_est / (np.linalg.norm(t_est) + 1e-10)
    cos_t = np.clip(np.dot(t_gt_norm.flatten(), t_est_norm.flatten()), -1.0, 1.0)
    e_t = np.degrees(np.arccos(np.abs(cos_t)))  # up to sign ambiguity

    return float(max(e_R, e_t))


def compute_pose_auc(
    errors: list,
    thresholds: list = None,
    num_bins: int = 200
) -> dict:
    """
    Compute Area Under Curve (AUC) for pose errors at specified angular thresholds
    (default: 5°, 10°, 20°), matching LoFTR and RIPE Table 2.
    """
    if thresholds is None:
        thresholds = [5, 10, 20]

    if len(errors) == 0:
        return {f"AUC@{t}deg": 0.0 for t in thresholds}

    errors_arr = np.array(errors, dtype=np.float64)
    total_pairs = len(errors_arr)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

    results = {}
    for t in thresholds:
        eval_t = np.linspace(0.0, float(t), num_bins)
        recalls = np.array([np.sum(errors_arr <= step) / total_pairs for step in eval_t])
        auc = trapz_fn(recalls, eval_t) / float(t) * 100.0
        results[f"AUC@{t}deg"] = float(auc)
        results[f"Acc@{t}deg"] = float(np.sum(errors_arr < t) / total_pairs * 100.0)

    return results


def print_megadepth_comparison_table(our_auc: dict, dim: int = 24):
    """Print side-by-side Table 2 (MegaDepth-1500) comparison with RIPE."""
    lines = []
    lines.append("=" * 78)
    lines.append("   COMPARISON: RIPE Paper (Table 2 Left - MegaDepth-1500 Benchmark)")
    lines.append("=" * 78)
    lines.append(f"  {'Category':<14} | {'Method':<25} | {'Dim':>5} | {'Label':^5} | {'AUC@5°':>7} | {'AUC@10°':>7} | {'AUC@20°':>7}")
    lines.append("  " + "-" * 74)

    curr_cat = None
    for method, (cat, m_dim, label, a5, a10, a20) in RIPE_MEGADEPTH_RESULTS.items():
        if cat != curr_cat:
            curr_cat = cat
            lines.append(f"  [{cat}]")
        lines.append(f"  {'':<14} | {method:<25} | {m_dim:>5} | {label:^5} | {a5:>6.2f}% | {a10:>6.2f}% | {a20:>6.2f}%")

    lines.append("  " + "=" * 74)
    a5_val = our_auc.get("AUC@5deg", 0.0)
    a10_val = our_auc.get("AUC@10deg", 0.0)
    a20_val = our_auc.get("AUC@20deg", 0.0)
    lines.append(f"  {'Proposed':<14} | {'QSED (Ours)':<25} | {dim:>5} | {'Entropy':^5} | {a5_val:>6.2f}% | {a10_val:>6.2f}% | {a20_val:>6.2f}%")
    lines.append("=" * 78)
    lines.append("  Supervision Labels: D=Depth, P=Pose, H=Homography Aug., Pair=Binary Pairs")
    lines.append("=" * 78)
    table_str = "\n".join(lines)
    print("\n" + table_str + "\n")
    return table_str


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def run_megadepth_eval(args):
    """Run QSED evaluation on MegaDepth-1500 pairs."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check for mounted Kaggle dataset if data_dir not found
    if not os.path.isdir(data_dir) and os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "megadepth" in root.lower() or "megadepth_test_1500" in dirs:
                candidate = os.path.join(root, "megadepth_test_1500") if "megadepth_test_1500" in dirs else root
                if os.path.isdir(candidate):
                    data_dir = candidate
                    print(f"  [Auto-detected Kaggle MegaDepth path]: {data_dir}")
                    break

    print("=" * 76)
    print("   QSED 24-D Evaluation on MegaDepth-1500 Benchmark")
    print("=" * 76)
    print(f"  Descriptor dimension : {args.dimension}-D")
    print(f"  Normalization        : {args.norm}")
    print(f"  Max keypoints        : {args.max_keypoints}")
    print(f"  Max image size       : {args.max_side}px")
    print(f"  Data directory       : {data_dir}")
    print("=" * 76)

    # Load pair list or indices
    pairs_file = os.path.join(data_dir, "pairs.txt")
    npz_file = os.path.join(data_dir, "megadepth_test_1500.npz")

    pose_errors = []
    
    if os.path.isfile(npz_file):
        print(f"  Loading test pairs from {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)
        pair_list = data["pair_names"]
        K1_list = data["K1"]
        K2_list = data["K2"]
        R_gt_list = data["R"]
        t_gt_list = data["t"]
        
        limit = min(len(pair_list), args.num_pairs) if args.num_pairs > 0 else len(pair_list)
        print(f"  Evaluating {limit} image pairs...")

        for i in tqdm(range(limit), desc="MegaDepth-1500"):
            p1, p2 = pair_list[i]
            img_path1 = os.path.join(data_dir, p1)
            img_path2 = os.path.join(data_dir, p2)
            
            if not os.path.isfile(img_path1) or not os.path.isfile(img_path2):
                pose_errors.append(float('inf'))
                continue

            img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                pose_errors.append(float('inf'))
                continue

            h1, w1 = img1.shape
            scale1 = args.max_side / max(h1, w1) if max(h1, w1) > args.max_side else 1.0
            if scale1 < 1.0:
                img1 = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)), interpolation=cv2.INTER_AREA)

            h2, w2 = img2.shape
            scale2 = args.max_side / max(h2, w2) if max(h2, w2) > args.max_side else 1.0
            if scale2 < 1.0:
                img2 = cv2.resize(img2, (int(w2 * scale2), int(h2 * scale2)), interpolation=cv2.INTER_AREA)

            # Adjust intrinsics for scale
            K1 = K1_list[i].copy()
            K1[:2] *= scale1
            K2 = K2_list[i].copy()
            K2[:2] *= scale2

            R_gt = R_gt_list[i]
            t_gt = t_gt_list[i]

            # Detect & Describe
            kps1 = detect_default_keypoints(img1.astype(np.float64), max_keypoints=args.max_keypoints)
            kps2 = detect_default_keypoints(img2.astype(np.float64), max_keypoints=args.max_keypoints)
            desc1 = extract_all_entropy_descriptors(img1.astype(np.float64), kps1, dimension=args.dimension, norm=args.norm)
            desc2 = extract_all_entropy_descriptors(img2.astype(np.float64), kps2, dimension=args.dimension, norm=args.norm)

            match_res = match_entropy_descriptors(desc1, desc2, method="mutual_nn")
            matches = match_res["matches"]

            if len(matches) < 5:
                pose_errors.append(float('inf'))
                continue

            pts1 = np.float32([[kps1[m[0]].x, kps1[m[0]].y] for m in matches])
            pts2 = np.float32([[kps2[m[1]].x, kps2[m[1]].y] for m in matches])

            # Normalize coordinates using K
            pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None)
            pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None)

            # Estimate Essential matrix
            E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.001)

            if E is None or E.shape != (3, 3):
                pose_errors.append(float('inf'))
                continue

            # Recover relative pose
            _, R_est, t_est, _ = cv2.recoverPose(E, pts1_norm, pts2_norm, mask=mask)

            err = compute_relative_pose_error(R_gt, t_gt, R_est, t_est)
            pose_errors.append(err)

    else:
        print(f"\n  [Info] MegaDepth-1500 archive not detected at: {data_dir}")
        print("  To run on MegaDepth-1500, please download the standard test archive (~624 MB):")
        print("    wget https://huggingface.co/datasets/shngjz/TwoViewBenchmark/resolve/main/megadepth_test_1500.tar.gz")
        print("    tar -xzf megadepth_test_1500.tar.gz -C data/")
        print("\n  Displaying baseline table from RIPE Paper Table 2 for reference:")
        print_megadepth_comparison_table({}, dim=args.dimension)
        return

    # Compute AUC
    auc_results = compute_pose_auc(pose_errors, thresholds=[5, 10, 20])
    
    print("\n" + "=" * 65)
    print("   RESULTS: MegaDepth-1500 Pose Estimation")
    print("=" * 65)
    print(f"  Evaluated image pairs  : {len(pose_errors)}")
    print(f"  AUC@5°                 : {auc_results['AUC@5deg']:.2f}% (Acc={auc_results['Acc@5deg']:.2f}%)")
    print(f"  AUC@10°                : {auc_results['AUC@10deg']:.2f}% (Acc={auc_results['Acc@10deg']:.2f}%)")
    print(f"  AUC@20°                : {auc_results['AUC@20deg']:.2f}% (Acc={auc_results['Acc@20deg']:.2f}%)")
    print("=" * 65)

    # Print Table 2 comparison
    table_str = print_megadepth_comparison_table(auc_results, dim=args.dimension)

    # Save results
    report_file = os.path.join(output_dir, "megadepth_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"  Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MegaDepth-1500 benchmark matching RIPE Table 2.")
    parser.add_argument("--data-dir", type=str, default=os.path.join("data", "megadepth_test_1500"),
                        help="Path to MegaDepth-1500 test directory")
    parser.add_argument("--num-pairs", type=int, default=1500, help="Number of pairs to evaluate (default: 1500, 0=all)")
    parser.add_argument("--max-keypoints", type=int, default=1024, help="Max keypoints (default: 1024, or 2048 for full)")
    parser.add_argument("--max-side", type=int, default=1200, help="Resize longer side (default: 1200, matching RIPE Sec 4.4)")
    parser.add_argument("--dimension", type=int, default=24, choices=[9, 24], help="Descriptor dimension")
    parser.add_argument("--norm", type=str, default="l2", choices=["none", "l2", "unit_entropy"], help="Normalization")
    parser.add_argument("--output-dir", type=str, default=os.path.join("outputs", "megadepth"), help="Output directory")
    args = parser.parse_args()

    run_megadepth_eval(args)


if __name__ == "__main__":
    main()
