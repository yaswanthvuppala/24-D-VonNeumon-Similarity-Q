"""
HPatches Baseline Comparison: SIFT and ORB vs QSED.

Runs classical SIFT (128-D) and ORB (256-bit) descriptors on the HPatches
dataset for homography estimation, providing baselines to compare against
the QSED 24-D Von Neumann Entropy descriptor.

Usage:
  python scripts/hpatches_baselines.py --dataset-dir data/hpatches-sequences-release
  python scripts/hpatches_baselines.py --max-keypoints 1024 --max-side 480
  python scripts/hpatches_baselines.py --help
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from tqdm import tqdm

from src.feature.hpatches_adapter import (
    load_hpatches_image,
    load_homography,
    scale_homography,
    compute_corner_error,
    compute_homography_auc,
    get_hpatches_sequences,
    get_sequence_pairs,
)


# =============================================================================
# SIFT BASELINE
# =============================================================================

def run_sift_baseline(sequences, max_keypoints, max_side, ransac_thresh=3.0):
    """
    Run SIFT (128-D) descriptor baseline on HPatches sequences.

    Args:
        sequences: List of sequence dicts from get_hpatches_sequences().
        max_keypoints: Maximum keypoints per image.
        max_side: Resize longest side to this value.
        ransac_thresh: RANSAC reprojection threshold.

    Returns:
        Dict of errors by category: {'illumination': [...], 'viewpoint': [...], 'all': [...]}
    """
    try:
        sift = cv2.SIFT_create(nfeatures=max_keypoints)
    except AttributeError:
        print("  [WARNING] SIFT not available in this OpenCV build. Skipping SIFT baseline.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    errors = {"illumination": [], "viewpoint": [], "all": []}

    for seq_info in tqdm(sequences, desc="SIFT baseline", unit="seq"):
        seq_path = seq_info["path"]
        seq_type = seq_info["type"]

        # Load reference
        img_ref, scale_ref = load_hpatches_image(os.path.join(seq_path, "1.ppm"), max_side=max_side)
        img_ref_u8 = img_ref.astype(np.uint8)
        kps_ref, desc_ref = sift.detectAndCompute(img_ref_u8, None)

        if desc_ref is None or len(kps_ref) == 0:
            for _ in range(5):
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
            continue

        for pair_info in get_sequence_pairs(seq_path):
            img_tgt, scale_tgt = load_hpatches_image(pair_info["tgt_path"], max_side=max_side)
            H_gt = load_homography(pair_info["H_path"])
            H_gt_scaled = scale_homography(H_gt, scale_ref, scale_tgt)

            img_tgt_u8 = img_tgt.astype(np.uint8)
            kps_tgt, desc_tgt = sift.detectAndCompute(img_tgt_u8, None)

            if desc_tgt is None or len(kps_tgt) < 4:
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
                continue

            matches = bf.match(desc_ref, desc_tgt)

            if len(matches) < 4:
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
                continue

            pts_ref = np.float32([kps_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_tgt = np.float32([kps_tgt[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H_est, mask = cv2.findHomography(pts_ref, pts_tgt, cv2.RANSAC, ransac_thresh)

            if H_est is not None:
                err = compute_corner_error(H_gt_scaled, H_est, img_ref.shape)
            else:
                err = float('inf')

            errors[seq_type].append(err)
            errors["all"].append(err)

    return errors


# =============================================================================
# ORB BASELINE
# =============================================================================

def run_orb_baseline(sequences, max_keypoints, max_side, ransac_thresh=3.0):
    """
    Run ORB (256-bit binary) descriptor baseline on HPatches sequences.

    Args:
        sequences: List of sequence dicts from get_hpatches_sequences().
        max_keypoints: Maximum keypoints per image.
        max_side: Resize longest side to this value.
        ransac_thresh: RANSAC reprojection threshold.

    Returns:
        Dict of errors by category: {'illumination': [...], 'viewpoint': [...], 'all': [...]}
    """
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    errors = {"illumination": [], "viewpoint": [], "all": []}

    for seq_info in tqdm(sequences, desc="ORB baseline", unit="seq"):
        seq_path = seq_info["path"]
        seq_type = seq_info["type"]

        # Load reference
        img_ref, scale_ref = load_hpatches_image(os.path.join(seq_path, "1.ppm"), max_side=max_side)
        img_ref_u8 = img_ref.astype(np.uint8)
        kps_ref, desc_ref = orb.detectAndCompute(img_ref_u8, None)

        if desc_ref is None or len(kps_ref) == 0:
            for _ in range(5):
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
            continue

        for pair_info in get_sequence_pairs(seq_path):
            img_tgt, scale_tgt = load_hpatches_image(pair_info["tgt_path"], max_side=max_side)
            H_gt = load_homography(pair_info["H_path"])
            H_gt_scaled = scale_homography(H_gt, scale_ref, scale_tgt)

            img_tgt_u8 = img_tgt.astype(np.uint8)
            kps_tgt, desc_tgt = orb.detectAndCompute(img_tgt_u8, None)

            if desc_tgt is None or len(kps_tgt) < 4:
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
                continue

            matches = bf.match(desc_ref, desc_tgt)

            if len(matches) < 4:
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
                continue

            pts_ref = np.float32([kps_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_tgt = np.float32([kps_tgt[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H_est, mask = cv2.findHomography(pts_ref, pts_tgt, cv2.RANSAC, ransac_thresh)

            if H_est is not None:
                err = compute_corner_error(H_gt_scaled, H_est, img_ref.shape)
            else:
                err = float('inf')

            errors[seq_type].append(err)
            errors["all"].append(err)

    return errors


# =============================================================================
# RESULTS DISPLAY & SAVING
# =============================================================================

def print_comparison_table(results_dict, output_dir, base_dir):
    """Print and save the baseline comparison table."""

    print("\n" + "=" * 75)
    print("   HPatches Baseline Comparison")
    print("=" * 75)

    categories = ["illumination", "viewpoint", "all"]

    for category in categories:
        print(f"\n  {category.upper()}")
        print("-" * 70)
        print(f"  {'Method':<12} | {'Dim':>6} | {'AUC@1px':>8} | {'AUC@3px':>8} | {'AUC@5px':>8} | {'AUC@10px':>9}")
        print("-" * 70)

        for method_name, method_errors in results_dict.items():
            if method_errors is None:
                continue
            cat_errors = method_errors.get(category, [])
            if len(cat_errors) == 0:
                continue

            auc = compute_homography_auc(cat_errors)
            dim_label = {"SIFT": "128", "ORB": "256b"}.get(method_name, "24")
            print(
                f"  {method_name:<12} | {dim_label:>6} "
                f"| {auc['AUC@1px']:>7.2f}% "
                f"| {auc['AUC@3px']:>7.2f}% "
                f"| {auc['AUC@5px']:>7.2f}% "
                f"| {auc['AUC@10px']:>8.2f}%"
            )

    print("\n" + "=" * 75)

    # Save text report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "hpatches_baselines.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HPatches Baseline Comparison Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for category in categories:
            f.write(f"\n{category.upper()}\n")
            f.write(f"{'Method':<12} | {'Dim':>6} | {'AUC@1px':>8} | {'AUC@3px':>8} | {'AUC@5px':>8} | {'AUC@10px':>9}\n")
            f.write("-" * 70 + "\n")
            for method_name, method_errors in results_dict.items():
                if method_errors is None:
                    continue
                cat_errors = method_errors.get(category, [])
                if len(cat_errors) == 0:
                    continue
                auc = compute_homography_auc(cat_errors)
                dim_label = {"SIFT": "128", "ORB": "256b"}.get(method_name, "24")
                f.write(
                    f"{method_name:<12} | {dim_label:>6} "
                    f"| {auc['AUC@1px']:>7.2f}% "
                    f"| {auc['AUC@3px']:>7.2f}% "
                    f"| {auc['AUC@5px']:>7.2f}% "
                    f"| {auc['AUC@10px']:>8.2f}%\n"
                )
    print(f"\n  Results saved to: {report_path}")

    # Save markdown summary
    docs_dir = os.path.join(base_dir, "docs", "paper_notes")
    os.makedirs(docs_dir, exist_ok=True)
    md_path = os.path.join(docs_dir, "hpatches_baselines.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPatches Baseline Comparison\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

        for category in categories:
            f.write(f"\n## {category.capitalize()}\n\n")
            f.write("| Method | Dim | AUC@1px | AUC@3px | AUC@5px | AUC@10px |\n")
            f.write("| :--- | ---: | ---: | ---: | ---: | ---: |\n")
            for method_name, method_errors in results_dict.items():
                if method_errors is None:
                    continue
                cat_errors = method_errors.get(category, [])
                if len(cat_errors) == 0:
                    continue
                auc = compute_homography_auc(cat_errors)
                dim_label = {"SIFT": "128", "ORB": "256b"}.get(method_name, "24")
                bold = "**" if method_name not in ("SIFT", "ORB") else ""
                f.write(
                    f"| {bold}{method_name}{bold} | {dim_label} "
                    f"| {auc['AUC@1px']:.2f}% "
                    f"| {auc['AUC@3px']:.2f}% "
                    f"| {auc['AUC@5px']:.2f}% "
                    f"| {auc['AUC@10px']:.2f}% |\n"
                )

    print(f"  Markdown saved to: {md_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SIFT and ORB baselines on HPatches for comparison with QSED.",
    )
    parser.add_argument(
        "--dataset-dir", type=str,
        default=os.path.join("data", "hpatches-sequences-release"),
        help="Path to HPatches sequences directory"
    )
    parser.add_argument(
        "--max-keypoints", type=int, default=1024,
        help="Maximum keypoints per image (default: 1024)"
    )
    parser.add_argument(
        "--max-side", type=int, default=480,
        help="Resize longest image side (default: 480)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join("outputs", "hpatches"),
        help="Output directory for results (default: outputs/hpatches)"
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, args.dataset_dir) if not os.path.isabs(args.dataset_dir) else args.dataset_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    print("=" * 75)
    print("   HPatches Baseline Evaluation (SIFT + ORB)")
    print("=" * 75)
    print(f"  Max keypoints : {args.max_keypoints}")
    print(f"  Max side      : {args.max_side}px")
    print(f"  Dataset       : {dataset_dir}")
    print("=" * 75)

    sequences = get_hpatches_sequences(dataset_dir)
    print(f"\n  Found {len(sequences)} sequences\n")

    results = {}

    # SIFT baseline
    print("\n--- Running SIFT baseline ---")
    start = time.time()
    sift_errors = run_sift_baseline(sequences, args.max_keypoints, args.max_side)
    if sift_errors is not None:
        print(f"  SIFT completed in {time.time() - start:.1f}s")
        results["SIFT"] = sift_errors

    # ORB baseline
    print("\n--- Running ORB baseline ---")
    start = time.time()
    orb_errors = run_orb_baseline(sequences, args.max_keypoints, args.max_side)
    print(f"  ORB completed in {time.time() - start:.1f}s")
    results["ORB"] = orb_errors

    # Print comparison
    print_comparison_table(results, output_dir, base_dir)


if __name__ == "__main__":
    main()
