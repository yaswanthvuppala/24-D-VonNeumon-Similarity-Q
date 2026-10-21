"""
Oxford Affine Covariant Regions Benchmark for QSED 24-D Entropy Descriptor.

Evaluates the QSED quantum descriptor on the classic Oxford Affine benchmark
(Mikolajczyk & Schmid, IJCV 2005) across 8 challenging real-world transformations:
  - Viewpoint variation: 'graf', 'wall'
  - Scale & Rotation:     'boat', 'bark'
  - Image Blur:          'bikes', 'trees'
  - Illumination change: 'leuven'
  - JPEG compression:    'ubc'

Features automated dataset download (~50 MB total from Oxford University),
full detect -> describe -> match -> RANSAC homography estimation, and
AUC calculation matching the HPatches / RIPE evaluation protocol.

Usage:
  python scripts/oxford_eval.py
  python scripts/oxford_eval.py --download
  python scripts/oxford_eval.py --dimension 24 --norm l2
"""

import os
import sys
import time
import argparse
import tarfile
import urllib.request
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
from src.feature.hpatches_adapter import (
    estimate_homography_ransac,
    compute_corner_error,
    compute_homography_auc,
)


OXFORD_SEQUENCES = {
    "graf":   "Viewpoint (Planar)",
    "wall":   "Viewpoint (Planar)",
    "boat":   "Scale & Rotation",
    "bark":   "Scale & Rotation",
    "bikes":  "Image Blur",
    "trees":  "Image Blur",
    "leuven": "Illumination Change",
    "ubc":    "JPEG Compression",
}

OXFORD_BASE_URL = "https://thor.robots.ox.ac.uk/affine/"


# =============================================================================
# DATASET DOWNLOAD & EXTRACTION
# =============================================================================

def download_oxford_dataset(target_dir: str):
    """Download and extract all 8 Oxford Affine sequences (~50 MB)."""
    os.makedirs(target_dir, exist_ok=True)
    print("\n  Downloading Oxford Affine Benchmark (~50 MB total)...")

    for seq_name in OXFORD_SEQUENCES:
        seq_folder = os.path.join(target_dir, seq_name)
        img1 = os.path.join(seq_folder, "img1.ppm")
        if os.path.isfile(img1):
            continue

        tar_filename = f"{seq_name}.tar.gz"
        tar_path = os.path.join(target_dir, tar_filename)
        url = f"{OXFORD_BASE_URL}{tar_filename}"

        print(f"    Fetching {seq_name} ({OXFORD_SEQUENCES[seq_name]})...")
        try:
            urllib.request.urlretrieve(url, tar_path)
            os.makedirs(seq_folder, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=seq_folder)
            os.remove(tar_path)
        except Exception as e:
            print(f"    [Error] Failed to download {seq_name}: {e}")

    print("  Oxford dataset ready!\n")


def find_image_file(seq_folder: str, idx: int) -> str:
    """Find image file by index supporting .ppm, .png, .pgm."""
    for ext in ["ppm", "png", "pgm"]:
        candidate = os.path.join(seq_folder, f"img{idx}.{ext}")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Image {idx} not found in {seq_folder}")


def load_oxford_homography(seq_folder: str, ref_idx: int, tgt_idx: int) -> np.ndarray:
    """Load 3x3 homography matrix (e.g. H1to2p)."""
    h_path = os.path.join(seq_folder, f"H{ref_idx}to{tgt_idx}p")
    if not os.path.isfile(h_path):
        raise FileNotFoundError(f"Homography file not found: {h_path}")
    return np.loadtxt(h_path).reshape(3, 3).astype(np.float64)


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def run_oxford_evaluation(args):
    """Run QSED evaluation across all 8 Oxford sequences."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, args.dataset_dir) if not os.path.isabs(args.dataset_dir) else args.dataset_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Auto-download if dataset directory is missing or empty
    if not os.path.isdir(dataset_dir) or len(os.listdir(dataset_dir)) == 0 or args.download:
        download_oxford_dataset(dataset_dir)

    print("=" * 76)
    print("   QSED 24-D Descriptor Evaluation on Oxford Affine Benchmark")
    print("=" * 76)
    print(f"  Descriptor dimension : {args.dimension}-D")
    print(f"  Normalization        : {args.norm}")
    print(f"  Matching method      : {args.matching}")
    print(f"  Max keypoints        : {args.max_keypoints}")
    print(f"  Dataset directory    : {dataset_dir}")
    print("=" * 76)

    seq_results = {}
    all_errors = []

    start_time = time.time()

    for seq_name, challenge in OXFORD_SEQUENCES.items():
        seq_folder = os.path.join(dataset_dir, seq_name)
        if not os.path.isdir(seq_folder):
            continue

        ref_img_path = find_image_file(seq_folder, 1)
        img_ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        kps_ref = detect_default_keypoints(img_ref, max_keypoints=args.max_keypoints)
        desc_ref = extract_all_entropy_descriptors(
            img_ref, kps_ref, ordering=args.ordering, norm=args.norm, dimension=args.dimension
        )

        seq_errors = []

        for tgt_idx in range(2, 7):
            tgt_img_path = find_image_file(seq_folder, tgt_idx)
            img_tgt = cv2.imread(tgt_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            H_gt = load_oxford_homography(seq_folder, 1, tgt_idx)

            kps_tgt = detect_default_keypoints(img_tgt, max_keypoints=args.max_keypoints)
            desc_tgt = extract_all_entropy_descriptors(
                img_tgt, kps_tgt, ordering=args.ordering, norm=args.norm, dimension=args.dimension
            )

            match_res = match_entropy_descriptors(desc_ref, desc_tgt, method=args.matching)
            H_est = estimate_homography_ransac(kps_ref, kps_tgt, match_res["matches"], ransac_thresh=args.ransac_thresh)

            if H_est is not None:
                err = compute_corner_error(H_gt, H_est, img_ref.shape)
            else:
                err = float('inf')

            seq_errors.append(err)
            all_errors.append(err)

        auc = compute_homography_auc(seq_errors, thresholds=[1, 3, 5])
        seq_results[seq_name] = {
            "challenge": challenge,
            "errors": seq_errors,
            "auc": auc
        }

    elapsed = time.time() - start_time

    # Overall AUC
    overall_auc = compute_homography_auc(all_errors, thresholds=[1, 3, 5])

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("   RESULTS: Oxford Affine Benchmark (Homography AUC)")
    print("=" * 76)
    print(f"  {'Sequence':<10} | {'Transformation / Challenge':<25} | {'AUC@1px':>8} | {'AUC@3px':>8} | {'AUC@5px':>8}")
    print("  " + "-" * 72)

    for seq_name, res in seq_results.items():
        a = res["auc"]
        print(f"  {seq_name:<10} | {res['challenge']:<25} | {a['AUC@1px']:>7.2f}% | {a['AUC@3px']:>7.2f}% | {a['AUC@5px']:>7.2f}%")

    print("  " + "=" * 72)
    print(f"  {'OVERALL':<10} | {'All 8 Sequences (40 pairs)':<25} | {overall_auc['AUC@1px']:>7.2f}% | {overall_auc['AUC@3px']:>7.2f}% | {overall_auc['AUC@5px']:>7.2f}%")
    print("=" * 76)
    print(f"  Completed in {elapsed:.1f}s\n")

    # Save to report file
    report_file = os.path.join(output_dir, "oxford_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("QSED 24-D Descriptor - Oxford Affine Benchmark Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Sequence':<10} | {'Challenge':<25} | {'AUC@1px':>8} | {'AUC@3px':>8} | {'AUC@5px':>8}\n")
        f.write("-" * 72 + "\n")
        for seq_name, res in seq_results.items():
            a = res["auc"]
            f.write(f"{seq_name:<10} | {res['challenge']:<25} | {a['AUC@1px']:>7.2f}% | {a['AUC@3px']:>7.2f}% | {a['AUC@5px']:>7.2f}%\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'OVERALL':<10} | {'All 8 Sequences':<25} | {overall_auc['AUC@1px']:>7.2f}% | {overall_auc['AUC@3px']:>7.2f}% | {overall_auc['AUC@5px']:>7.2f}%\n")
    print(f"  Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate QSED descriptor on Oxford Affine Benchmark.")
    parser.add_argument("--dataset-dir", type=str, default=os.path.join("data", "oxford-affine"),
                        help="Path to Oxford Affine directory")
    parser.add_argument("--download", action="store_true", help="Force re-downloading Oxford dataset")
    parser.add_argument("--max-keypoints", type=int, default=1024, help="Max keypoints per image")
    parser.add_argument("--dimension", type=int, default=24, choices=[9, 24], help="Descriptor dimension")
    parser.add_argument("--norm", type=str, default="l2", choices=["none", "l2", "unit_entropy"], help="Normalization")
    parser.add_argument("--ordering", type=str, default="raster", help="Probe ordering")
    parser.add_argument("--matching", type=str, default="mutual_nn", help="Matching method")
    parser.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC threshold")
    parser.add_argument("--output-dir", type=str, default=os.path.join("outputs", "oxford"), help="Output directory")
    args = parser.parse_args()

    run_oxford_evaluation(args)


if __name__ == "__main__":
    main()
