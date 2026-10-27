"""
HPatches Benchmark Evaluation for QSED 24-D Von Neumann Entropy Descriptor.

Evaluates the QSED quantum descriptor on the HPatches dataset for homography
estimation under viewpoint and illumination changes.

Pipeline per image pair:
  1. Load grayscale images, resize (longest side → max_side px)
  2. Detect keypoints using QSED gradient + directional variation pipeline
  3. Extract 24-D Von Neumann entropy descriptors
  4. Match descriptors via Mutual Nearest Neighbor
  5. Estimate homography via RANSAC
  6. Compute corner reprojection error vs ground-truth H
  7. Report AUC at {1, 3, 5, 10} pixel thresholds

Usage:
  python scripts/hpatches_eval.py --dataset-dir data/hpatches-sequences-release
  python scripts/hpatches_eval.py --dimension 9 --norm none
  python scripts/hpatches_eval.py --max-keypoints 512 --matching mutual_nn
  python scripts/hpatches_eval.py --help
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
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
    load_hpatches_image,
    load_homography,
    scale_homography,
    estimate_homography_ransac,
    compute_corner_error,
    compute_homography_auc,
    get_hpatches_sequences,
    get_sequence_pairs,
    format_ripe_comparison_table,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate QSED 24-D descriptor on HPatches for homography estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/hpatches_eval.py --dataset-dir data/hpatches-sequences-release
  python scripts/hpatches_eval.py --dimension 9 --norm none --ordering raster
  python scripts/hpatches_eval.py --max-keypoints 512 --matching mutual_nn
        """
    )
    parser.add_argument(
        "--dataset-dir", type=str,
        default=os.path.join("data", "hpatches-sequences-release"),
        help="Path to HPatches sequences directory (default: data/hpatches-sequences-release)"
    )
    parser.add_argument(
        "--max-keypoints", type=int, default=1024,
        help="Maximum keypoints per image (default: 1024)"
    )
    parser.add_argument(
        "--max-side", type=int, default=480,
        help="Resize longest image side to this value in pixels (default: 480)"
    )
    parser.add_argument(
        "--dimension", type=int, default=24, choices=[9, 24],
        help="Descriptor dimension: 9 or 24 (default: 24)"
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Odd outer probe-neighborhood size; produces n*n-1 values")
    parser.add_argument("--y", type=int, default=None,
                        help="Odd entropy-patch size used at every probe")
    parser.add_argument(
        "--norm", type=str, default="l2", choices=["none", "l2", "unit_entropy"],
        help="Descriptor normalization strategy (default: l2)"
    )
    parser.add_argument(
        "--ordering", type=str, default="raster", choices=["raster", "clockwise"],
        help="Position ordering for 24-D descriptor (default: raster)"
    )
    parser.add_argument(
        "--matching", type=str, default="mutual_nn",
        choices=["mutual_nn", "nn_a_to_b", "greedy"],
        help="Descriptor matching strategy (default: mutual_nn)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join("outputs", "hpatches"),
        help="Output directory for results (default: outputs/hpatches)"
    )
    parser.add_argument(
        "--ransac-thresh", type=float, default=3.0,
        help="RANSAC reprojection threshold in pixels (default: 3.0)"
    )
    args = parser.parse_args()
    if (args.n is None) != (args.y is None):
        parser.error("--n and --y must be supplied together")
    if args.n is not None and (args.n <= 0 or args.y <= 0 or args.n % 2 == 0 or args.y % 2 == 0 or args.y > args.n):
        parser.error("--n and --y must be positive odd integers with y <= n")
    return args


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_hpatches(args):
    """Run the full HPatches evaluation pipeline."""

    # Resolve paths relative to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, args.dataset_dir) if not os.path.isabs(args.dataset_dir) else args.dataset_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect if running on Kaggle with mounted dataset
    if not os.path.isdir(dataset_dir) and os.path.isdir("/kaggle/input"):
        print("  [Kaggle Environment Detected] Checking /kaggle/input for HPatches sequences...")
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "hpatches-sequences-release" in dirs:
                candidate = os.path.join(root, "hpatches-sequences-release")
                if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "v_weapons", "1.ppm")):
                    dataset_dir = candidate
                    print(f"  --> Found HPatches dataset at: {dataset_dir}")
                    break

    # -------------------------------------------------------------------------
    # 1. Discover sequences
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("   QSED HPatches Benchmark Evaluation")
    print("=" * 75)
    descriptor_dimension = args.n * args.n - 1 if args.n is not None else args.dimension
    print(f"  Descriptor dimension : {descriptor_dimension}-D")
    print(f"  Normalization        : {args.norm}")
    print(f"  Ordering             : {args.ordering}")
    print(f"  Matching method      : {args.matching}")
    print(f"  Max keypoints        : {args.max_keypoints}")
    print(f"  Max image side       : {args.max_side}px")
    print(f"  RANSAC threshold     : {args.ransac_thresh}px")
    print(f"  Dataset directory    : {dataset_dir}")
    print(f"  Output directory     : {output_dir}")
    print("=" * 75)

    sequences = get_hpatches_sequences(dataset_dir)
    n_illum = sum(1 for s in sequences if s["type"] == "illumination")
    n_view = sum(1 for s in sequences if s["type"] == "viewpoint")
    print(f"\n  Found {len(sequences)} sequences ({n_illum} illumination, {n_view} viewpoint)")
    print(f"  Total image pairs: {len(sequences) * 5}\n")

    # -------------------------------------------------------------------------
    # 2. Evaluation loop
    # -------------------------------------------------------------------------
    errors = {"illumination": [], "viewpoint": [], "all": []}
    match_counts = {"illumination": [], "viewpoint": [], "all": []}
    failed_homographies = {"illumination": 0, "viewpoint": 0, "all": 0}
    per_pair_records = []

    start_time = time.time()

    for seq_info in tqdm(sequences, desc="Evaluating sequences", unit="seq"):
        seq_name = seq_info["name"]
        seq_path = seq_info["path"]
        seq_type = seq_info["type"]

        # Load reference image and extract features once per sequence
        ref_path = os.path.join(seq_path, "1.ppm")
        img_ref, scale_ref = load_hpatches_image(ref_path, max_side=args.max_side)

        kps_ref = detect_default_keypoints(img_ref, max_keypoints=args.max_keypoints)
        if len(kps_ref) == 0:
            # Skip sequence if no keypoints found in reference
            for pair_info in get_sequence_pairs(seq_path):
                per_pair_records.append({
                    "sequence": seq_name,
                    "type": seq_type,
                    "pair": pair_info["pair_name"],
                    "num_kps_ref": 0,
                    "num_kps_tgt": 0,
                    "num_matches": 0,
                    "corner_error": float('inf'),
                })
                errors[seq_type].append(float('inf'))
                errors["all"].append(float('inf'))
                failed_homographies[seq_type] += 1
                failed_homographies["all"] += 1
            continue

        desc_ref = extract_all_entropy_descriptors(
            img_ref, kps_ref,
            ordering=args.ordering, norm=args.norm, dimension=args.dimension, n=args.n, y=args.y
        )

        # Evaluate each pair
        pairs = get_sequence_pairs(seq_path)
        for pair_info in pairs:
            # Load target image
            img_tgt, scale_tgt = load_hpatches_image(pair_info["tgt_path"], max_side=args.max_side)

            # Load and scale ground-truth homography
            H_gt = load_homography(pair_info["H_path"])
            H_gt_scaled = scale_homography(H_gt, scale_ref, scale_tgt)

            # Detect and describe target
            kps_tgt = detect_default_keypoints(img_tgt, max_keypoints=args.max_keypoints)
            desc_tgt = extract_all_entropy_descriptors(
                img_tgt, kps_tgt,
                ordering=args.ordering, norm=args.norm, dimension=args.dimension, n=args.n, y=args.y
            )

            # Match descriptors
            match_result = match_entropy_descriptors(
                desc_ref, desc_tgt, method=args.matching
            )
            num_matches = match_result["num_matches"]

            # Estimate homography
            H_est = estimate_homography_ransac(
                kps_ref, kps_tgt, match_result["matches"],
                ransac_thresh=args.ransac_thresh
            )

            # Compute error
            if H_est is not None:
                err = compute_corner_error(H_gt_scaled, H_est, img_ref.shape)
            else:
                err = float('inf')
                failed_homographies[seq_type] += 1
                failed_homographies["all"] += 1

            # Accumulate results
            errors[seq_type].append(err)
            errors["all"].append(err)
            match_counts[seq_type].append(num_matches)
            match_counts["all"].append(num_matches)

            per_pair_records.append({
                "sequence": seq_name,
                "type": seq_type,
                "pair": pair_info["pair_name"],
                "num_kps_ref": len(kps_ref),
                "num_kps_tgt": len(kps_tgt),
                "num_matches": num_matches,
                "corner_error": err,
            })

    elapsed = time.time() - start_time

    # -------------------------------------------------------------------------
    # 3. Compute and display results
    # -------------------------------------------------------------------------
    print("\n")
    print("=" * 75)
    print("   RESULTS: QSED {}-D Descriptor on HPatches".format(descriptor_dimension))
    print("=" * 75)

    results_text_lines = []
    results_text_lines.append("=" * 75)
    results_text_lines.append("   QSED HPatches Benchmark Results")
    results_text_lines.append("=" * 75)
    results_text_lines.append(f"  Date               : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text_lines.append(f"  Descriptor          : {descriptor_dimension}-D Von Neumann Entropy")
    results_text_lines.append(f"  Normalization       : {args.norm}")
    results_text_lines.append(f"  Ordering            : {args.ordering}")
    results_text_lines.append(f"  Matching            : {args.matching}")
    results_text_lines.append(f"  Max keypoints       : {args.max_keypoints}")
    results_text_lines.append(f"  Max image side      : {args.max_side}px")
    results_text_lines.append(f"  RANSAC threshold    : {args.ransac_thresh}px")
    results_text_lines.append(f"  Total time          : {elapsed:.1f}s")
    results_text_lines.append("")

    for category in ["illumination", "viewpoint", "all"]:
        cat_errors = errors[category]
        cat_matches = match_counts[category]
        cat_failed = failed_homographies[category]

        if len(cat_errors) == 0:
            continue

        auc = compute_homography_auc(cat_errors)
        n_pairs = len(cat_errors)

        header = f"\n  {category.upper()} ({n_pairs} pairs)"
        print(header)
        results_text_lines.append(header)

        print("-" * 55)
        results_text_lines.append("-" * 55)

        for key, val in auc.items():
            line = f"    {key:>10s} : {val:6.2f}%"
            print(line)
            results_text_lines.append(line)

        # Matching statistics
        if len(cat_matches) > 0:
            mc = np.array(cat_matches)
            stats_line = (
                f"    Matches: mean={np.mean(mc):.1f}, "
                f"median={np.median(mc):.0f}, "
                f"min={np.min(mc)}, max={np.max(mc)}"
            )
            print(stats_line)
            results_text_lines.append(stats_line)

        fail_line = f"    Failed homographies: {cat_failed}/{n_pairs}"
        print(fail_line)
        results_text_lines.append(fail_line)

    # -------------------------------------------------------------------------
    # Comparison against RIPE Paper Table 2
    # -------------------------------------------------------------------------
    all_auc = compute_homography_auc(errors["all"], thresholds=[1, 3, 5])
    ripe_table_str = format_ripe_comparison_table(
        our_auc1=all_auc.get("AUC@1px", 0.0),
        our_auc3=all_auc.get("AUC@3px", 0.0),
        our_auc5=all_auc.get("AUC@5px", 0.0),
        model_name=f"QSED-{descriptor_dimension}D (Ours)",
        dim=descriptor_dimension
    )
    print("\n" + ripe_table_str + "\n")
    results_text_lines.append("\n" + ripe_table_str + "\n")

    # -------------------------------------------------------------------------
    # 4. Save results
    # -------------------------------------------------------------------------

    # Save text report
    report_path = os.path.join(output_dir, "hpatches_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text_lines))
    print(f"\n  Results saved to: {report_path}")

    # Save per-pair CSV
    csv_path = os.path.join(output_dir, "hpatches_per_pair.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence", "type", "pair", "num_kps_ref", "num_kps_tgt",
            "num_matches", "corner_error"
        ])
        writer.writeheader()
        for record in per_pair_records:
            writer.writerow(record)
    print(f"  Per-pair CSV saved to: {csv_path}")

    # Save markdown summary
    docs_dir = os.path.join(base_dir, "docs", "paper_notes")
    os.makedirs(docs_dir, exist_ok=True)
    md_path = os.path.join(docs_dir, "hpatches_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPatches Benchmark Results\n\n")
        f.write(f"**Descriptor:** QSED {descriptor_dimension}-D Von Neumann Entropy  \n")
        f.write(f"**Normalization:** {args.norm}  \n")
        f.write(f"**Matching:** {args.matching}  \n")
        f.write(f"**Max keypoints:** {args.max_keypoints}  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

        f.write("## Homography Estimation AUC\n\n")
        f.write("| Category | Pairs | AUC@1px | AUC@3px | AUC@5px | AUC@10px | Failed H |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")

        for category in ["illumination", "viewpoint", "all"]:
            cat_errors = errors[category]
            if len(cat_errors) == 0:
                continue
            auc = compute_homography_auc(cat_errors)
            cat_failed = failed_homographies[category]
            f.write(
                f"| {category.capitalize()} | {len(cat_errors)} "
                f"| {auc['AUC@1px']:.2f}% "
                f"| {auc['AUC@3px']:.2f}% "
                f"| {auc['AUC@5px']:.2f}% "
                f"| {auc['AUC@10px']:.2f}% "
                f"| {cat_failed} |\n"
            )

        f.write("\n## Matching Statistics\n\n")
        f.write("| Category | Mean Matches | Median | Min | Max |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: |\n")
        for category in ["illumination", "viewpoint", "all"]:
            mc = match_counts[category]
            if len(mc) > 0:
                mc_arr = np.array(mc)
                f.write(
                    f"| {category.capitalize()} "
                    f"| {np.mean(mc_arr):.1f} "
                    f"| {np.median(mc_arr):.0f} "
                    f"| {np.min(mc_arr)} "
                    f"| {np.max(mc_arr)} |\n"
                )

    print(f"  Markdown summary saved to: {md_path}")
    print(f"\n  Evaluation completed in {elapsed:.1f}s")

    return errors, match_counts


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    args = parse_args()
    evaluate_hpatches(args)


if __name__ == "__main__":
    main()
