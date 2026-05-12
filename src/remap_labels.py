# ============================================================
# remap_labels.py
# GalaxEye Space — Standalone Label Remapping Utility
# ============================================================
# Use this script to verify or pre-apply the mandatory label
# remapping to .tif mask files BEFORE training.
#
# Remapping (assignment requirement):
#   0 (Background) → 0 (No-Change)
#   1 (Intact)     → 0 (No-Change)
#   2 (Damaged)    → 1 (Change)
#   3 (Destroyed)  → 1 (Change)
#
# Usage:
#   python src/remap_labels.py --verify          (just print statistics)
#   python src/remap_labels.py --apply           (write remapped .tif files)
#   python src/remap_labels.py --verify --split train
#
# NOTE: The dataset.py DataLoader applies this remapping on-the-fly
# during training — you do NOT need to run this script to train.
# It is provided for data inspection and pre-processing convenience.
# ============================================================

import os
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from collections import Counter


# Mandatory remapping table
REMAP = {0: 0, 1: 0, 2: 1, 3: 1}


def remap_array(arr: np.ndarray) -> np.ndarray:
    """Apply label remapping to a numpy array."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for orig, new in REMAP.items():
        out[arr == orig] = new
    return out


def verify_split(split_dir: str, split_name: str):
    """
    Print class distribution statistics for a dataset split.
    Shows both ORIGINAL and REMAPPED label counts.
    """
    mask_dir = os.path.join(split_dir, "target")
    if not os.path.isdir(mask_dir):
        print(f"[SKIP] {split_name}: target directory not found at {mask_dir}")
        return

    files = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    if not files:
        print(f"[SKIP] {split_name}: no .tif files in {mask_dir}")
        return

    orig_counter    = Counter()
    remapped_counter = Counter()
    total_pixels    = 0

    for fname in files:
        path = os.path.join(mask_dir, fname)
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.int32)

        orig_counter.update(arr.flatten().tolist())

        remapped = remap_array(arr)
        remapped_counter.update(remapped.flatten().tolist())
        total_pixels += arr.size

    print(f"\n{'='*55}")
    print(f"Split: {split_name.upper()}  |  Files: {len(files)}  |  Total pixels: {total_pixels:,}")
    print(f"\nOriginal label distribution:")
    for label in sorted(orig_counter):
        cnt  = orig_counter[label]
        pct  = 100.0 * cnt / total_pixels
        name = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}.get(label, "Unknown")
        print(f"   {label} ({name:12s}): {cnt:>10,} pixels  ({pct:.2f}%)")

    print(f"\nRemapped label distribution:")
    for label in sorted(remapped_counter):
        cnt  = remapped_counter[label]
        pct  = 100.0 * cnt / total_pixels
        name = {0: "No-Change", 1: "Change"}.get(label, "Unknown")
        print(f"   {label} ({name:10s}): {cnt:>10,} pixels  ({pct:.2f}%)")

    # Class imbalance ratio
    n_change    = remapped_counter.get(1, 0)
    n_no_change = remapped_counter.get(0, 0)
    if n_change > 0:
        ratio = n_no_change / n_change
        print(f"\nClass imbalance ratio (no-change / change): {ratio:.1f}x")
        print(f"Suggested pos_weight for BCE loss: {ratio:.1f}")
    else:
        print("\nWARNING: No 'change' pixels found in this split!")


def apply_remapping(split_dir: str, split_name: str, output_dir: str):
    """
    Read original masks, apply remapping, and save to output_dir.
    Only needed if you want remapped masks on disk.
    """
    mask_dir    = os.path.join(split_dir, "target")
    out_mask_dir = os.path.join(output_dir, split_name, "target")
    os.makedirs(out_mask_dir, exist_ok=True)

    files = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    print(f"\n[Apply] Remapping {len(files)} masks for split: {split_name}")

    for fname in files:
        src_path  = os.path.join(mask_dir, fname)
        dest_path = os.path.join(out_mask_dir, fname)

        with rasterio.open(src_path) as src:
            arr      = src.read(1).astype(np.int32)
            profile  = src.profile.copy()

        remapped = remap_array(arr)
        profile.update(dtype=rasterio.uint8, count=1)

        with rasterio.open(dest_path, "w", **profile) as dst:
            dst.write(remapped[np.newaxis, :, :])

    print(f"[Apply] Saved remapped masks to {out_mask_dir}")


# ── CLI entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify or apply mandatory GalaxEye label remapping."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Print label distribution before and after remapping"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write remapped .tif masks to --output_dir"
    )
    parser.add_argument(
        "--split", default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to process (default: all)"
    )
    parser.add_argument(
        "--data_root", default="data",
        help="Root data directory containing train/val/test (default: data)"
    )
    parser.add_argument(
        "--output_dir", default="data_remapped",
        help="Output directory for remapped masks (only used with --apply)"
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        split_dir = os.path.join(args.data_root, split)
        if args.verify:
            verify_split(split_dir, split)
        if args.apply:
            apply_remapping(split_dir, split, args.output_dir)

    if not args.verify and not args.apply:
        print("Nothing to do. Use --verify to inspect labels or --apply to remap.")


if __name__ == "__main__":
    main()
