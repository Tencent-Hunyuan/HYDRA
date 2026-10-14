# Copyright 2026 HYDRA Authors.
# Modified from pyiqa / pytorch-fid examples
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import glob
import sys

# rFID uses the FID-Inception network from pytorch-fid, the same extractor as
# tools/eval_top12.py. Weights are fetched once from the public release URL.
from pytorch_fid.fid_score import calculate_fid_given_paths


def main(args):
    # --- 1. Verify image folders exist ---
    originals_base_dir = os.path.join(args.path, 'originals')
    recons_base_dir = os.path.join(args.path, 'reconstructions')

    if not os.path.isdir(originals_base_dir) or not os.path.isdir(recons_base_dir):
        print(f"Error: 'originals' and/or 'reconstructions' folder not found under base directory '{args.path}'.")
        return

    recon_files = sorted(glob.glob(os.path.join(recons_base_dir, '*.png')))
    if not recon_files:
        print(f"Error: no image files found in '{recons_base_dir}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found {len(recon_files)} reconstructions | device: {device}")

    # --- 2. Compute and print rFID ---
    # consolidate_images() has already flattened every rank_* folder into the two
    # directories below, so a non-recursive glob sees all the images.
    print("Calculating rFID (this may take a while)...")
    try:
        rfid_score = calculate_fid_given_paths(
            [originals_base_dir, recons_base_dir],
            batch_size=args.batch_size,
            device=device,
            dims=2048,
            num_workers=args.num_workers,
        )
        print(f"  rFID ↓ : {rfid_score:.4f}")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"rFID computation failed: {e}")

    print("="*50)

    # --- 3. (optional) clean up folders ---
    if args.clean_up:
        print(f"Cleaning up consolidated image directories...")
        shutil.rmtree(originals_base_dir)
        shutil.rmtree(recons_base_dir)
        print("Done.")


def consolidate_images(base_path):
    originals_all_dir = os.path.join(base_path, 'originals')
    recons_all_dir = os.path.join(base_path, 'reconstructions')

    if os.path.exists(originals_all_dir) and os.path.exists(recons_all_dir):
        print("Consolidated folders already exist; skipping consolidation.")
        return

    print("Images not consolidated yet; consolidating images from all rank_* folders...")
    os.makedirs(originals_all_dir, exist_ok=True)
    os.makedirs(recons_all_dir, exist_ok=True)

    rank_dirs = sorted(glob.glob(os.path.join(base_path, 'rank_*')))
    if not rank_dirs:
        print(f"Error: no 'rank_*' subfolders found under '{base_path}'.")
        return

    for rank_dir in tqdm(rank_dirs, desc="Consolidating images"):
        for img_path in glob.glob(os.path.join(rank_dir, 'originals', '*.png')):
            shutil.copy2(img_path, originals_all_dir)
        for img_path in glob.glob(os.path.join(rank_dir, 'reconstructions', '*.png')):
            shutil.copy2(img_path, recons_all_dir)
    print("Image consolidation done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate metrics (LPIPS via PyIQA, rFID) from pre-saved image folders.")

    parser.add_argument('--path', type=str, required=True, 
                        help='Path to the BASE directory containing the rank_* subfolders.')

    parser.add_argument('--clean-up', action='store_true',
                        help='If set, deletes the consolidated image directories after evaluation.')
    parser.add_argument('--consolidate-only', action='store_true',
                        help='If set, only consolidates images and then exits.')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for the FID-Inception forward pass.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Data-loading workers for the FID-Inception forward pass.')

    args = parser.parse_args()

    consolidate_images(args.path)

    if not args.consolidate_only:
        main(args)
# python eval_fid.py \
#     --path /path/to/outputs/eval_64c_ddp_coco \
#     --consolidate-only
# python eval_fid.py \
#     --path /path/to/outputs/eval_64c_ddp_coco
