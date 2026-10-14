# Copyright 2026 HYDRA Authors.
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import shutil
import glob
import sys

# === DDP-related imports ===
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# =====================

# rFID uses the FID-Inception network from pytorch-fid (the TF-Inception port that
# every published FID number is based on). Weights are fetched once from the
# public release URL and cached under ~/.cache/torch/hub/checkpoints.
from pytorch_fid.fid_score import calculate_fid_given_paths

# skimage is used to compute PSNR and SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def import_model_from_config_path(config_path):
    """
    Dynamically import the config class and model class from config_path.
    
    Args:
        config_path: config directory path, e.g. '/path/to/hydra_tok' or 'hydra_tok'
    
    Returns:
        (HydraTokConfig, HydraTokVisionModel): the config class and model class
    """
    # Ensure the HYDRA-TOK root is in sys.path. tools/ lives directly under it,
    # so the root is the parent of this file's directory.
    hydra_tok_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if hydra_tok_root not in sys.path:
        sys.path.insert(0, hydra_tok_root)

    # Extract the module name and parent dir from config_path
    config_path = config_path.rstrip('/')

    if os.path.isabs(config_path):
        # Absolute path: /path/to/hydra_tok
        # Check whether it is under the HYDRA-TOK root
        if config_path.startswith(hydra_tok_root):
            # Under the HYDRA-TOK root: use the relative path as the module name
            module_name = os.path.relpath(config_path, hydra_tok_root).replace(os.sep, '.')
        else:
            # Outside the HYDRA-TOK root: add the parent dir to sys.path and use the directory name as the module name
            parent_dir = os.path.dirname(config_path)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            module_name = os.path.basename(config_path)
    else:
        # Relative path: hydra_tok -> hydra_tok
        module_name = config_path.replace(os.sep, '.')

    # Dynamic import
    try:
        config_module = __import__(f"{module_name}.configuration_hydra_tok", fromlist=['HydraTokConfig'])
        model_module = __import__(f"{module_name}.modeling_hydra_tok", fromlist=['HydraTokVisionModel'])

        HydraTokConfig = getattr(config_module, 'HydraTokConfig')
        HydraTokVisionModel = getattr(model_module, 'HydraTokVisionModel')

        return HydraTokConfig, HydraTokVisionModel
    except ImportError as e:
        raise ImportError(
            f"Failed to import the model from {config_path}. Make sure the path is correct and "
            f"contains configuration_hydra_tok.py and modeling_hydra_tok.py. "
            f"Extracted module name: {module_name}. Error: {e}"
        )

# Custom Dataset for reading the ImageNet validation set from a flat folder layout


class ImageNetValFlatDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_paths = sorted(glob.glob(os.path.join(self.root, '*.JPEG')))
        if not self.file_paths:
            print(f"Warning: no .JPEG files found in directory {self.root}.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, idx  # return image and its index for unique filenames

# Custom Dataset for reading the COCO validation set (multiple image formats supported)


class COCOValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Support common image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.file_paths = []
        for ext in extensions:
            self.file_paths.extend(glob.glob(os.path.join(self.root, ext)))
        self.file_paths = sorted(self.file_paths)
        if not self.file_paths:
            print(f"Warning: no image files found in directory {self.root}.")
        else:
            print(f"Found {len(self.file_paths)} images in directory {self.root}.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, idx  # return image and its index for unique filenames
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Warning: failed to load image {img_path}: {e}")
            # If the image is corrupt, fall back to the first image (avoids infinite recursion)
            if idx != 0:
                return self.__getitem__(0)
            else:
                # If the first image is also corrupt, create a black image
                image = Image.new('RGB', (224, 224), color='black')
                if self.transform:
                    image = self.transform(image)
                return image, idx


def setup_ddp():
    """Initialize the DDP environment."""
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up the DDP environment."""
    dist.destroy_process_group()


def main(args):
    # --- 0. Dynamically import the model from config_path ---
    if not hasattr(args, 'config_path') or args.config_path is None:
        raise ValueError("--config-path is required")

    print(f"Dynamically importing the model from {args.config_path}...")
    HydraTokConfig, HydraTokVisionModel = import_model_from_config_path(args.config_path)
    print(f"Model imported successfully")

    # --- 1. DDP environment setup ---
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Starting DDP evaluation on {world_size} GPUs.")
        print(f"Global batch size: {args.batch_size * world_size}, Per-GPU batch size: {args.batch_size}")
        print(f"Using model from config path: {args.config_path}")

    # --- 2. Model setup ---
    config = HydraTokConfig.from_pretrained(args.config_path)
    config.image_size = args.image_size
    config.patch_size = 14

    model = HydraTokVisionModel(config=config).to(device, dtype=torch.bfloat16).eval()

    # --- 3. Load model weights ---
    # All ranks load weights from the same file to keep model states consistent
    if rank == 0:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)  # load directly onto the current GPU
    if 'student_model' in checkpoint:
        state_dict = checkpoint['student_model']
    elif "model" in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    if rank == 0:
        print("Successfully loaded weights into the model on all ranks.")
    # exit(0)
    # Wrap the model with DDP
    model = DDP(model, device_ids=[local_rank])

    # --- 4. Prepare the data loader ---
    DATA_NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_NORM['mean'], std=DATA_NORM['std'])
    ])

    # Pick the dataset class based on the dataset type
    if args.dataset_type == 'coco':
        if not hasattr(args, 'coco_val_path') or args.coco_val_path is None:
            raise ValueError("--coco-val-path must be specified when using the COCO dataset")
        dataset = COCOValDataset(root=args.coco_val_path, transform=transform)
        dataset_name = "COCO validation"
    else:  # default to ImageNet
        if not hasattr(args, 'imagenet_val_path') or args.imagenet_val_path is None:
            raise ValueError("--imagenet-val-path must be specified when using the ImageNet dataset")
        dataset = ImageNetValFlatDataset(root=args.imagenet_val_path, transform=transform)
        dataset_name = "ImageNet validation"

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, shuffle=False)

    if rank == 0:
        print(f"Loaded {dataset_name} set with {len(dataset)} images.")
        print(f"Metrics will be calculated at evaluation size: {args.eval_size}x{args.eval_size}")

    # --- 5. Prepare rFID and metric computation ---
    # Paper protocol: downsample both the original and the reconstruction to
    # eval_size with PIL LANCZOS, write them to a temporary directory as PNG
    # (lossless), then compute rFID over those two directories. All ranks share
    # the same two output folders -- filenames are the global dataset index, so
    # there are no collisions and the sampler's padding duplicates collapse.
    local_psnr_scores, local_ssim_scores = [], []

    originals_dir = os.path.join(args.output_dir, 'originals')
    recons_dir = os.path.join(args.output_dir, 'reconstructions')
    if rank == 0 and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    dist.barrier()
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(recons_dir, exist_ok=True)

    mean = torch.tensor(DATA_NORM['mean']).view(1, 3, 1, 1)
    std = torch.tensor(DATA_NORM['std']).view(1, 3, 1, 1)

    # --- 6. Evaluation loop ---
    data_iterator = dataloader
    if rank == 0:
        data_iterator = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch, indices in data_iterator:
            batch = batch.to(device, dtype=torch.bfloat16)
            recons = model.module.sample(batch)

            originals_cpu = (batch.float().cpu() * std + mean).clamp(0, 1)
            recons_cpu = (recons.float().cpu() * std + mean).clamp(0, 1)

            originals_np = (originals_cpu.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            recons_np = (recons_cpu.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            for i in range(originals_np.shape[0]):
                orig_img_pil = Image.fromarray(originals_np[i])
                recon_img_pil = Image.fromarray(recons_np[i])

                if args.eval_size != args.image_size:
                    orig_img_pil = orig_img_pil.resize(
                        (args.eval_size, args.eval_size), Image.LANCZOS)
                    recon_img_pil = recon_img_pil.resize(
                        (args.eval_size, args.eval_size), Image.LANCZOS)

                orig_np_resized = np.array(orig_img_pil)
                recon_np_resized = np.array(recon_img_pil)

                local_psnr_scores.append(
                    psnr_metric(orig_np_resized, recon_np_resized, data_range=255))
                local_ssim_scores.append(
                    ssim_metric(orig_np_resized, recon_np_resized, channel_axis=-1,
                                data_range=255, win_size=7))

                global_idx = indices[i].item()
                orig_img_pil.save(os.path.join(originals_dir, f"{global_idx}.png"))
                recon_img_pil.save(os.path.join(recons_dir, f"{global_idx}.png"))

    # --- 7. Gather PSNR / SSIM from all ranks ---
    dist.barrier()  # ensure all ranks have finished the loop and flushed their images

    all_psnr_scores = [None] * world_size
    all_ssim_scores = [None] * world_size
    dist.all_gather_object(all_psnr_scores, local_psnr_scores)
    dist.all_gather_object(all_ssim_scores, local_ssim_scores)

    # Only the primary rank performs the final computation and printing
    if rank == 0:
        final_psnr = [s for sub in all_psnr_scores for s in sub]
        final_ssim = [s for sub in all_ssim_scores for s in sub]

        avg_psnr = np.mean(final_psnr)
        avg_ssim = np.mean(final_ssim)

        print("\n" + "="*50)
        dataset_name = "COCO" if args.dataset_type == 'coco' else "ImageNet"
        print(f"Evaluation Results on {dataset_name} at {args.eval_size}x{args.eval_size}:")
        print(f"  PSNR ↑ : {avg_psnr:.2f} dB")
        print(f"  SSIM ↑ : {avg_ssim:.4f}")

        # --- 8. Compute rFID over the two saved image directories ---
        print("Calculating rFID (this may take a while)...")
        rfid_score = calculate_fid_given_paths(
            [originals_dir, recons_dir],
            batch_size=args.fid_batch_size,
            device=device,
            dims=2048,
            num_workers=args.num_workers,
        )
        print(f"  rFID ↓ : {rfid_score:.4f}")
        print("="*50)

        # --- 9. Remove the temporary images ---
        if args.keep_images:
            print(f"Keeping temporary images under {args.output_dir} (--keep-images).")
        else:
            print("Cleaning up temporary directories...")
            shutil.rmtree(args.output_dir, ignore_errors=True)
            print("Done.")

    # Clean up the DDP environment. Note: no barrier here on purpose -- ranks 1..N
    # leave as soon as their images are on disk (the barrier above guarantees that),
    # so they do not sit in a collective while rank 0 runs the multi-minute FID pass
    # and trip the NCCL collective timeout.
    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate HYDRA-TOK model on ImageNet-1K or COCO with DDP.")

    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to model checkpoint (.pth).')
    parser.add_argument('--dataset-type', type=str, default='imagenet', choices=['imagenet', 'coco'], 
                        help='Dataset type: imagenet or coco (default: imagenet)')
    parser.add_argument('--imagenet-val-path', type=str, default=None, 
                        help='Path to the root of FLAT ImageNet-1K validation directory '
                             '(required when dataset-type=imagenet).')
    parser.add_argument('--coco-val-path', type=str, default='/path/to/public-datasets/coco_val',
                        help='Path to COCO validation directory (required when dataset-type=coco).')
    parser.add_argument('--config-path', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'hydra_tok'),
                        help='Path to model config directory (defaults to the hydra_tok/ '
                             'package next to this tools/ directory).')
    parser.add_argument('--output-dir', type=str, default='./eval_temp_ddp',
                        help='Temporary directory for the resized PNGs used by rFID. '
                             'Deleted when the run finishes unless --keep-images is set. '
                             'Needs ~20GB free for 50K images at 256x256.')
    parser.add_argument('--keep-images', action='store_true',
                        help='Keep the temporary images instead of deleting them '
                             '(useful for re-running tools/eval_fid.py).')

    parser.add_argument('--image-size', type=int, default=448, help='Image size for model INPUT.')
    parser.add_argument('--eval-size', type=int, default=256, help='Image size for METRIC CALCULATION.')
    parser.add_argument('--batch-size', type=int, default=32, help='Per-GPU batch size for evaluation.')
    parser.add_argument('--fid-batch-size', type=int, default=50,
                        help='Batch size for the FID-Inception forward pass.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Data-loading workers for the FID-Inception forward pass.')

    args = parser.parse_args()
    main(args)
# Evaluate on ImageNet with 8 GPUs and batch size 256
# torchrun --nproc_per_node=8 eval_top12.py \
#     --checkpoint-path /path/to/outputs/results/hydra-tok-64c-no-distill/final_model.pth \
#     --dataset-type imagenet \
#     --imagenet-val-path /path/to/dataset/Generation/ImageNet1k/output/ImageNet-1K/val \
#     --config-path ./hydra_tok \
#     --output-dir ./eval_temp_ddp_no_dist_have_init \
#     --batch-size 256 \
#     --image-size 448 \
#     --eval-size 256

# Evaluate on COCO with 8 GPUs and batch size 256
# torchrun --nproc_per_node=8 eval_top12.py \
#     --checkpoint-path /path/to/outputs/results/hydra_tok_24layer_64C/final_model.pth \
#     --dataset-type coco \
#     --coco-val-path /path/to/public-datasets/coco_val/val2017  \
#     --config-path ./hydra_tok \
#     --output-dir ./eval_64c_ddp_coco \
#     --batch-size 256 \
#     --image-size 448 \
#     --eval-size 256
