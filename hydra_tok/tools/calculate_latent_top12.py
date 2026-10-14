# Copyright 2026 HYDRA Authors.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoImageProcessor

# ==========================================
# 1. Import the model (adjust to your Top-12 path)
# ==========================================
# tools/ lives directly under the HYDRA-TOK root, which is the parent directory
# of the hydra_tok model package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: the top12 config is imported here


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


def setup_ddp():
    """Initialize the distributed environment."""
    if "RANK" not in os.environ:
        # Compatibility for single-GPU debugging
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def main(args):
    # --- 1. DDP initialization ---
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\nStarting DDP *3D (C,H,W)* Statistics Calculation")
        print(f"   GPUs: {world_size}")
        print(f"   Config Path: {args.config_path}")
        print(f"   Checkpoint: {args.checkpoint_path}")

    # --- 2. Model setup ---
    print(f"Dynamically importing the model from {args.config_path}...")
    HydraTokConfig, HydraTokVisionModel = import_model_from_config_path(args.config_path)
    print(f"Model imported successfully")
    config = HydraTokConfig.from_pretrained(args.config_path)
    config.image_size = args.image_size
    config.patch_size = 14

    # Initialize the model in bfloat16 to save memory (cast to float64 when computing statistics)
    model = HydraTokVisionModel(config=config).to(device, dtype=torch.bfloat16).eval()

    # --- 3. Load weights ---
    if rank == 0:
        print(f"Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")  # load to CPU first to save GPU memory

    if 'student_model' in checkpoint:
        state_dict = checkpoint['student_model']
    elif "model" in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Handle possible key prefix mismatch
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = DDP(model, device_ids=[local_rank])

    # --- 4. Data processing ---
    if rank == 0:
        image_processor = AutoImageProcessor.from_pretrained(args.processor_path, trust_remote_code=True)
    dist.barrier()
    if rank != 0:
        image_processor = AutoImageProcessor.from_pretrained(args.processor_path, trust_remote_code=True)

    transform = transforms.Compose([
        transforms.Resize(image_processor.size["shortest_edge"], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_processor.crop_size["height"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    dataset_train = datasets.ImageFolder(root=args.imagenet_train_path, transform=transform)
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        sampler=sampler_train, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        shuffle=False,
        drop_last=False
    )

    # --- 5. Initialize 3D statistics for the Welford algorithm ---
    # mean and M2 are initialized on the first forward pass, with shape [C, H, W]
    count = 0
    mean = None 
    M2 = None   

    # --- 6. Statistics loop ---
    data_iterator = dataloader_train
    if rank == 0:
        data_iterator = tqdm(dataloader_train, desc="Calculating 3D Stats")

    with torch.no_grad():
        for step, (images, _) in enumerate(data_iterator):
            images = images.to(device, dtype=torch.bfloat16)

            # Forward pass: extract latents
            # Shape is typically [Batch, Channels, Height, Width]
            latents = model.module.encode(images) 
            latents = latents.to(torch.float64)  # use float64 for precision

            batch_size_curr = latents.shape[0]
            if batch_size_curr == 0:
                continue

            # Lazily initialize the statistics tensors (on the first iteration)
            if mean is None:
                feature_shape = latents.shape[1:]  # get [C, H, W]
                if rank == 0:
                    print(f"\nDetected Feature Shape: {feature_shape}")
                    print(f"   Calculated Tensor Size: {feature_shape} (per rank)")

                mean = torch.zeros(feature_shape, device=device, dtype=torch.float64)
                M2 = torch.zeros(feature_shape, device=device, dtype=torch.float64)

            # --- Welford Online Algorithm (3D version) ---
            # 1. Compute the current batch mean, aggregating only over the batch dim (dim=0)
            #    Result shape: [C, H, W]
            batch_mean = torch.mean(latents, dim=0) 

            # 2. Compute the current batch M2 (sum of squared differences)
            #    (x - x_bar)^2
            #    Broadcasting: [B, C, H, W] - [C, H, W] -> [B, C, H, W]
            #    Sum dim=0 -> [C, H, W]
            batch_M2 = torch.sum((latents - batch_mean)**2, dim=0)

            # 3. Update the running statistics
            new_count = count + batch_size_curr
            delta = batch_mean - mean

            # Update mean
            mean = mean + delta * (batch_size_curr / new_count)

            # Update M2
            # All operations here are element-wise and work with [C, H, W]
            M2 = M2 + batch_M2 + (delta**2) * (count * batch_size_curr / new_count)

            count = new_count

            # --- Print progress (rank 0 only) ---
            if rank == 0 and step % args.log_interval == 0 and count > 1:
                # Compute the current mean std for display
                curr_std_avg = torch.sqrt(M2 / count).mean().item()
                tqdm.write(
                    f"[Step {step}] Total Samples: {int(count)} | "
                    f"Avg Mean: {mean.mean().item():.4f} | "
                    f"Avg Std: {curr_std_avg:.4f}"
                )

    # --- 7. DDP aggregation (gather & merge) ---
    dist.barrier() 
    if rank == 0:
        print("\nLocal calculation finished. Aggregating across ranks...")

    # 1. Gather sample counts
    count_tensor = torch.tensor([count], device=device, dtype=torch.float64)
    all_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
    dist.all_gather(all_counts, count_tensor)

    # 2. Gather mean and M2
    # If a rank got no data (empty batches), mean is still None and must be handled
    if mean is None:
        # Fallback: assume the shape matches other ranks (data is usually evenly distributed)
        # Simply init to zeros to avoid errors; count=0 does not affect the merge
        # Note: feature_shape is required here; if this rank never ran, the shape is unknown
        # In practice this rarely happens with drop_last=False and enough data
        print(f"Warning: Rank {rank} processed 0 samples.")
        pass 

    all_means = [torch.zeros_like(mean) for _ in range(world_size)]
    all_M2s = [torch.zeros_like(M2) for _ in range(world_size)]

    dist.all_gather(all_means, mean)
    dist.all_gather(all_M2s, M2)

    # --- 8. Global merge ---
    if rank == 0:
        print("Merging statistics...")
        global_count = 0.0
        # Initialize the global containers [C, H, W]
        global_mean = torch.zeros_like(mean)
        global_M2 = torch.zeros_like(M2)

        for i in range(world_size):
            c_i = all_counts[i].item()
            m_i = all_means[i]
            m2_i = all_M2s[i]

            if c_i == 0:
                continue

            # Formula for merging two sets
            new_global_count = global_count + c_i
            delta = m_i - global_mean

            global_mean = global_mean + delta * (c_i / new_global_count)
            global_M2 = global_M2 + m2_i + (delta**2) * (global_count * c_i / new_global_count)

            global_count = new_global_count

        if global_count < 2:
            print("Error: Total sample count < 2, cannot calculate variance.")
        else:
            # Compute the final variance and std [C, H, W]
            global_variance = global_M2 / global_count
            global_std = torch.sqrt(global_variance)

            print(f"\nCalculation Complete!")
            print(f"   Final Tensor Shape: {global_mean.shape}")
            print(f"   Total Samples Processed: {int(global_count)}")
            print(f"   Global Mean (Avg): {global_mean.mean().item():.6f}")
            print(f"   Global Std  (Avg): {global_std.mean().item():.6f}")

            # Save
            stat_dict = {
                'latent_mean': global_mean.cpu(),  # [C, H, W]
                'latent_var': global_variance.cpu(), 
                'latent_std': global_std.cpu()    # [C, H, W]
            }

            try:
                os.makedirs(os.path.dirname(args.stats_save_path), exist_ok=True)
                torch.save(stat_dict, args.stats_save_path)
                print(f"Statistics saved to: {args.stats_save_path}")
            except (OSError, RuntimeError) as e:
                print(f"Save failed: {e}")

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path arguments
    parser.add_argument('--checkpoint-path', type=str, required=True, help="Path to hydra_tok model checkpoint")
    parser.add_argument('--imagenet-train-path', type=str, required=True, help="Path to ImageNet train dataset")
    parser.add_argument('--config-path', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'hydra_tok'),
                        help="Path to model config directory (defaults to the hydra_tok/ "
                             "package next to this tools/ directory).")
    parser.add_argument('--processor-path', type=str,
                        default='OpenGVLab/InternViT-300M-448px-V2_5',
                        help="Image processor (a HF repo id works as-is).")

    # Output arguments
    parser.add_argument('--stats-save-path', type=str, default='./hydra_tok_latent_stats_3d_top12.pt')

    # Training arguments
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=32, help="Per-GPU batch size")
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--log-interval', type=int, default=20)

    args = parser.parse_args()
    main(args)
# Assume the code above is saved as calculate_stats_on_train.py

# torchrun --nproc_per_node=8 calculate_latent_top12.py \
#     --checkpoint-path /path/to/outputs/results/hydra-tok-64c-no-distill/final_model.pth \
#     \
#     --imagenet-train-path /path/to/dataset/Generation/ImageNet1k/output/ImageNet-1K/train \
#     --config-path ./hydra_tok \
#     --processor-path OpenGVLab/InternViT-300M-448px-V2_5 \
#     --stats-save-path /path/to/Nitro-E/hydra_tok_latent_64C_no_dist.pt \
#     \
#     --batch-size 256 \
#     --image-size 448 \
