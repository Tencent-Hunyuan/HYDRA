# Copyright 2026 HYDRA Authors.
# Modified from FlowMo / RAE (GAN recipe references)
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import sys
import time
import math
import random
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from PIL import Image

# === Logging ===
logger = logging.getLogger(__name__)

# === HuggingFace Transformers ===
from transformers import AutoModel, AutoImageProcessor

# === Distributed training modules ===
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# === TensorBoard Logging ===
from torch.utils.tensorboard import SummaryWriter

# === Evaluation metrics ===
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ==============================================================================
# 1. Path settings (adjust to your environment)
# ==============================================================================


# Import model definitions (single definition; the stage-2 GAN recipe is
# selected via forward args, see use_gan / lpips_weight in configs/stage2_gan.yaml)
from hydra_tok.modeling_hydra_tok import HydraTokVisionModel, Distill_Adapter
from hydra_tok.configuration_hydra_tok import HydraTokConfig


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'configs', 'stage2_gan.yaml')


def load_config() -> dict:
    """Load the training config from YAML, then apply --key=value CLI overrides.

    ``configs/stage2_gan.yaml`` is the single source of truth: it supplies every
    default and doubles as the key schema, so a typo in a config file or on the
    command line is rejected instead of silently ignored. A ``--config`` file may
    be partial; whatever it omits falls back to the shipped defaults.

    Usage:
        torchrun --nproc_per_node=8 train_stage2_gan.py                             # configs/stage2_gan.yaml
        torchrun --nproc_per_node=8 train_stage2_gan.py --config my_stage2.yaml
        torchrun --nproc_per_node=8 train_stage2_gan.py --config my_stage2.yaml --lr=5e-5
    """
    import argparse
    parser = argparse.ArgumentParser(description="HYDRA-TOK stage-2 GAN training")
    parser.add_argument('--config', type=str, default=None,
                        help=f'Path to a YAML training config (default: {DEFAULT_CONFIG_PATH})')
    args, unknown = parser.parse_known_args()

    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f) or {}

    if args.config and os.path.abspath(args.config) != DEFAULT_CONFIG_PATH:
        with open(args.config, 'r') as f:
            file_cfg = yaml.safe_load(f) or {}
        unknown_keys = set(file_cfg) - set(config)
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {args.config}: {sorted(unknown_keys)}")
        config.update(file_cfg)

    # CLI overrides: both --key=value and --key value forms are accepted.
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if not arg.startswith('--'):
            i += 1
            continue
        if '=' in arg:
            key, value = arg[2:].split('=', 1)
            i += 1
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
            key, value = arg[2:], unknown[i + 1]
            i += 2
        else:
            raise ValueError(f"Missing value for config override: {arg}")
        if key not in config:
            raise ValueError(f"Unknown config override: --{key}")
        config[key] = _parse_cli_value(value)

    return config


def _parse_cli_value(value):
    """Parse a CLI override value.

    yaml.safe_load handles ints/bools/lists, but not YAML-1.1-unfriendly
    scientific notation like 1e-4, so add a numeric fallback for strings.
    """
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return parsed

# Import GAN-related modules
from disc import DinoDiscriminator, DiffAug, hinge_d_loss, vanilla_g_loss, build_discriminator

# ==============================================================================
# 2. Custom datasets
# ==============================================================================


class StandardImageDataset(Dataset):
    """
    Dataset class for the standard image data format.
    Layout:
    - data_info.json: [{'image': '000000/0000001.jpg', 'annotation': '000000/0000001.json'},...]
    - local_folder/000000/0000001.jpg
    - cap_folder/000000/0000001.json: {'caption': '...'}
    """

    def __init__(self, dataset_root: str, data_info_file: str = 'data_info.json',
                 transform=None, min_resolution: int = None, sample_ratio: float = None):
        """
        Args:
            dataset_root: dataset root directory
            data_info_file: data_info.json file name ('data_info.json' or 'data_info_448.json')
            transform: image preprocessing transforms
            min_resolution: minimum resolution threshold (if given, only load images with resolution >= min_resolution)
            sample_ratio: random sampling ratio (if given, e.g. 0.5 means randomly sample 50% of the data)
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.min_resolution = min_resolution
        self.sample_ratio = sample_ratio
        self.image_paths = []

        rank = int(os.environ.get("RANK", 0))

        # Load data_info.json
        data_info_path = os.path.join(dataset_root, data_info_file)
        if not os.path.exists(data_info_path):
            if rank == 0:
                logger.warning(f"Data info file not found: {data_info_path}, trying data_info.json...")
            data_info_path = os.path.join(dataset_root, 'data_info.json')
            if not os.path.exists(data_info_path):
                raise FileNotFoundError(f"Data info file not found: {data_info_path}")

        if rank == 0:
            logger.info(f"Loading dataset from: {dataset_root}")
            logger.info(f"Using data info file: {data_info_path}")
            if min_resolution:
                logger.info(f"Filtering images with resolution >= {min_resolution}")
            if sample_ratio:
                logger.info(f"Will randomly sample {sample_ratio*100:.1f}% of the data")

        # Read data_info.json
        with open(data_info_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)

        # Build the image path list
        local_folder = os.path.join(dataset_root, 'local_folder')
        count = 0
        skipped_low_res = 0

        for item in data_info:
            if 'image' not in item:
                continue

            img_rel_path = item['image']
            img_abs_path = os.path.join(local_folder, img_rel_path)


            self.image_paths.append(img_abs_path)
            count += 1

        # Random sampling (if sample_ratio is specified)
        if sample_ratio is not None and sample_ratio < 1.0:
            original_count = len(self.image_paths)
            sample_size = int(original_count * sample_ratio)
            # Seed from the dataset path for reproducibility, with different sampling per dataset
            seed_value = hash(self.dataset_root) % (2**31) + 123  # part of the seed comes from the dataset path hash
            random.seed(seed_value)
            sampled_indices = random.sample(range(original_count), sample_size)
            self.image_paths = [self.image_paths[i] for i in sorted(sampled_indices)]
            if rank == 0:
                logger.info(
                    f"Randomly sampled {len(self.image_paths)} images from {original_count} "
                    f"(ratio: {sample_ratio})")

        if rank == 0:
            logger.info(f"Loaded {len(self.image_paths)} images from {os.path.basename(dataset_root)}")
            if skipped_low_res > 0:
                logger.info(f"Skipped {skipped_low_res} images with resolution < {min_resolution}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0
        except (OSError, ValueError, RuntimeError) as e:
            # On error: skip the corrupt image and try the next one
            new_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(new_idx)


class CustomJSONDataset(Dataset):
    def __init__(self, yaml_path, transform=None):
        """
        Args:
            yaml_path: path to single_image_noproblem.yaml
            transform: image preprocessing transforms
        """
        self.transform = transform
        self.image_paths = []

        # Prefix path for the special dataset
        self.NEXT_DATA_PREFIX = "/path/to/dataset/Understanding/LLaVA-NeXT-Data/llava_next_raw_format"
        self.NEXT_DATA_JSON_NAME = "llava_next_raw_format_processed.json"

        # 1. Read the YAML
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            logger.info(f"Loading data from YAML: {yaml_path}")

        # [Key fix]: parse the YAML structure
        # Expected YAML format: {'datasets': [List of Dicts]}
        if isinstance(yaml_content, dict) and 'datasets' in yaml_content:
            data_list = yaml_content['datasets']
        elif isinstance(yaml_content, list):
            # Backward compatibility (if it is directly a list)
            data_list = yaml_content
        else:
            raise ValueError(
                f"Unexpected YAML format. Expected dict with 'datasets' key or list, got {type(yaml_content)}")

        # 2. Iterate over the dataset list
        for entry in data_list:
            # Get json_path
            json_path = None
            if isinstance(entry, dict):
                json_path = entry.get('json_path')
            elif isinstance(entry, str):
                json_path = entry

            if not json_path or not os.path.exists(json_path):
                if rank == 0:
                    logger.warning(f"JSON path invalid or not found: {json_path}")
                continue

            # Check whether this is the special dataset that needs the prefix
            is_llava_next = self.NEXT_DATA_JSON_NAME in json_path

            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data_items = json.load(jf)

                # 3. Extract and process image paths
                count = 0
                for item in data_items:
                    if 'image' not in item:
                        continue

                    img_rel_path = item['image']

                    # --- Path handling logic ---
                    if is_llava_next:
                        if img_rel_path.startswith('/'):
                            # The original json has absolute paths, but we need to swap the root
                            full_path = os.path.join(self.NEXT_DATA_PREFIX, os.path.basename(img_rel_path))
                            if not os.path.exists(full_path):
                                # Fallback: try a direct join (if the json has relative paths)
                                full_path = os.path.join(self.NEXT_DATA_PREFIX, img_rel_path.lstrip('/'))
                        else:
                            full_path = os.path.join(self.NEXT_DATA_PREFIX, img_rel_path)
                    else:
                        # Other datasets are used as-is (assumed to be absolute paths)
                        full_path = img_rel_path

                    self.image_paths.append(full_path)
                    count += 1

                if rank == 0:
                    logger.info(f"Loaded {count} images from {os.path.basename(json_path)}")

            except (OSError, ValueError) as e:
                if rank == 0:
                    logger.error(f"Error reading JSON {json_path}: {e}")

        if rank == 0:
            logger.info(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0

        except (OSError, ValueError, RuntimeError) as e:
            # On error: skip the corrupt image and try the next one
            new_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(new_idx)


# ==============================================================================
# 3. Helper functions
# ==============================================================================


def print_trainable_modules(model, rank):
    """ Print trainable parameter statistics """
    if rank != 0:
        return
    trainable_modules = set()
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            module_name = name.split('.')[0]
            trainable_modules.add(module_name)

    print(f"\n[Model Stats] Total: {total_params:,} | Trainable: {trainable_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")


@torch.no_grad()


def evaluate(student_model, epoch, global_step, config, device, writer=None):
    """ Evaluation: compute PSNR and SSIM of reconstructed images """
    print(f"\n>>> Starting evaluation for Epoch {epoch + 1}...")
    # Save the original training state
    was_training = student_model.training
    student_model.eval()

    try:
        DATA_NORM = {"imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}

        # Simple center crop
        def get_eval_transform(size):
            return transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATA_NORM['imagenet']['mean'], std=DATA_NORM['imagenet']['std'])
            ])

        eval_dir = Path(config['eval_data_path'])
        all_images = list(eval_dir.glob('*.[jp][pn]g'))  # match jpg, png, jpeg
        if not all_images:
            print("Warning: No images found for evaluation.")
            return

        # Randomly sample 4 images for testing
        img_paths = random.sample(all_images, len(all_images))
        transform = get_eval_transform(config['image_size'])

        tensors = []
        raw_imgs = []
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            raw_imgs.append(img)
            tensors.append(transform(img))

        pv_batch = torch.stack(tensors).to(device, dtype=torch.bfloat16)

        # Call the model's sample method (reconstruction)
        # Note: per the model code, sample returns a [B, C, H, W] tensor
        with torch.no_grad():
            recons = student_model.module.sample(pv_batch)
        recons = recons.float().cpu()

        # De-normalize
        mean = torch.tensor(DATA_NORM['imagenet']['mean']).view(1, 3, 1, 1)
        std = torch.tensor(DATA_NORM['imagenet']['std']).view(1, 3, 1, 1)
        recons = (recons * std + mean).clamp(0, 1)
        recons_np = (recons.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        psnr_vals, ssim_vals = [], []
        for i, im_recon_np in enumerate(recons_np):
            # Resize the original image to the same size for comparison
            im_raw_np = np.array(raw_imgs[i].resize((config['image_size'], config['image_size']), Image.LANCZOS))
            # Simple check: ensure channel alignment
            if im_raw_np.shape != im_recon_np.shape:
                continue

            im_raw_np = im_raw_np.astype(np.float32) / 255.
            im_recon_float = im_recon_np.astype(np.float32) / 255.

            psnr_vals.append(psnr_metric(im_raw_np, im_recon_float, data_range=1.0))
            ssim_vals.append(ssim_metric(im_raw_np, im_recon_float, data_range=1.0, channel_axis=-1, win_size=7))

        psnr_mean = np.mean(psnr_vals) if psnr_vals else 0
        ssim_mean = np.mean(ssim_vals) if ssim_vals else 0

        print(f"Epoch {epoch + 1} Eval: PSNR = {psnr_mean:.2f} dB, SSIM = {ssim_mean:.4f}")
        if writer:
            writer.add_scalar('Eval/PSNR', psnr_mean, global_step)
            writer.add_scalar('Eval/SSIM', ssim_mean, global_step)
        print("="*50)
    finally:
        # Restore the original training state
        student_model.train(was_training)

# ==============================================================================
# 4. Main program
# ==============================================================================


def main():
    # --------------------------
    # A. Initialize the distributed environment
    # --------------------------
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Initialized DDP on {world_size} GPUs.")

    # --------------------------
    # B. Configuration (Config)
    # --------------------------
    # All hyperparameters live in configs/stage2_gan.yaml; --config picks a
    # different file and --key=value overrides individual fields.
    config = load_config()
    if rank == 0:
        print(json.dumps(config, indent=2, sort_keys=True, default=str))

    local_batch_size = config['batch_size'] // world_size
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    # --------------------------
    # C. Model initialization
    # --------------------------
    # Stage3: no teacher model needed (no distillation)

    # Student model
    # repository root (the directory containing this script and hydra_tok/)
    local_model_parent_dir = os.path.dirname(os.path.abspath(__file__))
    student_config_path = os.path.join(local_model_parent_dir, "hydra_tok/config.json")
    student_config = HydraTokConfig.from_pretrained(student_config_path)
    student_config.image_size = config['image_size']
    student_config.patch_size = config['patch_size']

    student_model = HydraTokVisionModel(student_config).to(device).to(torch.bfloat16)
    if rank == 0:
        print(student_model)
    if rank == 0:
        print(config)

    # 3. Weight loading - load the stage-2 checkpoint first
    start_epoch = 0
    global_step_start = 0

    if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
        if rank == 0:
            print(f"Loading checkpoint from: {config['resume_from_checkpoint']}")
        checkpoint = torch.load(config['resume_from_checkpoint'], map_location=device)

        # Load model weights
        if 'model' in checkpoint:
            student_model.load_state_dict(checkpoint['model'], strict=False)
            if rank == 0:
                print("Loaded student model weights from checkpoint")
        else:
            # If there is no 'model' key, try loading directly
            student_model.load_state_dict(checkpoint, strict=False)
            if rank == 0:
                print("Loaded student model weights from checkpoint (direct)")

        # Load training state (if present)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            global_step_start = checkpoint['global_step']
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, global_step {global_step_start}")

    else:
        if rank == 0:
            print("Warning: No checkpoint or teacher initialization specified!")

    # Stage3: no adapters needed (no distillation)

    # --------------------------
    # Stage3: freeze the encoder, train only the generation part
    # --------------------------
    # Freeze the encoder (embeddings + encoder)
    for param in student_model.embeddings.parameters():
        param.requires_grad = False
    for param in student_model.encoder.parameters():
        param.requires_grad = False

    # Train only generation-related modules
    # 1. Global blocks (last 12 Transformer layers)
    for param in student_model.global_blocks.parameters():
        param.requires_grad = True
    # 2. Generation Unproject Layer
    for param in student_model.chal_unproj_generation.parameters():
        param.requires_grad = True
    # 3. Flow Head
    for param in student_model.flow_head.parameters():
        param.requires_grad = True
    # 4. Position Embeddings
    student_model.global_block_pos_embed.requires_grad = True
    student_model.decoder_pos_embed.requires_grad = True

    if rank == 0:
        print("="*60)
        print("Stage3 Training Mode: Freezing Encoder, Training Generator")
        print("  - Frozen: embeddings, encoder")
        print("  - Trainable: global_blocks, chal_unproj_generation, flow_head")
        print("="*60)

    # Print parameter info
    print_trainable_modules(student_model, rank)

    # DDP wrap
    student_model = DDP(student_model, device_ids=[local_rank], find_unused_parameters=True)

    # --------------------------
    # GAN discriminator initialization
    # --------------------------
    discriminator = None
    disc_aug = None
    disc_optimizer = None

    if config.get('use_gan', False):
        if rank == 0:
            print("="*60)
            print("Initializing GAN Discriminator...")
            print("="*60)

        # Configure the discriminator
        disc_config = {
            "arch": {
                "dino_ckpt_path": config['dino_ckpt_path'],
                "ks": 3,
                "key_depths": (2, 5, 8, 11),
                "norm_type": "bn",
                "using_spec_norm": True,
                "norm_eps": 1e-6,
                "recipe": "S_8",  # S_8 recipe, suited for 448x448 input
            },
            "augment": {
                "prob": 1.0,
                "cutout": 0.0,
            }
        }

        # Build the discriminator and data augmentation
        discriminator, disc_aug = build_discriminator(disc_config, device)
        discriminator = discriminator.to(torch.bfloat16)

        # If resuming from a checkpoint, load discriminator weights
        if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
            checkpoint = torch.load(config['resume_from_checkpoint'], map_location=device)
            if 'discriminator' in checkpoint:
                checkpoint_state = checkpoint['discriminator']
                model_state = discriminator.state_dict()

                # Print detailed loading info
                if rank == 0:
                    print("="*60)
                    print("Loading Discriminator Weights from Checkpoint:")
                    print("="*60)
                    print(f"Checkpoint keys: {len(checkpoint_state)}")
                    print(f"Model keys: {len(model_state)}")

                    # Check for missing and unexpected keys
                    missing_keys = set(model_state.keys()) - set(checkpoint_state.keys())
                    unexpected_keys = set(checkpoint_state.keys()) - set(model_state.keys())

                    if missing_keys:
                        print(f"Missing keys ({len(missing_keys)}):")
                        for key in sorted(list(missing_keys)[:10]):  # show the first 10 only
                            print(f"  - {key}")
                        if len(missing_keys) > 10:
                            print(f"... and {len(missing_keys) - 10} more")

                    if unexpected_keys:
                        print(f"Unexpected keys ({len(unexpected_keys)}):")
                        for key in sorted(list(unexpected_keys)[:10]):  # show the first 10 only
                            print(f"  - {key}")
                        if len(unexpected_keys) > 10:
                            print(f"... and {len(unexpected_keys) - 10} more")

                    if not missing_keys and not unexpected_keys:
                        print("All keys matched perfectly!")

                    print("="*60)

                discriminator.load_state_dict(checkpoint_state, strict=False)
                if rank == 0:
                    print("Loaded discriminator weights from checkpoint")

        # Print details before DDP wrapping (discriminator is directly accessible)
        if rank == 0:
            print("="*60)
            print("Discriminator Architecture Details:")
            print("="*60)

            # Count parameters
            total_params = 0
            trainable_params = 0
            frozen_params = 0

            # Group stats by module
            module_stats = {}

            # Record detailed info for every layer
            layer_details = []

            for name, param in discriminator.named_parameters():
                param_count = param.numel()
                total_params += param_count

                if param.requires_grad:
                    trainable_params += param_count
                else:
                    frozen_params += param_count

                # Extract the module name (top level)
                module_name = name.split('.')[0]
                if module_name not in module_stats:
                    module_stats[module_name] = {'total': 0, 'trainable': 0, 'frozen': 0,
                                                 'layer_count': 0, 'layer_prefixes': set()}

                module_stats[module_name]['total'] += param_count
                if param.requires_grad:
                    module_stats[module_name]['trainable'] += param_count
                else:
                    module_stats[module_name]['frozen'] += param_count

                # Record details for each layer
                layer_details.append({
                    'name': name,
                    'module': module_name,
                    'shape': list(param.shape),
                    'params': param_count,
                    'trainable': param.requires_grad,
                    'dtype': str(param.dtype)
                })

                # Count unique layers per module (by layer prefix, excluding param name)
                layer_prefix = '.'.join(name.split('.')[:-1]) if '.' in name else name
                if layer_prefix not in module_stats[module_name]['layer_prefixes']:
                    module_stats[module_name]['layer_prefixes'].add(layer_prefix)
                    module_stats[module_name]['layer_count'] = len(module_stats[module_name]['layer_prefixes'])

            # Print overall statistics
            print(f"\nOverall Statistics:")
            print(f"  Total parameters: {total_params/1e6:.2f}M")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")
            print(f"  Frozen parameters: {frozen_params/1e6:.2f}M ({100*frozen_params/total_params:.2f}%)")

            # Print per-module details
            print(f"\nModule Details:")
            for module_name, stats in sorted(module_stats.items()):
                print(f"\n  [{module_name}]")
                print(f"    Total: {stats['total']/1e6:.2f}M | Trainable: {stats['trainable']/1e6:.2f}M "
                      f"| Frozen: {stats['frozen']/1e6:.2f}M | Layers: {stats['layer_count']}")

            # Print layer-by-layer details (grouped by module)
            print(f"\nLayer-by-Layer Details:")
            current_module = None
            for layer in sorted(layer_details, key=lambda x: (x['module'], x['name'])):
                if layer['module'] != current_module:
                    current_module = layer['module']
                    print(f"\n  [{current_module}]")

                status = " Trainable" if layer['trainable'] else " Frozen"
                shape_str = 'x'.join(map(str, layer['shape']))
                print(f"    {layer['name']:<60} | Shape: {shape_str:<25} "
                      f"| Params: {layer['params']/1e3:>8.1f}K | {layer['dtype']:<10} | {status}")

            print(f"\nTraining Configuration:")
            print(f"  GAN training starts at epoch: {config['gan_start_epoch']}")
            print(f"  Discriminator updates start at epoch: {config['disc_start_epoch']}")
            print(f"  Discriminator learning rate: {config.get('disc_lr', 1e-4):.2e}")
            print(f"  Discriminator updates per generator step: {config.get('disc_updates', 1)}")
            print("="*60)

        # DDP wrap the discriminator
        discriminator = DDP(discriminator, device_ids=[local_rank], broadcast_buffers=False)

        # Discriminator optimizer
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optimizer = optim.AdamW(
            disc_params,
            lr=config.get('disc_lr', 1e-4),
            betas=(config['adam_beta1'], config['adam_beta2'])
)

        # If resuming from a checkpoint, load the discriminator optimizer state (scheduler is created later)
        if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
            checkpoint = torch.load(config['resume_from_checkpoint'], map_location=device)
            if 'disc_optimizer' in checkpoint:
                disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
                if rank == 0:
                    print("Loaded discriminator optimizer state from checkpoint")

    # --------------------------
    # D. Data loader
    # --------------------------
    if rank == 0:
        print("Setting up data loader...")
        print("model")
        print(student_model)

    # Use the HuggingFace processor to get normalization parameters
    processor_path = config['processor_path']
    image_processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

    transform = transforms.Compose([
        transforms.Resize(image_processor.size["shortest_edge"]),
        transforms.CenterCrop(image_processor.crop_size["height"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    # Stage2: use stage2_datasets plus the YAML dataset
    dataset_list = []
    sampler = None

    # 1. Load the stage-2 standard-format datasets
    if 'stage2_datasets' in config and config['stage2_datasets']:
        if rank == 0:
            print("="*60)
            print("Loading Stage2 Datasets:")
            print("="*60)

        for ds_config in config['stage2_datasets']:
            ds_root = ds_config['root']
            data_info_file = ds_config.get('data_info_file', 'data_info.json')
            min_res = ds_config.get('min_resolution', None)
            sample_ratio = ds_config.get('sample_ratio', None)

            if rank == 0:
                print(f"  - {os.path.basename(ds_root)}")
                print(f"    Data info: {data_info_file}")
                if min_res:
                    print(f"    Min resolution: {min_res}")
                if sample_ratio:
                    print(f"    Sample ratio: {sample_ratio*100:.1f}%")

            try:
                dataset = StandardImageDataset(
                    dataset_root=ds_root,
                    data_info_file=data_info_file,
                    transform=transform,
                    min_resolution=min_res,
                    sample_ratio=sample_ratio
)
                dataset_list.append(dataset)
                if rank == 0:
                    print(f"    Loaded {len(dataset)} images")
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                if rank == 0:
                    logger.error(f"    Failed to load {ds_root}: {e}")
                continue

        if rank == 0:
            print("="*60)

    # 2. Load the YAML dataset (if enabled)
    if config.get('use_yaml_dataset', False) and 'yaml_path' in config and config['yaml_path']:
        if rank == 0:
            print("="*60)
            print("Loading YAML Dataset:")
            print("="*60)
            print(f"  - YAML Path: {config['yaml_path']}")

        try:
            yaml_dataset = CustomJSONDataset(
                yaml_path=config['yaml_path'],
                transform=transform
)
            dataset_list.append(yaml_dataset)
            if rank == 0:
                print(f"  Loaded {len(yaml_dataset)} images from YAML")
                print("="*60)
        except (OSError, ValueError, KeyError) as e:
            if rank == 0:
                logger.error(f"  Failed to load YAML dataset: {e}")
                print("="*60)

    # 3. Check that at least one dataset was loaded
    if len(dataset_list) == 0:
        raise ValueError("No datasets loaded! Please configure 'stage2_datasets' or set 'use_yaml_dataset=True'")

    # 5. Merge all datasets
    if len(dataset_list) == 1:
        combined_dataset = dataset_list[0]
    else:
        combined_dataset = ConcatDataset(dataset_list)

    if rank == 0:
        print(f"\nTotal dataset size: {len(combined_dataset)}")
        print("="*60)

    # 6. Create the DataLoader
    sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        combined_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        shuffle=False
)

    # --------------------------
    # Create the discriminator LR scheduler (must come after the dataloader)
    # --------------------------
    disc_scheduler = None
    if discriminator is not None and disc_optimizer is not None:
        disc_total_steps = len(dataloader) * config['epochs']
        disc_warmup_steps = config.get('disc_warmup_steps', 100)
        disc_min_lr = config.get('disc_min_lr', 2e-5)
        disc_max_lr = config.get('disc_lr', 1e-4)

        def disc_lr_lambda(current_step):
            if current_step < disc_warmup_steps:
                return float(current_step) / float(max(1, disc_warmup_steps))
            progress = float(current_step - disc_warmup_steps) / float(max(1, disc_total_steps - disc_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            target_ratio = disc_min_lr / disc_max_lr
            return target_ratio + (1.0 - target_ratio) * cosine_decay

        disc_scheduler = LambdaLR(disc_optimizer, disc_lr_lambda)

        # If resuming from a checkpoint, load the discriminator scheduler state
        if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
            checkpoint = torch.load(config['resume_from_checkpoint'], map_location=device)
            if 'disc_scheduler' in checkpoint:
                disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
                if rank == 0:
                    print("Loaded discriminator scheduler state from checkpoint")

        if rank == 0:
            print(f"Discriminator LR scheduler: {disc_max_lr:.2e} -> {disc_min_lr:.2e} "
                  f"(warmup: {disc_warmup_steps} steps)")

    # --------------------------
    # E. Optimizer & scheduler
    # --------------------------
    # Stage3: optimize only generation-related parameters (no adapters)
    gen_params = [p for p in student_model.module.parameters() if p.requires_grad]
    optimizer = optim.AdamW(gen_params, lr=config['lr'], betas=(config['adam_beta1'], config['adam_beta2']))

    # If resuming from a checkpoint, load the optimizer state
    if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
        checkpoint = torch.load(config['resume_from_checkpoint'], map_location=device)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0:
                print("Loaded optimizer state from checkpoint")

    total_steps = len(dataloader) * config['epochs']

    # LambdaLR: Warmup + Cosine Decay
    def lr_lambda(current_step):
        if current_step < config['warmup_steps']:
            return float(current_step) / float(max(1, config['warmup_steps']))
        progress = float(current_step - config['warmup_steps']) / float(max(1, total_steps - config['warmup_steps']))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_ratio = config['min_lr'] / config['lr']
        return target_ratio + (1.0 - target_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda)

    # --------------------------
    # F. Logging & checkpointing
    # --------------------------
    writer = None
    if rank == 0:
        os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'logs'))
        print(f"Training started. Steps per epoch: {len(dataloader)}")

    # ==========================================================================
    # 4. Training loop
    # ==========================================================================
    start_time = time.time()
    steps_per_epoch = len(dataloader)

    # GAN-related configuration
    gan_start_step = config.get('gan_start_epoch', 0) * steps_per_epoch
    disc_update_step = config.get('disc_start_epoch', 0) * steps_per_epoch

    for epoch in range(start_epoch, config['epochs']):
        student_model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        for i, batch_data in enumerate(dataloader):
            # Handle data formats
            # StandardImageDataset returns (image, label) tuples
            # Images are already tensors
            raw_images, _ = batch_data

            images = raw_images.to(device, dtype=torch.bfloat16)

            # Compute the current global step
            global_step = global_step_start + epoch * len(dataloader) + i

            # Decide whether GAN is enabled
            use_gan = config.get('use_gan', False) and discriminator is not None
            use_gan_loss = use_gan and global_step >= gan_start_step
            train_disc = use_gan and global_step >= disc_update_step

            # -----------------------------------------------------------
            # 1. Generator Forward Pass
            # -----------------------------------------------------------
            optimizer.zero_grad()

            # Put the discriminator in eval mode (while computing the generator loss)
            if use_gan:
                discriminator.eval()

            # Stage3: compute the flow loss only (GAN + LPIPS included), no distillation
            # Student: returns a flow loss dict (flow MSE + LPIPS + GAN + KL + BN)
            fake_images_for_disc = None
            if use_gan_loss:
                result = student_model(
                    pixel_values=images,
                    output_hidden_states=False,  # hidden states not needed
                    discriminator=discriminator,
                    disc_aug=disc_aug,
                    use_gan_loss=True,
                    return_fake_images=True,  # return fake images for the discriminator update
                    max_d_weight=config.get('max_d_weight', 1e4),  # adaptive weight cap
                    # whether to use the adaptive weight
                    use_adaptive_gan_weight=config.get('use_adaptive_gan_weight', False),
                    use_kl_loss=config['use_kl_loss'],
                    use_bn_loss=config['use_bn_loss'],
                    kl_weight=config['kl_weight'],
                    bn_weight=config['bn_weight'],
                    gan_weight_base=config['gan_weight_base'],
                    lpips_weight=config.get('lpips_weight', 1.0),
)
                if isinstance(result, tuple):
                    loss_dict, fake_images_for_disc = result
                else:
                    loss_dict = result
            else:
                loss_dict = student_model(
                    pixel_values=images,
                    output_hidden_states=False,
                    use_kl_loss=config['use_kl_loss'],
                    use_bn_loss=config['use_bn_loss'],
                    kl_weight=config['kl_weight'],
                    bn_weight=config['bn_weight'],
                    lpips_weight=config.get('lpips_weight', 1.0),
)

            # loss_dict is a dict containing: {'total', 'flow_mse', 'lpips', 'gan', 'adaptive_weight'}
            # Extract the loss components
            total_loss = config['lambda_f'] * loss_dict['total']
            flow_mse_loss = loss_dict['flow_mse']
            lpips_loss = loss_dict['lpips']
            gan_loss = loss_dict['gan']
            adaptive_weight = loss_dict.get('adaptive_weight', torch.tensor(0.5, device=device))  # adaptive weight

            # -----------------------------------------------------------
            # 2. Generator backward (Stage3: flow loss only)
            # -----------------------------------------------------------
            # Stage3: no distillation, flow loss only

            total_loss.backward()
            # gradient clipping for the generator (stabilizes training against loss spikes)
            if config.get('max_grad_norm', 1.0) is not None:
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(),
                    max_norm=config.get('max_grad_norm', 1.0))
            optimizer.step()
            scheduler.step()

            # -----------------------------------------------------------
            # 5. Discriminator update (GAN training)
            # -----------------------------------------------------------
            disc_loss = torch.tensor(0.0, device=device)
            disc_accuracy = torch.tensor(0.0, device=device)

            if train_disc:
                # Check whether fake_images_for_disc is available
                if fake_images_for_disc is None:
                    if rank == 0:
                        print("Warning: fake_images_for_disc is None, skipping discriminator update")
                    # Restore model states
                    student_model.train()
                    discriminator.eval()
                else:
                    # Set model states
                    student_model.eval()
                    discriminator.train()

                    # Crop size for discriminator input (224x224)
                    disc_crop_size = config.get('disc_crop_size', 224)

                    for _ in range(config.get('disc_updates', 1)):
                        disc_optimizer.zero_grad()

                        with torch.no_grad():
                            # fake_images_for_disc is the full 448x448 image in [-1, 1]
                            # Unify here: discretize, crop, DiffAug

                            fake_normed = fake_images_for_disc  # already in [-1, 1]

                            # Normalize real images to [-1, 1]
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device, images.dtype)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device, images.dtype)
                            real_denorm = images * std + mean
                            real_normed = real_denorm * 2.0 - 1.0

                            # Discretize (simulate quantization) - applied to both fake and real for consistency
                            fake_detached = fake_normed.clamp(-1.0, 1.0)
                            fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0

                            real_detached = real_normed.clamp(-1.0, 1.0)
                            real_detached = torch.round((real_detached + 1.0) * 127.5) / 127.5 - 1.0

                            # Random crop to disc_crop_size x disc_crop_size (default 224x224)
                            # Use the same crop position for fake and real to stay consistent
                            H, W = fake_detached.shape[2], fake_detached.shape[3]
                            if H > disc_crop_size and W > disc_crop_size:
                                # Random crop start position (shared by fake and real)
                                top = torch.randint(0, H - disc_crop_size + 1, (1,)).item()
                                left = torch.randint(0, W - disc_crop_size + 1, (1,)).item()
                                fake_detached = fake_detached[:, :, top:top+disc_crop_size, left:left+disc_crop_size]
                                real_detached = real_detached[:, :, top:top+disc_crop_size, left:left+disc_crop_size]
                            elif H != disc_crop_size or W != disc_crop_size:
                                # If the image is smaller than crop_size, resize (should not happen; fallback)
                                fake_detached = F.interpolate(
                                    fake_detached, size=(disc_crop_size, disc_crop_size),
                                    mode='bilinear', align_corners=False)
                                real_detached = F.interpolate(
                                    real_detached, size=(disc_crop_size, disc_crop_size),
                                    mode='bilinear', align_corners=False)

                        # Apply data augmentation
                        # Note: DiffAug.aug() forces float32, so cast again afterwards
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_detached)

                        # Ensure the input dtype matches the discriminator (bfloat16)
                        # DiffAug casts inputs to float32; cast back to bfloat16
                        try:
                            disc_dtype = next(discriminator.module.parameters()).dtype
                        except (StopIteration, AttributeError):
                            # Fallback: default to bfloat16 (set explicitly in the training script)
                            disc_dtype = torch.bfloat16
                        fake_input = fake_input.to(dtype=disc_dtype)
                        real_input = real_input.to(dtype=disc_dtype)

                        # Compute discriminator outputs
                        logits_fake, logits_real = discriminator(fake_input, real_input)

                        # Compute the discriminator loss (hinge loss)
                        disc_loss = hinge_d_loss(logits_real, logits_fake)

                        # Compute discriminator accuracy
                        disc_accuracy = (logits_real > logits_fake).float().mean()

                        # Discriminator backward
                        disc_loss.backward()
                        # gradient clipping for the discriminator
                        if config.get('disc_max_grad_norm', 1.0) is not None:
                            torch.nn.utils.clip_grad_norm_(
                                discriminator.parameters(),
                                max_norm=config.get('disc_max_grad_norm', 1.0))
                        disc_optimizer.step()

                    # Step the discriminator LR scheduler (once per batch)
                    if disc_scheduler is not None:
                        disc_scheduler.step()

                    # Restore model states
                    discriminator.eval()
                    student_model.train()

            # -----------------------------------------------------------
            # 6. Logging
            # -----------------------------------------------------------
            if rank == 0 and (i + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]

                # Compute timing info
                elapsed_time = time.time() - start_time
                steps_done = global_step + 1
                steps_per_sec = steps_done / elapsed_time if elapsed_time > 0 else 0
                remaining_steps = total_steps - steps_done
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

                # Format the time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                # Build the log string (Stage3: show each loss component)
                log_str = (
                    f"[{current_time_str}] "
                    f"Epoch [{epoch+1}/{config['epochs']}] Step [{i+1}/{len(dataloader)}] "
                    f"LR: {current_lr:.2e} | "
                    f"Total: {total_loss.item():.4f} | "
                    f"Flow_MSE: {flow_mse_loss.item():.4f} | "
                    f"LPIPS: {lpips_loss.item():.4f} | "
                    f"GAN: {gan_loss.item():.4f}"
)

                # If the adaptive weight is enabled, log it
                if use_gan_loss and config.get('use_adaptive_gan_weight', False):
                    if isinstance(adaptive_weight, torch.Tensor):
                        log_str += f" | GAN_Weight: {adaptive_weight.item():.4f}"
                    else:
                        log_str += f" | GAN_Weight: {adaptive_weight:.4f}"

                # If GAN is enabled, add GAN-related logs
                if train_disc:
                    disc_lr = (disc_scheduler.get_last_lr()[0]
                                   if disc_scheduler is not None else config.get('disc_lr', 1e-4))
                    log_str += (f" | D_Loss: {disc_loss.item():.4f} | D_Acc: {disc_accuracy.item():.2%}"
                                f" | D_LR: {disc_lr:.2e}")

                log_str += f" | Elapsed: {elapsed_str} | ETA: {eta_str}"
                print(log_str)

                if writer:
                    writer.add_scalar('Loss/Total', total_loss.item(), global_step)
                    writer.add_scalar('Loss/Flow_MSE', flow_mse_loss.item(), global_step)
                    writer.add_scalar('Loss/LPIPS', lpips_loss.item(), global_step)
                    writer.add_scalar('Loss/GAN', gan_loss.item(), global_step)
                    writer.add_scalar('Params/LR', current_lr, global_step)

                    # Adaptive weight logging
                    if use_gan_loss and config.get('use_adaptive_gan_weight', False):
                        if isinstance(adaptive_weight, torch.Tensor):
                            writer.add_scalar('Params/GAN_Weight_Adaptive', adaptive_weight.item(), global_step)
                        else:
                            writer.add_scalar('Params/GAN_Weight_Adaptive', adaptive_weight, global_step)

                    # GAN-related logging
                    if train_disc:
                        disc_lr = (disc_scheduler.get_last_lr()[0]
                                       if disc_scheduler is not None else config.get('disc_lr', 1e-4))
                        writer.add_scalar('Loss/Discriminator', disc_loss.item(), global_step)
                        writer.add_scalar('Disc/Accuracy', disc_accuracy.item(), global_step)
                        writer.add_scalar('Params/Disc_LR', disc_lr, global_step)

            # Evaluate every 2000 steps
            if (global_step + 1) % 2000 == 0:
                dist.barrier()
                if rank == 0:
                    evaluate(student_model, epoch, global_step, config, device, writer)
                dist.barrier()

            # Save a periodic checkpoint
            if rank == 0 and (global_step + 1) % 10000 == 0:
                ckpt_path = os.path.join(config['output_dir'], 'checkpoints', f'step_{global_step+1}.pth')
                ckpt_dict = {
                    'model': student_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'global_step': global_step,
                }
                # Save discriminator states
                if discriminator is not None:
                    ckpt_dict['discriminator'] = discriminator.module.state_dict()
                if disc_optimizer is not None:
                    ckpt_dict['disc_optimizer'] = disc_optimizer.state_dict()
                if disc_scheduler is not None:
                    ckpt_dict['disc_scheduler'] = disc_scheduler.state_dict()

                torch.save(ckpt_dict, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    # ==========================================================================
    # 5. Finish training
    # ==========================================================================
    if rank == 0:
        print(f"Training finished. Time: {(time.time()-start_time)/3600:.2f} hrs.")
        final_path = os.path.join(config['output_dir'], 'final_model.pth')
        final_ckpt = {
            'model': student_model.module.state_dict(),
            'config': config
        }
        # Save discriminator states
        if discriminator is not None:
            final_ckpt['discriminator'] = discriminator.module.state_dict()
        if disc_optimizer is not None:
            final_ckpt['disc_optimizer'] = disc_optimizer.state_dict()
        if disc_scheduler is not None:
            final_ckpt['disc_scheduler'] = disc_scheduler.state_dict()

        torch.save(final_ckpt, final_path)
        print(f"Saved final model to {final_path}")
        if writer:
            writer.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
