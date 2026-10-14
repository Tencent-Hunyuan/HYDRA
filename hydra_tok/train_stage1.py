# Copyright 2026 HYDRA Authors.
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


# Import model definitions from the specific folder
from hydra_tok.modeling_hydra_tok import HydraTokVisionModel, Distill_Adapter
from hydra_tok.configuration_hydra_tok import HydraTokConfig


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'configs', 'stage1.yaml')


def load_config() -> dict:
    """Load the training config from YAML, then apply --key=value CLI overrides.

    ``configs/stage1.yaml`` is the single source of truth: it supplies every
    default and doubles as the key schema, so a typo in a config file or on the
    command line is rejected instead of silently ignored. A ``--config`` file may
    be partial; whatever it omits falls back to the shipped defaults.

    Usage:
        torchrun --nproc_per_node=8 train_stage1.py                          # configs/stage1.yaml
        torchrun --nproc_per_node=8 train_stage1.py --config my_stage1.yaml
        torchrun --nproc_per_node=8 train_stage1.py --config my_stage1.yaml --lr=1e-4
    """
    import argparse
    parser = argparse.ArgumentParser(description="HYDRA-TOK stage-1 training")
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

# ==============================================================================
# 2. Custom datasets
# ==============================================================================


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


class SAM1BDataset(Dataset):
    """
    SAM-1B dataset class (optimized)
    """

    def __init__(self, image_root: str, caption_root: str, dataset_size: int = 10000000, transform=None):
        logger.info(f"--- Initializing SAM1BDataset ---")
        logger.info(f"Image root: {image_root}")
        logger.info(f"Caption root: {caption_root}")

        self.image_root = image_root
        self.caption_root = caption_root
        self.transform = transform
        self.dataset_size = dataset_size

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            logger.info(f"Dataset size: {self.dataset_size}")
            if self.transform:
                logger.info("Received an externally provided transform.")
            else:
                logger.warning("Warning: SAM1BDataset received no transform!")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int):
        """
        Fetch one data sample.
        """
        try:
            file_index = idx + 1
            file_basename = f"sa_{file_index}"

            # 1. Load the caption
            cap_abs_path = os.path.join(self.caption_root, f"{file_basename}.txt")
            with open(cap_abs_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()

            if not prompt:
                raise FileNotFoundError(f"Caption file {cap_abs_path} is empty.")

            # 2. Load the image
            img_abs_path = os.path.join(self.image_root, f"{file_basename}.jpg")
            image = Image.open(img_abs_path).convert('RGB')

            if self.transform:
                jpg_tensor = self.transform(image)
            else:
                jpg_tensor = transforms.ToTensor()(image)

            # 3. Return the image and label (1 means the SAM dataset)
            return jpg_tensor, 0

        except (FileNotFoundError, IOError, Image.UnidentifiedImageError, ValueError) as e:
            # On error: skip the corrupt sample and fetch a random one
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self[new_idx]

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logger.error(
                f"SAM1BDataset: unexpected error at index {idx} (sa_{idx+1}): {e}. Fetching a random sample...")
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self[new_idx]


# ==============================================================================
# 3. Helper functions
# ==============================================================================

def create_teacher_model(ckpt: str):
    """ Load the frozen InternViT teacher model """
    if not ckpt:
        raise ValueError(
            "teacher_path is not set. Point it at a local InternViT-300M-448px-V2.5 "
            "directory (see configs/stage1.yaml).")
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"Loading pretrained teacher model: {ckpt}")
    teacher_model = AutoModel.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    return teacher_model


def calculate_distillation_loss(student_features, teacher_features, beta=2.0, device='cuda'):
    """ Compute the layer-wise adaptive distillation loss """
    num_layers = min(len(student_features), len(teacher_features))
    if num_layers == 0:
        return torch.tensor(0.0, device=device)

    layer_losses, alignment_penalties = [], []
    for l in range(num_layers):
        s_feat, t_feat = student_features[l], teacher_features[l]
        # Basic shape check
        if s_feat.shape != t_feat.shape:
            # Handle a possible CLS token mismatch (they usually match here)
            min_len = min(s_feat.shape[1], t_feat.shape[1])
            s_feat = s_feat[:, -min_len:, :]
            t_feat = t_feat[:, -min_len:, :]

        cos_sim = F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
        penalty = 1.0 - cos_sim
        alignment_penalties.append(penalty)
        layer_losses.append(penalty)

    adaptive_weights = 1 / num_layers

    total_loss = torch.sum(adaptive_weights * torch.stack(layer_losses))
    return total_loss


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
    student_model.eval()

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
    # All hyperparameters live in configs/stage1.yaml; --config picks a different
    # file and --key=value overrides individual fields.
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
    # 1. Teacher model (frozen)
    teacher_model = create_teacher_model(config["teacher_path"]).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 2. Student model (trainable)
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
    # 3. Weight loading (teacher -> student) - initialize depending on config
    if config['init_from_teacher']:
        if rank == 0:
            print("Loading teacher weights into student...")
        teacher_sd = teacher_model.state_dict()
        student_model.load_state_dict(teacher_sd, strict=False)

        if rank == 0:
            print("Initializing Global Blocks from Student Encoder...")
        num_global = student_config.global_blocks_depth
        if num_global > 0:
            # source_layers = student_model.encoder.layers_second[-num_global:]
            source_layers = student_model.encoder.layers[-num_global:]
            target_layers = student_model.global_blocks
            for i in range(num_global):
                target_layers[i].load_state_dict(source_layers[i].state_dict())
    else:
        if rank == 0:
            print("Skipping teacher weight initialization, using random/pretrained weights...")

    # 4. Adapter initialization
    # 24 Transformer layers; create 24 adapters, one per layer 0-23
    teacher_dim = teacher_model.config.hidden_size
    student_dim = student_config.hidden_size
    num_layers = student_config.num_hidden_layers  # should be 24

    adapters = nn.ModuleList([
        Distill_Adapter(in_channels=student_dim, out_channels=teacher_dim) 
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)

    # Print parameter info
    print_trainable_modules(student_model, rank)

    # DDP broadcast initial weights


    # DDP wrap
    student_model = DDP(student_model, device_ids=[local_rank])
    adapters = DDP(adapters, device_ids=[local_rank])

    # --------------------------
    # D. Data loader
    # --------------------------
    if rank == 0:
        print("Setting up data loader...")
        print("model")
        print(student_model)

    # Use the HuggingFace processor to get normalization parameters
    processor_path = config["processor_path"]
    image_processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

    transform = transforms.Compose([
        transforms.Resize(image_processor.size["shortest_edge"]), 
        transforms.CenterCrop(image_processor.crop_size["height"]), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    # Select the dataset based on config
    if config['dataset_type'] == 'imagenet_custom':
        # Combine ImageNet + CustomJSONDataset (two datasets)
        if rank == 0:
            print(f"Using ImageNet + Custom Dataset")
            print(f"  - ImageNet: {config['imagenet_path']}")
            print(f"  - Custom YAML: {config['yaml_path']}")

        dataset_list = []

        # 1. Load the ImageNet dataset
        imagenet_dataset = datasets.ImageFolder(
            root=os.path.join(config['imagenet_path'], 'train'), 
            transform=transform
        )
        dataset_list.append(imagenet_dataset)

        if rank == 0:
            print(f"  - ImageNet size: {len(imagenet_dataset)}")

        # 2. Load the custom YAML dataset
        custom_dataset = CustomJSONDataset(
            yaml_path=config['yaml_path'], 
            transform=transform
        )
        dataset_list.append(custom_dataset)

        if rank == 0:
            print(f"  - Custom size: {len(custom_dataset)}")

        # 3. Merge the two datasets
        dataset = ConcatDataset(dataset_list)

        if rank == 0:
            print(f"  - Combined total size: {len(dataset)}")

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=local_batch_size, sampler=sampler, 
            num_workers=8, pin_memory=True, shuffle=False
        )

    elif config['dataset_type'] == 'sam':
        # Use the SAM-1B dataset only
        if rank == 0:
            print(f"Using SAM-1B Dataset")
            print(f"  - SAM Image Root: {config['sam_image_root']}")
            print(f"  - SAM Caption Root: {config['sam_caption_root']}")

        sam_dataset = SAM1BDataset(
            image_root=config['sam_image_root'],
            caption_root=config['sam_caption_root'],
            dataset_size=config['sam_dataset_size'],
            transform=transform
        )

        if rank == 0:
            print(f"  - SAM Dataset size: {len(sam_dataset)}")

        sampler = DistributedSampler(sam_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            sam_dataset, batch_size=local_batch_size, sampler=sampler, 
            num_workers=8, pin_memory=True, shuffle=False
        )

    elif config['dataset_type'] == 'imagenet_custom_sam':
        # Combine ImageNet + CustomJSONDataset + SAM-1B (three datasets)
        if rank == 0:
            print(f"Using ImageNet + Custom + SAM Dataset")
            print(f"  - ImageNet: {config['imagenet_path']}")
            print(f"  - Custom YAML: {config['yaml_path']}")
            print(f"  - SAM Image Root: {config['sam_image_root']}")

        dataset_list = []

        # 1. Load the ImageNet dataset
        imagenet_dataset = datasets.ImageFolder(
            root=os.path.join(config['imagenet_path'], 'train'), 
            transform=transform
        )
        dataset_list.append(imagenet_dataset)

        if rank == 0:
            print(f"  - ImageNet size: {len(imagenet_dataset)}")

        # 2. Load the custom YAML dataset
        custom_dataset = CustomJSONDataset(
            yaml_path=config['yaml_path'], 
            transform=transform
        )
        dataset_list.append(custom_dataset)

        if rank == 0:
            print(f"  - Custom size: {len(custom_dataset)}")

        # 3. Load the SAM-1B dataset
        sam_dataset = SAM1BDataset(
            image_root=config['sam_image_root'],
            caption_root=config['sam_caption_root'],
            dataset_size=config['sam_dataset_size'],
            transform=transform
        )
        dataset_list.append(sam_dataset)

        if rank == 0:
            print(f"  - SAM size: {len(sam_dataset)}")

        # 4. Merge all datasets
        dataset = ConcatDataset(dataset_list)

        if rank == 0:
            print(f"  - Combined total size: {len(dataset)}")

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=local_batch_size, sampler=sampler, 
            num_workers=8, pin_memory=True, shuffle=False
        )

    elif config['dataset_type'] == 'combined':
        # Combine ImageNet + CustomJSONDataset + SAM-1B (three datasets)
        if rank == 0:
            print(f"Using Combined Dataset: ImageNet + CustomJSONDataset + SAM-1B")
            print(f"  - ImageNet: {config['imagenet_path']}")
            print(f"  - Custom YAML: {config['yaml_path']}")
            print(f"  - SAM Image Root: {config['sam_image_root']}")

        dataset_list = []

        # 1. Load the ImageNet dataset
        imagenet_dataset = datasets.ImageFolder(
            root=os.path.join(config['imagenet_path'], 'train'),
            transform=transform
        )
        dataset_list.append(imagenet_dataset)

        if rank == 0:
            print(f"  - ImageNet size: {len(imagenet_dataset)}")

        # 2. Load the custom YAML dataset
        custom_dataset = CustomJSONDataset(
            yaml_path=config['yaml_path'],
            transform=transform
        )
        dataset_list.append(custom_dataset)

        if rank == 0:
            print(f"  - Custom size: {len(custom_dataset)}")

        # 3. Load the SAM-1B dataset
        sam_dataset = SAM1BDataset(
            image_root=config['sam_image_root'],
            caption_root=config['sam_caption_root'],
            dataset_size=config['sam_dataset_size'],
            transform=transform
        )
        dataset_list.append(sam_dataset)

        if rank == 0:
            print(f"  - SAM size: {len(sam_dataset)}")

        # 4. Merge all datasets
        dataset = ConcatDataset(dataset_list)

        if rank == 0:
            print(f"  - Combined total size: {len(dataset)}")

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=local_batch_size, sampler=sampler,
            num_workers=8,
            pin_memory=True, shuffle=False
        )

    elif config['dataset_type'] == 'custom':
        if rank == 0:
            print(f"Using CustomJSONDataset with YAML: {config['yaml_path']}")
        dataset = CustomJSONDataset(yaml_path=config['yaml_path'], transform=transform)
        if rank == 0:
            print(f"Dataset size: {len(dataset)}")

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=local_batch_size, sampler=sampler, 
            num_workers=8, pin_memory=True, shuffle=False
        )

    else:
        # Default to ImageNet
        if rank == 0:
            print(f"Using ImageNet dataset from: {config['imagenet_path']}")
        dataset = datasets.ImageFolder(root=os.path.join(config['imagenet_path'], 'train'), transform=transform)
        if rank == 0:
            print(f"Dataset size: {len(dataset)}")

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=local_batch_size, sampler=sampler, 
            num_workers=8, pin_memory=True, shuffle=False
        )

    # --------------------------
    # E. Optimizer & scheduler
    # --------------------------
    params = list(student_model.module.parameters()) + list(adapters.module.parameters())
    optimizer = optim.AdamW(params, lr=config['lr'], betas=(config['adam_beta1'], config['adam_beta2']))

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

    for epoch in range(config['epochs']):
        student_model.train()
        adapters.train()
        sampler.set_epoch(epoch)

        for i, batch_data in enumerate(dataloader):
            # Handle different data formats
            if isinstance(batch_data, dict):
                # Dict format: 'images' is a list of PIL images; apply the transform
                pil_images = batch_data['images']
                images_list = [transform(img) for img in pil_images]
                raw_images = torch.stack(images_list)
            else:
                # ImageNet/Custom/Combined datasets return (image, label) tuples
                # Images are already tensors
                raw_images, _ = batch_data

            images = raw_images.to(device, dtype=torch.bfloat16)
            optimizer.zero_grad()

            # -----------------------------------------------------------
            # 1. Forward Pass
            # -----------------------------------------------------------
            # Student: returns the flow loss and all layer outputs (index 0...24)
            loss_dict, s_hidden_states = student_model(
                pixel_values=images,
                output_hidden_states=True,
                use_kl_loss=config['use_kl_loss'],
                use_bn_loss=config['use_bn_loss'],
                kl_weight=config['kl_weight'],
                bn_weight=config['bn_weight'],
                lpips_weight=config['lpips_weight'],
            )
            loss_flow = loss_dict['total']

            # Teacher: returns all layer outputs (index 0...24)
            with torch.no_grad():
                t_outputs = teacher_model(pixel_values=images, output_hidden_states=True)
                t_hidden_states = t_outputs.hidden_states

            # -----------------------------------------------------------
            # 2. Feature slicing and alignment
            # -----------------------------------------------------------
            # Rationale: hidden_states[0] is the embedding, so we skip it
            # hidden_states[1] is layer 0, ..., hidden_states[24] is layer 23

            # Extract the valid Transformer layers (24 in total)
            s_layers = s_hidden_states[1:] 
            t_layers = t_hidden_states[1:]

            # Distill from all 24 layers
            s_all = s_layers
            t_all = t_layers
            adapters_all = adapters.module  # use all adapters

            # -----------------------------------------------------------
            # 3. Compute the distillation loss
            # -----------------------------------------------------------

            # >>> All Layers <<<
            adapted_s_all = [adapter(feat).float() for adapter, feat in zip(adapters_all, s_all)]
            t_all_fp32 = [feat.float() for feat in t_all]
            loss_dist = calculate_distillation_loss(
                adapted_s_all, t_all_fp32, beta=config['distillation_beta'], device=device
            )

            # -----------------------------------------------------------
            # 4. Backward
            # -----------------------------------------------------------
            # Combine into the total loss
            total_loss = (config['lambda_f'] * loss_flow) + \
                         (config['lambda_d_front'] * loss_dist)

            total_loss.backward()
            # gradient clipping (stabilizes training against loss spikes)
            if config.get('max_grad_norm', 1.0) is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(student_model.parameters()) + list(adapters.parameters()),
                    max_norm=config.get('max_grad_norm', 1.0))
            optimizer.step()
            scheduler.step()

            # -----------------------------------------------------------
            # 5. Logging
            # -----------------------------------------------------------
            global_step = epoch * len(dataloader) + i

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

                print(
                    f"[{current_time_str}] "
                    f"Epoch [{epoch+1}/{config['epochs']}] Step [{i+1}/{len(dataloader)}] "
                    f"LR: {current_lr:.2e} | Total: {total_loss.item():.4f} | "
                    f"Flow: {loss_flow.item():.4f} | "
                    f"Dist-A: {loss_dist.item():.4f} | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                )
                if writer:
                    writer.add_scalar('Loss/Total', total_loss.item(), global_step)
                    writer.add_scalar('Loss/Flow', loss_flow.item(), global_step)
                    writer.add_scalar('Loss/Dist_A', loss_dist.item(), global_step)
                    writer.add_scalar('Params/LR', current_lr, global_step)

            # Save a periodic checkpoint
            if rank == 0 and (global_step + 1) % 50000 == 0:
                ckpt_path = os.path.join(config['output_dir'], 'checkpoints', f'step_{global_step+1}.pth')
                torch.save({
                    'model': student_model.module.state_dict(),
                    'adapters': adapters.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # End of Epoch Evaluation
        dist.barrier()
        if rank == 0:
            evaluate(student_model, epoch, global_step, config, device, writer)

    # ==========================================================================
    # 5. Finish training
    # ==========================================================================
    if rank == 0:
        print(f"Training finished. Time: {(time.time()-start_time)/3600:.2f} hrs.")
        final_path = os.path.join(config['output_dir'], 'final_model.pth')
        torch.save({
            'model': student_model.module.state_dict(),
            'adapters': adapters.module.state_dict(),
            'config': config
        }, final_path)
        print(f"Saved final model to {final_path}")
        if writer:
            writer.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
