# coding=utf-8
# Copyright 2025 NUS Show Lab.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0

"""
T2I Prompts inference script - HYDRA-TOK version (8-GPU DDP)

Two input formats are supported:
1. Text file: one prompt per line, use the line number as prompt_id
2. CSV file: read the 'prompt' column, use the 'index' column (if present) or the line number as prompt_id

Refer to the generation code in train_stage_one.py (the generate_images function)

Usage (8-GPU DDP):
# Text file
torchrun --nproc_per_node=8 evaluation/inference_t2i.py \
    config=configs/inference.yaml t2i_prompts_file=prompts/t2i_prompts.txt

# CSV file
# GenEval prompts: t2i_prompts_file=prompts/geneval_meta_data_rewritten.jsonl
"""

import os
import sys

# Add the project root to the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm
import logging
import numpy as np
from models import HydraQwen2_5, HydraQwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from tools.utils import get_config, flatten_omega_conf, denorm, get_hyper_params, get_llm_name, load_state_dict
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import json
import csv
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from einops import rearrange

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from transport import Sampler, create_transport


# ==================== Compile utility functions ====================
def compile_model_for_inference(model, vae_model, vae_type, rank=0, compile_mode="reduce-overhead"):
    """
    Compile the model with torch.compile to accelerate inference

    Args:
        model: main model (HydraQwen2_5)
        vae_model: VAE model (Gen_ViT)
        vae_type: VAE type
        rank: process rank
        compile_mode: compile mode, one of "reduce-overhead", "max-autotune", "default"
            - "reduce-overhead": suited for inference, reduces CPU overhead
            - "max-autotune": maximum autotuning, longer compile time but potentially faster
            - "default": default mode

    Returns:
        compiled_model, compiled_vae_model
    """
    logger = get_logger(__name__, rank)

    if not torch.cuda.is_available():
        if rank == 0:
            logger.warning("CUDA is not available, skipping torch.compile")
        return model, vae_model

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Compiling the model with torch.compile (mode={compile_mode})...")

    # Compile the main model's generation method
    try:
        # Use fullgraph=False to allow graph breaks for better compatibility
        model.t2i_generate = torch.compile(
            model.t2i_generate,
            mode=compile_mode,
            fullgraph=False,
            dynamic=True  # support dynamic batch size
)
        if rank == 0:
            logger.info("model.t2i_generate compiled successfully")
    except (RuntimeError, OSError, ValueError) as e:
        if rank == 0:
            logger.warning(f"model.t2i_generate compilation failed: {e}")

    # Compile the VAE decode method
    try:
        if vae_type == 'hydra_tok':
            vae_model.decode = torch.compile(
                vae_model.decode,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True
)
            if rank == 0:
                logger.info("vae_model.decode (Gen_ViT) compiled successfully")
    except (RuntimeError, OSError, ValueError) as e:
        if rank == 0:
            logger.warning(f"VAE compilation failed: {e}")

    if rank == 0:
        logger.info("=" * 60)

    return model, vae_model


def warmup_compiled_model(model, vae_model, vae_type, device, weight_type,
                          image_latent_dim, latent_height, latent_width, patch_size,
                          text_tokenizer, num_t2i_image_tokens, bos_id, eos_id,
                          boi_id, eoi_id, pad_id, img_pad_id, max_text_len, max_seq_len,
                          config, sampler, rank=0):
    """
    Warm up the compiled model to trigger JIT compilation
    The first call to the compiled model is slow, so compilation is triggered in advance here
    """
    logger = get_logger(__name__, rank)

    if rank == 0:
        logger.info("Warming up the compiled model...")

    warmup_batch_size = 1
    warmup_prompt = ["A beautiful sunset over the ocean"]

    try:
        with torch.no_grad():
            # Prepare inputs
            batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
                prepare_gen_input(
                    warmup_prompt, text_tokenizer, num_t2i_image_tokens,
                    bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                    max_text_len, device
)

            # Initialize noise
            z = torch.randn((warmup_batch_size,
                             image_latent_dim, latent_height * patch_size,
                             latent_width * patch_size)).to(weight_type).to(device)

            # CFG setup
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)

            # Attention mask
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0),
                max_seq_len,
                modality_positions,
                device
).to(weight_type)

            model_kwargs = dict(
                text_tokens=torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0),
                attention_mask=block_mask,
                modality_positions=torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0),
                output_hidden_states=True,
                max_seq_len=max_seq_len,
                guidance_scale=config.transport.guidance_scale,
                _return_raw_cfg=True  # flag: return raw cond and uncond, do not apply CFG inside the model
)

            # Sampling (run only a few steps to trigger compilation)
            sample_fn = sampler.sample_ode(
                sampling_method=config.transport.sampling_method,
                num_steps=2,  # a small number of steps for warmup
                atol=config.transport.atol,
                rtol=config.transport.rtol,
                reverse=config.transport.reverse,
                time_shifting_factor=config.transport.time_shifting_factor
)

            samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
            samples = torch.chunk(samples, 2)[0]
            samples = samples.to(weight_type)

            # Warm up VAE decode
            if vae_type == 'hydra_tok':
                _ = vae_model.decode(samples)

        # Synchronize CUDA
        torch.cuda.synchronize()

        if rank == 0:
            logger.info("Model warmup complete")

    except (RuntimeError, OSError, ValueError) as e:
        if rank == 0:
            logger.warning(f"Model warmup failed (may affect the first batch's inference speed): {e}")


# ==================== DDP utility functions ====================
def setup_distributed():
    """Initialize the distributed process group"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up the distributed process group"""
    dist.destroy_process_group()


def get_logger(name, rank=0):
    """Get a logger; only rank 0 emits logs"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


def load_hydra_tok_vae(config, device, weight_type, rank=0):
    """
    Load the HYDRA-TOK Gen_ViT VAE model
    Refer to lines 225-306 of train_stage_one.py
    """
    from models.hydra_tok.modeling_hydra_tok import Gen_ViT
    from models.hydra_tok.configuration_hydra_tok import HydraTokConfig

    logger = get_logger(__name__, rank)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Initializing Gen_ViT (generation branch)...")

    # Load config
    hydra_tok_config = HydraTokConfig.from_pretrained(config.model.hydra_tok_model.hydra_tok_config)
    hydra_tok_config.gen_layers = getattr(config.model.hydra_tok_model, 'gen_layers', 12)
    hydra_tok_config.und_layers = getattr(config.model.hydra_tok_model, 'und_layers', 12)
    hydra_tok_config.latent_stats_path = None

    if rank == 0:
        logger.info(f"HYDRA-TOK config: {config.model.hydra_tok_model.hydra_tok_config}")
        logger.info(f"Gen layers: {hydra_tok_config.gen_layers}, Und layers: {hydra_tok_config.und_layers}")

    # Initialize model
    vae_model = Gen_ViT(hydra_tok_config).to(device).to(weight_type)

    # Load Gen_ViT pretrained weights (refer to lines 241-249 of train_stage_one.py)
    gen_vit_weights_path = os.path.join(config.model.hydra_tok_model.pretrained_model_path, "gen_vit_weights.pth")
    if os.path.exists(gen_vit_weights_path):
        if rank == 0:
            logger.info(f"Loading Gen_ViT weights: {gen_vit_weights_path}")
        gen_state_dict = torch.load(gen_vit_weights_path, map_location="cpu")
        missing_keys, unexpected_keys = vae_model.load_state_dict(gen_state_dict, strict=False)
        if rank == 0:
            if missing_keys:
                logger.warning(f"Gen_ViT missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Gen_ViT unexpected keys: {len(unexpected_keys)}")
            logger.info("Gen_ViT weights loaded successfully")
    else:
        if rank == 0:
            logger.warning(f"Gen_ViT weights file does not exist: {gen_vit_weights_path}")

    # Register processor buffers (refer to lines 251-256 of train_stage_one.py)
    DATA_NORM = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    p_mean = torch.tensor(DATA_NORM["mean"]).view(1, 3, 1, 1).to(device).to(weight_type)
    p_std = torch.tensor(DATA_NORM["std"]).view(1, 3, 1, 1).to(device).to(weight_type)
    vae_model.register_buffer('processor_mean', p_mean)
    vae_model.register_buffer('processor_std', p_std)
    if rank == 0:
        logger.info("Registered processor_mean and processor_std buffers")

    # Load latent statistics (refer to lines 258-296 of train_stage_one.py)
    latent_stats_path = getattr(config.model.hydra_tok_model, 'latent_stats_path', None)
    if latent_stats_path and os.path.exists(latent_stats_path):
        if rank == 0:
            logger.info(f"Loading Gen_ViT latent statistics: {latent_stats_path}")
        stats = torch.load(latent_stats_path, map_location="cpu")

        # Extract latent mean
        if 'latent_mean' in stats:
            latent_mean = stats['latent_mean']
        else:
            raise ValueError(f"Missing 'latent_mean' in {latent_stats_path}. Keys: {stats.keys()}")

        # Extract latent std
        if 'latent_std' in stats:
            latent_std = stats['latent_std']
        elif 'latent_std_dev' in stats:
            latent_std = stats['latent_std_dev']
        else:
            raise ValueError(f"Missing 'latent_std' in {latent_stats_path}. Keys: {stats.keys()}")

        # Extract eps
        if 'latent_eps' in stats:
            latent_eps = stats['latent_eps']
        else:
            if rank == 0:
                logger.warning("'latent_eps' not found, using default value 1e-6")
            latent_eps = 1e-6

        # Register as buffers
        vae_model.register_buffer('latent_mean', latent_mean.to(device).to(weight_type))
        vae_model.register_buffer('latent_std_dev', latent_std.to(device).to(weight_type))
        if isinstance(latent_eps, torch.Tensor):
            vae_model.register_buffer('latent_eps', latent_eps.to(device).to(weight_type))
        else:
            vae_model.register_buffer('latent_eps', torch.tensor(latent_eps, dtype=weight_type, device=device))

        if rank == 0:
            logger.info(f"Latent statistics loaded. Mean shape: {latent_mean.shape}, Std shape: {latent_std.shape}")
    else:
        if rank == 0:
            logger.warning(f"Latent statistics file does not exist: {latent_stats_path}")

    # Freeze Gen_ViT (no gradients needed for inference)
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()

    if rank == 0:
        logger.info("=" * 60)
    return vae_model


def load_und_vit(config, device, weight_type, load_pretrained=True, rank=0):
    """
    Load the HYDRA-TOK Und_ViT model
    Refer to lines 307-328 of train_stage_one.py

    Args:
        load_pretrained: whether to load pretrained weights. Set to False if the checkpoint
            already contains und_vit weights
    """
    from models.hydra_tok.modeling_hydra_tok import Und_ViT
    from models.hydra_tok.configuration_hydra_tok import HydraTokConfig

    logger = get_logger(__name__, rank)

    if rank == 0:
        logger.info("-" * 60)
        logger.info("Initializing Und_ViT (understanding branch)...")

    hydra_tok_config = HydraTokConfig.from_pretrained(config.model.hydra_tok_model.hydra_tok_config)
    hydra_tok_config.gen_layers = getattr(config.model.hydra_tok_model, 'gen_layers', 12)
    hydra_tok_config.und_layers = getattr(config.model.hydra_tok_model, 'und_layers', 12)

    und_vit = Und_ViT(hydra_tok_config).to(device).to(weight_type)

    # Load Und_ViT pretrained weights (only when load_pretrained=True)
    if load_pretrained:
        und_vit_weights_path = os.path.join(config.model.hydra_tok_model.pretrained_model_path, "und_vit_weights.pth")
        if os.path.exists(und_vit_weights_path):
            if rank == 0:
                logger.info(f"Loading Und_ViT pretrained weights: {und_vit_weights_path}")
            und_state_dict = torch.load(und_vit_weights_path, map_location="cpu")
            und_vit.load_state_dict(und_state_dict, strict=False)
            if rank == 0:
                logger.info("Und_ViT pretrained weights loaded successfully")
        else:
            if rank == 0:
                logger.warning(f"Und_ViT weights file does not exist: {und_vit_weights_path}")
    else:
        if rank == 0:
            logger.info("Skipping Und_ViT pretrained weights loading (they will be loaded from the checkpoint)")

    und_vit.eval()
    if rank == 0:
        logger.info("-" * 60)
    return und_vit


def decode_hydra_tok(vae_model, samples, weight_type):
    """
    Decode latents to images with the HYDRA-TOK Gen_ViT
    Refer to lines 1263-1269 of train_stage_one.py
    """
    with torch.no_grad():
        # Ensure dtype consistency
        samples = samples.to(weight_type)

        # Gen_ViT: decode latents [B, C, H, W] -> images [B, 3, H, W]
        images = vae_model.decode(samples)
        images = images.float().cpu()

        # Clamp to [0, 1]
        images = torch.clamp(images, 0, 1)

        # Convert to uint8 numpy array [B, H, W, 3]
        images = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    return images


def _load_checked(model, state_dict, source, rank=0):
    """`model.load_state_dict(..., strict=False)` with a sanity check.

    strict=False is required here (und_vit / vae weights are loaded separately),
    but it also silently accepts a state dict that matches *nothing* -- a wrong
    model_path or a stale `showo.*` prefix then leaves the whole model randomly
    initialized and inference emits pure noise with no error anywhere. Check that
    a sane fraction of the model's parameters actually got filled.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    total = len(model.state_dict())
    missing = len(incompatible.missing_keys)
    loaded = total - missing
    if rank == 0:
        print(f"Loaded {loaded}/{total} tensors from {source} "
              f"({missing} missing, {len(incompatible.unexpected_keys)} unexpected)")
    if loaded < 0.5 * total:
        raise RuntimeError(
            f"Only {loaded}/{total} model tensors were loaded from {source!r} -- the "
            f"checkpoint does not match this model. Check that model_path points at the "
            f"right directory and that the weight prefixes are 'llm.*' (legacy 'showo.*' "
            f"checkpoints need tools/convert_showo_prefix.py). "
            f"First few missing keys: {incompatible.missing_keys[:5]}"
        )
    return incompatible


def main():
    # ==================== Initialize distributed setup ====================
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    weight_type = torch.bfloat16

    logger = get_logger(__name__, rank)

    if rank == 0:
        print(f"Launching {world_size}-GPU DDP inference")
        print("=" * 60)

    # ==================== CUDA inference optimization settings ====================
    # Enable TensorFloat-32 (TF32) to accelerate matrix operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN autotuning (effective for fixed input sizes)
    torch.backends.cudnn.benchmark = True

    # Set float32 matmul precision to 'high' (uses TF32)
    torch.set_float32_matmul_precision('high')

    if rank == 0:
        logger.info("CUDA inference optimizations enabled (TF32, cuDNN benchmark)")

    # Synchronize all processes
    dist.barrier()

    config = get_config()

    vae_type = config.model.vae_model.type
    if rank == 0:
        logger.info(f"VAE type: {vae_type}")

    # ==================== Load VAE model ====================
    if vae_type == 'hydra_tok':
        vae_model = load_hydra_tok_vae(config, device, weight_type, rank)
        # Do not load pretrained weights, because the checkpoint already contains the full und_vit weights
        und_vit = load_und_vit(config, device, weight_type, load_pretrained=False, rank=rank)
    else:
        raise NotImplementedError(f"Unsupported VAE type: {vae_type}")

    # ==================== Initialize tokenizer ====================
    text_tokenizer, hydra_token_ids = get_text_tokenizer(
        config.model.llm.llm_model_path,
        add_hydra_tokens=True,
        return_hydra_token_ids=True,
        llm_name=get_llm_name(config.model.llm.llm_model_path)
)
    config.model.llm.llm_vocab_size = len(text_tokenizer)

    # ==================== Select model class ====================
    if vae_type == 'hydra_tok':
        ModelClass = HydraQwen2_5
        if rank == 0:
            logger.info("Using HydraQwen2_5 model")
    else:
        ModelClass = HydraQwen2_5
        if rank == 0:
            logger.info("Using HydraQwen2_5 model")

    # ==================== Initialize model ====================
    # Refer to lines 372-376 of train_stage_one.py
    if config.model.llm.load_from_llm:
        model = ModelClass.from_pretrained(config.model.llm.pretrained_model_path, use_safetensors=False).to(device)
    else:
        model = ModelClass(**config.model.llm).to(device)

    # ==================== Set up Und_ViT (before loading weights) ====================
    # und_vit must be set first so that load_state_dict can load the und_vit.* weights
    # Refer to lines 380-384 of train_stage_one.py
    if vae_type == 'hydra_tok' and und_vit is not None:
        model.set_und_vit(und_vit)
        if rank == 0:
            logger.info("Und_ViT has been set on the HydraQwen2_5 model (will be overwritten by the model weights)")

    # ==================== Load model weights ====================
    # Consistent with lines 771-805 of train_stage_one.py
    if not config.model.llm.load_from_llm:
        model_path = config.model_path

        # Determine whether model_path is a checkpoint directory or a specific file
        if os.path.isdir(model_path):
            # If it is a directory, assume checkpoint-xxx format
            checkpoint_dir = model_path
        elif "ema_model" in model_path and model_path.endswith(".pt"):
            # If it is an ema_model/ema_state.pt path
            ema_dir = os.path.dirname(model_path)
            checkpoint_dir = os.path.dirname(ema_dir)
        else:
            # If it is a specific.bin/.pt file
            checkpoint_dir = os.path.dirname(os.path.dirname(model_path))

        # Step 1: Load the base model
        base_model_path = os.path.join(checkpoint_dir, "unwrapped_model", "pytorch_model.bin")
        if os.path.exists(base_model_path):
            if rank == 0:
                logger.info(f"Loading base model from: {base_model_path}")
            state_dict = torch.load(base_model_path, map_location="cpu")
            _load_checked(model, state_dict, base_model_path, rank)
            del state_dict
            if rank == 0:
                logger.info(" Base model loaded")
        else:
            # Try loading directly from model_path
            if rank == 0:
                logger.info(f"Loading model from: {model_path}")
            state_dict = load_state_dict(model_path)
            _load_checked(model, state_dict, model_path, rank)
            del state_dict
            if rank == 0:
                logger.info(" Model loaded")

    model.to(weight_type)
    model.eval()

    # ==================== torch.compile optimization ====================
    # Note: the first torch.compile may take 5-10 minutes, but subsequent inference will be faster
    # To skip waiting for compilation, set use_compile: false
    use_compile = getattr(config, 'use_compile', False)  # compilation disabled by default (to avoid long waits)
    compile_mode = getattr(config, 'compile_mode', 'default')  # use default mode for faster compilation

    if use_compile:
        if rank == 0:
            logger.info("Starting torch.compile (the first run may take 5-10 minutes, please be patient)...")
            logger.info("To skip compilation, set use_compile: false in the config")
        model, vae_model = compile_model_for_inference(
            model, vae_model, vae_type, rank, compile_mode
)
    else:
        if rank == 0:
            logger.info("Skipping torch.compile (to enable, set use_compile: true)")

    # for time embedding
    if config.model.llm.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    # ==================== Get hyperparameters ====================
    num_t2i_image_tokens, _, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, hydra_token_ids)

    # Read inference parameters from config. Every sampling knob lives under
    # `transport` -- see configs/inference.yaml. Earlier revisions also had
    # top-level `guidance_scale` / `num_inference_steps` keys that shadowed the
    # `transport` ones; they are gone, so fail loudly instead of silently
    # ignoring a stale `guidance_scale=7.5` on the command line (OmegaConf would
    # happily merge it in as an unused key).
    for legacy_key in ('guidance_scale', 'num_inference_steps'):
        if legacy_key in config:
            raise ValueError(
                f"Top-level '{legacy_key}' is no longer supported -- it used to shadow "
                f"'transport.{legacy_key}'. Pass 'transport.{legacy_key}=...' instead."
            )
    batch_size = config.batch_size
    guidance_scale = config.transport.guidance_scale
    num_inference_steps = config.transport.num_inference_steps
    output_dir = getattr(config, 'output_dir', None)  # support output_dir or outdir
    t2i_prompts_file = getattr(config, 't2i_prompts_file', 'prompts/t2i_prompts.txt')  # path to the T2I prompts file
    n_samples = getattr(config, 'n_samples', 1)  # generate 1 image per prompt
    skip_grid = getattr(config, 'skip_grid', False)  # whether to skip saving the grid
    seed = getattr(config, 'seed', 42)  # random seed

    if rank == 0:
        logger.info(f"Batch size: {batch_size}, Guidance scale: {guidance_scale}")
        logger.info(f"Inference steps: {num_inference_steps}")
        logger.info(f"Generating {n_samples} images per prompt")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Prompts file: {t2i_prompts_file}")
        logger.info(f"Skip saving grid: {skip_grid}")
        logger.info(f"World size: {world_size}")

    # ==================== Load T2I prompts ====================
    # Support both text files and CSV files
    if rank == 0:
        logger.info(f"Loading from the T2I prompts file: {t2i_prompts_file}")

    # Ensure the file path is absolute or relative to the project root
    if not os.path.isabs(t2i_prompts_file):
        t2i_prompts_file = os.path.join(PROJECT_ROOT, t2i_prompts_file)

    # Detect file type
    file_ext = os.path.splitext(t2i_prompts_file)[1].lower()

    if file_ext == '.csv':
        # CSV file: read the prompt column
        if rank == 0:
            logger.info("Detected CSV file format, reading the 'prompt' column")

        all_data = []
        with open(t2i_prompts_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                prompt = row.get('prompt', '').strip()
                if not prompt:
                    continue

                # Prefer the 'index' column as prompt_id, otherwise use the line number (starting from 1)
                if 'index' in row and row['index'].strip():
                    try:
                        prompt_id = str(int(row['index'].strip()))
                    except ValueError:
                        prompt_id = str(idx + 1)
                else:
                    prompt_id = str(idx + 1)

                all_data.append([prompt, prompt_id])

        if rank == 0:
            logger.info(f"Read {len(all_data)} prompts from the CSV file")
    elif file_ext == '.json' or os.path.isdir(t2i_prompts_file):
        # WISE benchmark format: a json file (or a directory of json files),
        # each a list of {"Prompt":..., "prompt_id":...} dicts.
        if rank == 0:
            logger.info("Detected WISE json format (Prompt/prompt_id)")

        from glob import glob
        if os.path.isdir(t2i_prompts_file):
            json_paths = sorted(glob(os.path.join(t2i_prompts_file, "*.json")))
        else:
            json_paths = [t2i_prompts_file]

        wise_data = []
        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as f:
                wise_data += json.load(f)
        all_data = [[i['Prompt'], str(i['prompt_id'])] for i in wise_data]

        if rank == 0:
            logger.info(f"Read {len(all_data)} prompts from {len(json_paths)} WISE json file(s)")
    else:
        # Text file: one prompt per line
        if rank == 0:
            logger.info("Detected text file format, one prompt per line")

        with open(t2i_prompts_file, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f.readlines() if line.strip()]

        # Use the line number (starting from 1) as prompt_id
        all_data = [[prompt, str(idx + 1)] for idx, prompt in enumerate(all_prompts)]

        if rank == 0:
            logger.info(f"Read {len(all_data)} prompts from the text file")

    if rank == 0:
        logger.info(f"Loaded {len(all_data)} prompts in total")

    # DDP data sharding: each rank processes different prompts
    data_for_this_rank = all_data[rank::world_size]

    if rank == 0:
        logger.info(f"{len(all_data)} prompts in total, each GPU processes about {len(data_for_this_rank)}")

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # ==================== Create Transport Sampler ====================
    # Refer to lines 845-856 of train_stage_one.py
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=config.dataset.preprocessing.num_t2i_image_tokens,
        cfg_start=getattr(config.transport, 'cfg_start', 0.0),
        cfg_end=getattr(config.transport, 'cfg_end', 1.0),
        rescale_factor=getattr(config.transport, 'rescale_factor', 1.0),
)

    sampler = Sampler(transport)

    # ==================== Model warmup (trigger JIT compilation) ====================
    # Warmup is only needed when compile is enabled
    do_warmup = getattr(config, 'do_warmup', True) and use_compile
    if do_warmup:
        if rank == 0:
            logger.info("Starting model warmup (triggering JIT compilation, may take a few minutes)...")
        warmup_compiled_model(
            model, vae_model, vae_type, device, weight_type,
            image_latent_dim, latent_height, latent_width, patch_size,
            text_tokenizer, num_t2i_image_tokens, bos_id, eos_id,
            boi_id, eoi_id, pad_id, img_pad_id, max_text_len, max_seq_len,
            config, sampler, rank
)
    else:
        if rank == 0 and use_compile:
            logger.info("Skipping model warmup (the first batch of inference may be slower)")

    # Synchronize all processes before starting inference
    dist.barrier()

    # ==================== Inference loop ====================
    with torch.no_grad():
        for prompt, prompt_id in tqdm(
            data_for_this_rank,
            total=len(data_for_this_rank),
            desc=f"Rank {rank} Generating",
            disable=(rank != 0)
):
            if not prompt:
                continue

            # Set the base random seed for reproducibility
            # Use prompt_id as part of the seed so the same prompt_id yields the same result
            try:
                prompt_id_int = int(prompt_id)
            except ValueError:
                prompt_id_int = hash(prompt_id) % (2**31)

            # Create the prompt directory (same format as geneval: 5 digits, zero-padded)
            try:
                prompt_dir = os.path.join(output_dir, f"{prompt_id_int:0>5}")
            except (TypeError, ValueError):
                # If prompt_id is not a number, use the original string
                prompt_dir = os.path.join(output_dir, str(prompt_id))

            sample_path = os.path.join(prompt_dir, "samples")
            os.makedirs(sample_path, exist_ok=True)

            # Save metadata.jsonl
            metadata = {
                'prompt': prompt,
                'prompt_id': prompt_id,
                'n_samples': n_samples
            }
            metadata_file = os.path.join(prompt_dir, "metadata.jsonl")
            with open(metadata_file, "w", encoding='utf-8') as fp:
                json.dump(metadata, fp, ensure_ascii=False)

            # Check which images already exist, to support resuming
            existing_files = set()
            for sample_idx in range(n_samples):
                sample_file = os.path.join(sample_path, f"{sample_idx:05}.png")
                if os.path.exists(sample_file):
                    existing_files.add(sample_idx)

            # If all images already exist, skip
            if len(existing_files) == n_samples:
                if rank == 0:
                    logger.debug(f"Skipping existing files: {prompt_id} ({n_samples} images in total)")
                continue

            # Count the images that still need to be generated
            need_generate = n_samples - len(existing_files)
            if rank == 0:
                logger.debug(f"Prompt {prompt_id}: {len(existing_files)} already exist, {need_generate} to generate")

            # Generate n_samples images in batches
            sample_count = 0
            all_generated_images = []  # stored as (sample_idx, image) tuples

            for batch_idx in range((n_samples + batch_size - 1) // batch_size):
                current_bsz = min(batch_size, n_samples - sample_count)
                if current_bsz <= 0:
                    break

                # Check which images in the current batch need to be generated
                batch_need_generate = []
                batch_indices = []
                for i in range(current_bsz):
                    sample_idx = sample_count + i  # zero-based index
                    if sample_idx not in existing_files:
                        batch_need_generate.append(i)
                        batch_indices.append(sample_idx)

                # If no images in the current batch need generation, skip
                if not batch_need_generate:
                    sample_count += current_bsz
                    continue

                # Only generate the needed images
                actual_batch_size = len(batch_need_generate)
                prompts = [prompt] * actual_batch_size

                # Prepare inputs (refer to lines 1198-1202 of train_stage_one.py)
                batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
                    prepare_gen_input(
                        prompts, text_tokenizer, num_t2i_image_tokens,
                        bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                        max_text_len, device
)

                # Initialize noise (refer to lines 1204-1206 of train_stage_one.py)
                # Initialize noise with a different seed per image to ensure diversity
                z_list = []
                for sample_idx in batch_indices:
                    # Create an independent generator with a different seed per image
                    generator = torch.Generator(device=device)
                    # +1 keeps this consistent with the previous logic
                    generator.manual_seed(seed + prompt_id_int * 10000 + sample_idx + 1)
                    z = torch.randn((1,
                                     image_latent_dim, latent_height * patch_size,
                                     latent_width * patch_size),
                                     generator=generator, device=device, dtype=weight_type)
                    z_list.append(z)
                z = torch.cat(z_list, dim=0)

                # CFG setup (refer to lines 1208-1243 of train_stage_one.py)
                # Note: the guidance_scale > 0 branch and the model_kwargs logic in the training code
                # z needs to be concatenated, and so do text_tokens and modality_positions
                z = torch.cat([z, z], dim=0)
                text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
                modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)

                # Use the naive attention mask (more stable)
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    max_seq_len,
                    modality_positions,
                    device
).to(weight_type)

                # Model kwargs (fully consistent with lines 1235-1243 of train_stage_one.py)
                # Note: set _return_raw_cfg=True so that model.t2i_generate returns the raw [v_cond, v_uncond]
                # so that apply_cfg_tricks in transport.py can handle CFG tricks uniformly
                model_kwargs = dict(
                    text_tokens=torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0),
                    attention_mask=block_mask,
                    modality_positions=torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0),
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    # keep the original value, but set _return_raw_cfg=True so the model returns raw outputs
                    guidance_scale=guidance_scale,
                    _return_raw_cfg=True  # flag: return raw cond and uncond, do not apply CFG inside the model
)

                # Sampling (refer to lines 1245-1254 of train_stage_one.py)
                sample_fn = sampler.sample_ode(
                    sampling_method=config.transport.sampling_method,
                    num_steps=num_inference_steps,
                    atol=config.transport.atol,
                    rtol=config.transport.rtol,
                    reverse=config.transport.reverse,
                    time_shifting_factor=config.transport.time_shifting_factor
)

                samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
                # Consistent with train_stage_one.py: always chunk, because model_kwargs always uses the concatenated
                # tokens
                samples = torch.chunk(samples, 2)[0]
                # Ensure samples have the correct dtype for VAE model
                samples = samples.to(weight_type)

                # ==================== VAE decode ====================
                # Refer to lines 1258-1269 of train_stage_one.py
                if vae_type == 'hydra_tok':
                    images = decode_hydra_tok(vae_model, samples, weight_type)
                else:
                    raise NotImplementedError(f"Unsupported VAE type: {vae_type}")

                # Convert to PIL and collect
                pil_images = [Image.fromarray(image) for image in images]

                # Store the generated images in order
                for local_idx, sample_idx in enumerate(batch_indices):
                    all_generated_images.append((sample_idx, pil_images[local_idx]))

                sample_count += current_bsz

            # ==================== Save images ====================
            # Save all generated images (same format as geneval)
            if all_generated_images:
                # Sort by sample_idx
                all_generated_images.sort(key=lambda x: x[0])

                # Collect images for the grid
                all_samples_for_grid = []

                for sample_idx, image in all_generated_images:
                    # Save to the samples directory, named {sample_idx:05}.png
                    sample_file = os.path.join(sample_path, f"{sample_idx:05}.png")
                    image.save(sample_file)

                    # Collect images for the grid
                    if not skip_grid:
                        all_samples_for_grid.append(ToTensor()(image))

                    if rank == 0:
                        logger.debug(f"Saved: {prompt_id}/samples/{sample_idx:05}.png")

                # ==================== Save grid ====================
                if not skip_grid and all_samples_for_grid:
                    grid = make_grid(all_samples_for_grid, nrow=int(np.ceil(np.sqrt(len(all_samples_for_grid)))))
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid_img = Image.fromarray(grid.astype(np.uint8))
                    grid_file = os.path.join(prompt_dir, 'grid.png')
                    grid_img.save(grid_file)

                    if rank == 0:
                        logger.debug(f"Saved: {prompt_id}/grid.png")

                if rank == 0:
                    logger.info(
                        f"Prompt {prompt_id}: generated and saved {len(all_generated_images)} "
                        f"images to {prompt_dir}")

    # Wait for all processes to finish
    dist.barrier()
    cleanup_distributed()

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Generation complete! Output directory: {output_dir}")
        logger.info(f"All {world_size} GPUs have finished inference")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
