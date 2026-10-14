# coding=utf-8
# Copyright 2025 NUS Show Lab, HuggingFace.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0
"""Unified trainer for HYDRA - all three stages of the paper's progressive
pipeline run through this single script:

=====  ===========================  ========================================
stage  paper                        what happens
=====  ===========================  ========================================
1      Stage I                      image-text alignment, flow-head training
                                    (LLM frozen, 150K steps, data 0:1:3)
2      Stage II                     joint training, all parameters
                                    (100K steps, data 0:2:2)
3      Stage III                    quality refinement, instruction tuning
                                    (20K steps, data 1:2:2)
=====  ===========================  ========================================

The stage is inferred from the config filename (configs/hydra_*_stageN.yaml)
or passed explicitly (``stage=N``). Differences between stages are carried by
the YAML configs: frozen_params, optimizer learning rates, data mixing
(samp_probs), ntp/flow coefficients.

Launched via the train.py entry point or directly:

    accelerate launch --config_file configs/zero2.yaml train_stage.py \
        config=configs/hydra_1.5b_stage1.yaml
"""

import os
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from models import HydraQwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from torch.nn.attention.flex_attention import flex_attention


os.environ["TOKENIZERS_PARALLELISM"] = "true"

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from datasets import create_imagetext_dataloader, MixedDataLoader, create_parquet_mmu_dataloader
from datasets.multi_source_utils import _ensure_list
from tools.utils import get_config, flatten_omega_conf, AverageMeter, denorm, denorm_vid, get_hyper_params, \
    get_llm_name, _freeze_params
from tools.model_utils import print_parameter_gradient_status

from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Memory optimization: set matmul precision to 'high' for speedup without accuracy loss
    torch.set_float32_matmul_precision('high')

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
)

    bs_t2i = config.training.batch_size_t2i
    bs_mmu = config.training.batch_size_mmu
    # optional: train on T2I data only (skips the MMU dataloader entirely)
    t2i_only = getattr(config.dataset, 't2i_only', False)

    if "concat" in config.dataset.mixed_loader_mode:
        assert config.dataset.accumulation == 1, "No need to enable accumulation in mixed-dataloader!"
        total_batch_size_per_gpu = bs_t2i if t2i_only else (bs_t2i + bs_mmu)
        total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
        total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps
    else:
        assert t2i_only or bs_t2i == bs_mmu, \
            "We should ensure batch size is consistent at each iteration if we use FlexAttention!"
        total_batch_size_per_gpu = bs_t2i * config.dataset.accumulation
        total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
        total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
)

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
)
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
)

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"[{current_time}] Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Setting training seed to {config.training.seed}")
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Loading models and optimizer")

    weight_type = get_weight_type(config)

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'hydra_tok':
        if config.model.vae_model.type == 'hydra_tok':
            from models.hydra_tok.modeling_hydra_tok import Gen_ViT, Und_ViT
            from models.hydra_tok.configuration_hydra_tok import HydraTokConfig

            hydra_tok_config = HydraTokConfig.from_pretrained(config.model.hydra_tok_model.hydra_tok_config)
            hydra_tok_config.gen_layers = 12
            hydra_tok_config.und_layers = 12
            hydra_tok_config.latent_stats_path = None
        # ========== 1. Initialize Gen_ViT (generation branch) ==========
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] " + "=" * 60)
        logger.info(f"[{current_time}] Initializing Gen_ViT (generation branch)...")
        vae_model = Gen_ViT(hydra_tok_config).to(accelerator.device).to(weight_type)

        # Load Gen_ViT pretrained weights
        gen_vit_weights_path = os.path.join(config.model.hydra_tok_model.pretrained_model_path, "gen_vit_weights.pth")
        if os.path.exists(gen_vit_weights_path):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{current_time}] Loading Gen_ViT weights: {gen_vit_weights_path}")
            gen_state_dict = torch.load(gen_vit_weights_path, map_location="cpu")
            vae_model.load_state_dict(gen_state_dict, strict=False)
        else:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.warning(
                f"[{current_time}] Gen_ViT weight file not found: {gen_vit_weights_path}, "
                f"using random initialization")

        # Register processor buffers and latent statistics
        if config.model.vae_model.type == 'hydra_tok':
            DATA_NORM = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            p_mean = torch.tensor(DATA_NORM["mean"]).view(1, 3, 1, 1).to(accelerator.device).to(weight_type)
            p_std = torch.tensor(DATA_NORM["std"]).view(1, 3, 1, 1).to(accelerator.device).to(weight_type)
            vae_model.register_buffer('processor_mean', p_mean)
            vae_model.register_buffer('processor_std', p_std)

        # Load Gen_ViT latent statistics (mean, std, eps)
        latent_stats_path = getattr(config.model.hydra_tok_model, 'latent_stats_path', None)
        if latent_stats_path and os.path.exists(latent_stats_path):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{current_time}] Loading Gen_ViT latent statistics: {latent_stats_path}")
            stats = torch.load(latent_stats_path, map_location="cpu")

            if 'latent_mean' in stats:
                latent_mean = stats['latent_mean']
            else:
                raise ValueError(f"Missing 'latent_mean' in {latent_stats_path}. Keys: {stats.keys()}")

            if 'latent_std' in stats:
                latent_std = stats['latent_std']
            elif 'latent_std_dev' in stats:
                latent_std = stats['latent_std_dev']
            else:
                raise ValueError(f"Missing 'latent_std' in {latent_stats_path}. Keys: {stats.keys()}")

            if 'latent_eps' in stats:
                latent_eps = stats['latent_eps']
            else:
                logger.warning(f"[{current_time}] 'latent_eps' not found, using the default value 1e-6")
                latent_eps = 1e-6

            vae_model.register_buffer('latent_mean', latent_mean.to(accelerator.device).to(weight_type))
            vae_model.register_buffer('latent_std_dev', latent_std.to(accelerator.device).to(weight_type))
            if isinstance(latent_eps, torch.Tensor):
                vae_model.register_buffer('latent_eps', latent_eps.to(accelerator.device).to(weight_type))
            else:
                vae_model.register_buffer(
                    'latent_eps', torch.tensor(latent_eps, dtype=weight_type, device=accelerator.device))

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"[{current_time}] Latent statistics loaded. Mean shape: {latent_mean.shape}, "
                f"Std shape: {latent_std.shape}")
        else:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.warning(f"[{current_time}] Latent statistics file not found, skipping loading")

        # Freeze all Gen_ViT parameters
        for param in vae_model.parameters():
            param.requires_grad = False
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Gen_ViT: all parameters frozen")

        # ========== 2. Initialize Und_ViT (understanding branch) ==========
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] " + "-" * 60)
        logger.info(f"[{current_time}] Initializing Und_ViT (understanding branch)...")
        und_vit = Und_ViT(hydra_tok_config).to(accelerator.device).to(weight_type)

        # Load Und_ViT pretrained weights
        und_vit_weights_path = os.path.join(config.model.hydra_tok_model.pretrained_model_path, "und_vit_weights.pth")
        if os.path.exists(und_vit_weights_path):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{current_time}] Loading Und_ViT weights: {und_vit_weights_path}")
            und_state_dict = torch.load(und_vit_weights_path, map_location="cpu")
            und_vit.load_state_dict(und_state_dict, strict=False)
        else:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.warning(
                f"[{current_time}] Und_ViT weight file not found: {und_vit_weights_path}, "
                f"using random initialization")

        # Und_ViT is fully trainable by default (last 12 layers + chal_unproj_understanding)
        for param in und_vit.parameters():
            param.requires_grad = True
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Und_ViT: all parameters trainable")

        # ========== 3. Print parameter statistics ==========
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] " + "=" * 60)
        logger.info(f"[{current_time}] HYDRA-TOK model parameter training status:")
        logger.info(f"[{current_time}] " + "=" * 60)

        # Gen_ViT statistics
        gen_total = sum(p.numel() for p in vae_model.parameters())
        gen_trainable = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]   Gen_ViT: {gen_trainable:,} trainable / {gen_total:,} total parameters")

        # Und_ViT statistics
        und_total = sum(p.numel() for p in und_vit.parameters())
        und_trainable = sum(p.numel() for p in und_vit.parameters() if p.requires_grad)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]   Und_ViT: {und_trainable:,} trainable / {und_total:,} total parameters")

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] " + "-" * 60)
        total_params = gen_total + und_total
        trainable_total = gen_trainable + und_trainable
        logger.info(f"[{current_time}]   Total: {trainable_total:,} trainable / {total_params:,} total parameters "
                    f"({100 * trainable_total / total_params:.2f}%)")
        logger.info(f"[{current_time}] " + "=" * 60)
    else:
        raise NotImplementedError

    # Compile VAE model for faster inference (before setting to main model)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Compiling VAE model with torch.compile for faster inference...")
        vae_model = torch.compile(vae_model, mode='reduce-overhead')
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]  VAE model compiled successfully")

    # Initialize the HYDRA model
    text_tokenizer, hydra_token_ids = get_text_tokenizer(config.model.llm.llm_model_path, add_hydra_tokens=True,
                                                         return_hydra_token_ids=True,
                                                         llm_name=get_llm_name(config.model.llm.llm_model_path))
    config.model.llm.llm_vocab_size = len(text_tokenizer)

    if config.model.llm.load_from_llm:
        model = HydraQwen2_5.from_pretrained(config.model.llm.pretrained_model_path, use_safetensors=False).to(
            accelerator.device)
    else:
        model = HydraQwen2_5(**config.model.llm).to(accelerator.device)



    # For HYDRA-TOK/Siglip3D, set Und_ViT on the HydraQwen2_5 model
    if config.model.vae_model.type == 'hydra_tok' and und_vit is not None:
        model.set_und_vit(und_vit)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Und_ViT set on the HydraQwen2_5 model")

    # Choose layers to freeze
    _freeze_params(model, config.model.llm.frozen_params)

    # Print parameter gradient status
    print_parameter_gradient_status(model, logger_instance=logger)

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # for time embedding
    if config.model.llm.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        # config.dataset.preprocessing.num_hq_image_tokens += 1
        # config.dataset.preprocessing.num_video_tokens += 1
        # config.dataset.preprocessing.num_mixed_modal_tokens += 1

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    optimizer_type = config.optimizer.name

    # Create parameter groups with grouped learning rates
    # Get the default learning rate (if group learning rates are undefined)
    default_lr = getattr(optimizer_config, 'learning_rate', 1e-4)

    optimizer_grouped_parameters = [
        {
            # Parameter group 1: Und_ViT (understanding module)
            "params": [p for n, p in model.named_parameters() if
                       (('und_vit' in n)
                        and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": getattr(optimizer_config, 'learning_rate_ve', default_lr),
        },
        {
            # Parameter group 2: semantic_proj (semantic projection layer)
            "params": [p for n, p in model.named_parameters() if
                       (('semantic_proj' in n) and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": getattr(optimizer_config, 'learning_rate_proj', default_lr)
        },
        {
            # Parameter group 3: diffusion_head, diff_proj, time_embed (diffusion-related modules, unified learning
            # rate)
            "params": [p for n, p in model.named_parameters() if
                       (('diffusion_head' in n or 'diff_proj' in n or 'time_embed' in n)
                        and p.requires_grad
                        and 'und_vit' not in n
                        and 'semantic_proj' not in n)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": getattr(optimizer_config, 'learning_rate_dit', default_lr)
        },
        {
            # Parameter group 4: the HYDRA LLM backbone
            "params": [p for n, p in model.named_parameters() if
                       (('llm' in n)
                        and p.requires_grad
                        and 'und_vit' not in n
                        and 'semantic_proj' not in n)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": getattr(optimizer_config, 'learning_rate_llm', default_lr)
        },
    ]

    # Drop empty groups (e.g. the LLM group when Stage I freezes the LLM)
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Print optimizer configuration
    if accelerator.is_main_process:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] ==================== Optimizer configuration ====================")
        logger.info(f"[{current_time}] Optimizer type: {optimizer_type}")
        logger.info(f"[{current_time}] Beta1: {optimizer_config.beta1}, Beta2: {optimizer_config.beta2}")
        logger.info(f"[{current_time}] Weight Decay: {optimizer_config.weight_decay}")
        logger.info(f"[{current_time}] Epsilon: {optimizer_config.epsilon}")
        logger.info(f"[{current_time}] ")

        # Print the learning rate of each parameter group

    ##################################
    #         DATALOADER             #
    #################################
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Creating dataloaders and lr_scheduler")

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is
    # stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.


    max_aspect_ratio = getattr(preproc_config, 'max_aspect_ratio', 2.0)

    # Get the max training examples limit
    max_train_examples_t2i = getattr(config.experiment, 'max_train_examples_t2i', None)
    max_train_examples_mmu = getattr(config.experiment, 'max_train_examples_mmu', None)


    # Use the existing imagetext dataloader - T2I
    # Support multiple data sources: convert shards and data_root to lists
    t2i_shards_list = _ensure_list(dataset_config.train_t2i_shards_path_or_url)
    t2i_roots_list = _ensure_list(dataset_config.train_t2i_data_root)

    if len(t2i_shards_list) != len(t2i_roots_list):
        raise ValueError(
            f"T2I data source count mismatch: {len(t2i_shards_list)} shards but {len(t2i_roots_list)} data_roots"
)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Loading {len(t2i_shards_list)} T2I data source(s):")
    for i, (shard, root) in enumerate(zip(t2i_shards_list, t2i_roots_list)):
        logger.info(f"[{current_time}]   [{i+1}] {shard} -> {root}")

    # Create multiple dataloaders and merge the datasets
    from torch.utils.data import ConcatDataset
    from datasets.image_text_dataset import ImageTextDataset

    t2i_datasets = []
    for shard, root in zip(t2i_shards_list, t2i_roots_list):
        ds = ImageTextDataset(
            anno_path=shard,
            data_root=root,
            text_tokenizer=text_tokenizer,
            max_seq_len=preproc_config.max_seq_length,
            image_size=preproc_config.resolution,
            latent_height=preproc_config.latent_height,
            latent_width=preproc_config.latent_width,
            cond_dropout_prob=config.training.cond_dropout_prob,
            num_image_tokens=preproc_config.num_t2i_image_tokens,
            is_captioning=False,
            min_res=preproc_config.min_res,
            random_und_or_gen=preproc_config.random_und_or_gen,
            hydra_token_ids=hydra_token_ids,
            system=("", "", ""),
            use_hydra_tok=(config.model.vae_model.type == 'hydra_tok'),
)
        t2i_datasets.append(ds)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]   Dataset loaded: {len(ds)} samples")

    # Merge the multiple datasets
    if len(t2i_datasets) == 1:
        combined_t2i_dataset = t2i_datasets[0]
    else:
        combined_t2i_dataset = ConcatDataset(t2i_datasets)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]  Combined T2I dataset: {len(combined_t2i_dataset)} total samples")

    # Create DataLoader
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            combined_t2i_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            drop_last=True,
            seed=48
)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Use the collate_fn of the first dataset
    collate_fn = t2i_datasets[0].collate_fn

    train_dataloader_t2i = DataLoader(
        combined_t2i_dataset,
        batch_size=config.training.batch_size_t2i,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=dataset_config.num_workers,
        drop_last=True
)

   # Data for understanding
    use_parquet_mmu = getattr(config.dataset, 'use_parquet_mmu', False) and not t2i_only

    # Parquet MMU dataset configuration
    parquet_mmu_index_json = getattr(config.dataset, 'parquet_mmu_index_json', None)
    parquet_mmu_kwargs = dict(getattr(config.dataset, 'parquet_mmu_kwargs', {}))

    if use_parquet_mmu and parquet_mmu_index_json:
        # Use Parquet MMU dataset (LLaVA-One-Vision style)
        # Advantage: fast loading, suitable for large-scale training
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Using Parquet MMU dataset")
        logger.info(f"[{current_time}] Index JSON: {parquet_mmu_index_json}")
        logger.info(f"[{current_time}] Initializing Parquet MMU DataLoader...")

        # Get Parquet-specific parameters from the config
        parquet_image_col = parquet_mmu_kwargs.get('image_col', 'image')
        parquet_caption_col = parquet_mmu_kwargs.get('caption_col', 'caption')
        parquet_id_col = parquet_mmu_kwargs.get('id_col', 'id')
        parquet_max_files = parquet_mmu_kwargs.get('max_files', None)
        # Prefetch factor to speed up data loading
        parquet_prefetch_factor = parquet_mmu_kwargs.get('prefetch_factor', 4)

        train_dataloader_mmu = create_parquet_mmu_dataloader(
            index_json_path=parquet_mmu_index_json,
            batch_size=config.training.batch_size_mmu,
            text_tokenizer=text_tokenizer,
            accelerator=accelerator,
            image_size=preproc_config.resolution,
            latent_width=preproc_config.latent_width,
            latent_height=preproc_config.latent_height,
            cond_dropout_prob=0.0,  # No dropout in MMU mode
            max_seq_len=preproc_config.max_seq_length,
            num_image_tokens=preproc_config.num_mmu_image_tokens,
            num_workers=dataset_config.num_workers,
            is_captioning=True,  # MMU mode
            min_res=preproc_config.min_res,
            shuffle=True,
            random_und_or_gen=0.0,
            drop_last=True,
            hydra_token_ids=hydra_token_ids,
            system=("", "", ""),
            use_hydra_tok=(config.model.vae_model.type == 'hydra_tok'),
            seed=43,
            image_col=parquet_image_col,
            caption_col=parquet_caption_col,
            id_col=parquet_id_col,
            max_files=parquet_max_files,
)

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"[{current_time}] Parquet MMU DataLoader created successfully! "
            f"Dataset size: {len(train_dataloader_mmu.dataset)}")

    elif not t2i_only:
        # MMU data (LLaVA-style conversations; question-aware chat format).
        # The config fields are lists for consistency with the t2i side -
        # MMUDataset takes a single annotation (json / yaml), so take the first.
        _mmu_annos = _ensure_list(dataset_config.train_mmu_shards_path_or_url)
        mmu_anno_path = _mmu_annos[0]
        _mmu_roots = _ensure_list(dataset_config.train_mmu_data_root)
        mmu_data_root = _mmu_roots[0]

        # max_text_len = max_seq_len - num_image_tokens - 4 (bos, eos, boi, eoi)
        max_text_len = preproc_config.max_seq_length - preproc_config.num_mmu_image_tokens - 4
        # source and target each take half, leaving some margin for role tokens
        source_max_len = max_text_len // 2 - 50
        target_max_len = max_text_len // 2 - 50

        from datasets.mmu_dataset import MMUDataset

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Loading MMU dataset: {mmu_anno_path} -> {mmu_data_root}")

        combined_mmu_dataset = MMUDataset(
            root=mmu_data_root,
            text_tokenizer=text_tokenizer,
            max_seq_len=preproc_config.max_seq_length,
            source_max_len=source_max_len,
            target_max_len=target_max_len,
            image_size=preproc_config.resolution,
            num_image_tokens=preproc_config.num_mmu_image_tokens,
            latent_width=preproc_config.latent_width,
            latent_height=preproc_config.latent_height,
            cond_dropout_prob=config.training.cond_dropout_prob,
            annotation_path=mmu_anno_path,
            stage=getattr(config.training, 'stage', 'tuning'),
            hydra_token_ids=hydra_token_ids,
            default_system_prompt=getattr(
                dataset_config, 'default_system_prompt', "system\nYou are a helpful assistant.<|im_end|>"),
)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]   Dataset loaded: {len(combined_mmu_dataset)} samples")

        # Create DataLoader
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        if accelerator is not None and accelerator.num_processes > 1:
            sampler = DistributedSampler(
                combined_mmu_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                drop_last=True,
                seed=41
)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        # Use the collate_fn of the dataset
        collate_fn = combined_mmu_dataset.collate_fn

        train_dataloader_mmu = DataLoader(
            combined_mmu_dataset,
            batch_size=config.training.batch_size_mmu,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=dataset_config.num_workers,
            drop_last=True
)

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] MMU DataLoader created successfully! Dataset size: {len(combined_mmu_dataset)}")

    num_update_steps_per_epoch = len(train_dataloader_t2i)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    if config.experiment.resume_from_checkpoint:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Resuming from checkpoint...")
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        # Filter out "checkpoint-final" type checkpoints
        dirs = [d for d in dirs if d.split("-")[1].isdigit()]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            unwrapped_model_dir = f'{path}/unwrapped_model'

            # Check if sharded weights exist
            sharded_files = []
            if os.path.exists(unwrapped_model_dir):
                files = os.listdir(unwrapped_model_dir)
                # Look for sharded weight files like pytorch_model-00001-of-00002.bin
                sharded_files = sorted(
                    [f for f in files if f.startswith('pytorch_model-') and f.endswith('.bin') and 'of' in f])

            if len(sharded_files) > 0:
                # Load sharded weights
                accelerator.print(
                    f"[{current_time}] Resuming from checkpoint {unwrapped_model_dir} "
                    f"(sharded weights: {len(sharded_files)} files)")
                state_dict = {}
                for shard_file in sharded_files:
                    shard_path = os.path.join(unwrapped_model_dir, shard_file)
                    accelerator.print(f"[{current_time}] Loading shard: {shard_file}")
                    shard_dict = torch.load(shard_path, map_location="cpu")
                    state_dict.update(shard_dict)
                    del shard_dict
            else:
                # Load single weight file
                single_weight_path = f'{unwrapped_model_dir}/pytorch_model.bin'
                accelerator.print(f"[{current_time}] Resuming from checkpoint {single_weight_path}")
                state_dict = torch.load(single_weight_path, map_location="cpu")

            # not load some parameters
            if config.model.llm.params_not_load is not None:
                params_to_delete = []
                for k in state_dict:
                    for n in config.model.llm.params_not_load:
                        if n in k:
                            params_to_delete.append(k)
                for k in params_to_delete:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False if config.model.llm.params_not_load is not None else True)
            del state_dict

            if accelerator.is_main_process:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"[{current_time}] Load checkpoint successfully")

        # we recommend save and load the dataloader state
        # these save and recover functions are based on our internal packages
        # please modified them when necessary
        # recover_dataloader_state(accelerator.process_index, train_dataloader_t2i,
        # config.experiment.output_dataloader_state_dir)

    # Combine these dataloaders into a single iterable model
    if t2i_only:
        mixed_loader = train_dataloader_t2i
    else:
        mixed_loader = MixedDataLoader(
            loader_list=[train_dataloader_t2i, train_dataloader_mmu],
            samp_probs=config.dataset.samp_probs,
            accumulation=config.dataset.accumulation,
            mode=config.dataset.mixed_loader_mode
)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps - global_step,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
)

    ##################################
    #       Prepare accelerator     #
    #################################
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Preparing model, optimizer and dataloaders")

    # Key fix: no longer save the model reference in advance
    # Reason: accelerator.prepare() and torch.compile() create new wrapper objects
    # A reference saved in advance points to the stale model, which causes incorrect EMA updates
    # The correct approach is to dynamically fetch the model currently being trained at each EMA update

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # Key optimization: use torch.compile to accelerate the main model (Qwen2.5 LLM)
    # The first run is very slow (about 5-10 minutes to compile), but after compilation the speed usually improves by
    # 20%-40%
    use_torch_compile = getattr(config.training, 'use_torch_compile', False)
    if use_torch_compile and torch.cuda.is_available() and hasattr(torch, 'compile'):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] ⏳ First compilation may take 5-10 minutes, please wait...")
        logger.info(f"[{current_time}]  Tip: If stuck, check CPU usage with 'top' - high CPU means compiling")
        model = torch.compile(model)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]  Main model compiled successfully")

    ##################################
    #             Training          #
    #################################
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] ***** Running training *****")
    logger.info(f"[{current_time}]   Num training steps = {config.training.max_train_steps}")
    logger.info(f"[{current_time}]   Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(
        f"[{current_time}]   Total train batch size (w. parallel, distributed & accumulation) "
        f"= {total_batch_size}")
    logger.info(f"[{current_time}]   Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=preproc_config.num_t2i_image_tokens*config.training.noise_shift,
)  # default: velocity;

    sampler = Sampler(transport)

    @torch.no_grad()
    def prepare_latents_and_labels(
            pixel_values: Union[torch.FloatTensor, torch.LongTensor],
            data_type,
            shape,
            image_masks,
            modality_positions
):


        if config.model.vae_model.type == 'hydra_tok':
            image_latents = vae_model.encode(pixel_values)
            recons_images = image_latents
        else:
            raise NotImplementedError

        c, h, w = image_latents.shape[1:]
        # timesteps, noise, original image
        # each for loop takes around 0.002, which is affordable
        t_list, xt_list, ut_list, masks = [], [], [], []
        for i, tp in enumerate(data_type):
            # x0->noise x1->image
            if tp in ['mmu', 'mmu_vid']:
                x1_input = image_latents[i][None]
            elif hasattr(vae_model, 'normalize_latents'):
                x1_input = vae_model.normalize_latents(image_latents[i][None])
            else:
                x1_input = (image_latents[i][None] - vae_model.latent_mean) / \
                    (vae_model.latent_std_dev + vae_model.latent_eps)
            t, x0, x1 = transport.sample(x1_input,
                                         config.training.und_max_t0 if tp in ['mmu', 'mmu_vid'] else None)
            # timesteps, noised image, velocity
            t, xt, ut = transport.path_sampler.plan(t, x0, x1)

            t_list.append(t)
            xt_list.append(xt)
            ut_list.append(ut)
            if data_type[0] != 'interleaved_data':
                if tp in ['mmu', 'mmu_vid'] and config.training.und_max_t0 == 1.0:
                    masks.append(image_masks[i][None] * 0.0)
                else:
                    masks.append(image_masks[i][None])

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)

        if len(masks) != 0:
            masks = torch.cat(masks, dim=0)
        else:
            masks = image_masks

        return xt, t, ut, recons_images, masks

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Data loading timeout threshold (seconds)
    DATA_LOAD_TIMEOUT_WARNING = 60  # Print a warning if it exceeds 60 seconds

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Starting epoch {epoch}/{num_train_epochs}")

        data_load_start = time.time()
        for batch_idx, batch in enumerate(mixed_loader):
            data_load_time = time.time() - data_load_start

            # Detect abnormally slow data loading
            if data_load_time > DATA_LOAD_TIMEOUT_WARNING:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.warning(
                    f"[{current_time}]  Rank {accelerator.process_index}: Data loading took "
                    f"{data_load_time:.1f}s (batch {batch_idx})")

            if batch_idx == 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"[{current_time}] First batch loaded successfully! Starting training...")
                logger.info(f"[{current_time}] ⏳ First forward pass may take 1-3 minutes due to torch.compile...")

            # Print detailed progress for the first 10 steps (for debugging)
            if batch_idx < 10 and accelerator.is_main_process:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"[{current_time}]  Processing batch {batch_idx + 1}...")

            text_tokens = batch['text_tokens'].to(accelerator.device)
            text_labels = batch['text_labels'].to(accelerator.device)
            pixel_values = batch['images'].to(accelerator.device).to(weight_type)
            if batch['data_type'][0] == 'interleaved_data':
                b, n = pixel_values.shape[:2]
                pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
                batch['data_type'] = batch['data_type'] * n
            else:
                b, n = 0, 0

            text_masks = batch['text_masks'].to(accelerator.device)
            image_masks = batch['image_masks'].to(accelerator.device)
            modality_positions = batch['modality_positions'].to(accelerator.device)
            # prepare image latents and labels
            image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(
                pixel_values,
                batch['data_type'],
                (b, n),
                image_masks,
                modality_positions
)


            # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
            # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
            # omni_mask_fn = omni_attn_mask(modality_positions)
            # block_mask = create_block_mask(omni_mask_fn, B=text_tokens.shape[0], H=None,
            #                                Q_LEN=preproc_config.max_seq_length,
            #                                KV_LEN=preproc_config.max_seq_length, device=accelerator.device)
            # or use naive omni attention mask, which is more stable
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                              text_tokens.size(1),
                                              modality_positions,
                                              accelerator.device).to(weight_type)

            logits, loss_ntp, loss_flow = model(text_tokens=text_tokens,
                                                image_latents=image_latents,
                                                t=t.to(weight_type),
                                                attention_mask=block_mask,
                                                text_masks=text_masks,
                                                image_masks=image_masks,
                                                text_labels=text_labels,
                                                image_labels=image_labels,
                                                modality_positions=modality_positions,
                                                output_hidden_states=True,
                                                max_seq_len=text_tokens.size(1),
                                                device=accelerator.device,
)

            # Print forward completion for the first 10 steps (to help debug torch.compile compilation time)
            if batch_idx < 10 and accelerator.is_main_process:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"[{current_time}]  Batch {batch_idx + 1} forward done, Loss_NTP: {loss_ntp.item():.4f}, "
                    f"Loss_FLOW: {loss_flow.item():.4f}")

            # Gather the losses across all processes for logging (if we use distributed training).
            # T2I text_labels are all -100 -> cross_entropy returns nan; replace with 0
            # to prevent nan from poisoning the total loss and gradients.
            if torch.isnan(loss_ntp):
                loss_ntp = torch.zeros_like(loss_ntp)
            avg_loss_ntp = accelerator.gather(loss_ntp.repeat(total_batch_size_per_gpu)).mean()
            avg_loss_flow = accelerator.gather(loss_flow.repeat(total_batch_size_per_gpu)).mean()
            loss = config.training.ntp_coeff * loss_ntp + config.training.flow_coeff * loss_flow

            accelerator.backward(loss.to(weight_type) / config.training.gradient_accumulation_steps)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

            # log gradient norm before zeroing it
            if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
):
                log_grad_norm(model, accelerator, global_step + 1)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
)
                    lr = [group["lr"] for group in optimizer.param_groups]
                    if len(lr) == 4:
                        # The optimizer has 4 parameter groups:
                        # lr[0] = Und_ViT (learning_rate_ve)
                        # lr[1] = semantic_proj (learning_rate_proj)
                        # lr[2] = diffusion_head/diff_proj/time_embed (learning_rate_dit)
                        # lr[3] = LLM (learning_rate_llm)
                        logs = {
                            "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr_ve": lr[0],
                            "lr_proj": lr[1],
                            "lr_dit": lr[2],
                            "lr_llm": lr[3],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                        }
                        accelerator.log(logs, step=global_step + 1)
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(
                            f"[{current_time}] "
                            f"Epoch: {epoch} "
                            f"Step: {global_step + 1} "
                            f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                            f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR_ve: {lr[0]:0.6f} "
                            f"LR_proj: {lr[1]:0.6f} "
                            f"LR_dit: {lr[2]:0.6f} "
                            f"LR_llm: {lr[3]:0.6f}"
)
                    else:
                        logs = {
                            "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr": lr[0],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                        }
                        accelerator.log(logs, step=global_step + 1)
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(
                            f"[{current_time}] "
                            f"Epoch: {epoch} "
                            f"Step: {global_step + 1} "
                            f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                            f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR: {lr[0]:0.6f}"
)
                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"[{current_time}] Saving checkpoint at step {global_step + 1}")
                    save_checkpoint(model, config, accelerator, global_step + 1)
                    # save_dataloader_state(accelerator.process_index, train_dataloader_t2i,
                    # config.experiment.output_dataloader_state_dir)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:

                    generate_images(
                        model,
                        vae_model,
                        text_tokenizer,
                        config,
                        global_step + 1,
                        accelerator.device,
                        weight_type,
                        sampler,
                        hydra_token_ids
)

                    if batch['data_type'][0] != "interleaved_data":
                            with torch.no_grad():
                                recons_images_vis = vae_model.decode(recons_images)
                                if recons_images_vis.dim() == 5:
                                    recons_images_vis = recons_images_vis.squeeze(2)
                                if hasattr(vae_model, 'denormalize_image'):
                                    recons_images_vis = vae_model.denormalize_image(recons_images_vis)
                            visualize_reconstruction(
                                pixel_values,
                                recons_images_vis,
                                batch['texts'],
                                global_step + 1,
                                vae_type=config.model.vae_model.type,
)

                global_step += 1

            # Record the start time of the next data loading
            data_load_start = time.time()

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Saving final checkpoint")
    save_checkpoint(model, config, accelerator, "final")
    # we recommend save and load the dataloader state
    # these save and recover functions are based on our internal packages
    # please modified them when necessary
    # save_dataloader_state(accelerator.process_index, train_dataloader_t2i,
    # config.experiment.output_dataloader_state_dir)
    # logger.info(f"Saved dataloader state to {config.experiment.output_dataloader_state_dir}")

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Saving final model")
        unwrapped_model = accelerator.unwrap_model(model)

        # Save the final model
        unwrapped_model.save_pretrained(config.experiment.output_dir, safe_serialization=False)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}]  Final model saved to {config.experiment.output_dir}")

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Final model saved successfully")

    accelerator.end_training()


@torch.no_grad()
def generate_images(
        model,
        vae_model,
        text_tokenizer,
        config,
        global_step,
        device,
        weight_type,
        sampler,
        hydra_token_ids,
):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        prompts = f.read().splitlines()[:config.training.batch_size_t2i]

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, \
    image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, image_pad_id, video_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, hydra_token_ids, is_hq=False)  # hq means high resolution

    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
        prepare_gen_input(
            prompts, text_tokenizer, num_t2i_image_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, image_pad_id,
            max_text_len, device
)

    z = torch.randn((len(prompts),
                     image_latent_dim, latent_height * patch_size,
                     latent_width * patch_size)).to(weight_type).to(device)

    if guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        # omni_mask_fn = omni_attn_mask(modality_positions)
        # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #                                KV_LEN=max_seq_len, device=device)
        # or use naive omni attention mask, which is more stable
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          max_seq_len,
                                          modality_positions,
                                          device).to(weight_type)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions
        # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        # omni_mask_fn = omni_attn_mask(modality_positions)
        # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #                                KV_LEN=max_seq_len, device=device)
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          max_seq_len,
                                          modality_positions,
                                          device).to(weight_type)

    model_kwargs = dict(
        text_tokens=torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0),
        attention_mask=block_mask,
        modality_positions=torch.cat([batch_modality_positions,
                                      batch_modality_positions_null], dim=0),
        output_hidden_states=True,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale
)

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
)
    samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
    samples = torch.chunk(samples, 2)[0]
    # Ensure samples have the correct dtype for VAE model
    samples = samples.to(weight_type)

    if config.model.vae_model.type == 'hydra_tok':
        if hasattr(vae_model, 'denormalize_latents'):
            samples = vae_model.denormalize_latents(samples)
        images = vae_model.decode(samples)
        images = images.float().cpu()
        if images.dim() == 5:
            images = images.squeeze(2)
        if hasattr(vae_model, 'denormalize_image'):
            images = vae_model.denormalize_image(images)
        else:
            images = torch.clamp(images, 0, 1)
        images = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    else:
        raise NotImplementedError

    model.train()
    pil_images = [Image.fromarray(image) for image in images]

    # 3. Key fix: build the image list to ensure it is not overwritten
    logger.info(f"Successfully generated {len(pil_images)} images. Uploading to WandB...")

    wandb_list = []
    for i, image in enumerate(pil_images):
        # Use min to prevent out-of-bounds access when the prompt and image counts differ
        current_caption = prompts[i] if i < len(prompts) else f"Gen_{i}"
        wandb_list.append(wandb.Image(image, caption=current_caption))

    # 4. Upload the whole list at once
    if len(wandb_list) > 0:
        wandb.log({"Generated Validation Images": wandb_list}, step=global_step)
    # Convert to PIL images
    # for image in images:
    #     print(image.shape)
    # pil_images = [Image.fromarray((image)) for image in images]
    # print(f"len(pil_images): {len(pil_images)}")

    # # Log images
    # for i, image in enumerate(pil_images):
    #     caption = prompts[i % len(prompts)] if i < len(prompts) else f"Image {i}"

    #     try:
    #         wandb_image = wandb.Image(image, caption=caption)
    #         wandb.log({"Generated images": wandb_image}, step=global_step)
    #     except Exception as e:
    #         logger.warning(f"Failed to log image {i}: {e}")


@torch.no_grad()
def visualize_reconstruction(
        pixel_values,
        recons_images,
        captions,
        global_step,
        vae_type='hydra_tok',
):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{current_time}] Visualizing images...")

    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.float().cpu().numpy()
    if isinstance(recons_images, torch.Tensor):
        recons_images = recons_images.float().cpu().numpy()

    HYDRA_TOK_NORM_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    HYDRA_TOK_NORM_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    if pixel_values.ndim == 4 and pixel_values.shape[1] == 3:
        pixel_values = pixel_values * HYDRA_TOK_NORM_STD + HYDRA_TOK_NORM_MEAN
    pixel_values = np.clip(pixel_values, 0, 1)
    recons_images = np.clip(recons_images, 0, 1)

    if pixel_values.ndim == 4 and pixel_values.shape[1] == 3:
        pixel_values = np.transpose(pixel_values, (0, 2, 3, 1))
    if recons_images.ndim == 4 and recons_images.shape[1] == 3:
        recons_images = np.transpose(recons_images, (0, 2, 3, 1))

    pixel_values = (pixel_values * 255).astype(np.uint8)
    recons_images = (recons_images * 255).astype(np.uint8)

    visualized_images = np.concatenate((pixel_values, recons_images), axis=2)
    pil_images = [Image.fromarray(image) for image in visualized_images]

    wandb_images = [wandb.Image(image, caption=captions[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Original images vs. Reconstructed": wandb_images}, step=global_step)




def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        # Filter out "checkpoint-final" type checkpoints
        checkpoints = [d for d in checkpoints if d.split("-")[1].isdigit()]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"[{current_time}] {len(checkpoints)} checkpoints already exist, "
                f"removing {len(removing_checkpoints)} checkpoints"
)
            logger.info(f"[{current_time}] removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using
    # stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
)

        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            grad_norms[name] = grad_norm
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

    if grad_norms:
        mean_grad_norm = np.mean(list(grad_norms.values()))
        max_grad_norm = np.max(list(grad_norms.values()))
        accelerator.log({"grad_norm/mean": mean_grad_norm, "grad_norm/max": max_grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
