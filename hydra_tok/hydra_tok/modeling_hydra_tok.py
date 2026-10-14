# --------------------------------------------------------
# Copyright (c) 2025, OpenGVLab. All rights reserved. 
# Licensed under The MIT License [see LICENSE for details]
# HYDRA-TOK-(InternViT)
# --------------------------------------------------------

import ast
import math
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Optional, Tuple, Union
import torch
from math import sqrt  # <-- added here
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Block
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_hydra_tok import HydraTokConfig
logger = logging.get_logger(__name__)

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except ImportError:
    print('FlashAttention is not installed.')
    has_flash_attn = False


class HydraTokRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    HydraTokRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of HydraTokRMSNorm')
except ImportError:
    # using the normal HydraTokRMSNorm
    pass
except (RuntimeError, OSError, ValueError):
    logger.warning('discovered apex but it failed to load, falling back to HydraTokRMSNorm')
    pass


import warnings
warnings.filterwarnings("ignore")

# === LPIPS for Perceptual Loss ===
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("LPIPS not installed. Install via: pip install lpips")
    LPIPS_AVAILABLE = False

# === GAN Discriminator for Adversarial Loss ===
try:
    from disc import DinoDiscriminator, DiffAug, vanilla_g_loss
    GAN_AVAILABLE = True
except ImportError:
    print("GAN discriminator not available. Check disc module.")
    GAN_AVAILABLE = False


#------------------------------------------------------------
#                 HYDRA-TOK Modules
#------------------------------------------------------------


def p2l_transform_tensor(x, patch_size):
    """
    Transform from pixel space to latent space
    [B, C, H, W] -> [B, * H//patch_size * W//patch_size, C*patch_size*patch_size]
    """
    B, C, H, W = x.shape
    return rearrange(
        x,
        "b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)",
        h1=H // patch_size,
        h2=patch_size,
        w1=W // patch_size,
        w2=patch_size,
    )


def l2p_transform_tensor(x, patch_size, img_size):
    """
    Transform from latent space to pixel space
    [B, H//patch_size * W//patch_size, C*tubelet_size*patch_size*patch_size] -> [B, C, H, W]
    """
    B = x.shape[0]
    C = x.shape[2] // (patch_size * patch_size)
    return rearrange(
        x,
        "b (h1 w1) (c h2 w2) -> b c (h1 h2) (w1 w2)",
        h1=img_size // patch_size,
        h2=patch_size,
        w1=img_size // patch_size,
        w2=patch_size,
        c=C,
    )


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


NORM2FN = {
    'rms_norm': HydraTokRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class HydraTokVisionEmbeddings(nn.Module):
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        # shape: [batch*temporal, channel, width, height] / [b*t, c*patch*patch, w//patch, h//patch]
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [batch, seq_le=1024, dim]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class HydraTokAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = HydraTokRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = HydraTokRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(
        self, x,
        attn_mask=None,
    ):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn_bias = torch.zeros(N, N, dtype=q.dtype, device=q.device)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn += attn_bias  # masking
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(
        self, x, key_padding_mask=None, need_weights=False,
        attn_mask=None,
    ):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False,
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(
        self, hidden_states: torch.Tensor,
        attn_mask=None,
    ) -> torch.Tensor:
        x = self._naive_attn(hidden_states, attn_mask=attn_mask) if not self.use_flash_attn \
            else self._flash_attn(hidden_states, attn_mask=attn_mask)
        return x


class HydraTokMLP(nn.Module):
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class HydraTokVisionEncoderLayer(nn.Module):
    def __init__(self, config: HydraTokConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = HydraTokAttention(config)
        self.mlp = HydraTokMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_mask=None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the
            layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states), attn_mask=attn_mask) * self.ls1)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states

    
class HydraTokVisionEncoder(nn.Module):
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.latent_ch = config.latent_ch        # 64
        self.vit_hidden_size = config.vit_hidden_size  # 1024

        # --- 1. Define the Bottleneck: VAE Encoder (Proj) ---
        # VAE structure: 1024 -> 1024 -> GELU -> 128 (outputs μ and logvar, 64 dims each)
        self.chal_proj = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.vit_hidden_size, self.vit_hidden_size)),  # 1024 -> 1024
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(self.vit_hidden_size, self.latent_ch * 2)),  # 1024 -> 128 (μ: 64, logvar: 64)
        ]))

        # --- 2. Define the Bottleneck: Decoder (Unproj) ---
        # Structure: 64 -> 1024 (Xavier) -> GELU -> 1024 -> 1024 (Identity)
        self.chal_unproj_understanding = nn.Sequential(OrderedDict([
            ("c_proj", nn.Linear(self.latent_ch, self.vit_hidden_size)),     # 64 -> 1024
            ("gelu", nn.GELU()),
            ("c_fc", nn.Linear(self.vit_hidden_size, self.vit_hidden_size)),  # 1024 -> 1024
        ]))
       
        # --- Initialization ---
        self._init_bottleneck_weights(self.chal_proj)
        self._init_bottleneck_weights(self.chal_unproj_understanding)
    def _init_bottleneck_weights(self, module):
        """
        Bottleneck initialization optimized for understanding tasks
        Goal: make the features restored by the Bottleneck as close as possible to the original features
        
        Strategy:
        - c_fc (1024→1024): Identity + small perturbation, keeps information flowing but breaks symmetry
        - c_proj (1024↔64): Xavier + adaptive gain, no LAPACK needed, close to Orthogonal
        """
        for name, m in module.named_modules():
            if isinstance(m, nn.Linear):
                if "c_fc" in name:
                    # 1024 -> 1024: Identity + small perturbation
                    nn.init.eye_(m.weight)
                    m.weight.data += 0.01 * torch.randn_like(m.weight.data)
                    if m.bias is not None: 
                        nn.init.constant_(m.bias, 0)
                    logger.info(f"Initialized {name} to Identity + noise.")
                
                elif "c_proj" in name:
                    # Dimension reduce/increase: Xavier init + adaptive gain
                    # The gain compensates for the dimension change, mimicking the scaling of Orthogonal
                    in_dim, out_dim = m.in_features, m.out_features
                    gain = math.sqrt(2.0 * min(in_dim, out_dim) / max(in_dim, out_dim))
                    nn.init.xavier_uniform_(m.weight, gain=gain)  # no LAPACK needed
                    
                    if m.bias is not None: 
                        nn.init.constant_(m.bias, 0)
                    logger.info(f"Initialized {name} to Xavier (gain={gain:.4f}).")

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            attn_mask=None,
            deterministic: bool = False,  # set to True at inference to use μ directly
    ) -> Union[Tuple, BaseModelOutput]:
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        # hidden_states = inputs_embeds
        hidden_states = inputs_embeds[:, 1:, :]
        
        saved_latents = None 

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer, hidden_states, attn_mask,
                )
            else:
                layer_outputs = encoder_layer(hidden_states, attn_mask)
            
            hidden_states = layer_outputs

            # 3. Core logic: insert the Bottleneck after layer 12 (idx=11)
            
            if idx == 11:
                # 1. Split CLS and Patches
                # hidden_states shape: [B, N+1, C]
                input_tokens = hidden_states

                # 2. A. VAE Proj: 1024 -> 1024 -> 128 (μ and logvar)
                h = self.chal_proj(input_tokens)         # [B, N, 128]
                mu, logvar = h.chunk(2, dim=-1)          # each [B, N, 64]
                
                # 2. B. Compute the KL divergence: KL(N(μ, σ²) || N(0, 1))
                # Formula: KL = -0.5 * mean(1 + logvar - μ² - exp(logvar))
                # Averaged directly over all dims (B, N, C); every latent element contributes equally
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # 2. C. Key step: separate the understanding and generation branches
                if deterministic:
                    # Inference mode: use the mean directly
                    patch_latents_for_understanding = mu
                    patch_latents_for_generation = mu
                else:
                    # Training mode:
                    # - Understanding branch (distilled by the last 12 layers): use μ (noise-free, aligned with the
                    # Teacher)
                    # - Generation branch (reconstruction + Flow): use z = μ + σ * ε (noisy, VAE training)
                    patch_latents_for_understanding = mu  # clean, used for distillation
                    
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    patch_latents_for_generation = mu + eps * std  # noisy, used for generation
                
                # 2. D. Save the latents (used by the generation branch)
                saved_latents = patch_latents_for_generation

                # 2. E. Unproj: 64 -> 1024 -> 1024 (the understanding branch uses the clean latent)
                reconstructed_patches = self.chal_unproj_understanding(patch_latents_for_understanding)

                # 3. Concatenate back
                hidden_states = reconstructed_patches
        # End of the loop
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        base_output = BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )
        
        # Returns: ViT output, compressed latents, [original patch features, reconstructed patch features], KL loss
        return base_output, saved_latents, [input_tokens, reconstructed_patches], kl_loss


class FlowDecoder(nn.Module):
    """ patch-wise pixel flow decoder (rectified flow) """
    def __init__(
        self, target_channels, z_channels, depth, width, grad_checkpointing=False, num_sampling_steps='10',
        train_schedule='fat_lognormal', use_cfg=False, noise_concat=False, patch_size=14, img_size=224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        # configs
        self.use_cfg = use_cfg
        self.train_schedule = train_schedule
        self.num_sampling_steps = int(num_sampling_steps)
        self.noise_concat = noise_concat
        print(f"Sampling Step: {self.num_sampling_steps}")

        # mlp head (latent to pixel)
        self.in_channels = target_channels + z_channels if noise_concat else target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

    @torch.no_grad()
    def _get_lpips(self):
        """Lazily build the LPIPS (VGG) module; training paths only."""
        if self._lpips is None:
            logger.info("Initializing LPIPS loss (VGG backbone) for perceptual training")
            self._lpips = lpips.LPIPS(net='vgg').eval()
            for param in self._lpips.parameters():
                param.requires_grad = False
        return self._lpips

    def sample(self, z, schedule="linear", cfg=1.0, cfg_interval=None):
        
        b, n, c_z = z.shape
        z = z.reshape(b*n, c_z)
        sample_steps = self.num_sampling_steps

        # get all timesteps ts and intervals Δts
        if schedule == "linear":
            ts = torch.arange(1, sample_steps + 1).flip(0) / sample_steps
            dts = torch.ones_like(ts) * (1.0 / sample_steps)
        elif schedule.startswith("pow"):  # "pow_0.25"
            p = float(schedule.split("_")[1])
            ts = torch.arange(0, sample_steps + 1).flip(0) ** (1 / p) / sample_steps ** (
                1 / p
            )
            dts = ts[:-1] - ts[1:]
        else:
            raise NotImplementedError
        ts = 1 - ts
        
        # cfg interval
        if cfg_interval is None:  # cfg_interval = "(.17,1.02)"
            interval = None
        else:
            cfg_lo, cfg_hi = ast.literal_eval(cfg_interval)
            interval = self._edm_to_flow_convention(cfg_lo), self._edm_to_flow_convention(cfg_hi)

        # sampling (sample_steps) steps: noise X0 -> clean X1
        trajs = []
        x = torch.randn(b*n, self.in_channels, device=z.device, dtype=z.dtype)  # noise start [b,n,c]
        
        null_z = z.clone() * 0.0 if cfg != 1.0 else None
        for i, (t, dt) in enumerate((zip(ts, dts))):
            timesteps = torch.tensor([t] * (b*n)).to(z.device)

            xc = x
            if self.noise_concat:
                xc = torch.cat([x, z], dim=-1)  # c: 192 + 768 = 960
            
            vc = self.net(x=xc, t=1000*timesteps, c=z)  # conditional v

            # classifier free guidance
            if null_z is not None and (interval is None or ((t.item() >= interval[0]) and (t.item() <= interval[1]))):
                xu = x
                if self.noise_concat:
                    xu = torch.cat([x, null_z], dim=-1)  # c: 192 + 768=960
                vu = self.net(x=xu, t=1000*timesteps, c=null_z)  # unconditional v
                vc = vu + cfg * (vc - vu)
            
            # update x
            x = x + dt * vc
            trajs.append(x)
        
        sampled_token = trajs[-1]
        sampled_image = l2p_transform_tensor(
            sampled_token.reshape(b, n, self.in_channels),
            patch_size=self.patch_size, img_size=self.img_size)
        return sampled_image


class Distill_Adapter(nn.Module):
    def __init__(self, in_channels=1408, out_channels=3200, 
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.head = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        
        x = self.input_proj(x)
        
        t = self.time_embed(t)
        c = self.cond_embed(c)
        
        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)


#         return reconstructed_image
class HydraTokVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = HydraTokConfig

    def __init__(self, config: HydraTokConfig):
        super().__init__(config)
        self.config = config
        vit_hidden_size = config.vit_hidden_size
        llm_hidden_size = config.llm_hidden_size
        self.use_disp_loss = config.use_disp_loss

        # 1. ViT Encoder (first 12 layers + Bottleneck)
        self.embeddings = HydraTokVisionEmbeddings(config)
        self.encoder = HydraTokVisionEncoder(config)
        
        self.latent_ch = config.latent_ch

        # 2. Global Transformer Blocks (Semantic Decoder / last 12 layers)
        self.global_blocks_depth = config.global_blocks_depth
        self.global_block_pos_embed = nn.Parameter(torch.randn(1, self.embeddings.num_patches, vit_hidden_size))
        
        # Use HydraTokVisionEncoderLayer to keep the structure consistent
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        global_dpr = dpr[-self.global_blocks_depth:] if self.global_blocks_depth > 0 else []

        self.global_blocks = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, global_dpr[idx]) 
            for idx in range(self.global_blocks_depth)
        ])
        
 

        # 3. Generation Unproject Layer
        # Keep it purely linear + Identity initialization
        self.chal_unproj_generation = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(self.latent_ch, vit_hidden_size)),
                ("gelu", nn.GELU()),
                ("c_proj", nn.Linear(vit_hidden_size, vit_hidden_size)),
        ]))
        
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.embeddings.num_patches, vit_hidden_size))
        
        # 4. Flow Head
        self.flow_head = FlowDecoder(
            target_channels=3 * config.patch_size * config.patch_size,
            z_channels=config.vit_hidden_size,
            width=config.vit_hidden_size,
            depth=config.num_decoder_layers,
            num_sampling_steps=config.num_sampling_steps,
            grad_checkpointing=False,
            patch_size=config.patch_size,
            img_size=config.image_size,
            use_cfg=config.use_cfg,
        )

        # Init Pos Embeddings
        logger.info("Init pos_embed from sincos pos_embed")
        pos_embed_spatial = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.embeddings.num_patches**0.5),  
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed_spatial).float())
        self.global_block_pos_embed.data.copy_(torch.from_numpy(pos_embed_spatial).float())
        
        self.flow_head.apply(self._init_weights)

        # === New: LPIPS (following FlowMo Stage 1A) ===
        # Created lazily by _get_lpips() on first use, so pure-reconstruction
        # and inference paths never load the VGG backbone.
        self._lpips = None
        self.use_lpips = LPIPS_AVAILABLE
        if not LPIPS_AVAILABLE:
            logger.warning("LPIPS not available, skipping perceptual loss")
        
        # === New: initialize the GAN discriminator ===
        # Note: the discriminator must be initialized and optimized separately in the training script
        # Here we only flag whether the GAN is available
        self.use_gan = GAN_AVAILABLE
        if self.use_gan:
            logger.info("GAN discriminator available for adversarial training")
        else:
            logger.warning("GAN discriminator not available, skipping adversarial loss")
        DATA_NORM = {"processor": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
        proc_mean = torch.tensor(DATA_NORM["processor"]["mean"]).view(1, 3, 1, 1)
        proc_std = torch.tensor(DATA_NORM["processor"]["std"]).view(1, 3, 1, 1)
        
        # Register as buffers so they move with the model (e.g. .to(device))
        self.register_buffer('processor_mean', proc_mean)
        self.register_buffer('processor_std', proc_std)
   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {}

    def get_input_embeddings(self):
        return self.embeddings

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pass


    def calculate_bottleneck_loss(self, student_feats, teacher_feats):
        """
        Compute the cosine distance between the reconstructed (Student) and original (Teacher) features.
        Corresponds to the paper formula: L = 1 - CosineSimilarity(H_student, H_teacher)
        
        Args:
            student_feats: reconstructed features [B, N, C] (reconstructed_patches)
            teacher_feats: original features [B, N, C] (patch_tokens)
        """
        # 1. L2 normalization (the ||H|| denominator in the formula)
        #    eps avoids division by zero
        student_norm = F.normalize(student_feats, p=2, dim=-1, eps=1e-6)
        teacher_norm = F.normalize(teacher_feats, p=2, dim=-1, eps=1e-6)
        
        # 2. Compute the dot product (the <H_U, H_T> numerator in the formula)
        #    Result shape: [B, N]
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
        
        # 3. Loss = 1 - mean similarity
        #    Ideally the similarity is 1 and the loss is 0
        loss = 1.0 - cosine_sim.mean()
        
        return loss
    
    def calculate_adaptive_weight(
        self,
        recon_loss: torch.Tensor,
        gan_loss: torch.Tensor,
        layer: torch.nn.Parameter,
        max_d_weight: float = 1e4,
    ) -> torch.Tensor:
        """
        Compute the adaptive GAN loss weight (following RAE Stage-1 training)
        
        Adjusts the GAN loss weight automatically by comparing the gradient norms of the
        reconstruction and GAN losses, keeping both gradients on the same scale so that the
        GAN loss does not dominate training.
        
        Args:
            recon_loss: reconstruction loss (Flow MSE + LPIPS)
            gan_loss: GAN generator loss
            layer: layer parameters used for the gradient (usually the last layer of the generator)
            max_d_weight: upper bound of the adaptive weight (prevents overly large weights)
            
        Returns:
            adaptive_weight: the adaptive weight scalar
        """
        # Compute the gradient of the reconstruction loss
        recon_grads = torch.autograd.grad(
            recon_loss, layer, retain_graph=True, create_graph=False
        )[0]
        
        # Compute the gradient of the GAN loss
        gan_grads = torch.autograd.grad(
            gan_loss, layer, retain_graph=True, create_graph=False
        )[0]
        
        # Compute the gradient norm ratio: recon_grad_norm / gan_grad_norm
        # If the reconstruction gradient is larger, raise the GAN weight; if the GAN gradient is larger, lower it
        d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
        
        # Clamp the weight range to keep training stable if the weight gets too large
        d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
        
        return d_weight.detach()
    def calculate_lpips_loss(self, xt, model_output, pixel_values, t_broadcast, B, N):
        """
        Compute the LPIPS perceptual loss (following FlowMo Stage 1A)
        Uses the straight-line property of Rectified Flow for one-step prediction
        
        Args:
            xt: noised image [B*N, C]
            model_output: predicted velocity field [B*N, C]
            pixel_values: ground-truth image [B, 3, H, W]
            t_broadcast: timestep [B, 1, 1]
            B: Batch size
            N: number of tokens
            
        Returns:
            lpips_val: LPIPS loss value
        """
        if not self.use_lpips or not self.training:
            return torch.tensor(0.0, device=pixel_values.device)
        
        # 1. Reshape from (B*N, C) back to (B, N, C)
        xt_reshaped = rearrange(xt, '(b n) c -> b n c', b=B)
        pred_velocity = rearrange(model_output, '(b n) c -> b n c', b=B)
        
        # 2. One-step prediction: x_1 (target) = x_t + (1-t) * velocity
        # Formula follows the linear interpolation property of Rectified Flow
        pred_latents = xt_reshaped + (1.0 - t_broadcast) * pred_velocity
        
        # 3. Map the tokens back to image space (unpatchify)
        pred_images = l2p_transform_tensor(
            pred_latents, 
            patch_size=self.config.patch_size, 
            img_size=self.config.image_size
        )
        
        # 4. Normalize to [-1, 1] (the input range LPIPS expects)
        # ImageNet normalization params: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(
            pixel_values.device, pixel_values.dtype
        )
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(
            pixel_values.device, pixel_values.dtype
        )
        
        # Denormalize to [0, 1]
        gt_images = pixel_values * std + mean
        pred_images_denorm = pred_images * std + mean
        
        # Scale to [-1, 1]
        gt_images_scaled = gt_images * 2.0 - 1.0
        pred_images_scaled = pred_images_denorm * 2.0 - 1.0
        
        # 5. Make sure LPIPS is on the correct device and dtype
        lpips_fn = self._get_lpips()
        lpips_device = next(lpips_fn.parameters()).device
        lpips_dtype = next(lpips_fn.parameters()).dtype

        if lpips_device != pixel_values.device:
            lpips_fn.to(pixel_values.device)
        
        # 6. Compute the LPIPS loss
        # FlowMo Stage 1A recommended weight: lambda = 0.1
        # Ensure the input dtype matches the LPIPS model
        lpips_val = lpips_fn(
            pred_images_scaled.to(lpips_dtype), 
            gt_images_scaled.to(lpips_dtype)
        ).mean()
        
        return lpips_val
    
    def calculate_gan_loss(self, xt, model_output, pixel_values, t_broadcast, B, N, 
                           discriminator, disc_aug, disc_crop_size=224, return_fake_images=False):
        """
        Compute the GAN generator loss (following FlowMo Stage 1A)
        Uses the straight-line property of Rectified Flow for one-step prediction
        
        Args:
            xt: noised image [B*N, C]
            model_output: predicted velocity field [B*N, C]
            pixel_values: ground-truth image [B, 3, H, W]
            t_broadcast: timestep [B, 1, 1]
            B: Batch size
            N: number of tokens
            discriminator: DINO discriminator (DDP-wrapped)
            disc_aug: DiffAug data augmentation
            disc_crop_size: random crop size before feeding the discriminator (default 224)
            return_fake_images: whether to also return the fake images (used for the discriminator update)
            
        Returns:
            gan_loss: GAN generator loss value
            fake_images: (optional) normalized fake images [B, 3, H, W], in the range [-1, 1]
        """
        if not self.use_gan or not self.training:
            # If return_fake_images=True, a tuple must be returned
            if return_fake_images:
                return torch.tensor(0.0, device=pixel_values.device), None
            return torch.tensor(0.0, device=pixel_values.device)
        
        # 1. Reshape from (B*N, C) back to (B, N, C)
        xt_reshaped = rearrange(xt, '(b n) c -> b n c', b=B)
        pred_velocity = rearrange(model_output, '(b n) c -> b n c', b=B)
        
        # 2. One-step prediction: x_1 (target) = x_t + (1-t) * velocity
        # Formula follows the linear interpolation property of Rectified Flow
        pred_latents = xt_reshaped + (1.0 - t_broadcast) * pred_velocity
        
        # 3. Map the tokens back to image space (unpatchify)
        pred_images = l2p_transform_tensor(
            pred_latents, 
            patch_size=self.config.patch_size, 
            img_size=self.config.image_size
        )
        
        # 4. Normalize to [-1, 1] (the input range the discriminator expects)
        # ImageNet normalization params: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(
            pixel_values.device, pixel_values.dtype
        )
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(
            pixel_values.device, pixel_values.dtype
        )
        
        # Denormalize to [0, 1]
        pred_images_denorm = pred_images * std + mean
        
        # Scale to [-1, 1]
        pred_images_scaled = pred_images_denorm * 2.0 - 1.0
        
        # 5. Save the original image (returned as is, to avoid duplicate processing)
        # Note: the returned fake images should be the full 448x448 images, without crop or DiffAug
        # Crop and DiffAug should be applied uniformly in the discriminator-update part of the training script
        fake_images_for_return = pred_images_scaled.clone()
        
        # 6. Random crop to disc_crop_size x disc_crop_size (default 224x224)
        # Random-crop the fake images (only for the generator loss computation)
        H, W = pred_images_scaled.shape[2], pred_images_scaled.shape[3]
        if H > disc_crop_size and W > disc_crop_size:
            # Randomly pick the crop start position
            top = torch.randint(0, H - disc_crop_size + 1, (1,)).item()
            left = torch.randint(0, W - disc_crop_size + 1, (1,)).item()
            pred_images_scaled_cropped = pred_images_scaled[:, :, top:top+disc_crop_size, left:left+disc_crop_size]
        else:
            # If the image is already <= crop_size, use it as is
            pred_images_scaled_cropped = pred_images_scaled
        
        # 7. Apply DiffAug data augmentation
        # Note: DiffAug.aug() forces a cast to float32 (see diffaug.py lines 34-35), so convert again afterwards
        fake_aug = disc_aug.aug(pred_images_scaled_cropped)
        
        # Ensure the input dtype matches the discriminator (the discriminator is bfloat16)
        # DiffAug turns the input into float32; convert it back to bfloat16
        try:
            if hasattr(discriminator, 'module'):
                # DDP-wrapped case
                disc_dtype = next(discriminator.module.parameters()).dtype
            else:
                # Non-DDP case
                disc_dtype = next(discriminator.parameters()).dtype
        except (StopIteration, AttributeError):
            # Fallback: default to bfloat16 (set explicitly in the training script)
            disc_dtype = torch.bfloat16
        fake_aug = fake_aug.to(dtype=disc_dtype)
        
        # 8. Get the logits from the discriminator
        # Set the discriminator to eval mode (it is not updated here; only the generator loss is computed)
        discriminator.eval()
        logits_fake, _ = discriminator(fake_aug, None)
        
        # 9. Compute the generator loss (vanilla_g_loss: -mean(logits_fake))
        gan_loss = vanilla_g_loss(logits_fake)
        
        if return_fake_images:
            # Return the full normalized fake images (448x448, no crop, no DiffAug)
            # For unified handling during the discriminator update
            return gan_loss, fake_images_for_return
        else:
            return gan_loss

# =======================================================
    # 1. Training Forward (modified)
    # =======================================================
    def forward(self, pixel_values, output_hidden_states=False, skip_flow_loss=False,
                discriminator=None, disc_aug=None, use_gan_loss=False, return_fake_images=False,
                max_d_weight=1e4, use_adaptive_gan_weight=False,
                use_kl_loss=True, use_bn_loss=True, lpips_weight=0.1,
                kl_weight=1e-4, bn_weight=0.5, gan_weight_base=0.5):
        """
        skip_flow_loss: used in Stage 2, skips Flow Head computation to save memory and avoid
            updating the generation branch
        discriminator: GAN discriminator (DDP-wrapped), used to compute the GAN loss
        disc_aug: DiffAug data augmentation, used for GAN training
        use_gan_loss: whether to enable the GAN loss
        return_fake_images: whether to return the fake images (for the discriminator update, avoids regenerating them)
        max_d_weight: upper bound of the adaptive GAN weight (default 1e4)
        use_adaptive_gan_weight: whether to use the adaptive GAN weight (default False, i.e. a fixed weight)
        use_kl_loss: include the VAE KL regularization (stage-1 recipe, default True)
        use_bn_loss: include the bottleneck consistency loss (stage-1 recipe, default True)
        lpips_weight: weight of the one-step LPIPS perceptual loss (stage-1 uses 0.1, stage-2 uses 1.0)
        kl_weight: VAE KL loss weight (default 1e-4; start small to prevent posterior collapse)
        bn_weight: bottleneck consistency loss weight (default 0.5)
        gan_weight_base: base GAN weight (default 0.5; kept below 1 so the GAN loss cannot dominate)

        Loss recipes:
            stage 1 (default): flow MSE + 1e-4*KL + 0.5*BN + 0.1 * LPIPS
            stage 2 (GAN):     flow MSE + 1.0 * LPIPS + adaptive * GAN
                              (pass use_kl_loss=False, use_bn_loss=False, lpips_weight=1.0)
        """
        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)
            B, N_plus_1, C = hidden_states.shape
        else:
            raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        # 1. Encoder Forward
        # This covers the Bottleneck processing (VAE: μ/logvar -> reparameterization)
        encoder_outputs, latents, bottleneck_feats, kl_loss = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True, 
        )
        # import ipdb; ipdb.set_trace()
        # Get the features used for distillation (all outputs of Layers 0-23)
        features_for_distill = encoder_outputs.hidden_states

        # --- Stage 2 fast path: skip the Flow computation ---
        if skip_flow_loss:
            # Only return a 0.0 loss dict and the distillation features
            # unproj_bias has already taken part in the computation inside the encoder (Layer 11->12)
            dummy_loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=True)
            dummy_loss_dict = {
                'total': dummy_loss,
                'flow_mse': dummy_loss,
                'lpips': dummy_loss,
                'gan': dummy_loss,
            }
            if output_hidden_states:
                return dummy_loss_dict, features_for_distill
            else:
                return dummy_loss_dict

        # --- Stage 1 full path: compute the Flow Loss ---
        if latents.shape[1] > self.embeddings.num_patches:
            gen_latents = latents[:, 1:, :] 
        else:
            gen_latents = latents
        # 2. Generation Path (chal_unproj_generation -> Global Blocks)
        condition_tokens = self.chal_unproj_generation(gen_latents)
        _, N, _ = condition_tokens.shape 

        # Global Pos Embed
        global_block_pos_embed = self.global_block_pos_embed.repeat(B, 1, 1).view(B, -1, C)
        condition_tokens = condition_tokens + global_block_pos_embed[:, :N]
        
        # Global Blocks (Enhancement)
        for block in self.global_blocks:
            condition_tokens = block(condition_tokens)

        # Flow Head Pos Embed
        decoder_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1).view(B, -1, C)
        condition_tokens = condition_tokens + decoder_pos_embed[:, :N]
        
        # --- Loss Calculation ---
        target_tokens = p2l_transform_tensor(pixel_values, self.config.patch_size)
        noise = torch.randn_like(target_tokens)
        nt = torch.randn((B,), device=pixel_values.device, dtype=target_tokens.dtype)
        t = torch.sigmoid(nt) 
        t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))
        t = t.view(B, 1, 1)

        t_broadcast = t
        xt = t_broadcast * target_tokens + (1 - t_broadcast) * noise
        ut = target_tokens - noise 
        
        xt = rearrange(xt, 'b n c -> (b n) c')
        ut = rearrange(ut, 'b n c -> (b n) c')
        condition_tokens_flat = rearrange(condition_tokens, 'b n c -> (b n) c')
        
        t_for_net = t.view(-1).repeat_interleave(N) * 1000

        model_output = self.flow_head.net(xt, t_for_net, condition_tokens_flat)
        loss = F.mse_loss(model_output.float(), ut.float(), reduction='mean')
        
        # === Compute the one-step perceptual loss (LPIPS) ===
        # Following FlowMo Stage 1A: exploit the straight-line property of Rectified Flow
        lpips_val = self.calculate_lpips_loss(
            xt=xt,
            model_output=model_output,
            pixel_values=pixel_values,
            t_broadcast=t_broadcast,
            B=B,
            N=N
        )
        
        # === Compute the one-step GAN loss ===
        # Following FlowMo Stage 1A: exploit the straight-line property of Rectified Flow
        fake_images_for_disc = None
        if use_gan_loss and discriminator is not None and disc_aug is not None:
            gan_result = self.calculate_gan_loss(
                xt=xt,
                model_output=model_output,
                pixel_values=pixel_values,
                t_broadcast=t_broadcast,
                B=B,
                N=N,
                discriminator=discriminator,
                disc_aug=disc_aug,
                return_fake_images=return_fake_images
            )
            if return_fake_images:
                gan_loss, fake_images_for_disc = gan_result
            else:
                gan_loss = gan_result
        else:
            # Create a zero loss that still carries gradients so the adaptive-weight computation cannot fail
            gan_loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=True)
        
        # Compute the other losses (Bottleneck, etc.)
        target_patch_tokens, recon_patch_tokens = bottleneck_feats
        bn_loss = self.calculate_bottleneck_loss(recon_patch_tokens, target_patch_tokens) if use_bn_loss \
            else torch.tensor(0.0, device=pixel_values.device)
        
        # VAE KL divergence loss (standard implementation)
        # kl_loss was already computed in the encoder: KL(N(μ, σ²) || N(0, 1))
        # Formula: KL = -0.5 * sum(1 + logvar - μ² - exp(logvar))
        # Computed elementwise, summed over the token/channel dims, averaged over the batch
        
        # Loss weights are passed in via forward args (kl_weight / bn_weight /
        # gan_weight_base / lpips_weight) and controlled by the training configs
        # (configs/stage1.yaml, configs/stage2_gan.yaml).

        # Compute the reconstruction loss (Flow MSE + LPIPS + optional KL/BN)
        recon_total = loss + lpips_weight * lpips_val
        if use_kl_loss:
            recon_total = recon_total + kl_weight * kl_loss
        if use_bn_loss:
            recon_total = recon_total + bn_weight * bn_loss
        
        # Adaptive GAN weight computation (following RAE Stage-1 training)
        # Only computed when the GAN loss is enabled and the adaptive weight is used
        adaptive_weight = gan_weight_base
        if use_gan_loss and use_adaptive_gan_weight and gan_loss.requires_grad and gan_loss.item() > 0:
            # Use the last layer of flow_head for the gradient
            # Use the final_layer linear weight as the reference layer
            last_layer = self.flow_head.net.final_layer.linear.weight
            try:
                # Ensure both recon_total and gan_loss have gradients
                if recon_total.requires_grad and gan_loss.requires_grad:
                    adaptive_weight_tensor = self.calculate_adaptive_weight(
                        recon_total, gan_loss, last_layer, max_d_weight
                    )
                    adaptive_weight = (
                        adaptive_weight_tensor.item()
                        if isinstance(adaptive_weight_tensor, torch.Tensor) else adaptive_weight_tensor)
                else:
                    # If there is no gradient, use the base weight
                    adaptive_weight = gan_weight_base
            except (RuntimeError, ValueError) as e:
                # If the computation fails (e.g. some layer has no gradient), fall back to the fixed weight
                if self.training:
                    import warnings
                    warnings.warn(
                        f"Failed to calculate adaptive weight: {e}, using base weight {gan_weight_base} instead.")
                adaptive_weight = gan_weight_base
        else:
            # When the adaptive weight is not used, fall back to the base weight
            adaptive_weight = gan_weight_base
        
        # Convert to a tensor (for the loss computation)
        if not isinstance(adaptive_weight, torch.Tensor):
            adaptive_weight = torch.tensor(adaptive_weight, device=pixel_values.device)
        
        # Compute the total loss: recon_total + adaptive_weight * gan_loss
        total_loss = recon_total + adaptive_weight * gan_loss
        
        # Build the loss dict with the individual components
        loss_dict = {
            'kl': kl_loss,
            'bn': bn_loss,
            'total': total_loss,
            'flow_mse': loss,  # Flow MSE loss
            'lpips': lpips_val,
            'gan': gan_loss,
            'adaptive_weight': (adaptive_weight.detach()
                               if isinstance(adaptive_weight, torch.Tensor) else adaptive_weight),
        }
        
        # If the fake images must be returned (for the discriminator update)
        if return_fake_images and fake_images_for_disc is not None:
            if output_hidden_states:
                return loss_dict, features_for_distill, fake_images_for_disc
            else:
                return loss_dict, fake_images_for_disc
            
        if output_hidden_states:
            return loss_dict, features_for_distill 
        else:
            return loss_dict
    # =======================================================
    # 2. Inference & Sampling (Global Blocks enabled)
    # =======================================================
    def encode(self, pixel_values):        
        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)
        # Use deterministic mode at inference: use μ directly
        _, latents, _, _ = self.encoder(inputs_embeds=hidden_states, output_hidden_states=False, deterministic=True)
        if latents.shape[1] > self.embeddings.num_patches:
            latents = latents[:, 1:, :] 
        else:
            latents = latents
        b, n, c = latents.shape
        h = int(sqrt(n)) 
        w = n // h
        latents = latents.transpose(1, 2).view(b, c, h, w)
        return latents

    def decode(self, latent_tokens):
        
        B, C, H, W = latent_tokens.shape
        N = H * W
        
        # [B, C, H*W] -> [B, H*W, C]
        latent_tokens = latent_tokens.view(B, C, N).transpose(1, 2)

        condition_tokens = self.chal_unproj_generation(latent_tokens)

        B, N, C = condition_tokens.shape 

        global_block_pos_embed = self.global_block_pos_embed.repeat(B, 1, 1).view(B, -1, C)
        condition_tokens = condition_tokens + global_block_pos_embed[:, :N]
        for block in self.global_blocks:
            condition_tokens = block(condition_tokens)
        
        decoder_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1).view(B, -1, C)
        condition_tokens = condition_tokens + decoder_pos_embed[:, :N]
        
        # Assume flow_head takes [B, N, C] and outputs an image [B, C, H, W]
        
        reconstructed_image = self.flow_head.sample(z=condition_tokens)
        output_dtype = reconstructed_image.dtype
        
        # Convert the buffers to the same dtype
        proc_mean = self.processor_mean.to(dtype=output_dtype)
        proc_std = self.processor_std.to(dtype=output_dtype)

        # (x * std) + mean
        reconstructed_image = (reconstructed_image * proc_std) + proc_mean
        # --- [modification] end ---  
        
        return reconstructed_image
    def sample(self, pixel_values):
        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)
            B, N_plus_1, C = hidden_states.shape

        # 1. Encode (use deterministic mode at inference: use μ directly)
        _, latents, _, _ = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=False,
            deterministic=True,
        )
        if latents.shape[1] > self.embeddings.num_patches:
            gen_latents = latents[:, 1:, :] 
        else:
            gen_latents = latents
        # 2. Unproject
        condition_tokens = self.chal_unproj_generation(gen_latents)
        _, N, _ = condition_tokens.shape

        # 3. Global Pos Embed
        global_block_pos_embed = self.global_block_pos_embed.repeat(B, 1, 1).view(B, -1, C)
        condition_tokens = condition_tokens + global_block_pos_embed[:, :N]

        # 4. Run the last 12 layers
        for block in self.global_blocks:
            condition_tokens = block(condition_tokens)

        # 5. Decoder Pos Embed & Reconstruct
        decoder_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1).view(B, -1, C)
        condition_tokens = condition_tokens + decoder_pos_embed[:, :N]

        reconstructed_image = self.flow_head.sample(z=condition_tokens) 
        return reconstructed_image
