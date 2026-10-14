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
    print(" LPIPS not installed. Install via: pip install lpips")
    LPIPS_AVAILABLE = False


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


class Generation_Encoder(nn.Module):
    """First 12 encoder layers, used by Generation_ViT (KL/VAE paradigm)"""
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]

        # Only initialize the first 12 layers


        self.layers = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, dpr[idx]) for idx in range(12)])
        self.gradient_checkpointing = False

        self.latent_ch = config.latent_ch        # 64
        self.vit_hidden_size = config.vit_hidden_size  # 1024

        # --- Bottleneck: VAE Encoder (Proj) ---
        # VAE structure: 1024 -> 1024 -> GELU -> 128 (outputs mu and logvar, each 64-dim)
        self.chal_proj = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.vit_hidden_size, self.vit_hidden_size)),  # 1024 -> 1024
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(self.vit_hidden_size, self.latent_ch * 2)),  # 1024 -> 128 (μ: 64, logvar: 64)
        ]))


    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            attn_mask=None,
            deterministic: bool = False,  # set True at inference to use mu directly
) -> Tuple:
        """
        First 12 encoding layers + VAE bottleneck projection

        Args:
            deterministic: set True at inference to use mu directly; False during training to resample

        Returns:
            patch_latents_for_generation: [B, N, 64] - latents for the generation branch
                (noisy during training, mu at inference)
            patch_latents_for_understanding: [B, N, 64] - latents for the understanding branch (always mu, no noise)
            cls_token: [B, 1, C] - CLS token
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds[:, 1:, :]

        # Run the first 12 layers
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


        patch_tokens = hidden_states
        # VAE Proj: 1024 -> 128 (mu and logvar)
        h = self.chal_proj(patch_tokens)         # [B, N, 128]
        mu, logvar = h.chunk(2, dim=-1)          # each [B, N, 64]
        patch_latents = mu
        return patch_latents


class Understanding_Encoder(nn.Module):
    """Last 12 encoder layers, used by Understanding_ViT (KL/VAE paradigm)"""
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]

        # Initialize the last 12 layers (indices 12-23)
        self.layers = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, dpr[idx + 12]) for idx in range(12)])
        self.gradient_checkpointing = False

        self.latent_ch = config.latent_ch        # 64
        self.vit_hidden_size = config.vit_hidden_size  # 1024

        # --- Bottleneck: Decoder (Unproj) ---
        # Structure: 64 -> 1024 (Xavier) -> GELU -> 1024 -> 1024 (Identity)
        self.chal_unproj_understanding = nn.Sequential(OrderedDict([
            ("c_proj", nn.Linear(self.latent_ch, self.vit_hidden_size)),     # 64 -> 1024
            ("gelu", nn.GELU()),
            ("c_fc", nn.Linear(self.vit_hidden_size, self.vit_hidden_size)),  # 1024 -> 1024
        ]))


    def forward(
            self,
            latents,
            output_hidden_states: Optional[bool] = None,
            attn_mask=None,
) -> Tuple:
        """
        Start from latents and go through Unproj + the last 12 layers

        Args:
            latents: [B, N, 64] - compressed latents (from mu, used for understanding-branch distillation)

        Returns:
            hidden_states: [B, N, C] or [B, N+1, C] - output of the last 12 layers
            encoder_states: tuple - intermediate layer outputs (if output_hidden_states=True)
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)

        encoder_states = () if output_hidden_states else None

        # Unproj: 64 -> 1024
        patch_features = self.chal_unproj_understanding(latents)  # [B, N, 1024]


        hidden_states = patch_features

        # Run the last 12 layers
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

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states


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
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        x = torch.randn(b*n, self.in_channels).cuda()  # noise start [b,n,c]
        x = x.to(z.dtype)

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


# --------------------------------------------------------
# 1. Clean Encoder Gen (first 12 layers)
# --------------------------------------------------------
class HydraTokVisionEncoder_Gen(nn.Module):
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config

        # Drop path rates for the first 12 generation layers
        # Note: computed over the total number of layers, then the first half is taken,
        # keeping the linear increase in probability consistent
        total_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # Assume gen_layers is 12
        my_dpr = total_dpr[:config.gen_layers]

        self.layers = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, my_dpr[idx]) for idx in range(config.gen_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            attn_mask=None,
) -> Union[Tuple, BaseModelOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

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

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
)


# --------------------------------------------------------
# 2. Clean Encoder Und (last 12 layers)
# --------------------------------------------------------
class HydraTokVisionEncoder_Und(nn.Module):
    def __init__(self, config: HydraTokConfig):
        super().__init__()
        self.config = config

        # Fix the DPR computation: computed over the total number of layers, taking the second half
        total_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # Slice [12:24]
        my_dpr = total_dpr[config.gen_layers : config.gen_layers + config.und_layers]

        self.layers = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, my_dpr[idx]) for idx in range(config.und_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            attn_mask=None,
) -> Union[Tuple, BaseModelOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

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

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
)


# --------------------------------------------------------
# 3. Generation ViT (generation branch, KL/VAE paradigm)
# # --------------------------------------------------------
class Gen_ViT(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = HydraTokConfig

    def __init__(self, config: HydraTokConfig):
        super().__init__(config)
        self.config = config
        self.vit_hidden_size = config.vit_hidden_size
        self.latent_ch = config.latent_ch

        # 1. Base encoder (first 12 layers) - with a built-in VAE bottleneck
        self.embeddings = HydraTokVisionEmbeddings(config)
        self.encoder = Generation_Encoder(config)  # use Generation_Encoder (already includes VAE Proj)

        # 2. Generation Unproj & Global Blocks
        self.global_blocks_depth = config.global_blocks_depth

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        global_dpr = dpr[-self.global_blocks_depth:] if self.global_blocks_depth > 0 else []

        self.global_blocks = nn.ModuleList([
            HydraTokVisionEncoderLayer(config, global_dpr[idx])
            for idx in range(self.global_blocks_depth)
        ])

        self.chal_unproj_generation = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(self.latent_ch, self.vit_hidden_size)),
                ("gelu", nn.GELU()),
                ("c_proj", nn.Linear(self.vit_hidden_size, self.vit_hidden_size)),
        ]))

        # Pos Embeddings & Flow Head
        self.global_block_pos_embed = nn.Parameter(torch.randn(1, self.embeddings.num_patches, self.vit_hidden_size))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.embeddings.num_patches, self.vit_hidden_size))

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

        DATA_NORM = {"processor": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
        proc_mean = torch.tensor(DATA_NORM["processor"]["mean"]).view(1, 3, 1, 1)
        proc_std = torch.tensor(DATA_NORM["processor"]["std"]).view(1, 3, 1, 1)

        # Register as buffers so they move with the model (e.g. via.to(device))
        self.register_buffer('processor_mean', proc_mean)
        self.register_buffer('processor_std', proc_std)
        self.register_buffer_stats(config)


    def register_buffer_stats(self, config):
        if config.latent_stats_path is not None:
            stats = torch.load(config.latent_stats_path, map_location="cpu")
            self.register_buffer('latent_mean', stats['latent_mean'].reshape(1, -1, 1, 1))
            self.register_buffer('latent_std_dev', stats['latent_std'].reshape(1, -1, 1, 1))
            self.register_buffer('latent_eps', torch.tensor(1e-6))

            DATA_NORM = {"processor": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
            self.register_buffer('processor_mean', torch.tensor(DATA_NORM["processor"]["mean"]).view(1, 3, 1, 1))
            self.register_buffer('processor_std', torch.tensor(DATA_NORM["processor"]["std"]).view(1, 3, 1, 1))

    def encode(self, pixel_values, deterministic: bool = True):
        """
        Encode an image into latent space

        Args:
            pixel_values: [B, 3, H, W] - input image
            deterministic: set True at inference to use mu directly

        Returns:
            latents: [B, C, h, w] - encoded latents
        """
        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)
        else:
            raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        # 1. Encoder (first 12 layers) + VAE bottleneck

        patch_latents_for_generation = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
)

        # 2. Use the generation-branch latents (mu at inference)
        latents = patch_latents_for_generation

        # 3. Reshape to [B, C, H, W]
        b, n, c = latents.shape
        h = int(sqrt(n))
        w = n // h
        latents = latents.transpose(1, 2).view(b, c, h, w)
        # latents = (latents - self.latent_mean) / (self.latent_std_dev + self.latent_eps)
        return latents

    def decode(self, latent_tokens):
        B, C, H, W = latent_tokens.shape
        N = H * W

        # [B, C, H*W] -> [B, H*W, C]
        latent_tokens = latent_tokens * (self.latent_std_dev + self.latent_eps) + self.latent_mean
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

        return reconstructed_image


# --------------------------------------------------------
# 4. Understanding ViT (understanding branch, KL/VAE paradigm)
# --------------------------------------------------------
class Und_ViT(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = HydraTokConfig

    def __init__(self, config: HydraTokConfig):
        super().__init__(config)
        self.config = config
        self.latent_ch = config.latent_ch
        self.vit_hidden_size = config.vit_hidden_size

        # Use Understanding_Encoder (with built-in Unproj + last 12 layers)
        self.encoder = Understanding_Encoder(config)


    def forward(self, image_latents):
        """
        Understanding-branch forward pass

        Args:
            image_latents: [B, C, H, W] or [B, N, C] - latents from mu (noise-free, used for distillation)


        Returns:
            hidden_states: [B, N, C] - output of the last 12 layers
        """
        if image_latents.dim() == 4:
            B, C, H, W = image_latents.shape
            image_latents = image_latents.flatten(2).transpose(1, 2)

        # Call Understanding_Encoder directly (handles Unproj + last 12 layers internally)
        hidden_states, _ = self.encoder(
            latents=image_latents,
)

        return hidden_states
# --------------------------------------------------------
