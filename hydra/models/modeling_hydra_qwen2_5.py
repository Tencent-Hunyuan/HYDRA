# coding=utf-8
# Copyright 2025 NUS Show Lab.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange
from transformers import AutoConfig
from torch.nn.attention.flex_attention import BlockMask
from .misc import velocity_prediction, next_token_prediction, interpolate_pos_encoding
from .hydra_tok.modeling_hydra_tok import Und_ViT
from .hydra_tok.configuration_hydra_tok import HydraTokConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig
from .modules import ModulatedAttentionBlock, RMSNorm, PatchEmbed, TimestepEmbedder, FinalLayer
from .qwen2 import Qwen2ForCausalLM


class HydraQwen2_5(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            llm_vocab_size=None,
            llm_model_path='',
            load_from_llm=False,
            image_latent_dim=64,  # HYDRA-TOK latent_ch
            image_latent_height=32,
            image_latent_width=32,
            video_latent_height=16,
            video_latent_width=16,
            patch_size=1,  # HYDRA-TOK needs no patching; use latent tokens directly
            hidden_size=1536,
            hydra_tok_hidden_size=1024,  # HYDRA-TOK vit_hidden_size
            num_diffusion_layers=10,
            add_time_embeds=True,
            add_qk_norm=False,
            **kwargs,
    ):
        super().__init__()

        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if load_from_llm:
            self.llm = Qwen2ForCausalLM(llm_config)
        else:
            self.llm = Qwen2ForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.llm.resize_token_embeddings(llm_vocab_size)

        # Keep a reference to the Und_ViT model (used for understanding feature extraction)
        # Note: und_vit is set via set_und_vit() after initialization
        # Initialized as None and set later via set_und_vit
        self.und_vit = None

        # Single-path projection: HYDRA-TOK vit_hidden_size -> LLM hidden size
        # HYDRA-TOK semantic_forward outputs [B, N, vit_hidden_size=1024]
        # which must be projected to the LLM hidden_size (2048 for the 7B model)
        self.semantic_proj = nn.Sequential(
            RMSNorm(hydra_tok_hidden_size), 
            nn.Linear(hydra_tok_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # adjust for diffusion head
        self.diffusion_head_config = DiffusionHeadConfig()
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.diffusion_head_config.hidden_size, self.diffusion_head_config.hidden_size)
            )
            self.time_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(self.diffusion_head_config, layer_idx) for layer_idx in
             range(num_diffusion_layers)]
        )
        self.diffusion_head_b = FinalLayer(self.diffusion_head_config.hidden_size, patch_size, image_latent_dim)

        self.reset_parameters()

    def set_und_vit(self, und_vit):
        # Fix: remove the underscore, or use self.add_module
        # so that model.parameters() includes und_vit's parameters
        self.und_vit = und_vit
        print(f"Und_ViT registered as a submodule; its parameters will be included in the optimizer")


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def reset_parameters(self):
        # Initialize semantic projection
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        _basic_init(self.semantic_proj)
        _basic_init(self.diffusion_head_a)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def unpatchify(self, x, h, w, T=0):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.config.image_latent_dim
        p = self.config.patch_size
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs
    


    def forward_und_only(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            **kwargs,
    ):
        """
        Single-path understanding forward: extract semantic features with Und_ViT

        Flow:
        image_latents [B, C, H, W]
            → Und_ViT.forward [B, N, vit_hidden_size]
            → semantic_proj [B, N, hidden_size]
            → LLM
        """
        T = 0
        input_embeds = self.llm.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        
        if len(image_latents.shape) != 4:
            b, c, T, h, w = image_latents.shape
        else:
            b, c, h, w = image_latents.shape

        # Single path: latents → Und_ViT → proj
        if T == 0:
            # Und_ViT expects image_latents [B, C, H, W]
            if self.und_vit is None:
                raise RuntimeError("Und_ViT model not set! Call model.set_und_vit(und_vit) first.")
            
           
            
            # Und_ViT understanding feature extraction: [B, C, H, W] + [B, 1, D] -> [B, N, vit_hidden_size]
            semantic_feats = self.und_vit(image_latents=image_latents.to(dtype))
            # Project to the LLM hidden dimension: [B, N, vit_hidden_size] -> [B, N, hidden_size]
            image_embeds = self.semantic_proj(semantic_feats)
        else:
            # video: [B, C, T, H, W] -> [B*T, C, H, W]
            image_latents_flat = rearrange(image_latents, 'b c t h w -> (b t) c h w')
            
       
            # Und_ViT last 12 layers
            semantic_feats = self.und_vit(image_latents=image_latents_flat.to(dtype))
            # Projection
            image_embeds = self.semantic_proj(semantic_feats)
            # [B*T, N, D] -> [B, T*N, D]
            image_embeds = image_embeds.reshape(b, T, -1, self.config.hidden_size)
            image_embeds = rearrange(image_embeds, 'b t l d -> b (t l) d')

        h_, w_ = h, w  # no patching needed; use the latent resolution directly

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = \
                        image_embeds[i * modality_positions.size(1) + j, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j, :length]

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits

    def forward(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            guidance_scale=0.0,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            **kwargs,
    ):
        """
        Single-path forward: extract semantic features with Und_ViT

        Flow:
        image_latents [B, C, H, W]
            → Und_ViT.forward [B, N, vit_hidden_size]
            → semantic_proj [B, N, hidden_size]
            → LLM → Diffusion Head
        """
        
        
        T = 0
        if image_latents is None:
            # text-only
            logits = self.llm(input_ids=text_tokens, attention_mask=attention_mask)
            return logits
        else:
            # multimodal understanding and generation
            input_embeds = self.llm.model.embed_tokens(text_tokens)
            dtype = input_embeds.dtype
            if len(image_latents.shape) != 4:
                b, c, T, h, w = image_latents.shape
            else:
                b, c, h, w = image_latents.shape

            # Single path: latents → Und_ViT → proj
            if T == 0:
                
                if self.und_vit is None:
                    raise RuntimeError("Und_ViT model not set! Call model.set_und_vit(und_vit) first.")
                
                
                
                # Und_ViT last-12-layer semantic extraction: [B, C, H, W] + [B, 1, D] -> [B, N, vit_hidden_size]
                semantic_feats = self.und_vit(image_latents=image_latents.to(dtype))
                # Project to the LLM hidden dimension
                image_embeds = self.semantic_proj(semantic_feats)
            else:
                # video: [B, C, T, H, W] -> [B*T, C, H, W]
                image_latents_flat = rearrange(image_latents, 'b c t h w -> (b t) c h w')
                
                
                
                semantic_feats = self.und_vit(image_latents=image_latents_flat.to(dtype))
                image_embeds = self.semantic_proj(semantic_feats)
                image_embeds = image_embeds.reshape(b, T, -1, self.config.hidden_size)
                image_embeds = rearrange(image_embeds, 'b t l d -> b (t l) d')

            # Use the latent resolution directly (patch_size=1)
            p = self.config.patch_size
            h_, w_ = h, w

            # Handle image_labels (velocity target)
            if image_labels is not None:
                if T == 0:
                    # [B, C, H, W] -> [B, N, C*p*p]
                    image_labels = rearrange(image_labels, 'b c h w -> b (h w) c')
                    image_labels = image_labels.reshape(shape=(b, h_ * w_, p * p * c))
                else:
                    image_labels = rearrange(image_labels, 'b c t h w -> b (t h w) c')
                    image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))

            time_embeds = self.time_embed(t, dtype)
            if hasattr(self, 'time_embed_proj'):
                time_embeds_proj = self.time_embed_proj(time_embeds)
            else:
                time_embeds_proj = time_embeds

            # structure text and image embeddings into sequences
            if image_labels is not None:
                new_image_labels = torch.zeros([b, max_seq_len, p * p * c], device=device, dtype=dtype)
                image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    # Fix: if length == 0, skip the injection logic (text-only data; no need to inject time embedding
                    # or image embedding)
                    if length == 0:
                        continue
                    
                    if self.config.add_time_embeds:
                        input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                        # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                        # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                        input_embeds[i, offset + 1:offset + 1 + length - 1] = image_embeds[
                                                                              i * modality_positions.size(1) + j,
                                                                              :max(length - 1, 0)]
                        if image_labels is not None:
                            # mask the position of time embedding
                            image_masks[i, offset] = 0
                            # it's necessary to include :length-1, as sometimes we may skip some idle images when
                            # length=0
                            new_image_labels[i, offset + 1:offset + 1 + length - 1] = image_labels[
                                                                                      i * modality_positions.size(
                                                                                          1) + j, :max(length - 1, 0)]
                    else:
                        input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j,
                                                                  :length]
                        if image_labels is not None:
                            new_image_labels[i, offset:offset + length] = image_labels[
                                                                          i * modality_positions.size(1) + j, :length]

            # DEBUG: use module hook on diff_proj to capture LLM output gradient
            if getattr(self, '_debug_capture_llm_intermediates', False):
                if hasattr(self, 'diff_proj'):
                    def _s2_diff_proj_bwd(module, grad_input, grad_output):
                        if grad_input and grad_input[0] is not None:
                            self._debug_llm_output_grad_norm = grad_input[0].detach().float().norm().item()
                        if grad_output and grad_output[0] is not None:
                            self._debug_diff_proj_grad_output_norm = grad_output[0].detach().float().norm().item()
                    self._diff_proj_hook_handle = self.diff_proj.register_full_backward_hook(_s2_diff_proj_bwd)

            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                output_hidden_states=output_hidden_states
            )

            logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

            if getattr(self, '_debug_capture_llm_output', False):
                self._debug_llm_normed_output = last_hidden_states.detach()

            # diffusion head to predict vector fields
            _dbg_dh = getattr(self, '_debug_capture_diffhead', False)
            if hasattr(self, 'diff_proj'):
                last_hidden_states = self.diff_proj(last_hidden_states)
            if _dbg_dh:
                self._dbg_dh_after_proj = last_hidden_states.detach()
            position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)
            for li, layer in enumerate(self.diffusion_head_a):
                last_hidden_states = layer(hidden_states=last_hidden_states,
                                           adaln_input=time_embeds,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           modality_positions=modality_positions,
                                           )[0]
                if _dbg_dh:
                    setattr(self, f'_dbg_dh_after_layer{li}', last_hidden_states.detach())
            v_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)
            if _dbg_dh:
                self._dbg_dh_v_pred = v_pred.detach()
            # print(f"DEBUG: v_pred shape: {v_pred.shape}, labels shape: {new_image_labels.shape}, mask sum:
            # {image_masks.sum()}")
            # [:v_pred.shape[0]] is the valid image labels (special case for interleaved data training)
            if text_labels is not None and image_labels is not None:
                loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
                loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
                return logits, loss_ntp, loss_flow

            elif image_labels is not None:
                loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
                return logits, loss_flow

            elif text_labels is not None:
                loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
                return logits, loss_ntp

            else:
                v_pred_ = []
                num_imgs = 0
                for i, modality_batch in enumerate(modality_positions):
                    for j, (offset, length) in enumerate(modality_batch):
                        if length == 0:
                            break
                        else:
                            v_pred_.append(v_pred[i, offset:offset + length])
                            num_imgs += 1
                v_pred_ = torch.stack(v_pred_)

                # remove the time embedding
                if self.config.add_time_embeds:
                    v_pred_ = v_pred_[:, 1:, :]

                # unpatchify
                v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

                if T == 0:
                    v_pred_ = rearrange(v_pred_, 'i j k -> i k j')
                    v_pred_ = v_pred_.reshape(num_imgs, c, h, w)
                else:
                    v_pred_ = rearrange(v_pred_, 'b t l c -> b c t l')
                    v_pred_ = v_pred_.reshape(num_imgs, c, T, h, w)

                # specific for image-to-video generation
                if first_frame_as_cond:
                    # zero the v-prediction for the first frame
                    v_pred_ = torch.cat([
                        torch.zeros_like(v_pred_)[:, :, :1],
                        v_pred_[:, :, 1:]
                    ], dim=2)

                # specific for mixed-modality generation
                if only_denoise_last_image:
                    if guidance_scale > 0:
                        v_pred_cond, v_pred_uncond = torch.chunk(v_pred_, 2)

                        v_pred_cond = torch.cat([
                            torch.zeros_like(v_pred_cond)[:-1, :, :],
                            v_pred_cond[-1:, :, :]
                        ], dim=0)

                        v_pred_uncond = torch.cat([
                            torch.zeros_like(v_pred_uncond)[:-1, :, :],
                            v_pred_uncond[-1:, :, :]
                        ], dim=0)

                        v_pred_ = torch.cat([v_pred_cond, v_pred_uncond], dim=0)
                    else:
                        v_pred_ = torch.cat([
                            torch.zeros_like(v_pred_)[:-1, :, :],
                            v_pred_[-1:, :, :]
                        ], dim=0)

                return logits, v_pred_

    @torch.no_grad()
    def t2i_generate(
            self,
            image_latents=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            max_seq_len=None,
            guidance_scale=0.0,
            **kwargs,
    ):
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            v_cond, v_uncond = torch.chunk(v, 2)
            # If kwargs contains _return_raw_cfg=True, return the raw cond and uncond so transport.py handles CFG
            # uniformly
            return torch.cat([v_cond, v_uncond], dim=0)
            # if kwargs.get("_return_raw_cfg", False):
            #     return torch.cat([v_cond, v_uncond], dim=0)
            # v = v_uncond + guidance_scale * (v_cond - v_uncond)
            # return torch.cat([v, v], dim=0)

        else:
            if t.shape[-1] != text_tokens.shape[0]:
                t[:-1] = 1.0
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            return v

    @torch.no_grad()
    def mmu_generate(
            self,
            input_embeds=None,
            attention_mask=None,
            max_new_tokens=100,
            temperature=1.0,
            top_k=None,
            eos_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        device = input_embeds.device
        dtype = input_embeds.dtype
        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask.to(dtype))['logits']

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            idx_next_embeds = self.llm.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1).to(dtype)

            if eos_token is not None and idx_next.cpu() == eos_token:
                break

        return result

    @torch.no_grad()
    def lm_generate(
            self,
            input_ids=None,
            attention_mask=None,
            tokenizer=None,
            max_new_tokens=100,
            boi_token=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            device=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_tokens = input_ids
        output_tokens = []
        for _ in range(max_new_tokens):
            # Generate the next token
            outputs = self.llm(
                input_ids=torch.tensor([generated_tokens]).to(device),
                attention_mask=attention_mask,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the next token to the sequence
            generated_tokens.append(next_token)
            output_tokens.append(next_token)

            # Check if the `eos_token_id` is generated
            if next_token == tokenizer.eos_token_id or next_token == boi_token:  # EOS token ID
                break

            # Decode the generated tokens
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        return generated_text

    @torch.no_grad()
    def mm_generate(
            self,
            input_ids=None,
            image_latents=None,
            t=None,
            modality_positions=None,
            attention_mask=None,
            tokenizer=None,
            max_new_tokens=100,
            boi_token=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            device=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_tokens = input_ids
        output_tokens = []
        if attention_mask is not None and type(attention_mask) == BlockMask:
            raise NotImplementedError

        for _ in range(max_new_tokens):
            # Generate the next token
            logits = self.forward_und_only(
                text_tokens=torch.tensor([generated_tokens]).to(device),
                image_latents=image_latents,
                t=t,
                attention_mask=attention_mask,
                modality_positions=modality_positions,
            )

            next_token_logits = logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the next token to the sequence
            generated_tokens.append(next_token)
            output_tokens.append(next_token)

            # Check if the `eos_token_id` is generated
            if next_token == tokenizer.eos_token_id or next_token == boi_token:  # EOS token ID
                break

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(next_token_logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L + 1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b.to(image_latents.dtype)

            # Decode the generated tokens
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        return generated_text
