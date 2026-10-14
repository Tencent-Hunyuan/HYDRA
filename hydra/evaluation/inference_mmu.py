# coding=utf-8
# Copyright 2025 NUS Show Lab, HuggingFace.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0
"""HYDRA multimodal-understanding (MMU) inference.

Answers questions about one or more images:

    python inference_mmu.py \
        config=configs/mmu.yaml \
        mmu_image_path=./test.jpg \
        question='Describe this image.'

    # multiple questions separated by ' *** '
    python inference_mmu.py \
        config=configs/mmu.yaml \
        mmu_image_path=./test.jpg \
        question='What is in this image? *** What color is the sky?'

The chat template must stay identical to the one used during training
(default_system_prompt in the training YAML).
If mmu_image_path is a directory, every image inside is processed.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from PIL import Image
from tqdm import tqdm
import logging

from models import HydraQwen2_5, omni_attn_mask_naive
from models.hydra_tok.modeling_hydra_tok import Gen_ViT, Und_ViT
from models.hydra_tok.configuration_hydra_tok import HydraTokConfig
from models.misc import get_text_tokenizer
from tools.utils import get_config, get_llm_name, set_seed
from datasets.utils import image_transform_hydra_tok

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Must match default_system_prompt in the training config.
SYSTEM_PROMPT = "system\nYou are a helpful assistant.<|im_end|>"

# ImageNet normalization (the same statistics the tokenizer was trained with).
DATA_NORM = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def load_state_dict(model_path):
    """Load a single or sharded pytorch_model checkpoint into a state dict."""
    from collections import OrderedDict
    if model_path.endswith(('.bin', '.pth', '.pt')):
        return torch.load(model_path, map_location="cpu")
    files = os.listdir(model_path)
    sharded = sorted(f for f in files
                     if f.startswith('pytorch_model-') and f.endswith('.bin') and 'of' in f)
    state_dict = OrderedDict()
    if sharded:
        print(f"Loading sharded weights from {model_path} ({len(sharded)} files)")
        for f in sharded:
            state_dict.update(torch.load(os.path.join(model_path, f), map_location="cpu"))
        return state_dict
    single = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(single):
        print(f"Loading weights from {single}")
        return torch.load(single, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {model_path}")


if __name__ == '__main__':
    config = get_config()
    if config.training.seed is not None:
        set_seed(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # flash attention requires fp16/bf16 inputs
    weight_type = torch.bfloat16

    # ---------------- HYDRA-TOK vision tower ----------------
    hydra_tok_config = HydraTokConfig.from_pretrained(config.model.hydra_tok_model.hydra_tok_config)
    hydra_tok_config.gen_layers = getattr(config.model.hydra_tok_model, 'gen_layers', 12)
    hydra_tok_config.und_layers = getattr(config.model.hydra_tok_model, 'und_layers', 12)
    hydra_tok_config.latent_stats_path = None  # not needed for inference

    vae_model = Gen_ViT(hydra_tok_config).to(device).to(weight_type)
    gen_vit_weights_path = os.path.join(config.model.hydra_tok_model.pretrained_model_path,
                                        "gen_vit_weights.pth")
    if os.path.exists(gen_vit_weights_path):
        vae_model.load_state_dict(torch.load(gen_vit_weights_path, map_location="cpu"), strict=False)
    else:
        logger.warning(f"Gen_ViT weights not found at {gen_vit_weights_path}, using random init")

    vae_model.register_buffer(
        'processor_mean', torch.tensor(DATA_NORM["mean"]).view(1, 3, 1, 1).to(device).to(weight_type))
    vae_model.register_buffer(
        'processor_std', torch.tensor(DATA_NORM["std"]).view(1, 3, 1, 1).to(device).to(weight_type))
    vae_model.eval()

    und_vit = Und_ViT(hydra_tok_config).to(device).to(weight_type)
    und_vit_weights_path = os.path.join(config.model.hydra_tok_model.pretrained_model_path,
                                        "und_vit_weights.pth")
    if os.path.exists(und_vit_weights_path):
        und_vit.load_state_dict(torch.load(und_vit_weights_path, map_location="cpu"), strict=False)
    else:
        logger.warning(f"Und_ViT weights not found at {und_vit_weights_path}, using random init")
    und_vit.eval()

    # ---------------- HYDRA model ----------------
    text_tokenizer, hydra_token_ids = get_text_tokenizer(config.model.llm.llm_model_path,
                                                          add_hydra_tokens=True,
                                                          return_hydra_token_ids=True,
                                                          llm_name=get_llm_name(config.model.llm.llm_model_path))
    config.model.llm.llm_vocab_size = len(text_tokenizer)

    model = HydraQwen2_5(**config.model.llm).to(device).to(weight_type)
    # und_vit must be attached BEFORE load_state_dict so that the
    # und_vit.* keys in the checkpoint have somewhere to land.
    model.set_und_vit(und_vit)
    state_dict = load_state_dict(config.model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if 'position_ids' not in k]
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:10]} ...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]} ...")
    model.eval()

    num_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens

    top_k = 1
    max_new_tokens = 1024

    if not (str(config.mmu_image_path).endswith('.jpg') or str(config.mmu_image_path).endswith('.png')):
        file_list = [os.path.join(config.mmu_image_path, fn) for fn in os.listdir(config.mmu_image_path)]
    else:
        file_list = [config.mmu_image_path]

    questions = str(config.question).split(' *** ')

    sys_prompt_ids = text_tokenizer(SYSTEM_PROMPT, add_special_tokens=False)['input_ids']
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
    role_b = text_tokenizer("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

    for step, image_path in enumerate(tqdm(file_list)):
        image_ori = Image.open(image_path).convert("RGB")
        v_tsr = image_transform_hydra_tok(image_ori,
                                          resolution=config.dataset.preprocessing.resolution).unsqueeze(0)
        v_tsr = v_tsr.to(device).to(weight_type)  # [1, 3, H, W]

        with torch.no_grad():
            # 1. tokenize to latents, 2. understanding branch, 3. project to LLM dims
            image_latents = vae_model.encode(v_tsr)                 # [1, C, h, w]
            semantic_feats = model.und_vit(image_latents=image_latents)
            image_embeds = model.semantic_proj(semantic_feats)       # [1, N, D]

        for question in questions:
            # training chat format: the question is preceded by a newline
            prompt = question if question.startswith("\n") else "\n" + question
            input_ids = text_tokenizer(prompt, add_special_tokens=False).input_ids
            text_tokens_a = torch.tensor(
                [hydra_token_ids['bos_id']] + sys_prompt_ids + role_a).to(device)[None, :]
            text_tokens_b = torch.tensor(
                [hydra_token_ids['boi_id'], hydra_token_ids['eoi_id']] + input_ids + role_b
            ).to(device)[None, :]

            text_embeds_a = model.llm.model.embed_tokens(text_tokens_a)
            text_embeds_b = model.llm.model.embed_tokens(text_tokens_b)

            if config.model.llm.add_time_embeds:
                time_embeds = model.time_embed(torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype)
                if hasattr(model, 'time_embed_proj'):
                    time_embeds = model.time_embed_proj(time_embeds)
                input_embeds = torch.cat([
                    text_embeds_a,
                    text_embeds_b[:, :1],  # [BOI]
                    time_embeds,
                    image_embeds,
                    text_embeds_b[:, 1:]   # [EOI] + question + role_b
                ], dim=1).to(weight_type)
                modality_positions = torch.tensor(
                    [text_tokens_a.shape[1] + 2, num_image_tokens],
                    dtype=torch.long)[None, None, :].to(device)
            else:
                input_embeds = torch.cat([
                    text_embeds_a,
                    text_embeds_b[:, :1],  # [BOI]
                    image_embeds,
                    text_embeds_b[:, 1:]   # [EOI] + question + role_b
                ], dim=1).to(weight_type)
                modality_positions = torch.tensor(
                    [text_tokens_a.shape[1] + 1, num_image_tokens],
                    dtype=torch.long)[None, None, :].to(device)

            attention_mask = omni_attn_mask_naive(
                B=input_embeds.size(0),
                LEN=input_embeds.size(1),
                modalities=modality_positions,
                device=device, inverted=True
            ).to(input_embeds.dtype)

            with torch.no_grad():
                cont_toks = model.mmu_generate(input_embeds=input_embeds,
                                                attention_mask=attention_mask,
                                                top_k=top_k,
                                                max_new_tokens=max_new_tokens,
                                                eos_token=text_tokenizer.eos_token_id)
            output_tokens = torch.tensor(cont_toks).unsqueeze(0)
            text = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            print(f'\nUser: {question}\nHYDRA: {text[0]}\n{"-" * 60}')
