# coding=utf-8
# Copyright 2025 NUS Show Lab.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Any, Callable, List, Optional, Union
import math
import random

import torch
from torchvision.datasets.folder import default_loader

from datasets.utils import resize_and_pad_image, to_tensor_and_normalize
# from utils import image_transform_vae, image_transform_hydra_tok, resize_and_pad_image, to_tensor_and_normalize

import os
import json
import yaml
import torch.utils.data as data
import copy
from PIL import Image
IGNORE_INDEX = -100


def rank0_print(*args, **kwargs):
    """Only print on rank 0 for distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args, **kwargs)


class MMUDataset(data.Dataset):
    """
    MMU Dataset supporting YAML configuration for loading multiple datasets.

    Supports two modes:
    1. YAML mode: annotation_path points to a YAML file with 'datasets' key
    2. JSON mode (legacy): annotation_path points to a JSON file directly

    YAML format example:
    ```yaml
    datasets:
      - json_path: /path/to/data1.json
        sampling_strategy: "all"  # or "first:10%", "random:1000", etc.
      - json_path: /path/to/data2.json
        sampling_strategy: "first:50%"
    ```
    """
    def __init__(
            self,
            root: str,
            text_tokenizer,
            max_seq_len=1024,
            source_max_len=512,
            target_max_len=512,
            image_size=768,
            latent_height=24,
            latent_width=24,
            num_image_tokens=576,
            cond_dropout_prob=0.1,
            loader: Callable[[str], Any] = default_loader,
            annotation_path="",
            default_system_prompt="system\nYou are a helpful assistant.<|im_end|>",
            stage='pre-training',
            hydra_token_ids=None,
            filter_no_image=True,  # new: filter out records without images
):

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = hydra_token_ids['bos_id']
        self.eos_id = hydra_token_ids['eos_id']
        self.boi_id = hydra_token_ids['boi_id']
        self.eoi_id = hydra_token_ids['eoi_id']
        self.img_pad_id = hydra_token_ids['img_pad_id']
        self.img_id = hydra_token_ids['img_id']
        self.data_type = "mmu"
        self.max_seq_len = max_seq_len
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        # 4 for bos, eos, boi, and eoi tokens
        self.max_text_len = max_seq_len - num_image_tokens - 4


        self.stage = stage
        if self.stage.startswith('pre-training'):
            self.default_system_prompt = None
        else:
            self.default_system_prompt = default_system_prompt

        self.root = root
        self.loader = loader
        self.filter_no_image = filter_no_image
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Load data: supports YAML or JSON format
        self.samples = self._load_data(annotation_path)

        # Filter out records without images
        # if self.filter_no_image:
        #     original_count = len(self.samples)
        #     self.samples = [
        #         record for record in self.samples
        #         if record.get("image") is not None
        #     ]
        #     new_count = len(self.samples)
        #     if original_count != new_count:
        #         rank0_print(f"[Data Filtering] Removed {original_count - new_count} records without 'image'. "
        #                     f"Dataset size: {original_count} -> {new_count}")

        rank0_print(f"MMUDataset loaded. Total {len(self.samples)} samples!")

    def _load_data(self, annotation_path: str) -> List[dict]:
        """
        Load data from YAML config or JSON file.

        Args:
            annotation_path: Path to YAML config file or JSON annotation file

        Returns:
            List of data samples
        """
        if annotation_path.endswith('.yaml') or annotation_path.endswith('.yml'):
            return self._load_from_yaml(annotation_path)
        elif annotation_path.endswith('.json') or annotation_path.endswith('.jsonl'):
            return self._load_from_json(annotation_path)
        else:
            # Try to detect the file type
            try:
                with open(annotation_path, 'r') as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '{' or first_char == '[':
                        return self._load_from_json(annotation_path)
                    else:
                        return self._load_from_yaml(annotation_path)
            except (OSError, ValueError) as e:
                raise ValueError(f"Cannot determine file type for {annotation_path}: {e}")

    def _load_from_yaml(self, yaml_path: str) -> List[dict]:
        """
        Load data from YAML configuration file.

        YAML format:
        ```yaml
        datasets:
          - json_path: /path/to/data.json
            sampling_strategy: "all"  # Options: "all", "first:N", "first:N%", "end:N", "random:N"
        ```
        """
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        datasets_config = yaml_data.get('datasets', [])
        all_samples = []

        for dataset_cfg in datasets_config:
            json_path = dataset_cfg.get('json_path')
            sampling_strategy = dataset_cfg.get('sampling_strategy', 'all')

            if not json_path:
                rank0_print(f"[Warning] Skipping dataset config without 'json_path': {dataset_cfg}")
                continue

            # Load a single JSON file
            cur_data = self._load_from_json(json_path)

            # Apply the sampling strategy
            cur_data = self._apply_sampling_strategy(cur_data, sampling_strategy)

            rank0_print(f"Loaded {len(cur_data)} samples from {json_path} (strategy: {sampling_strategy})")
            all_samples.extend(cur_data)

        return all_samples

    def _load_from_json(self, json_path: str) -> List[dict]:
        """Load data from JSON or JSONL file."""
        if json_path.endswith('.jsonl'):
            samples = []
            with open(json_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            return samples
        else:
            with open(json_path, 'r') as f:
                return json.load(f)

    def _apply_sampling_strategy(self, data: List[dict], strategy: str) -> List[dict]:
        """
        Apply sampling strategy to data.

        Supported strategies:
        - "all": Use all data
        - "first:N": Use first N samples
        - "first:N%": Use first N% of samples
        - "end:N": Use last N samples
        - "end:N%": Use last N% of samples
        - "random:N": Randomly sample N samples
        - "random:N%": Randomly sample N% of samples
        """
        if strategy == "all":
            return data

        if ":" not in strategy:
            rank0_print(f"[Warning] Unknown sampling strategy '{strategy}', using 'all'")
            return data

        strategy_type, sampling_number = strategy.split(":", 1)

        # Compute the sample count
        if "%" in sampling_number:
            percentage = int(sampling_number.replace("%", ""))
            sample_count = math.ceil(percentage * len(data) / 100)
        else:
            sample_count = int(sampling_number)

        # Cap the sample count at the total data size
        sample_count = min(sample_count, len(data))

        if strategy_type == "first":
            return data[:sample_count]
        elif strategy_type == "end":
            return data[-sample_count:]
        elif strategy_type == "random":
            if sample_count >= len(data):
                return data
            indices = random.sample(range(len(data)), sample_count)
            return [data[i] for i in sorted(indices)]
        else:
            rank0_print(f"[Warning] Unknown strategy type '{strategy_type}', using 'all'")
            return data

    def __len__(self):
        return len(self.samples)

    def format_multi_sequence_und_qwen2_5(self, sources, targets, ignore_question=True):

        text_tokens = []
        text_labels = []
        modality_positions = []

        if not self.stage.startswith('pre-training'):
            default_system_prompt = self.text_tokenizer(
                self.default_system_prompt,
                max_length=100,
                truncation=True,
                add_special_tokens=False,
)['input_ids']
            role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
            role_b = self.text_tokenizer("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

        cur_len = 1  # <|begin_of_text|>
        for source_ids, target_ids in zip(sources, targets):
            if not self.stage.startswith('pre-training'):
                # Fix: end the user turn with <|im_end|> instead of eos_id
                source_ids = role_a + source_ids + role_b
            # NOTE: only support one image
            if cur_len == 1 and not self.stage.startswith('pre-training'):
                source_ids = default_system_prompt + source_ids
            if self.img_id in source_ids:
                image_id_index = source_ids.index(self.img_id)
                source_ids = source_ids[:image_id_index] + \
                             [self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id] \
                             + source_ids[image_id_index + 1:]

                # +1 for one <|im_start|> token
                modality_positions.append((cur_len + image_id_index + 1, self.num_image_tokens))

            # Fix: ensure every assistant reply ends with eos_id (used for loss computation)
            if not self.stage.startswith('pre-training'):
                # Append <|im_end|> to the end of the target so the model learns to stop its reply
                target_ids = target_ids + [self.eos_id]  # <|im_end|>

            text_tokens.extend(source_ids + target_ids)
            # ignore question - note that target_ids already contains eos_id
            if ignore_question:
                text_labels.extend(
                    [IGNORE_INDEX for _ in range(len(source_ids))] + target_ids
)
            else:
                text_labels.extend(source_ids + target_ids)

            cur_len = len(text_tokens) + 1

        text_labels = [IGNORE_INDEX] + text_labels
        text_tokens = [self.bos_id] + text_tokens
        text_labels = text_labels + [IGNORE_INDEX] * (self.max_seq_len - len(text_labels))
        text_tokens = text_tokens + [self.pad_id] * (self.max_seq_len - len(text_tokens))

        text_tokens = torch.tensor(text_tokens)[:self.max_seq_len]
        text_labels = torch.tensor(text_labels)[:self.max_seq_len]

        # Change 3: explicitly mark this as a sequence with no valid image region
        if len(modality_positions) == 0:
            modality_positions = torch.tensor([[0, 0]])
        else:
            modality_positions = torch.tensor(modality_positions)

        text_mask = torch.where((text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
        image_mask = torch.where(text_tokens == self.img_pad_id,
                                 torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]

            # Change 1: determine whether an image exists and is not None
            has_image = 'image' in sample and sample['image'] is not None
            # print(f"image: {sample['image']}")
            if has_image:
                try:
                    image_path = os.path.join(self.root, sample['image'])
                    # Check whether the file exists
                    if not os.path.exists(image_path):
                        rank0_print(f"[Warning] Image file not found: {image_path}, using dummy image")
                        image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
                    elif os.path.exists(sample['image']):
                        image = self.loader(sample['image']).convert('RGB')
                    else:
                        # Try to load the image; fall back to a dummy image on failure
                        image = self.loader(image_path).convert('RGB')
                except (OSError, ValueError, RuntimeError) as img_error:
                    # The image file is corrupted or another error occurred; use a dummy image but keep the text
                    # rank0_print(f"[Warning] Failed to load image for sample {idx} (path: {sample.get('image',
                    # 'N/A')}): {img_error}, using dummy image")
                    # image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
                    rank0_print(
                        f"[Warning] Failed to load image for sample {idx} "
                        f"(path: {sample.get('image', 'N/A')}): {img_error}, retrying with next sample")
                    return self.__getitem__((idx + 1) % len(self.samples))
                    # rank0_print(f"[Warning] Failed to load image for sample {idx} (path: {sample.get('image',
                    # 'N/A')}): {img_error}, skipping this sample")
                    # raise RuntimeError(f"Failed to load image for sample {idx}: {img_error}") from img_error
            else:
                # Change 2: in text-only mode, generate an all-black dummy image to keep tensor shapes consistent
                image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

            # Unified transform handling
            image = resize_and_pad_image(image, target_resolution=(self.image_size, self.image_size))
            image = to_tensor_and_normalize(image, mean=self.mean, std=self.std)

            conversation = []
            is_first_user = True  # flag for the first user turn
            for conv in sample['conversations']:
                if conv['from'] == 'human':
                    # Unified handling: if there is an image, remove all <image> tags first, then add one at the start
                    # of the first turn
                    if has_image:
                        # Remove all <image> and \n<image> tags
                        conv['value'] = conv['value'].replace('<image>', '')
                        # Add <image> only at the start of the first user turn
                        if is_first_user:
                            conv['value'] = '<image>\n' + conv['value']
                            is_first_user = False

                    # remove question from the first stage (in pre-training, keep only <image>)
                    if self.stage.startswith('pre-training'):
                        if '<image>' in conv['value']:
                            conv['value'] = '<image>'

                    conversation.append({"role": "user", "content": conv['value']})
                else:
                    conversation.append({"role": "assistant", "content": conv['value']})

            sources = [f"{conv['content']}" for conv in conversation if conv["role"] == "user"]
            # eos_token is no longer added here; it is handled uniformly in format_multi_sequence_und_qwen2_5
            targets = [f"{conv['content']}" for conv in conversation if
                       conv["role"] == "assistant"]

            # Tokenize
            # import ipdb
            # ipdb.set_trace()
            sources = [self.text_tokenizer(
                source,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False
).input_ids for source in sources]

            targets = [self.text_tokenizer(
                target,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False
).input_ids for target in targets]

            text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                self.format_multi_sequence_und_qwen2_5(sources, targets)

            ret = {'text_tokens': text_tokens, 'text_labels': text_labels, 'images': image,
                   'modality_positions': modality_positions, 'text_masks': text_mask, 'data_type': self.data_type,
                   'image_masks': image_mask, 'texts': self.text_tokenizer.batch_decode(text_tokens)}

            return ret

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            rank0_print(f"[Warning] Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type'):
                batched[k] = torch.stack(v, dim=0)

        return batched

# if __name__ == '__main__':
#     import sys
#     sys.path.insert(0, "/path/to/show-o2")
#     from torch.utils.data import DataLoader
#     from models.misc import get_text_tokenizer

#     text_tokenizer, hydra_token_ids = get_text_tokenizer(
#         "Qwen/Qwen2.5-1.5B-Instruct",
#         add_hydra_tokens=True,
#         return_hydra_token_ids=True,
#         llm_name="qwen2_5"
#)

#     # Test mode selection: yaml or json
#     USE_YAML_MODE = True

#     if USE_YAML_MODE:
#         # Mode 1: load multiple datasets from a YAML config file
#         print("=" * 80)
#         print("Testing YAML mode: Loading multiple datasets from YAML config")
#         print("=" * 80)

#         dataset_mmu = MMUDataset(
# root="/path/to/dataset/Understanding/LLaVA-NeXT-Data/llava_next_raw_format",  # in YAML mode, each JSON file path is
# absolute
#             image_size=448,
#             num_image_tokens=1024,
#             max_seq_len=1560,
#             annotation_path="./datasets/example_mmu_datasets.yaml",
#             text_tokenizer=text_tokenizer,
#             stage='tuning',
#             hydra_token_ids=hydra_token_ids,
#             filter_no_image=True,  # filter out records without images
#)
#     else:
#         # Mode 2: use a JSON file directly (backward compatible)
#         print("=" * 80)
#         print("Testing JSON mode: Loading from single JSON file")
#         print("=" * 80)

#         dataset_mmu = MMUDataset(
#             root="/path/to/dataset/Understanding/LLaVA-NeXT-Data/llava_next_raw_format",
#             image_size=448,
#             num_image_tokens=1024,
#             max_seq_len=1560,
#             annotation_path="/path/to/LLaVA-NeXT-Data/llava_next_raw_format_processed.json",
#             text_tokenizer=text_tokenizer,
#             stage='tuning',
#             hydra_token_ids=hydra_token_ids,
#             filter_no_image=True,
#)

#     print(f"\nTotal samples: {len(dataset_mmu)}")

#     train_dataloader = DataLoader(
#         dataset_mmu,
#         batch_size=16,
#         collate_fn=dataset_mmu.collate_fn,
#         shuffle=True,
#         num_workers=4,  # fewer workers for faster testing
#         drop_last=True
#)

#     def show_model_inputs(batch, hydra_token_ids, text_tokenizer, sample_idx=0):
#         """Simple display of the model inputs from the dataloader"""
#         tokens = batch['text_tokens'][sample_idx]
#         labels = batch['text_labels'][sample_idx]
#         decoded_tokens = batch['texts'][sample_idx]

#         # Fix: get pad_id from text_tokenizer, not from hydra_token_ids
#         pad_id = text_tokenizer.pad_token_id

#         print("\n" + "="*60)
#         print("Model input format")
#         print("="*60)

#         # 1. Basic info
#         print(f"\n[Input data shapes]")
#         print(f"  text_tokens: {batch['text_tokens'].shape}")
#         print(f"  text_labels: {batch['text_labels'].shape}")
#         print(f"  images: {batch['images'].shape}")
#         print(f"  modality_positions: {batch['modality_positions'].shape}")
#         print(f"  text_masks: {batch['text_masks'].shape}")
#         print(f"  image_masks: {batch['image_masks'].shape}")

#         # 2. Token sequence overview
#         print(f"\n[Token sequence overview (sample {sample_idx})]")

#         # Fix: use the correct pad_id
#         non_pad_mask = tokens != pad_id

#         non_pad_tokens = tokens[non_pad_mask]
#         non_pad_labels = labels[non_pad_mask]
#         # decoded_tokens is already a list of strings, no need to decode again; index by position
#         # Note: this assumes len(decoded_tokens) equals len(tokens) (including padding)
#         # If decoded_tokens is the result of tokenizer.batch_decode, it usually does not include padding
#         # (depending on skip_special_tokens); for safety, only a simple length check is done here

#         # Statistics
#         loss_tokens = (non_pad_labels != -100).sum().item()
#         ignore_tokens = (non_pad_labels == -100).sum().item()
#         image_tokens = (non_pad_tokens == hydra_token_ids['img_pad_id']).sum().item()

#         print(f"  Total valid tokens: {len(non_pad_tokens)}")
#         print(f"  Loss computed on: {loss_tokens} tokens (what the model learns)")
#         print(f"  Ignored: {ignore_tokens} tokens (system prompt / user questions)")
#         print(f"  Image placeholders: {image_tokens} tokens")

#         # 3. Show the token sequence (simplified)
#         print(f"\n[Token sequence (first 100 valid tokens)]")
#         print("  Format: [position] token text | loss status")
#         print("  " + "-"*56)

#         count = 0
#         for i in range(len(tokens)):
#             if count >= 100000:
#                 print("... (more tokens)")
#                 break

#             # Fix: use the correct pad_id
#             if tokens[i] == pad_id:
#                 continue

#             tid = tokens[i].item()
#             lid = labels[i].item()

#             # Safely get the token text
#             if i < len(decoded_tokens):
#                 token_str = decoded_tokens[i]
#             else:
#                 # If batch['texts'] is a whole-sentence string instead of a token list,
#                 # a single token needs to be decoded here
#                 token_str = text_tokenizer.decode([tid])

#             # Simplified display of image placeholders
#             if tid == hydra_token_ids['img_pad_id']:
#                 # This is an image token
#                 if i == 0 or tokens[i-1].item() != hydra_token_ids['img_pad_id']:
#                     # This is the first of a run of image tokens
#                     print(f"  [{i:4d}] <image_pad>... ({image_tokens} image tokens)")
#                 count += 1
#                 continue

#             # The logic here: if the previous token was an image token and the current one is too,
#             # skip printing (already handled above). For simplicity, the if above covers the
#             # case where the current token is img_pad_id.

#             status = "Loss" if lid != -100 else "ignored"
#             # Handle newlines so the print format is not broken
#             token_display = token_str.replace('\n', '\\n')
#             token_display = token_display[:20] if len(token_display) <= 20 else token_display[:17] + "..."

#             print(f"  [{i:4d}] {token_display:<20} | {status}")
#             count += 1

#         # 4. Image position info
#         print(f"\n[Image position info]")
#         modality_pos = batch['modality_positions'][sample_idx]
#         if len(modality_pos.shape) > 1 and modality_pos.shape[0] > 0:
#             for j, pos_info in enumerate(modality_pos):
#                 start_pos, num_tokens = pos_info[0].item(), pos_info[1].item()
#                 if num_tokens > 0:
#                     print(f"  Image {j+1}: start position={start_pos}, placeholder count={num_tokens}")
#         else:
#             print("  No image or format mismatch")

#         print("="*60 + "\n")

#     from tqdm import tqdm
#     for i, batch in enumerate(tqdm(train_dataloader)):
#         print(f"\nBatch {i}:")
#         print(f"  text_tokens shape: {batch['text_tokens'].shape}")
#         print(f"  images shape: {batch['images'].shape}")

#         # Check image token related info
#         print(f"  modality_positions: {batch['modality_positions'][0]}")
#         print(f"  text_masks shape: {batch['text_masks'].shape}, sum: {batch['text_masks'][0].sum().item()}")
#         print(f"  image_masks shape: {batch['image_masks'].shape}, sum: {batch['image_masks'][0].sum().item()}")

#         # Verify the image_pad token count is correct
#         text_tokens = batch['text_tokens'][0]
#         img_pad_count = (text_tokens == hydra_token_ids['img_pad_id']).sum().item()
#         print(f"  image_pad token count: {img_pad_count} (expected: 1024)")

#         # Check whether boi and eoi tokens exist
#         has_boi = (text_tokens == hydra_token_ids['boi_id']).any().item()
#         has_eoi = (text_tokens == hydra_token_ids['eoi_id']).any().item()
#         print(f"  has boi token: {has_boi}, has eoi token: {has_eoi}")

#         # Show the model inputs
#         if i == 0:
#             # Pass in text_tokenizer
#             show_model_inputs(batch, hydra_token_ids, text_tokenizer, sample_idx=0)

#         if i >= 2:
#             break

#     print("\n Test completed successfully!")


def calculate_sample_length(args):
    """
    Helper function: compute the final token length of a single sample
    (mimics the logic of __getitem__ but does not load the image)
    """
    sample, tokenizer, config = args

    # 1. Base token count (system prompt + special tokens)
    # <|im_start|>system...<|im_end|> + <|im_start|>user...
    # Rough estimate: System(15) + Special(~10) = 25
    base_overhead = 25

    # 2. Image tokens
    img_tokens = 0
    if sample.get('image') is not None:
        img_tokens = config['num_image_tokens'] + 2  # +2 for boi/eoi

    # 3. Text tokens
    conversation_text = ""
    for conv in sample['conversations']:
        val = conv['value']
        # Mimic the replacement logic in the code
        if '<image>' in val:
            val = val.replace('<image>', '')  # images are counted separately, remove them here
        if '\n<image>' in val:
            val = val.replace('\n<image>', '')

        conversation_text += val + " "  # simple concatenation for estimation

    # Tokenize
    # Note: for speed, the text is encoded once after concatenation; the difference from
    # encoding segment by segment is negligible
    text_ids = tokenizer.encode(conversation_text, add_special_tokens=False)

    total_len = len(text_ids) + img_tokens + base_overhead
    return total_len


def analyze_dataset_lengths(dataset, num_workers=16):
    import numpy as np
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    import multiprocessing

    print(f"\nStarting token length statistics over {len(dataset)} samples...")
    print(f"Using {num_workers} processes for tokenization...")

    # Prepare the arguments
    config = {
        'num_image_tokens': dataset.num_image_tokens,
    }

    # Build the task list
    tasks = [(s, dataset.text_tokenizer, config) for s in dataset.samples]

    lengths = []

    # Parallel processing
    # Note: the tokenizer may need the environment variable TOKENIZERS_PARALLELISM=false in multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use chunksize to improve performance
        results = list(tqdm(executor.map(calculate_sample_length, tasks), total=len(tasks), unit="sample"))
        lengths = results

    lengths = np.array(lengths)

    # === Statistics report ===
    print("\n" + "="*40)
    print("Dataset Token Length Statistics Report")
    print("="*40)
    print(f"Total samples: {len(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean: {np.mean(lengths):.2f}")
    print(f"Median: {np.median(lengths):.2f}")
    print("-" * 40)
    print(f"P75  (75% coverage): {np.percentile(lengths, 75):.2f}")
    print(f"P90  (90% coverage): {np.percentile(lengths, 90):.2f}")
    print(f"P95  (95% coverage): {np.percentile(lengths, 95):.2f}")
    print(f"P99  (99% coverage): {np.percentile(lengths, 99):.2f}")
    print(f"P99.9(99.9%):       {np.percentile(lengths, 99.9):.2f}")
    print("="*40)

    # === Truncation analysis ===
    current_max_len = dataset.max_seq_len
    truncated_count = np.sum(lengths > current_max_len)
    print(f"\nCurrent setting: max_seq_len = {current_max_len}")
    print(f"Samples that will be truncated: {truncated_count} ({truncated_count/len(lengths)*100:.2f}%)")

    # === Plot the histogram ===
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=current_max_len, color='r', linestyle='--', label=f'Current Max Len ({current_max_len})')
        plt.title('Dataset Token Length Distribution')
        plt.xlabel('Token Length')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)

        save_path = "dataset_length_dist.png"
        plt.savefig(save_path)
        print(f"\nLength distribution histogram saved to: {save_path}")
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        print(f"Plotting failed (matplotlib may not be installed): {e}")

# if __name__ == '__main__':
#     import sys
#     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from models.misc import get_text_tokenizer

#     # 1. Load the tokenizer
#     model_path = "Qwen/Qwen2.5-1.5B-Instruct"
#     print(f"Loading tokenizer from {model_path}...")

#     text_tokenizer, hydra_token_ids = get_text_tokenizer(
#         model_path,
#         add_hydra_tokens=True,
#         return_hydra_token_ids=True,
#         llm_name="qwen2_5"
#)

#     # 2. Initialize the Dataset (only loads the list, does not load images)
#     # Use YAML mode
#     dataset_mmu = MMUDataset(
#         root="/path/to/dataset/Understanding/LLaVA-NeXT-Data/llava_next_raw_format",
#         image_size=448,
#         num_image_tokens=1024, # make sure this matches your training setting
#         max_seq_len=4096,      # set this larger so the true lengths are measured
#         annotation_path="./datasets/example_mmu_datasets.yaml",
#         text_tokenizer=text_tokenizer,
#         stage='tuning',
#         hydra_token_ids=hydra_token_ids,
#         filter_no_image=True,
#)

#     # 3. Start the statistics
#     # Adjust num_workers based on your CPU core count
#     analyze_dataset_lengths(dataset_mmu, num_workers=32)


if __name__ == '__main__':
    import sys
    # datasets/ sits directly under hydra/, which is where models/ lives.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.misc import get_text_tokenizer
    from torch.utils.data import DataLoader
    # 1. Load the tokenizer
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading tokenizer from {model_path}...")
    text_tokenizer, hydra_token_ids = get_text_tokenizer(
        model_path,
        add_hydra_tokens=True,
        return_hydra_token_ids=True,
        llm_name="qwen2_5"
)
    dataset_mmu = MMUDataset(
        root= "/path/to/datasets/LLaVA-OneVision-Data/llava_processed_data/images",
        # root="/path/to/data/",
        # root= "/path/to/data/datasets/LLaVA-OneVision-Data-SI/images",
        image_size=448,
        num_image_tokens=1024,
        max_seq_len=1536,
        # annotation_path="/path/to/data/blip_laion_cc_sbu_558k.json",
        # # annotation_path="/path/to/data/data/LLaVA-OneVision/stage1-5/OneVision_stage_1_5_all_with_densefusion.json",
        # annotation_path="/path/to/LLaVA-NeXT-Data/llava_next_raw_format_processed.json",
        annotation_path= "/path/to/data/llava_onevision_merged.json",
        # annotation_path="/path/to/LLaVA-OneVision-Data/json_files/FigureQA_MathV360K.json",
        text_tokenizer=text_tokenizer,
        # stage='pre-training',
        # stage='pre-training-1-5',
        stage='tuning',
        hydra_token_ids=hydra_token_ids
)
    train_dataloader_t2i = DataLoader(dataset_mmu, batch_size=1, collate_fn=dataset_mmu.collate_fn,
                                      shuffle=True, num_workers=0, drop_last=True)
    # print(len(train_dataloader_t2i))
    # import ipdb
    # ipdb.set_trace()
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(train_dataloader_t2i)):
        print(text_tokenizer.batch_decode(batch['text_tokens']))
        batch['text_labels'][batch['text_labels'] == -100] = 100
        print(text_tokenizer.batch_decode(batch['text_labels']))
        print(f"modality_positions: {batch['modality_positions']}")
        if i == 100:
            exit()
        # import ipdb
        # ipdb.set_trace()
        # aa = text_tokenizer.batch_decode(batch['text_tokens'])
        # for j in range(len(aa)):
        #     if not aa[j].endswith("[PAD]"):
        #         import ipdb
        #         ipdb.set_trace()
        # continue
        # if (batch['modality_positions'][:, :, 1] == 576).sum() + (
        #         batch['modality_positions'][:, :, 1] == 0).sum() == 64 and batch['text_tokens'].shape[1] == 896:
        #     continue
        # else:
        #     import ipdb
        #     ipdb.set_trace()
        # print(f'[{i}/{len(train_dataloader_t2i)}]')
        # print(batch['text_tokens'])
        # print(batch['images'].shape)
        # import ipdb
        # ipdb.set_trace()

# PYTHONPATH=. python3 datasets/mmu_dataset.py
