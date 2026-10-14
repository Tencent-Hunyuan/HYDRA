import collections
import random
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import (
    image_transform_vae, image_transform_hydra_tok,
    remove_prefix, format_sequence_und, format_sequence_gen_qwen2_5
)


class SAM1BDataset(Dataset):
    """
    Dataset class for the SAM-1B dataset.
    Keeps the same processing as ImageTextDataset, supporting both T2I and MMU modes.
    """

    def __init__(self,
                 image_root: str,
                 caption_root: str,
                 text_tokenizer: Any,
                 dataset_size: int = 10_000_000,
                 max_seq_len: int = 318 + 576 + 2,
                 image_size: int = 768,
                 latent_height: int = 24,
                 latent_width: int = 24,
                 num_image_tokens: int = 576,
                 is_captioning: bool = False,
                 cond_dropout_prob: float = 0.1,
                 min_res: List[int] = None,
                 random_und_or_gen: float = 0.0,
                 hydra_token_ids: Optional[Dict[str, int]] = None,
                 system: Tuple[str, str, str] = ("", "", ""),
                 use_hydra_tok: bool = False,):
        if min_res is None:
            min_res = [512, 512]
        super().__init__()

        self.image_root = image_root
        self.caption_root = caption_root
        self.dataset_size = dataset_size
        self.use_hydra_tok = use_hydra_tok
        
        print(f"--- Initializing SAM1BDataset ---")
        print(f"Image root (image_root): {image_root}")
        print(f"Caption root (caption_root): {caption_root}")
        print(f"Dataset size (dataset_size): {dataset_size}")

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = hydra_token_ids['bos_id']
        self.eos_id = hydra_token_ids['eos_id']
        self.boi_id = hydra_token_ids['boi_id']
        self.eoi_id = hydra_token_ids['eoi_id']
        self.img_pad_id = hydra_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.is_captioning = is_captioning
        self.data_type = "mmu" if self.is_captioning else "t2i"
        self.image_transform = image_transform_hydra_tok if use_hydra_tok else image_transform_vae
        
        self.clip_image_size = 384  # Hard code resolution
        self.clip_mean = (0.5, 0.5, 0.5)
        self.clip_std = (0.5, 0.5, 0.5)
        self.min_res = min_res  # h, w
        self.random_und_or_gen = random_und_or_gen
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)
        if len(self.system_tokens[0]) == 0:
            # 4 for bos, eos, boi, and eoi tokens
            self.max_text_len = max_seq_len - num_image_tokens - 4
        else:
            # 4 for bos, eos, boi, and eoi tokens
            # 1 for eos after text token (a bit tricky)
            self.max_text_len = max_seq_len - num_image_tokens - 4 - self.system_token_len - 1

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            # SAM-1B file naming rule: sa_1.jpg, sa_1.txt, sa_2.jpg, sa_2.txt, ...
            file_index = idx + 1
            file_basename = f"sa_{file_index}"
            
            # 1. Load the image
            img_abs_path = os.path.join(self.image_root, f"{file_basename}.jpg")
            img = Image.open(img_abs_path)
            width, height = img.size

            # Resolution filtering
            if height < self.min_res[0] or width < self.min_res[1]:
                new_idx = random.randint(0, self.dataset_size - 1)
                return self.__getitem__(new_idx)

            # Enable random understanding or generation data
            if self.random_und_or_gen > 0:
                self.is_captioning = random.random() < self.random_und_or_gen
                self.data_type = "mmu" if self.is_captioning else "t2i"

            mode = img.mode
            if mode == 'RGBA':
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif img.mode == "P" or "transparency" in img.info:
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif mode == 'RGB':
                img = img.convert('RGB')
                img = np.array(img)
            elif mode == 'L':
                img = np.array(img.convert('L'))
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            else:
                new_idx = random.randint(0, self.dataset_size - 1)
                return self.__getitem__(new_idx)

            img = Image.fromarray(img)
            img_clip = self.image_transform(
                img, resolution=self.clip_image_size, mean=self.clip_mean, std=self.clip_std)
            img = self.image_transform(img, resolution=self.image_size)

            # 2. Load the caption
            cap_abs_path = os.path.join(self.caption_root, f"{file_basename}.txt")
            with open(cap_abs_path, 'r', encoding='utf-8') as f:
                text = f.read().strip().replace('\n', '')
            
            if not text:
                new_idx = random.randint(0, self.dataset_size - 1)
                return self.__getitem__(new_idx)
            
            if not self.is_captioning:
                text = remove_prefix(text)

            # Tokenization with truncation
            if random.random() < self.cond_dropout_prob and not self.is_captioning:
                text_tokens = self.text_tokenizer('',
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_text_len
                                                  ).input_ids
            else:
                text_tokens = self.text_tokenizer(text,
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_text_len
                                                  ).input_ids

            if self.is_captioning:
                # System prompt is not supported in this dataset
                text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                    format_sequence_und(text_tokens, self.bos_id, self.eos_id,
                                        self.boi_id, self.eoi_id, self.pad_id, self.img_pad_id,
                                        self.num_image_tokens, self.max_seq_len)
            else:
                text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                    format_sequence_gen_qwen2_5(text_tokens, self.system_tokens, self.bos_id, self.eos_id,
                                                self.boi_id, self.eoi_id, self.pad_id, self.img_pad_id,
                                                self.num_image_tokens, self.max_seq_len, self.system_token_len)

            sample = {
                'text_tokens': text_tokens, 'text_labels': text_labels, 'images': img,
                'modality_positions': modality_positions, 'text_masks': text_mask,
                'image_masks': image_mask, 'images_clip': img_clip, 'texts': text, 'data_type': self.data_type
            }

            return sample

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # Error handling: try reading another sample
            # print(f"SAM1BDataset: Error reading item {idx} (sa_{idx+1}): {e}")
            new_idx = random.randint(0, self.dataset_size - 1)
            if new_idx != idx:
                return self.__getitem__(new_idx)
            else:
                return self.__getitem__((idx + 1) % self.dataset_size)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type'):
                batched[k] = torch.stack(v, dim=0)

        return batched


def create_sam_dataloader(image_root: str,
                          caption_root: str,
                          dataset_size: int,
                          batch_size: int,
                          text_tokenizer: Any,
                          accelerator: Optional[Any] = None,
                          image_size: int = 384,
                          latent_width: int = 24,
                          latent_height: int = 24,
                          cond_dropout_prob: float = 0.1,
                          max_seq_len: int = 768,
                          num_image_tokens: int = 576,
                          num_workers: int = 8,
                          is_captioning: bool = False,
                          min_res: Optional[List[int]] = None,
                          shuffle: bool = True,
                          random_und_or_gen: float = 0.0,
                          drop_last: bool = True,
                          hydra_token_ids: Optional[Dict[str, int]] = None,
                          system: Tuple[str, str, str] = ("", "", ""),
                          use_hydra_tok: bool = False,
                          seed: int = 42) -> DataLoader:
    """
    Create a DataLoader for the SAM-1B dataset.
    """
    if min_res is None:
        min_res = [512, 512]
    dataset = SAM1BDataset(
        image_root=image_root,
        caption_root=caption_root,
        text_tokenizer=text_tokenizer,
        dataset_size=dataset_size,
        max_seq_len=max_seq_len,
        image_size=image_size,
        latent_height=latent_height,
        latent_width=latent_width,
        cond_dropout_prob=cond_dropout_prob,
        num_image_tokens=num_image_tokens,
        is_captioning=is_captioning,
        min_res=min_res,
        random_und_or_gen=random_und_or_gen,
        hydra_token_ids=hydra_token_ids,
        system=system,
        use_hydra_tok=use_hydra_tok,
    )

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataloader
