import argparse
import collections
import json
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

# from utils import (
#     image_transform_vae, image_transform_hydra_tok,
#     remove_prefix, format_sequence_und, format_sequence_gen_qwen2_5
#)


class ImageTextDataset(Dataset):
    """Dataset for loading image-text pairs from separated folders."""

    def __init__(self,
                 anno_path: str,
                 text_tokenizer: Any,
                 data_root: str,  # dataset root directory
                 max_seq_len: int = 318 + 576 + 2,
                 image_size: int = 768,
                 latent_height: int = 24,
                 latent_width: int = 24,
                 num_image_tokens: int = 576,
                 is_captioning: bool = False,
                 aes_score: Optional[float] = None,
                 cond_dropout_prob: float = 0.1,
                 min_res: List[int] = None,
                 random_und_or_gen: float = 0.0,
                 hydra_token_ids: Optional[Dict[str, int]] = None,
                 system: Tuple[str, str, str] = ("", "", ""),
                 use_hydra_tok: bool = False,):
        if min_res is None:
            min_res = [512, 512]
        super().__init__()

        self.data_root = data_root
        self.use_hydra_tok = use_hydra_tok

        # [Modified] Use json.load to read the standard JSON list file (data_info.json)
        print(f"Loading annotations from {anno_path}...")
        try:
            with open(anno_path, 'r') as f:
                self.anno = json.load(f)
            print(f"Loaded {len(self.anno)} samples.")
        except (OSError, ValueError) as e:
            print(f"Error loading annotation file {anno_path}: {e}")
            raise e

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

        self.aes_score = aes_score

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
        return len(self.anno)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            data = self.anno[idx]

            # [Modified] Build the image path: root/local_folder/000/xxx.jpg
            # Assume data['image'] is a relative path like "000000/0000001.jpg"
            image_path = os.path.join(self.data_root, 'local_folder', data['image'])

            img = Image.open(image_path)
            width, height = img.size

            # Resolution filtering
            if height < self.min_res[0] or width < self.min_res[1]:
                return self.__getitem__(idx + 1)

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
                return self.__getitem__(idx + 1)

            img = Image.fromarray(img)
            img = self.image_transform(img, resolution=self.image_size)

            # [Modified] Build the annotation file path and read its content
            # Assume data['annotation'] is a relative path like "000000/0000001.json"
            cap_path = os.path.join(self.data_root, 'cap_folder', data['annotation'])

            with open(cap_path, 'r') as f:
                cap_data = json.load(f)

            # Get the caption field
            text = cap_data.get('caption', "").replace('\n', '')

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
                'image_masks': image_mask,  'texts': text, 'data_type': self.data_type
            }

            return sample

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # Simple error handling: print the error and try reading the next item
            # A more robust retry mechanism may be needed in production to avoid deep recursion
            print(f"Error reading item {idx}: {e}")
            if idx + 1 < len(self.anno):
                return self.__getitem__(idx + 1)
            else:
                # If the last sample is also broken, try reading a random one
                return self.__getitem__(random.randint(0, len(self.anno) - 1))

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type'):
                batched[k] = torch.stack(v, dim=0)

        return batched


def create_imagetext_dataloader(train_shards_path_or_url: str,
                                data_root: str,  # dataset root directory
                                batch_size: int,
                                text_tokenizer: Any,
                                accelerator: Optional[Any] = None,
                                image_size: int = 384,
                                latent_width: int = 24,
                                latent_height: int = 24,
                                cond_dropout_prob: float = 0.1,
                                max_seq_len: int = 768,
                                num_image_tokens: int = 576,
                                num_workers: int = 64,
                                is_captioning: bool = False,
                                min_res: Optional[List[int]] = None,
                                shuffle: bool = True,
                                random_und_or_gen: float = 0.0,
                                drop_last: bool = True,
                                hydra_token_ids: Optional[Dict[str, int]] = None,
                                system: Tuple[str, str, str] = ("", "", ""),
                                use_hydra_tok: bool = False,
                                seed: int = 42,) -> DataLoader:

    if min_res is None:
        min_res = [512, 512]
    dataset = ImageTextDataset(train_shards_path_or_url,
                               text_tokenizer=text_tokenizer,
                               data_root=data_root,
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
                               use_hydra_tok=use_hydra_tok,)

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset,
                                     num_replicas=accelerator.num_processes,
                                     rank=accelerator.process_index,
                                     shuffle=shuffle,
                                     drop_last=drop_last,
                                     seed=seed)
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, collate_fn=dataset.collate_fn,
                            shuffle=shuffle, num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader


def example():
    """
    Test data loading in MMU (Multimodal Understanding) mode
    When data_type = 'mmu', is_captioning = True
    """
    import sys
    sys.path.insert(0, "/path/to/show-o2")

    from models.misc import get_text_tokenizer

    # Note: the model paths here are examples only and should be adjusted as needed
    print("=" * 60)
    print(" Testing MMU (Multimodal Understanding) DataLoader")
    print("=" * 60)

    try:
        text_tokenizer, hydra_token_ids = get_text_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct",
            add_hydra_tokens=True,
            return_hydra_token_ids=True,
            llm_name="qwen2_5"
)
        print(f" Tokenizer loaded. Vocab size: {len(text_tokenizer)}")
        print(f" Showo token IDs: {hydra_token_ids}")
    except (OSError, ValueError, RuntimeError) as e:
        print(f" Model tokenizer not found: {e}")
        return

    # Set your paths - MMU dataset (for understanding tasks)
    data_root_dir = "/path/to/dataset/llava_recap_3M_harmon"
    anno_file_path = os.path.join(data_root_dir, "data_info.json")

    # Check that the file exists to avoid errors
    if not os.path.exists(anno_file_path):
        print(f" Example file not found at {anno_file_path}, skipping dataloader creation.")
        return

    print(f"\n Data root: {data_root_dir}")
    print(f" Annotation file: {anno_file_path}")

    # Key point: is_captioning=True makes data_type='mmu'
    loader = create_imagetext_dataloader(
        train_shards_path_or_url=anno_file_path,
        data_root=data_root_dir,
        text_tokenizer=text_tokenizer,
        batch_size=4,
        max_seq_len=1536,
        image_size=448,
        latent_height=32,
        latent_width=32,
        cond_dropout_prob=0.0,  # no dropout in MMU mode
        num_image_tokens=1024,
        is_captioning=True,     # MMU mode: is_captioning=True
        min_res=[256, 256],
        random_und_or_gen=0.0,
        num_workers=0,
        hydra_token_ids=hydra_token_ids,
        system=("", "", ""),
        use_hydra_tok=True
)

    print("\n Dataloader created successfully!")
    print(f" Dataset size: {len(loader.dataset)}")
    print("\n" + "=" * 60)
    print(" Testing first batch (MMU mode)...")
    print("=" * 60)

    for i, data in enumerate(loader):
        print(f"\n Batch {i}:")
        print(f"    Images shape: {data['images'].shape}")
        print(f"   Text tokens shape: {data['text_tokens'].shape}")
        print(f"    Text labels shape: {data['text_labels'].shape}")
        print(f"   Modality positions shape: {data['modality_positions'].shape}")
        print(f"   Text masks shape: {data['text_masks'].shape}")
        print(f"    Image masks shape: {data['image_masks'].shape}")
        print(f"   Data type: {data['data_type']}")
        print(f"\n   Sample Text (first in batch):")
        print(f"     {data['texts'][0][:200]}..." if len(data['texts'][0]) > 200 else f"     {data['texts'][0]}")

        # Print token sequence info
        print(f"\n   Token sequence info (first sample):")
        text_tokens = data['text_tokens'][0]
        print(f"     First 20 tokens: {text_tokens[:20].tolist()}")
        print(f"     Last 20 tokens: {text_tokens[-20:].tolist()}")

        # Check special token positions
        boi_pos = (text_tokens == hydra_token_ids['boi_id']).nonzero(as_tuple=True)[0]
        eoi_pos = (text_tokens == hydra_token_ids['eoi_id']).nonzero(as_tuple=True)[0]
        print(f"     BOI position: {boi_pos.tolist() if len(boi_pos) > 0 else 'Not found'}")
        print(f"     EOI position: {eoi_pos.tolist() if len(eoi_pos) > 0 else 'Not found'}")

        # Check modality_positions
        print(f"\n   Modality positions (first sample): {data['modality_positions'][0].tolist()}")

        break

    print("\n" + "=" * 60)
    print(" MMU DataLoader test completed!")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-readers", type=int, default=1)
    parser.add_argument("--data-path", type=str, default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    example()
