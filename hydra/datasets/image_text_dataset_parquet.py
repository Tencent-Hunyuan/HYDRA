# coding=utf-8
# Copyright 2025 NUS Show Lab.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
#
# MMU Caption Dataset for Parquet Format (LLaVA-One-Vision style)
# Dedicated to loading MMU captioning data in parquet format
#

import argparse
import bisect
import collections
import json
import random
import os
import io
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import (
    image_transform_vae, image_transform_hydra_tok,
    remove_prefix, format_sequence_und, format_sequence_gen_qwen2_5
)

# from utils import (
#     image_transform_vae, image_transform_hydra_tok,
#     remove_prefix, format_sequence_und, format_sequence_gen_qwen2_5
#)


class LRUCache:
    """Simple LRU cache for caching multiple row groups"""
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> Optional[Dict]:
        if key in self.cache:
            # Move to the end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: int, value: Dict):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove the least recently used entry
                self.cache.popitem(last=False)
            self.cache[key] = value


class StreamingParquetMMUDataset(Dataset):
    """
    MMU dataset that streams Parquet files.

    Core optimization: data is not loaded in __init__; instead it is read on demand in __getitem__.
    - Initialization only reads metadata (row counts, row group info)
    - Data is read row group by row group, leveraging Parquet's columnar storage
    - Memory mapping is supported to reduce memory usage
    - An LRU cache holds multiple row groups to improve cache hit rate
    - Optimized for distributed file systems (pre_buffer)

    Suitable for large-scale datasets (thousands of parquet files), reducing
    initialization time from minutes to seconds.
    """

    def __init__(self,
                 parquet_path: str,
                 text_tokenizer: Any,
                 max_seq_len: int = 318 + 576 + 2,
                 image_size: int = 768,
                 latent_height: int = 24,
                 latent_width: int = 24,
                 num_image_tokens: int = 576,
                 is_captioning: bool = True,
                 aes_score: Optional[float] = None,
                 cond_dropout_prob: float = 0.0,
                 min_res: Optional[List[int]] = None,
                 random_und_or_gen: float = 0.0,
                 hydra_token_ids: Optional[Dict[str, int]] = None,
                 system: Tuple[str, str, str] = ("", "", ""),
                 use_hydra_tok: bool = False,
                 image_col: str = "image",
                 caption_col: str = "caption",
                 id_col: str = "id",
                 seed: int = 42,
                 row_group_cache_size: int = 3):
        super().__init__()

        self.parquet_path = parquet_path
        self.use_hydra_tok = use_hydra_tok
        self.image_col = image_col
        self.caption_col = caption_col
        self.id_col = id_col
        self.seed = seed  # seed isolation for distributed environments

        # Optimization 1: only open the file handle and read metadata, without loading data
        # Use pre_buffer=True to optimize read performance on distributed file systems (e.g. CephFS)
        try:
            self.parquet_file = pq.ParquetFile(
                parquet_path,
                memory_map=True,
                pre_buffer=True,  # pre-buffer to reduce seek overhead on distributed file systems
)
            self.num_rows = self.parquet_file.metadata.num_rows

            # Build the row group index (for fast row lookup)
            self._build_row_group_index()

            # Columns to read
            self.columns = [self.image_col, self.caption_col]
            if self.id_col:
                self.columns.append(self.id_col)

        except (OSError, ValueError, RuntimeError) as e:
            print(f"Error opening parquet file {parquet_path}: {e}")
            raise e

        # Tokenizer and special token IDs
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = hydra_token_ids['bos_id']
        self.eos_id = hydra_token_ids['eos_id']
        self.boi_id = hydra_token_ids['boi_id']
        self.eoi_id = hydra_token_ids['eoi_id']
        self.img_pad_id = hydra_token_ids['img_pad_id']

        # Sequence parameters
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.is_captioning = is_captioning
        self.data_type = "mmu" if self.is_captioning else "t2i"

        # Image filtering parameters
        self.aes_score = aes_score
        self.clip_image_size = 384
        self.clip_mean = (0.5, 0.5, 0.5)
        self.clip_std = (0.5, 0.5, 0.5)
        if min_res is None:
            min_res = [256, 256]
        self.min_res = min_res
        self.random_und_or_gen = random_und_or_gen

        # System tokens
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)
        if len(self.system_tokens[0]) == 0:
            self.max_text_len = max_seq_len - num_image_tokens - 4
        else:
            self.max_text_len = max_seq_len - num_image_tokens - 4 - self.system_token_len - 1
        self.image_transform = image_transform_hydra_tok if use_hydra_tok else image_transform_vae

        # Optimization 2: LRU cache for multiple row groups (improves cache hit rate in multi-worker settings)
        self._row_group_cache = LRUCache(capacity=row_group_cache_size)

    def _build_row_group_index(self):
        """Build the row group index for quickly locating the row group of a global row number"""
        metadata = self.parquet_file.metadata
        self.num_row_groups = metadata.num_row_groups

        # Cumulative row count index: row_group_offsets[i] = total rows of the first i row groups
        self.row_group_offsets = [0]
        for i in range(self.num_row_groups):
            rg_rows = metadata.row_group(i).num_rows
            self.row_group_offsets.append(self.row_group_offsets[-1] + rg_rows)

    def _find_row_group(self, global_idx: int) -> Tuple[int, int]:
        """
        Find the row group and in-group offset for a global row index.

        Returns:
            (row_group_idx, local_idx): row group index and in-group row index
        """
        rg_idx = bisect.bisect_right(self.row_group_offsets, global_idx) - 1
        local_idx = global_idx - self.row_group_offsets[rg_idx]
        return rg_idx, local_idx

    def _get_row_data(self, global_idx: int, max_retries: int = 3) -> Dict[str, Any]:
        """
        Optimization 3: read the requested row on demand, with an LRU cache and retry logic.

        Parquet is organized by row group, so reading a single row requires reading the whole group.
        The LRU cache avoids repeated reads of the same row group.
        Retries handle transient I/O errors on distributed file systems.
        """
        rg_idx, local_idx = self._find_row_group(global_idx)

        # Check the LRU cache
        cached_data = self._row_group_cache.get(rg_idx)
        if cached_data is None:
            # Cache miss: read a new row group (with retries)
            last_error = None
            for retry in range(max_retries):
                try:
                    table = self.parquet_file.read_row_group(rg_idx, columns=self.columns)
                    cached_data = table.to_pydict()
                    self._row_group_cache.put(rg_idx, cached_data)
                    break
                except (OSError, ValueError, RuntimeError) as e:
                    last_error = e
                    if retry < max_retries - 1:
                        import time
                        wait_time = (retry + 1) * 2  # backoff: 2s, 4s, 6s
                        print(
                            f"Parquet read retry {retry + 1}/{max_retries} for row_group {rg_idx}: {e}. "
                            f"Waiting {wait_time}s...")
                        time.sleep(wait_time)
            else:
                # All retries failed
                raise RuntimeError(f"Failed to read row_group {rg_idx} after {max_retries} retries: {last_error}")

        # Get the data from the cache
        row_data = {}
        for col in self.columns:
            row_data[col] = cached_data[col][local_idx]

        return row_data

    def __len__(self) -> int:
        return self.num_rows

    def _load_image_from_bytes(self, image_data) -> Optional[Image.Image]:
        """Load an image from bytes"""
        try:
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            else:
                return None

            img = Image.open(io.BytesIO(image_bytes))
            return img
        except (OSError, ValueError, RuntimeError) as e:
            return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            # Optimization 4: seed isolation for distributed environments
            # Ensure each worker's randomness is independent, avoiding random retry conflicts in multi-GPU training
            worker_info = torch.utils.data.get_worker_info()
            worker_seed = self.seed + idx + (worker_info.id if worker_info else 0)
            random.seed(worker_seed)

            # Core: read data on demand
            row = self._get_row_data(idx)

            # Load the image from bytes
            image_data = row[self.image_col]
            img = self._load_image_from_bytes(image_data)

            if img is None:
                return self._retry(idx)

            width, height = img.size

            # Resolution filtering
            if height < self.min_res[0] or width < self.min_res[1]:
                return self._retry(idx)

            # Enable random understanding or generation data
            is_captioning = self.is_captioning
            if self.random_und_or_gen > 0:
                is_captioning = random.random() < self.random_und_or_gen
            data_type = "mmu" if is_captioning else "t2i"

            # Handle image modes
            mode = img.mode
            if mode == 'RGBA':
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif mode == "P" or "transparency" in img.info:
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif mode == 'RGB':
                img = img.convert('RGB')
                img = np.array(img)
            elif mode == 'L':
                img = np.array(img.convert('L'))
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            else:
                return self._retry(idx)

            img = Image.fromarray(img)
            img_clip = self.image_transform(img, resolution=self.clip_image_size,
                                            mean=self.clip_mean, std=self.clip_std)
            img = self.image_transform(img, resolution=self.image_size)

            # Get the caption
            caption_data = row[self.caption_col]
            text = str(caption_data).replace('\n', '') if caption_data is not None else ""

            if not text.strip():
                return self._retry(idx)

            if not is_captioning:
                text = remove_prefix(text)

            # Tokenization with truncation
            if random.random() < self.cond_dropout_prob and not is_captioning:
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

            if is_captioning:
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
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': img,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'images_clip': img_clip,
                'texts': text,
                'data_type': data_type
            }

            return sample

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            return self._retry(idx)

    def _retry(self, failed_idx: int) -> Dict[str, Any]:
        """Retry with a random index"""
        random_index = random.randint(0, len(self) - 1)
        if random_index == failed_idx:
            random_index = (random_index + 1) % len(self)
        return self.__getitem__(random_index)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type'):
                batched[k] = torch.stack(v, dim=0)
        return batched


# Keep the original ParquetMMUDataset as an alias (full-load mode)
ParquetMMUDataset = StreamingParquetMMUDataset


def _load_single_parquet_dataset(args):
    """Helper function: load a single parquet dataset (for parallel loading)"""
    parquet_path, dataset_kwargs = args
    try:
        ds = ParquetMMUDataset(parquet_path=parquet_path, **dataset_kwargs)
        return ds, os.path.basename(parquet_path), len(ds), None
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        return None, os.path.basename(parquet_path), 0, str(e)


def create_parquet_mmu_dataloader(
    index_json_path: str,
    batch_size: int,
    text_tokenizer: Any,
    accelerator: Optional[Any] = None,
    image_size: int = 384,
    latent_width: int = 24,
    latent_height: int = 24,
    cond_dropout_prob: float = 0.0,
    max_seq_len: int = 768,
    num_image_tokens: int = 576,
    num_workers: int = 64,
    is_captioning: bool = True,
    min_res: Optional[List[int]] = None,
    shuffle: bool = True,
    random_und_or_gen: float = 0.0,
    drop_last: bool = True,
    hydra_token_ids: Optional[Dict[str, int]] = None,
    system: Tuple[str, str, str] = ("", "", ""),
    use_hydra_tok: bool = False,
    seed: int = 42,
    image_col: str = "image",
    caption_col: str = "caption",
    id_col: str = "id",
    max_files: Optional[int] = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    parallel_load: bool = True,
    parallel_load_workers: int = 16,
) -> DataLoader:
    """
    Create a Parquet MMU DataLoader from an index JSON file.

    Index JSON format:
    {
        "files": [
            {"absolute_path": "/path/to/file1.parquet",...},
            {"absolute_path": "/path/to/file2.parquet",...},
...
        ]
    }

    Args:
        index_json_path: index JSON file path
        batch_size: batch size
        text_tokenizer: text tokenizer
        accelerator: optional accelerator (for distributed training)
        image_size: image size
        latent_width: latent width
        latent_height: latent height
        cond_dropout_prob: conditional dropout probability
        max_seq_len: maximum sequence length
        num_image_tokens: number of image tokens
        num_workers: number of data loading workers
        is_captioning: whether this is a captioning task (MMU mode)
        min_res: minimum image resolution [height, width]
        shuffle: whether to shuffle the data
        random_und_or_gen: probability of randomly switching between MMU/T2I
        drop_last: whether to drop the last incomplete batch
        hydra_token_ids: dict of special token IDs
        system: system prompt tuple
        use_hydra_tok: whether to use HYDRA-TOK preprocessing
        seed: random seed
        image_col: image column name in the parquet file
        caption_col: caption column name in the parquet file
        id_col: id column name in the parquet file
        max_files: maximum number of files to load (for testing)
        prefetch_factor: prefetch factor, number of batches prefetched per worker (speeds up data loading)
        persistent_workers: whether to keep worker processes alive (avoids re-creation overhead)
        parallel_load: whether to load parquet files in parallel
        parallel_load_workers: number of parallel loading workers

    Returns:
        A DataLoader instance
    """
    if min_res is None:
        min_res = [256, 256]

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_time = time.time()

    # Load the index JSON
    print(f"Loading index from {index_json_path}...")
    with open(index_json_path, 'r') as f:
        index_data = json.load(f)

    files = index_data.get('files', [])
    if max_files is not None:
        files = files[:max_files]

    print(f"Found {len(files)} parquet files in index.")

    # Prepare the dataset arguments (shared by all files)
    dataset_kwargs = {
        'text_tokenizer': text_tokenizer,
        'max_seq_len': max_seq_len,
        'image_size': image_size,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'num_image_tokens': num_image_tokens,
        'is_captioning': is_captioning,
        'cond_dropout_prob': cond_dropout_prob,
        'min_res': min_res,
        'random_und_or_gen': random_und_or_gen,
        'hydra_token_ids': hydra_token_ids,
        'system': system,
        'use_hydra_tok': use_hydra_tok,
        'image_col': image_col,
        'caption_col': caption_col,
        'id_col': id_col,
    }

    # Filter for valid files
    valid_files = []
    for file_info in files:
        parquet_path = file_info.get('absolute_path', '')
        if parquet_path and os.path.exists(parquet_path):
            valid_files.append(parquet_path)
        else:
            print(f"Warning: Parquet file not found: {parquet_path}")

    if not valid_files:
        raise ValueError("No valid parquet files found!")

    print(f"Using streaming mode: initializing {len(valid_files)} parquet files (metadata only)...")

    # Create datasets for all parquet files (streaming mode: metadata only)
    datasets = []
    failed_files = []
    total_samples = 0

    if parallel_load and len(valid_files) > 1:
        # Parallel initialization (very fast in streaming mode, only metadata is read)
        print(f"Using parallel initialization with {parallel_load_workers} workers...")

        load_args = [(path, dataset_kwargs) for path in valid_files]

        with ThreadPoolExecutor(max_workers=parallel_load_workers) as executor:
            futures = {executor.submit(_load_single_parquet_dataset, arg): arg[0] for arg in load_args}

            for i, future in enumerate(as_completed(futures)):
                ds, filename, count, error = future.result()
                if error:
                    failed_files.append(f"{filename} ({error})")
                elif ds is not None:
                    datasets.append(ds)
                    total_samples += count

                # Print progress every 2000 files (streaming mode is faster, so print less often)
                if (i + 1) % 2000 == 0 or (i + 1) == len(valid_files):
                    print(f"  [{i+1}/{len(valid_files)}] Initialized {len(datasets)} files, {total_samples:,} samples")
    else:
        # Serial initialization
        for parquet_path in valid_files:
            try:
                ds = StreamingParquetMMUDataset(parquet_path=parquet_path, **dataset_kwargs)
                datasets.append(ds)
                total_samples += len(ds)
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                failed_files.append(f"{os.path.basename(parquet_path)} ({e})")
                continue

    if not datasets:
        raise ValueError("No valid parquet files loaded!")

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)

    load_time = time.time() - start_time
    print(f" Total samples: {len(combined_dataset):,}")
    print(f" Load time: {load_time:.2f}s ({len(datasets) / load_time:.1f} files/s)")

    if failed_files:
        print(f" Failed to load {len(failed_files)} files")

    # Create the sampler
    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            combined_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
)
        shuffle = False
    else:
        sampler = None

    # Use the first dataset's collate_fn
    collate_fn = datasets[0].collate_fn

    # Create the DataLoader (with tuned parameters)
    dataloader_kwargs = {
        'batch_size': batch_size,
        'sampler': sampler,
        'collate_fn': collate_fn,
        'shuffle': shuffle if sampler is None else False,
        'num_workers': num_workers,
        'drop_last': drop_last,
        'pin_memory': True,
    }

    # Add prefetching and persistent workers (speeds up training)
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = persistent_workers

    dataloader = DataLoader(combined_dataset, **dataloader_kwargs)

    return dataloader


def create_single_parquet_mmu_dataloader(
    parquet_path: str,
    batch_size: int,
    text_tokenizer: Any,
    accelerator: Optional[Any] = None,
    image_size: int = 384,
    latent_width: int = 24,
    latent_height: int = 24,
    cond_dropout_prob: float = 0.0,
    max_seq_len: int = 768,
    num_image_tokens: int = 576,
    num_workers: int = 64,
    is_captioning: bool = True,
    min_res: Optional[List[int]] = None,
    shuffle: bool = True,
    random_und_or_gen: float = 0.0,
    drop_last: bool = True,
    hydra_token_ids: Optional[Dict[str, int]] = None,
    system: Tuple[str, str, str] = ("", "", ""),
    use_hydra_tok: bool = False,
    seed: int = 42,
    image_col: str = "image",
    caption_col: str = "caption",
    id_col: str = "id",
) -> DataLoader:
    """
    Create an MMU DataLoader from a single Parquet file.

    Args:
        parquet_path: Parquet file path
        Other arguments are the same as create_parquet_mmu_dataloader

    Returns:
        A DataLoader instance
    """
    if min_res is None:
        min_res = [256, 256]

    dataset = ParquetMMUDataset(
        parquet_path=parquet_path,
        text_tokenizer=text_tokenizer,
        max_seq_len=max_seq_len,
        image_size=image_size,
        latent_height=latent_height,
        latent_width=latent_width,
        num_image_tokens=num_image_tokens,
        is_captioning=is_captioning,
        cond_dropout_prob=cond_dropout_prob,
        min_res=min_res,
        random_und_or_gen=random_und_or_gen,
        hydra_token_ids=hydra_token_ids,
        system=system,
        use_hydra_tok=use_hydra_tok,
        image_col=image_col,
        caption_col=caption_col,
        id_col=id_col,
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
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
)

    return dataloader


def count_parquet_rows_fast(parquet_path: str) -> int:
    """
    Quickly count the rows of a parquet file (reads metadata only, no data loading)
    """
    import pyarrow.parquet as pq
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        return parquet_file.metadata.num_rows
    except (OSError, ValueError, RuntimeError):
        # If pyarrow fails, fall back to pandas (slower)
        df = pd.read_parquet(parquet_path, columns=[])  # read schema only, no data
        return len(df)


def count_parquet_rows_parallel(file_paths: List[str], num_workers: int = 32) -> Tuple[int, int, List[str]]:
    """
    Count rows of multiple parquet files in parallel

    Args:
        file_paths: list of parquet file paths
        num_workers: number of parallel workers

    Returns:
        (total_samples, valid_files, failed_files)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_samples = 0
    valid_files = 0
    failed_files = []

    def count_single_file(path: str) -> Tuple[str, int, Optional[str]]:
        """Count the rows of a single file"""
        if not path or not os.path.exists(path):
            return path, 0, "File not found"
        try:
            count = count_parquet_rows_fast(path)
            return path, count, None
        except (OSError, ValueError, RuntimeError) as e:
            return path, 0, str(e)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(count_single_file, path): path for path in file_paths}

        for i, future in enumerate(as_completed(futures)):
            path, count, error = future.result()
            if error:
                failed_files.append(f"{os.path.basename(path)} ({error})")
            else:
                total_samples += count
                valid_files += 1

            # Print progress every 100 files
            if (i + 1) % 100 == 0 or (i + 1) == len(file_paths):
                print(f"  [{i+1}/{len(file_paths)}] Processed {valid_files} files, {total_samples:,} samples so far...")

    return total_samples, valid_files, failed_files


def example():
    """
    Count the total number of samples in a Parquet dataset (accelerated with parallelism)
    """
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print(" Counting Parquet MMU Dataset Samples (Parallel)")
    print("=" * 60)

    # Index file path
    index_json_path = "/path/to/llava_en_parquet_index.json"

    if not os.path.exists(index_json_path):
        print(f" Index file not found at {index_json_path}")
        return

    print(f"\n Index file: {index_json_path}")

    # Load the index JSON
    with open(index_json_path, 'r') as f:
        index_data = json.load(f)

    files = index_data.get('files', [])
    total_files = len(files)
    print(f" Total parquet files: {total_files}")

    # Extract all file paths
    file_paths = [f.get('absolute_path', '') for f in files]

    print(f"\n Counting rows in all parquet files (parallel)...")
    print("-" * 60)

    start_time = time.time()

    # Count rows using parallelism
    total_samples, valid_files, failed_files = count_parquet_rows_parallel(
        file_paths,
        num_workers=32  # adjust based on the number of CPU cores
)

    elapsed_time = time.time() - start_time

    print("-" * 60)
    print(f"\n" + "=" * 60)
    print(f" DATASET STATISTICS")
    print("=" * 60)
    print(f"   Total parquet files:    {total_files}")
    print(f"   Successfully loaded:    {valid_files}")
    print(f"   Failed to load:         {len(failed_files)}")
    print(f"   Total image-text pairs: {total_samples:,}")
    print(f"  ⏱  Time elapsed:           {elapsed_time:.2f}s")
    print(f"   Speed:                  {valid_files / elapsed_time:.1f} files/s")
    print("=" * 60)

    if failed_files:
        print(f"\n Failed files ({len(failed_files)}):")
        for f in failed_files[:10]:  # show only the first 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"... and {len(failed_files) - 10} more")


class DataLoadingMonitor:
    """
    Data loading latency monitor.

    Monitors data loading performance during training and helps identify I/O bottlenecks.

    Usage:
        monitor = DataLoadingMonitor(log_every=100)
        for batch in dataloader:
            with monitor:
                # data preprocessing
                pass
            # training logic
    """

    def __init__(self, log_every: int = 100, logger=None):
        self.log_every = log_every
        self.logger = logger
        self.step = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.min_time = float('inf')
        self.times = []
        self._start_time = None

    def __enter__(self):
        import time
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.perf_counter() - self._start_time
        self.step += 1
        self.total_time += elapsed
        self.max_time = max(self.max_time, elapsed)
        self.min_time = min(self.min_time, elapsed)
        self.times.append(elapsed)

        if self.step % self.log_every == 0:
            avg_time = self.total_time / self.step
            recent_avg = sum(self.times[-self.log_every:]) / min(len(self.times), self.log_every)

            msg = (
                f" Data Loading Stats (step {self.step}): "
                f"avg={avg_time*1000:.2f}ms, "
                f"recent_avg={recent_avg*1000:.2f}ms, "
                f"min={self.min_time*1000:.2f}ms, "
                f"max={self.max_time*1000:.2f}ms"
)

            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

        return False

    def reset(self):
        """Reset the statistics"""
        self.step = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.min_time = float('inf')
        self.times = []

    def get_stats(self) -> Dict[str, float]:
        """Get the statistics"""
        if self.step == 0:
            return {}
        return {
            'avg_ms': (self.total_time / self.step) * 1000,
            'min_ms': self.min_time * 1000,
            'max_ms': self.max_time * 1000,
            'total_steps': self.step,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--index-json", type=str,
                        default="/path/to/llava_en_parquet_index.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    example()
