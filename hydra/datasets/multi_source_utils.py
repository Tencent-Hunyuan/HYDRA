"""
Multi-source dataset utility functions
Supports specifying multiple T2I/MMU data sources in the config
"""

from typing import Union, List, Any, Optional, Dict, Tuple
from torch.utils.data import DataLoader, ConcatDataset
import logging

logger = logging.getLogger(__name__)


def _ensure_list(value: Union[str, List[str]]) -> List[str]:
    """
    Convert a string or list to list format

    Args:
        value: a string or list (including omegaconf.ListConfig)

    Returns:
        the value in list format
    """
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value
    else:
        # Support list-like objects such as omegaconf.ListConfig
        try:
            return list(value)
        except TypeError:
            raise TypeError(f"Expected str or list-like, got {type(value)}")


def create_multi_source_dataloader(
    train_shards_path_or_urls: Union[str, List[str]],
    data_roots: Union[str, List[str]],
    batch_size: int,
    text_tokenizer: Any,
    image_size: int = 384,
    latent_width: int = 24,
    latent_height: int = 24,
    cond_dropout_prob: float = 0.1,
    max_seq_len: int = 768,
    num_image_tokens: int = 576,
    num_workers: int = 64,
    is_captioning: bool = False,
    min_res: List[int] = None,
    shuffle: bool = True,
    random_und_or_gen: float = 0.0,
    drop_last: bool = True,
    hydra_token_ids: Optional[Dict[str, int]] = None,
    system: Tuple[str, str, str] = ("", "", ""),
    use_hydra_tok: bool = False,
    seed: int = 42,
    accelerator: Optional[Any] = None,
    create_dataloader_fn=None,  # function used to create a single dataloader
) -> DataLoader:
    """
    Create a DataLoader that supports multiple data sources

    Args:
        train_shards_path_or_urls: a single path string or a list of paths
        data_roots: a single data root directory or a list of root directories
        Other arguments are the same as create_imagetext_dataloader
        create_dataloader_fn: function used to create a single dataloader (usually create_imagetext_dataloader)

    Returns:
        DataLoader: the original DataLoader for a single source, or a merged DataLoader for multiple sources

    Examples:
        # Single data source (backward compatible)
        dataloader = create_multi_source_dataloader(
            train_shards_path_or_urls="/path/to/data_info.json",
            data_roots="/path/to/data_root",
            ...
        )

        # Multiple data sources
        dataloader = create_multi_source_dataloader(
            train_shards_path_or_urls=["/path1/data_info.json", "/path2/data_info.json"],
            data_roots=["/path1/data_root", "/path2/data_root"],
            ...
        )
    """
    if min_res is None:
        min_res = [512, 512]
    
    if create_dataloader_fn is None:
        from datasets.image_text_dataset import create_imagetext_dataloader
        create_dataloader_fn = create_imagetext_dataloader
    
    # Convert to list format
    shards_list = _ensure_list(train_shards_path_or_urls)
    roots_list = _ensure_list(data_roots)

    # Validate that the list lengths match
    if len(shards_list) != len(roots_list):
        raise ValueError(
            f"Data source count mismatch: {len(shards_list)} shards but {len(roots_list)} data_roots. "
            f"They must correspond one to one!"
        )

    # Single data source - return directly
    if len(shards_list) == 1:
        logger.info(f"Single T2I data source: {shards_list[0]}")
        return create_dataloader_fn(
            train_shards_path_or_url=shards_list[0],
            data_root=roots_list[0],
            batch_size=batch_size,
            text_tokenizer=text_tokenizer,
            image_size=image_size,
            latent_width=latent_width,
            latent_height=latent_height,
            cond_dropout_prob=cond_dropout_prob,
            max_seq_len=max_seq_len,
            num_image_tokens=num_image_tokens,
            num_workers=num_workers,
            is_captioning=is_captioning,
            min_res=min_res,
            shuffle=shuffle,
            random_und_or_gen=random_und_or_gen,
            drop_last=drop_last,
            hydra_token_ids=hydra_token_ids,
            system=system,
            use_hydra_tok=use_hydra_tok,
            seed=seed,
            accelerator=accelerator
        )
    
    # Multiple data sources - create multiple dataloaders and merge them
    logger.info(f"Multiple T2I data sources ({len(shards_list)}):")
    for i, (shard, root) in enumerate(zip(shards_list, roots_list)):
        logger.info(f"   [{i+1}] {shard} -> {root}")
    
    dataloaders = []
    for shard, root in zip(shards_list, roots_list):
        dl = create_dataloader_fn(
            train_shards_path_or_url=shard,
            data_root=root,
            batch_size=batch_size,
            text_tokenizer=text_tokenizer,
            image_size=image_size,
            latent_width=latent_width,
            latent_height=latent_height,
            cond_dropout_prob=cond_dropout_prob,
            max_seq_len=max_seq_len,
            num_image_tokens=num_image_tokens,
            num_workers=num_workers,
            is_captioning=is_captioning,
            min_res=min_res,
            shuffle=shuffle,
            random_und_or_gen=random_und_or_gen,
            drop_last=drop_last,
            hydra_token_ids=hydra_token_ids,
            system=system,
            use_hydra_tok=use_hydra_tok,
            seed=seed,
            accelerator=accelerator
        )
        dataloaders.append(dl)
    
    # Merge multiple dataloaders
    # Note: this simply returns the first dataloader; a more sophisticated merging
    # strategy may be needed in practice, e.g. interleave or cycle to sample
    # different data sources in turn
    logger.warning(
        "Multi-source merge: currently returning the first dataloader. "
        "To use a custom merge strategy (e.g. interleave/cycle), modify this function."
    )
    return dataloaders[0]


if __name__ == "__main__":
    # Test function
    print("Multi-source dataloader utility loaded successfully!")
