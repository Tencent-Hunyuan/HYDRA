from typing import Any, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from collections import OrderedDict
import random
from decord import VideoReader, cpu

##################################################
#              config utils
##################################################
HF_SCHEME = "hf://"

# Human-facing repo holding the released weights (see ckpt/README.md).
DEFAULT_HF_REPO = "Xuerui123/HYdra"


def resolve_hf_spec(spec: str) -> str:
    """Resolve an `hf://` checkpoint spec to a local path on disk.

    Syntax::

        hf://<repo_id>                       # whole repo (or a single file)
        hf://<repo_id>/<subfolder>           # one subfolder, e.g. hf://user/repo/hydra_1.5b
        hf://<repo_id>/<subfolder>@<revision>
        hf://<repo_id>/<a>,<b>               # several subfolders / files at once

    Examples::

        hf://Xuerui123/HYdra/hydra_1.5b
        hf://Xuerui123/HYdra/hydra_tok_gan
        hf://Xuerui123/HYdra/hydra_tok_latent_stats.pt
        hf://Xuerui123/HYdra/hydra_1.5b@main

    Returns the local snapshot directory for a repo or subfolder spec, or the
    local file path when the spec names exactly one file. Files land in the
    standard HuggingFace cache (`$HF_HOME`, default `~/.cache/huggingface`), so
    this downloads once and is a no-op on later runs.

    A non-`hf://` string is returned unchanged, so local paths keep working.
    """
    if not isinstance(spec, str) or not spec.startswith(HF_SCHEME):
        return spec

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to resolve 'hf://' checkpoint paths. "
            "Install it with `pip install -U huggingface_hub`, or point the "
            "config field at a local path instead."
        ) from exc

    body = spec[len(HF_SCHEME):]
    revision = None
    if "@" in body:
        body, revision = body.rsplit("@", 1)

    # A repo id is exactly two segments: "user/name". Anything after that is a
    # subfolder, and the subfolder may be a comma-separated list.
    parts = body.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Malformed hf:// spec {spec!r}: expected at least 'hf://<user>/<repo>', "
            f"e.g. hf://{DEFAULT_HF_REPO}/hydra_1.5b"
        )
    repo_id = "/".join(parts[:2])
    patterns = [p for p in "/".join(parts[2:]).split(",") if p]

    if len(patterns) == 1:
        try:
            return hf_hub_download(
                repo_id=repo_id, filename=patterns[0], revision=revision,
                repo_type="model",
            )
        except Exception:
            # Not a file -- treat it as a subfolder and fall through.
            pass

    allow_patterns = None
    if patterns:
        allow_patterns = [p for pat in patterns for p in (pat, f"{pat}/*")]

    snapshot = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        repo_type="model",
    )

    # snapshot_download returns the repo root; callers asking for a single
    # subfolder want that subfolder (they os.path.join weight filenames onto
    # it). Returning the root here silently yields a directory with no weights
    # in it, which load_state_dict used to turn into an empty state dict.
    if len(patterns) == 1:
        subdir = os.path.join(snapshot, patterns[0])
        if os.path.isdir(subdir):
            return subdir
    return snapshot


def _resolve_hf_paths(node) -> None:
    """Walk a config tree in place, replacing `hf://` strings with local paths."""
    if isinstance(node, DictConfig):
        for key in node.keys():  # noqa: SIM118 -- cfg.keys() avoids resolving interpolations
            try:
                value = node[key]
            except Exception:
                continue  # unresolvable interpolation -- leave it alone
            if isinstance(value, (DictConfig, ListConfig)):
                _resolve_hf_paths(value)
            elif isinstance(value, str) and value.startswith(HF_SCHEME):
                node[key] = resolve_hf_spec(value)
    elif isinstance(node, ListConfig):
        for idx in range(len(node)):
            value = node[idx]
            if isinstance(value, (DictConfig, ListConfig)):
                _resolve_hf_paths(value)
            elif isinstance(value, str) and value.startswith(HF_SCHEME):
                node[idx] = resolve_hf_spec(value)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    resolve_hf_config(conf)
    return conf


def resolve_hf_config(conf):
    """Download + rewrite every `hf://` path in a config.

    Every entry point funnels through `get_config()`, so this is what makes a
    config whose paths read `hf://Xuerui123/HYdra/...` work with no local
    checkpoint at all -- useful for verifying inference without a manual
    download. Set `HYDRA_HF_SKIP_DOWNLOAD=1` to leave the strings untouched.

    Safe under `torchrun`: every rank calls this, and `huggingface_hub` guards
    each blob with a file lock, so exactly one rank downloads a given file
    while the others block and then read it from the shared cache. Pre-warm
    with `huggingface-cli download` to skip the wait entirely.
    """
    if os.environ.get("HYDRA_HF_SKIP_DOWNLOAD", "0") in ("1", "true", "True"):
        return conf
    _resolve_hf_paths(conf)
    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              misc
##################################################

def _count_params(module):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in module.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def _freeze_params(model, frozen_params=None):
    if frozen_params is not None:
        for n, p in model.named_parameters():
            for name in frozen_params:
                if name in n:
                    p.requires_grad = False


def get_llm_name(llm_model_path) -> str:
    """Infer the LLM family name ('qwen2_5' | 'llama3') from a model path.

    Matches common HF hub ids and local directory layouts first, then falls
    back to reading the model's own config.json. Works with any user-supplied
    local path instead of a hard-coded path table.
    """
    import json as _json

    path = str(llm_model_path)
    lowered = path.lower()
    if "qwen2.5" in lowered or "qwen2_5" in lowered or "qwen2.5" in lowered:
        return 'qwen2_5'
    if "llama-3" in lowered or "llama3" in lowered:
        return 'llama3'

    cfg_path = os.path.join(path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = _json.load(f)
            arch = (cfg.get("architectures") or [""])[0].lower()
            model_type = cfg.get("model_type", "").lower()
            if "qwen2" in arch or model_type == "qwen2":
                return 'qwen2_5'
            if "llama" in arch or model_type == "llama":
                return 'llama3'
        except (KeyError, AttributeError, OSError, ValueError):
            pass

    raise ValueError(
        f"Cannot infer LLM family from path: {path}. "
        "Expected a Qwen2.5 or Llama-3 family model."
    )


# Backwards-compatible alias for older call sites.
path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": 'qwen2_5',
    "Qwen/Qwen2.5-1.5B-Instruct": 'qwen2_5',
    "meta-llama/Llama-3.2-1B-Instruct": 'llama3',
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denorm(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images

def denorm_vid(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    # B, C, T, H, W --> B, T, C, H, W
    images = images.permute(0, 2, 1, 3, 4).cpu().numpy().astype(np.uint8)
    return images


def get_hyper_params(config, text_tokenizer, hydra_token_ids, is_video=False, is_hq=False):
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    max_seq_len = config.dataset.preprocessing.max_seq_length
    num_video_tokens = config.dataset.preprocessing.num_video_tokens
    if is_video:
        max_text_len = max_seq_len - num_video_tokens - 4
        latent_width = config.dataset.preprocessing.video_latent_width
        latent_height = config.dataset.preprocessing.video_latent_height
        num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
        num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
    else:
        if is_hq:
            latent_width = config.dataset.preprocessing.hq_latent_width
            latent_height = config.dataset.preprocessing.hq_latent_height
            num_t2i_image_tokens = config.dataset.preprocessing.num_hq_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            max_seq_len = config.dataset.preprocessing.max_hq_seq_length
            max_text_len = max_seq_len - num_t2i_image_tokens - 4
        else:
            num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            latent_width = config.dataset.preprocessing.latent_width
            latent_height = config.dataset.preprocessing.latent_height
            max_text_len = max_seq_len - num_t2i_image_tokens - 4

    image_latent_dim = config.model.llm.image_latent_dim
    patch_size = config.model.llm.patch_size

    pad_id = text_tokenizer.pad_token_id
    bos_id = hydra_token_ids['bos_id']
    eos_id = hydra_token_ids['eos_id']
    boi_id = hydra_token_ids['boi_id']
    eoi_id = hydra_token_ids['eoi_id']
    bov_id = hydra_token_ids['bov_id']
    eov_id = hydra_token_ids['eov_id']
    img_pad_id = hydra_token_ids['img_pad_id']
    vid_pad_id = hydra_token_ids['vid_pad_id']

    guidance_scale = config.transport.guidance_scale

    return (num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len,
            image_latent_dim, patch_size,
            latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id,
            vid_pad_id, guidance_scale)


# these save and recover functions are based on our internal packages
# please modified them when necessary
def save_dataloader_state(rank, loader, ckpt_path="./"):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    saved_state = deepcopy(loader.__getstate__())
    torch.save(saved_state, ckpt_path)

def recover_dataloader_state(rank, loader, ckpt_path='./'):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            loader_state_dict = torch.load(f)
            loader.__setstate__(loader_state_dict)
        print(f"rank {rank} loader state dict loaded successfully!")


def save_images_as_grid(pil_images, fn, path, grid_size=(2, 2)):

    os.makedirs(path, exist_ok=True)

    rows, cols = grid_size

    num_images = len(pil_images)
    if num_images > rows * cols:
        raise ValueError(f"Number of images ({num_images}) exceeds grid capacity ({rows * cols}).")

    img_width, img_height = pil_images[0].size

    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))  # white background

    for idx, image in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x_offset = col * img_width
        y_offset = row * img_height
        grid_image.paste(image, (x_offset, y_offset))

    grid_image.save(os.path.join(path, f"{fn}.png"))

    return grid_image


def load_state_dict(model_path):
    if model_path.endswith(".bin"):
        state_dict = torch.load(model_path)
    else:
        checkpoint_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.bin')]
        )

        # Fail loudly. Callers pass the result to `load_state_dict(..., strict=False)`,
        # which happily accepts an empty dict -- the model then runs with randomly
        # initialized weights and emits pure noise with no error anywhere.
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No '*.bin' checkpoint files found in {model_path!r}. Point model_path at "
                f"the directory that directly contains pytorch_model.bin (e.g. "
                f"ckpt/hydra_1.5b or hf://{DEFAULT_HF_REPO}/hydra_1.5b), not at its parent. "
                f"Found instead: {sorted(os.listdir(model_path))}"
            )

        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            state_dict.update(checkpoint)

    return state_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_video(video_path, max_frames_num, fps, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 432, 432, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = [Image.fromarray(item) for item in vr.get_batch(frame_idx).asnumpy()]

    return spare_frames, frame_time, video_time
