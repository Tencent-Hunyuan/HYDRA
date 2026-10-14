import os
import torch
import numpy as np
import gc
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

# Assume your model files are in the models/hydra_tok directory
from models.hydra_tok.modeling_hydra_tok import HydraTokConfig, Gen_ViT, Und_ViT

# Try to import metric computation libraries
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_loss
    from skimage.metrics import structural_similarity as ssim_loss
except ImportError:
    print(" skimage not found. Installing: pip install scikit-image")

    def psnr_loss(img1, img2):
        return 0.0

    def ssim_loss(img1, img2, **kwargs):
        return 0.0

# ================= Configuration =================
CKPT_PATH = '/path/to/outputs/hydra_tok/final_model.pth'
LATENT_STATS_PATH = '/path/to/Nitro-E/hydra_tok_latent_1_1_1_24layer_64C_2weight_2linear_disp_norm_12_13_4M_dataset.pt'
TARGET_IMAGE_PATH = '/path/to/test_image_512x512_2.png'
CONFIG_DIR = 'models/hydra_tok'
IMAGE_SIZE = 448

# Auto-detect the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DATA_NORM = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# ================= Helper Functions =================


def center_crop_arr(pil_image, image_size):
    """Center crop and resize"""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def get_transform(image_size):
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_NORM['mean'], std=DATA_NORM['std'], inplace=True)
    ])

import os
import torch
from collections import OrderedDict
import torch
import os


def verify_gen_weights_against_ckpt(gen_model, checkpoint_path):
    """
    Compare the weight values of Gen_ViT against the original checkpoint, parameter by parameter.
    Ensures the refactored model parameters were not mistakenly initialized or loaded incorrectly.
    """
    print(f"\n [Verification] Starting Deep Weight Inspection for Gen_ViT...")

    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint not found: {checkpoint_path}")
        return

    # 1. Load the original checkpoint
    print(f"   Loading original checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if 'student_model' in ckpt:
        state_dict = ckpt['student_model']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Remove the module. prefix (if present)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 2. Iterate over all Gen_ViT parameters
    gen_params = dict(gen_model.named_parameters())

    matched_count = 0
    mismatched_keys = []
    missing_keys_in_ckpt = []

    print("   Comparing parameters...")

    for name, param in gen_params.items():
        # --- 3. Build the corresponding checkpoint key (mapping rules) ---
        ckpt_key = name

        # Rule A: chal_proj is top-level in Gen_ViT but belongs to the encoder in the ckpt
        # Gen: chal_proj.c_fc.weight -> Ckpt: encoder.chal_proj.c_fc.weight
        if name.startswith("chal_proj."):
            ckpt_key = "encoder." + name

        # Rule B: encoder.layers (0-11) names are usually identical
        # Gen: encoder.layers.0.ls1 -> Ckpt: encoder.layers.0.ls1

        # Rule C: global_blocks names are usually identical
        # Gen: global_blocks.0.ls1 -> Ckpt: global_blocks.0.ls1

        # Rule D: chal_unproj_generation names are usually identical (depends on ckpt structure)
        # Gen: chal_unproj_generation.c_fc.weight -> Ckpt: chal_unproj_generation.c_fc.weight

        # --- 4. Perform the comparison ---
        if ckpt_key not in state_dict:
            missing_keys_in_ckpt.append((name, ckpt_key))
            continue

        ckpt_param = state_dict[ckpt_key]

        # Ensure device and dtype match
        current_data = param.data.cpu().float()
        ckpt_data = ckpt_param.data.cpu().float()

        # Numeric comparison (allclose guards against tiny floating-point differences)
        if torch.allclose(current_data, ckpt_data, atol=1e-5):
            matched_count += 1
            # Optional: print some success cases
            # if matched_count % 50 == 0:
            #     print(f"       {name} matches.")
        else:
            diff = (current_data - ckpt_data).abs().max().item()
            mismatched_keys.append((name, diff))

    # --- 5. Print the report ---
    print("\n   [Verification Report]")
    print(f"   --------------------------------------------------")
    print(f"    Total Matched Parameters: {matched_count}")

    if len(mismatched_keys) > 0:
        print(f"    MISMATCHED VALUES found: {len(mismatched_keys)}")
        for key, diff in mismatched_keys[:10]:  # print only the first 10
            print(f"      - {key}: Max Diff = {diff:.6f}")
        if len(mismatched_keys) > 10:
            print("... (and more)")
    else:
        print(f"    No value mismatches found.")

    if len(missing_keys_in_ckpt) > 0:
        print(f"     KEYS NOT FOUND in Checkpoint: {len(missing_keys_in_ckpt)}")
        # This usually happens because the Gen model has buffers or parameters
        # that are not part of the original ckpt
        for name, expected_key in missing_keys_in_ckpt[:5]:
            print(f"      - Model: {name}  -> Expected Ckpt: {expected_key}")
    else:
        print(f"    All Gen_ViT keys found in checkpoint.")

    # Spot-check ls1
    print("\n   [Spot Check: LayerScale]")
    try:
        ls1_val = gen_model.encoder.layers[0].ls1.mean().item()
        print(f"   - Gen Enc Layer 0 ls1: {ls1_val:.6f}")
        if abs(ls1_val) < 1e-4:
            print("     ALERT: Layer 0 ls1 is close to zero!")
    except (AttributeError, IndexError, RuntimeError, ValueError):
        pass


def load_split_weights(gen_model, und_model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f" Loading monolithic checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Support different checkpoint formats
    if 'student_model' in ckpt:
        state_dict = ckpt['student_model']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    gen_dict = OrderedDict()
    und_dict = OrderedDict()

    # Counters: make sure every section has enough LayerScale entries loaded
    stats = {
        "gen_enc_ls1": 0, "gen_enc_ls2": 0,  # expect 12, 12
        "und_enc_ls1": 0, "und_enc_ls2": 0,  # expect 12, 12
        "global_ls1": 0,  "global_ls2": 0    # expect 12, 12
    }

    print("   Splitting and Remapping weights...")

    for key, value in state_dict.items():
        # 1. Strip the DDP prefix
        if key.startswith("module."):
            key = key[7:]

        # =========================================================
        # A. Global Blocks (belong to Gen_ViT)
        # Key format: global_blocks.0.ls1... global_blocks.11.ls2
        # =========================================================
        if key.startswith("global_blocks."):
            gen_dict[key] = value
            # Count
            if "ls1" in key:
                stats["global_ls1"] += 1
            if "ls2" in key:
                stats["global_ls2"] += 1
            continue

        # =========================================================
        # B. Encoder Layers (0-11 -> Gen, 12-23 -> Und)
        # Key format: encoder.layers.X.ls1...
        # =========================================================
        if key.startswith("encoder.layers."):
            # Parse the layer index: encoder.layers.12.ls1 -> idx=12, suffix="ls1"
            parts = key.split(".")
            try:
                layer_idx = int(parts[2])
                suffix = ".".join(parts[3:])  # ls1, attn.qkv.weight, etc.
            except ValueError:
                continue  # skip keys that cannot be parsed

            # --- Case 1: first 12 layers (0-11) -> Gen_ViT ---
            if layer_idx < 12:
                gen_dict[key] = value  # copy as-is
                if "ls1" in suffix:
                    stats["gen_enc_ls1"] += 1
                if "ls2" in suffix:
                    stats["gen_enc_ls2"] += 1

            # --- Case 2: last 12 layers (12-23) -> Und_ViT ---
            # Must rename: 12->0, 13->1... 23->11
            else:
                new_layer_idx = layer_idx - 12
                new_key = f"encoder.layers.{new_layer_idx}.{suffix}"

                und_dict[new_key] = value
                if "ls1" in suffix:
                    stats["und_enc_ls1"] += 1
                if "ls2" in suffix:
                    stats["und_enc_ls2"] += 1

            continue

        # =========================================================
        # C. Distribute the remaining components (embeddings, heads, bottlenecks)
        # =========================================================

        # 1. Pos Embeds -> Gen
        if key in ["global_block_pos_embed", "decoder_pos_embed"]:
            gen_dict[key] = value
            continue

        # 2. Embeddings -> Gen
        if key.startswith("embeddings."):
            gen_dict[key] = value
            continue

        # 3. Flow Head -> Gen
        if key.startswith("flow_head."):
            gen_dict[key] = value
            continue

        # 4. Proj (1024->64) -> Gen
        if key.startswith("encoder.chal_proj."):
            new_key = key.replace("encoder.chal_proj.", "chal_proj.")
            gen_dict[new_key] = value
            continue

        # 5. Unproj Generation (64->1024) -> Gen
        if key.startswith("chal_unproj_generation."):
            gen_dict[key] = value
            continue

        # 6. Unproj Understanding (64->1024) -> Und
        if key.startswith("encoder.chal_unproj_understanding."):
            new_key = key.replace("encoder.chal_unproj_understanding.", "chal_unproj_understanding.")
            und_dict[new_key] = value
            continue

    # =========================================================
    # Verification report
    # =========================================================
    print("\n [Precise LayerScale Verification]")
    print(f"   Gen Encoder (0-11)  | ls1: {stats['gen_enc_ls1']}/12  | ls2: {stats['gen_enc_ls2']}/12")
    print(f"   Und Encoder (12-23) | ls1: {stats['und_enc_ls1']}/12  | ls2: {stats['und_enc_ls2']}/12")
    print(f"   Global Blocks       | ls1: {stats['global_ls1']}/12  | ls2: {stats['global_ls2']}/12")

    # Strict check: if any count is not 12, raise or warn
    if any(v != 12 for v in stats.values()):
        print(" WARNING: LayerScale parameters count mismatch! Check your checkpoint.")
    else:
        print(" All LayerScale parameters mapped perfectly.")

    # =========================================================
    # Perform the loading
    # =========================================================
    print("\n Loading Gen_ViT...")
    # strict=False ignores the und-related parts absent from the Gen model
    missing_gen, _ = gen_model.load_state_dict(gen_dict, strict=False)
    # Filter out unimportant keys
    real_missing_gen = [k for k in missing_gen if "LPIPS" not in k and "latent_" not in k and "processor_" not in k]
    if real_missing_gen:
        print(f"    Gen Missing: {real_missing_gen[:5]}")
    else:
        print("    Gen Loaded OK.")

    print("\n Loading Und_ViT...")
    # strict=False ignores the gen-related parts absent from the Und model
    missing_und, _ = und_model.load_state_dict(und_dict, strict=False)
    if missing_und:
        print(f"    Und Missing: {missing_und[:5]}")
    else:
        print("    Und Loaded OK.")

    # =========================================================
    # Final numeric verification (a few sample points)
    # =========================================================
    print("\n Value Check (Sample):")
    verify_gen_weights_against_ckpt(gen_model, checkpoint_path)
    return gen_model, und_model

# [Change point] adapt to your keys: latent_mean, latent_std


def load_latent_stats(model, stats_path, device, dtype):
    """Load precomputed latent mean/std/eps"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Latent stats file not found: {stats_path}")

    print(f" Loading latent statistics from: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")

    # 1. Mean
    if 'latent_mean' in stats:
        mean = stats['latent_mean']
    else:
        raise ValueError(f"Missing 'latent_mean' in {stats_path}. Keys: {stats.keys()}")

    # 2. Std (the file uses latent_std)
    if 'latent_std' in stats:
        std = stats['latent_std']
    elif 'latent_std_dev' in stats:
        std = stats['latent_std_dev']
    else:
        raise ValueError(f"Missing 'latent_std' in {stats_path}. Keys: {stats.keys()}")

    # 3. Eps (fall back to a default if absent from the file)
    if 'latent_eps' in stats:
        eps = stats['latent_eps']
    else:
        print("    'latent_eps' not found, using default 1e-6")
        eps = 1e-6

    # Register the latent statistics
    # Note: the model's internal buffer may be named 'latent_std_dev'; we assign the
    # file's 'latent_std' to it
    model.register_buffer('latent_mean', mean.to(device).to(dtype))
    model.register_buffer('latent_std_dev', std.to(device).to(dtype))

    if isinstance(eps, torch.Tensor):
        model.register_buffer('latent_eps', eps.to(device).to(dtype))
    else:
        model.register_buffer('latent_eps', torch.tensor(eps).to(device).to(dtype))

    # Register the processor (ImageNet) statistics
    p_mean = torch.tensor(DATA_NORM["mean"]).view(1, 3, 1, 1).to(device).to(dtype)
    p_std = torch.tensor(DATA_NORM["std"]).view(1, 3, 1, 1).to(device).to(dtype)
    model.register_buffer('processor_mean', p_mean)
    model.register_buffer('processor_std', p_std)

    print(f" Latent stats loaded. Mean: {mean.shape}, Std: {std.shape}")


def compute_metrics(img_origin_pil, img_recon_np):
    img1_np = np.asarray(img_origin_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS), dtype=np.float32) / 255.
    img2_np = np.asarray(img_recon_np, dtype=np.float32) / 255.
    ssim = ssim_loss(img1_np, img2_np, multichannel=True, data_range=1.0, channel_axis=-1)
    psnr = psnr_loss(img1_np, img2_np)
    return psnr, ssim

# ================= Main Logic =================

if __name__ == "__main__":
    print(f" Running on device: {DEVICE} ({DTYPE})")

    # 1. Config Setup
    try:
        config = HydraTokConfig.from_pretrained(CONFIG_DIR)
    except OSError:
        print(" Config not found, creating default config.")
        config = HydraTokConfig()

    # Force-override the layer count config
    config.gen_layers = 12
    config.und_layers = 12
    config.latent_stats_path = None

    # CPU compatibility
    if DEVICE == "cpu":
        config.use_flash_attn = False
        print(" Flash Attention disabled for CPU.")
    else:
        config.use_flash_attn = True

    # 2. Init Models
    print("Initializing Models...")
    gen_vit = Gen_ViT(config).to(DEVICE).to(DTYPE).eval()
    und_vit = Und_ViT(config).to(DEVICE).to(DTYPE).eval()

    # Load the statistics
    load_latent_stats(gen_vit, LATENT_STATS_PATH, DEVICE, DTYPE)

    # 3. Load Weights
    load_split_weights(gen_vit, und_vit, CKPT_PATH)
    # print("Gen_VIT",gen_vit)

    # exit(0)
    # 4. Prepare Image
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f" Image not found: {TARGET_IMAGE_PATH}")
        exit()

    print(f"Processing Image: {TARGET_IMAGE_PATH}")
    img_raw = Image.open(TARGET_IMAGE_PATH).convert('RGB')
    transform = get_transform(IMAGE_SIZE)
    input_tensor = transform(img_raw).unsqueeze(0).to(DEVICE).to(DTYPE)  # [1, 3, 448, 448]

    # 5. Inference Pipeline
    with torch.no_grad():
        print("\n--- [Step 1] Gen_ViT: Encoding ---")
        latents, cls_token = gen_vit.encode(input_tensor)
        print(f"  Latents Shape: {latents.shape}")
        print(f"  CLS Token Shape: {cls_token.shape}")

        print("\n--- [Step 2] Und_ViT: Understanding (Validation) ---")
        und_features = und_vit(image_latents=latents, cls_token=cls_token)
        print(f"  Understanding Feats Shape: {und_features.shape}")

        print("\n--- [Step 3] Gen_ViT: Decoding (Reconstruction) ---")
        recon_tensor = gen_vit.decode(latents)

        # Post-process
        recon_tensor = recon_tensor.float().cpu()
        recon_np = (recon_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]

    # 6. Save & Metrics
    psnr, ssim = compute_metrics(img_raw, recon_np)
    save_path = "reconstruction_split_test.png"
    Image.fromarray(recon_np).save(save_path)

    print(f"\n Reconstruction Complete!")
    print(f" Metrics -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    print(f" Saved to: {os.path.abspath(save_path)}")

    # Save Gen_ViT weights
    gen_vit_save_path = "gen_vit_weights.pth"
    print(f"\n Saving Gen_ViT weights...")
    torch.save(gen_vit.state_dict(), gen_vit_save_path)
    print(f" Gen_ViT weights saved to: {os.path.abspath(gen_vit_save_path)}")

    und_vit_save_path = "und_vit_weights.pth"
    print(f"\n Saving Und_ViT weights...")
    torch.save(und_vit.state_dict(), und_vit_save_path)
    print(f" Und_ViT weights saved to: {os.path.abspath(und_vit_save_path)}")

    # Cleanup
    del gen_vit, und_vit
    torch.cuda.empty_cache()
    gc.collect()
