# HYDRA-TOK: Representation-Harmonized ViT Tokenizer

A pure-ViT tokenizer that serves **both** visual understanding and
generation from one representation — no quantization, no separate
encoders.

<div align="center">
<img src="../assets/architecture.png" width="95%">
<br>
<sub>Gen-ViT → GSB → Sem-ViT, with the Pixel Flow Decoder branching
off the bottleneck.</sub>
</div>

**Data flow** (448×448 RGB, 14×14 patches → 1024 tokens):

1. `Gen-ViT` (layers 1-12, 1024-d) — structure-preserving primitives
2. `GSB` — compress 1024 → **64 ch**, KL + cosine-consistency, then restore
3. two consumers of the 64-ch latent:
   - `Pixel Flow Decoder` → pixels (flow + LPIPS + GAN)
   - `Sem-ViT` (layers 13-24) → semantics for the LLM

The bottleneck width is the crux: 64 channels is enough for robust
synthesis, small enough to filter the noise that would otherwise hurt
semantics. Restoring dimensionality afterwards lets Sem-ViT build rich
representations on top of the same latent the decoder reads.

**Training — two stages** (`configs/stage1.yaml`, `configs/stage2_gan.yaml`):

| Stage | What trains | Losses |
|---|---|---|
| 1 | Full backbone, distilled from a frozen **InternViT-300M-448px-V2.5** teacher (per-layer cosine alignment on Gen-ViT *and* Sem-ViT) | flow + KL + bottleneck consistency + 0.1·LPIPS |
| 2 | Pixel Flow Decoder refinement at scale | flow + 1.0·LPIPS + adaptive DINO-discriminator GAN |

## Setup

```bash
pip install -r requirements.txt
```

## Training

Every hyperparameter lives in `configs/stage1.yaml` / `configs/stage2_gan.yaml` —
the training scripts hold no defaults of their own.

```bash
# stage 1 — set your dataset paths in configs/stage1.yaml first
bash scripts/train.sh stage1 --config configs/stage1.yaml

# stage 2 — resume from the stage-1 checkpoint
bash scripts/train.sh stage2_gan --config configs/stage2_gan.yaml
```

`--config` is optional (it defaults to the config shown above) and may point at a
partial YAML — omitted fields fall back to the shipped values. All fields can
also be overridden on the command line:

```bash
bash scripts/train.sh stage1 --config configs/stage1.yaml --lr=1e-4 --epochs=10
```

Unknown keys are rejected rather than silently ignored, in both the config file
and the CLI overrides.

Datasets (`dataset_type` in the YAML):

| `dataset_type` | data |
|---|---|
| `imagenet` | ImageNet-1K (`<imagenet_path>/train`) |
| `custom` | JSON-annotation datasets listed in a YAML (`--yaml_path`) |
| `sam` | SAM-1B (public, Meta) |
| `imagenet_custom`, `imagenet_custom_sam`, `combined` | unions of the above |

Multi-node: fill `hosts.txt` (one hostname per line, passwordless SSH)
and run `bash scripts/train_multinode.sh`.

## Evaluation

```bash
bash tools/eval_all.sh <experiment_name> hydra_tok
# 1) PSNR/SSIM + rFID on ImageNet val  (eval_top12.py)
# 2) rFID from saved image folders     (eval_fid.py)
# 3) latent statistics                 (calculate_latent_top12.py)
```

Requires `pyiqa` and `pytorch-fid` (`pip install pyiqa pytorch-fid`).

rFID uses the **FID-Inception** network (`pytorch_fid.inception.InceptionV3`) — the
TensorFlow-Inception port that published FID numbers are computed against, not
torchvision's Inception. Weights are downloaded once from the public
`pytorch-fid` release URL and cached in `~/.cache/torch/hub/checkpoints`.
Both eval paths share this extractor, so their rFID numbers are comparable:
`eval_top12.py` featurizes in memory during reconstruction, `eval_fid.py`
re-reads the PNGs written to disk (the two differ only by PNG round-tripping).

Reported numbers use 448×448 bicubic resize + center crop. Interpolation matters
— bilinear pre-smooths the reference image and inflates PSNR by roughly 1.5 dB.

## Model definition

One model definition, `hydra_tok/` — `HydraTokVisionModel`, built from
`hydra_tok/config.json`: 550M parameters, 24 layers, 1024-d hidden,
448×448 input, 14×14 patches, 64-channel latents. The stage-1 and
stage-2 recipes are not separate architectures; they differ only in the
config (`use_gan`, `lpips_weight`, `use_kl_loss`, `use_bn_loss`).

A released checkpoint is two files:

| File | Contents |
|---|---|
| `gen_vit_weights.pth` | Gen-ViT + GSB + Pixel Flow Decoder |
| `und_vit_weights.pth` | Sem-ViT half (`Und_ViT`, consumed by the LLM) |

## License

See the repository root `LICENSE`.
