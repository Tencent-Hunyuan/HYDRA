#!/bin/bash
# ==========================================================
# HYDRA-TOK evaluation script: runs eval_top12, eval_fid, calculate_latent in sequence
# Order: PSNR/SSIM -> rFID -> Latent Stats
# Continue with the next task on error
# ==========================================================

# set -e is intentionally not used; keep going on error
# ==========================================================
# Configuration - only `name` needs to be changed
# Dependencies: pip install pyiqa pytorch-fid (install beforehand)
# ==========================================================
name="${1:-hydra-tok-stage2-64c-kl-no-cls-40M-gan}"  # from CLI arg, with default
config_name="${2:-hydra_tok}"
# Base path (defaults to this script's directory, i.e. the tokenizer repo root)
HYDRA_TOK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_BASE="${HYDRA_TOK_ROOT}/results"
IMAGENET_TRAIN="/path/to/dataset/Generation/ImageNet1k/output/ImageNet-1K/train"
IMAGENET_VAL="/path/to/dataset/Generation/ImageNet1k/output/ImageNet-1K/val"
PROCESSOR_PATH="OpenGVLab/InternViT-300M-448px-V2_5"

# Paths are derived from name
# CHECKPOINT_PATH="${CHECKPOINT_BASE}/${name}/final_model.pth"
CHECKPOINT_PATH="${CHECKPOINT_BASE}/${name}/checkpoints/step_10000.pth"
CONFIG_PATH="${HYDRA_TOK_ROOT}/${config_name}"
STATS_SAVE_PATH="${HYDRA_TOK_ROOT}/stats/${name}_latent_stats.pt"
EVAL_OUTPUT_DIR="${HYDRA_TOK_ROOT}/eval_output/${name}"

# Training/evaluation parameters
IMAGE_SIZE=448
EVAL_SIZE=256
BATCH_SIZE=256
NUM_WORKERS=8
N_GPUS=8

# Track per-task status
EVAL_TOP12_STATUS="skipped"
EVAL_FID_STATUS="skipped"
LATENT_STATS_STATUS="skipped"

# ==========================================================
# Print configuration
# ==========================================================
echo "=========================================="
echo "HYDRA-TOK Evaluation Pipeline"
echo "=========================================="
echo "Experiment Name: ${name}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Stats Output: ${STATS_SAVE_PATH}"
echo "Eval Output: ${EVAL_OUTPUT_DIR}"
echo "Image Size: ${IMAGE_SIZE}, Eval Size: ${EVAL_SIZE}"
echo "Batch Size: ${BATCH_SIZE}, GPUs: ${N_GPUS}"
echo "=========================================="

# Check that the checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "   Available checkpoints in ${CHECKPOINT_BASE}/${name}/:"
    ls -la "${CHECKPOINT_BASE}/${name}/" 2>/dev/null || echo "   Directory does not exist"
    exit 1
fi

# Create output directories
mkdir -p "$(dirname ${STATS_SAVE_PATH})"
mkdir -p "${EVAL_OUTPUT_DIR}"

cd "${HYDRA_TOK_ROOT}"

# ==========================================================
# Task selection
# ==========================================================
RUN_EVAL_TOP12="${RUN_EVAL_TOP12:-true}"
RUN_EVAL_FID="${RUN_EVAL_FID:-true}"
RUN_LATENT_STATS="${RUN_LATENT_STATS:-true}"

# ==========================================================
# 1. Evaluate reconstruction quality (PSNR, SSIM, save images)
# ==========================================================
if [ "${RUN_EVAL_TOP12}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Evaluating Reconstruction (PSNR, SSIM)"
    echo "=========================================="
    
    torchrun --nproc_per_node=${N_GPUS} tools/eval_top12.py \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --imagenet-val-path "${IMAGENET_VAL}" \
        --config-path "${CONFIG_PATH}" \
        --output-dir "${EVAL_OUTPUT_DIR}" \
        --keep-images \
        --batch-size ${BATCH_SIZE} \
        --image-size ${IMAGE_SIZE} \
        --eval-size ${EVAL_SIZE}
    
    if [ $? -eq 0 ]; then
        EVAL_TOP12_STATUS="success"
        echo "Reconstruction evaluation complete"
    else
        EVAL_TOP12_STATUS="failed"
        echo "Reconstruction evaluation failed, continuing to next task..."
    fi
fi

# ==========================================================
# 2. Compute rFID (on the saved images)
# ==========================================================
if [ "${RUN_EVAL_FID}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Calculating rFID"
    echo "=========================================="
    
    # Consolidate images first
    echo "Consolidating images..."
    python tools/eval_fid.py \
        --path "${EVAL_OUTPUT_DIR}" \
        --consolidate-only
    
    consolidate_status=$?
    
    # Compute rFID
    echo "Calculating rFID..."
    python tools/eval_fid.py \
        --path "${EVAL_OUTPUT_DIR}"
    
    fid_status=$?
    
    if [ $consolidate_status -eq 0 ] && [ $fid_status -eq 0 ]; then
        EVAL_FID_STATUS="success"
        echo "rFID calculation complete"
    else
        EVAL_FID_STATUS="failed"
        echo "rFID calculation failed, continuing to next task..."
    fi
fi

# ==========================================================
# 3. Compute latent statistics
# ==========================================================
if [ "${RUN_LATENT_STATS}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Step 3: Calculating Latent Statistics"
    echo "=========================================="
    
    torchrun --nproc_per_node=${N_GPUS} tools/calculate_latent_top12.py \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --imagenet-train-path "${IMAGENET_TRAIN}" \
        --config-path "${CONFIG_PATH}" \
        --processor-path "${PROCESSOR_PATH}" \
        --stats-save-path "${STATS_SAVE_PATH}" \
        --batch-size ${BATCH_SIZE} \
        --image-size ${IMAGE_SIZE} \
        --num-workers ${NUM_WORKERS}
    
    if [ $? -eq 0 ]; then
        LATENT_STATS_STATUS="success"
        echo "Latent statistics saved to: ${STATS_SAVE_PATH}"
    else
        LATENT_STATS_STATUS="failed"
        echo "Latent statistics calculation failed"
    fi
fi

# ==========================================================
# Done - show all task statuses
# ==========================================================
echo ""
echo "=========================================="
echo "Evaluation Pipeline Complete!"
echo "=========================================="
echo "Task Status Summary:"
echo "  - PSNR/SSIM (eval_top12):  ${EVAL_TOP12_STATUS}"
echo "  - rFID (eval_fid):         ${EVAL_FID_STATUS}"
echo "  - Latent Stats:            ${LATENT_STATS_STATUS}"
echo ""
echo "Output Paths:"
echo "  - Eval Images:  ${EVAL_OUTPUT_DIR}"
echo "  - Latent Stats: ${STATS_SAVE_PATH}"
echo "=========================================="
