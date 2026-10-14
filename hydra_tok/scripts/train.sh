#!/bin/bash
# ==========================================================
# HYDRA-TOK single-node training launcher
#
# Usage:
#   bash train.sh <stage1|stage2_gan> [extra args passed to the trainer...]
#
# Hyperparameters come from configs/<stage>.yaml. --config is optional (that file
# is the default) and any field can be overridden with --key=value.
#
# Examples:
#   bash train.sh stage1
#   bash train.sh stage1 --config configs/stage1.yaml
#   bash train.sh stage1 --config configs/stage1.yaml --lr=1e-4
#   bash train.sh stage2_gan --config configs/stage2_gan.yaml
#
# Environment overrides:
#   N_GPUS  number of GPUs (default: all visible)
#   PORT    torchrun master port (default: 29500)
# ==========================================================
set -e

STAGE="${1:?Usage: bash train.sh <stage1|stage2_gan> [extra args...]}"
shift

case "$STAGE" in
    stage1)    SCRIPT="train_stage1.py" ;;
    stage2_gan) SCRIPT="train_stage2_gan.py" ;;
    *) echo "Unknown stage: $STAGE (expected stage1 or stage2_gan)"; exit 1 ;;
esac

N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
PORT=${PORT:-29500}

cd "$(dirname "$0")/.."

echo " HYDRA-TOK training | script: $SCRIPT | GPUs: $N_GPUS | args: $*"

torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=$PORT \
    "$SCRIPT" "$@"
