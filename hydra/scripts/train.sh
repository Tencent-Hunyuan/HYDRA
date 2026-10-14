#!/bin/bash
# ==========================================================
# HYDRA single-node training launcher
#
# Usage:
#   bash train.sh <config.yaml> [key=value overrides...]
#
# Examples:
#   bash train.sh configs/hydra_1.5b_stage1.yaml
#   bash train.sh configs/hydra_1.5b_stage2.yaml training.lr=1e-5
#
# Environment overrides:
#   NUM_GPUS          number of GPUs (default: all visible)
#   ACCELERATE_CONFIG accelerate config (default: configs/zero2.yaml)
# ==========================================================
set -e

cd "$(dirname "$0")/.."

CONFIG="${1:?Usage: bash train.sh <config.yaml> [key=value overrides...]}"
shift

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/zero2.yaml}

echo " HYDRA training | config: $CONFIG | GPUs: $NUM_GPUS | accelerate: $ACCELERATE_CONFIG"

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    train.py "config=$CONFIG" "$@"
