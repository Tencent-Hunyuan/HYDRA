#!/bin/bash
# ==========================================================
# HYDRA multi-node training launcher (SSH-based)
#
# Usage:
#   bash train_multinode.sh <config.yaml> [key=value overrides...]
#
# Prerequisites:
#   -./hosts.txt: one hostname or IP per line (all nodes)
#   - passwordless SSH from this node to every host
#   - the repository mounted at the same absolute path on every node
#
# Environment overrides:
#   HOST_FILE          host list file           (default:./hosts.txt)
#   N_GPUS_PER_NODE    GPUs per node             (default: 8)
#   ACCELERATE_CONFIG  accelerate config         (default: configs/zero2.yaml)
#   MASTER_PORT        torch/accelerate port     (default: 29500)
# ==========================================================
set -e

CONFIG="${1:?Usage: bash train_multinode.sh <config.yaml> [key=value overrides...]}"
shift

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

HOST_FILE=${HOST_FILE:-./hosts.txt}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/zero2.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$HOST_FILE" ]; then
    echo " Host file '$HOST_FILE' not found. Create it with one hostname/IP per line."
    exit 1
fi

HOSTS=($(cat "$HOST_FILE" | grep -v '^\s*#' | grep -v '^\s*$' | sort | uniq))
export NNODES=${#HOSTS[@]}
export MASTER_ADDR=${HOSTS[0]}

if [ "$NNODES" -eq 0 ]; then
    echo " Host file is empty or invalid."
    exit 1
fi

echo " Nodes (NNODES): $NNODES"
echo " Master (MASTER_ADDR): $MASTER_ADDR"
echo " Port (MASTER_PORT): $MASTER_PORT"
echo " GPUs per node: $N_GPUS_PER_NODE"
echo "---"

NODE_RANK=0
for HOSTNAME in "${HOSTS[@]}"; do
    echo "  Launching node $HOSTNAME (RANK: $NODE_RANK)"
    ssh -n "$HOSTNAME" "
        cd '$PROJECT_DIR'
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=$OMP_NUM_THREADS
        export WANDB_MODE=\${WANDB_MODE:-offline}

        LOG_DIR='$PROJECT_DIR/logs/$(basename "$CONFIG".yaml)'
        mkdir -p \$LOG_DIR
        LOG_FILE=\"\$LOG_DIR/train_node_${NODE_RANK}.log\"
        TOTAL_PROCESSES=$(($NNODES * $N_GPUS_PER_NODE))

        nohup accelerate launch \
            --config_file $ACCELERATE_CONFIG \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            --num_machines $NNODES \
            --machine_rank $NODE_RANK \
            --num_processes \$TOTAL_PROCESSES \
            train.py config=$CONFIG $@ > \$LOG_FILE 2>&1 &
    "

    if [ $? -eq 0 ]; then
        echo "     [node $NODE_RANK] launched -> logs/$(basename "$CONFIG".yaml)/train_node_${NODE_RANK}.log"
    else
        echo "     [node $NODE_RANK] launch FAILED ($HOSTNAME)"
    fi
    NODE_RANK=$((NODE_RANK + 1))
done

echo "--------------------------------------------------"
echo " Monitor with: tail -f logs/$(basename "$CONFIG".yaml)/train_node_0.log"
