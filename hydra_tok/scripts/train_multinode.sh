#!/bin/bash

# ==========================================================
# Section 1: user configuration (edit for your project)
# ==========================================================
# Optional debugging flags - uncomment if you need to diagnose NCCL or
# distributed-setup issues:
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# [Edit 1]: path of the host list file (one hostname/IP per line) - the only thing you must change
# Example host file content:
# host_or_ip_1
# host_or_ip_2
# host_or_ip_3
HOST_FILE="./hosts.txt"

# [Edit 2]: repository root (defaults to this script's directory)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# [Edit 3]: training script to run
PYTHON_SCRIPT="train_stage2_gan.py"

# [Edit 4]: arguments for the training script
TRAINING_ARGS=""


# ==========================================================
# Section 2: automated setup (usually no changes needed)
# ==========================================================

if [ ! -f "$HOST_FILE" ]; then  # check the host file exists
    echo "ERROR: host file '$HOST_FILE' not found. Point HOST_FILE at your real host list."
    exit 1
fi

# --- 1. Read the host file and set distributed-training variables ---
HOSTS=($(cat "$HOST_FILE" | grep -v '^\s*#' | grep -v '^\s*$' | sort | uniq))
export NNODES=${#HOSTS[@]}         # number of nodes (--nnodes), computed automatically
export MASTER_ADDR=${HOSTS[0]}     # master address (--master_addr), first host in the list by default
export N_GPUS_PER_NODE=8           # GPUs per node (--nproc_per_node), adjust if needed

if [ "$NNODES" -eq 0 ]; then
    echo "ERROR: host file is empty or invalid."
    exit 1
fi

echo "Total nodes (NNODES): $NNODES"
echo "Master address (MASTER_ADDR): $MASTER_ADDR"
echo "GPUs per node: $N_GPUS_PER_NODE"


# --- 2. A fixed master port ---
export MASTER_PORT=29123
echo "Using master port: $MASTER_PORT"


# ==========================================================
# Section 3: launch on every node via ssh, in order
# ==========================================================
echo "Launching on all nodes..."

NODE_RANK=0
for HOSTNAME in "${HOSTS[@]}"; do

    echo "--------------------------------------------------"
    echo "Starting node: $HOSTNAME (rank: $NODE_RANK)"

    # send a command block to HOSTNAME via ssh for remote execution
    ssh -n "$HOSTNAME" "
        # --- the following block runs on remote node '$HOSTNAME' ---

        # a. enter the project directory
        cd \"$PROJECT_DIR\"

        # b. python environment
        export PYTHONPATH=\"$PROJECT_DIR\"

        # c. per-node log file
        LOG_DIR=\"$PROJECT_DIR/logs/\$(basename $PYTHON_SCRIPT .py)\"
        mkdir -p \$LOG_DIR
        LOG_FILE=\"\$LOG_DIR/train_node_${NODE_RANK}.log\"

        # d. launch torchrun in the background with nohup
        #    torchrun picks up MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE automatically
        nohup torchrun \\
            --nproc_per_node=$N_GPUS_PER_NODE \\
            --nnodes=$NNODES \\
            --node_rank=$NODE_RANK \\
            --master_addr=$MASTER_ADDR \\
            --master_port=$MASTER_PORT \\
            $PYTHON_SCRIPT $TRAINING_ARGS > \$LOG_FILE 2>&1 &
    "

    # check the ssh command was dispatched successfully
    if [ $? -eq 0 ]; then
        echo "    OK: command dispatched to $HOSTNAME"
    else
        echo "    FAILED to dispatch to $HOSTNAME - check passwordless SSH setup"
    fi

    # advance the node rank
    NODE_RANK=$((NODE_RANK + 1))
done

echo "--------------------------------------------------"
echo "All nodes launched."
echo "Monitor training by tailing the per-node logs:"
echo "   tail -f $PROJECT_DIR/logs/$(basename $PYTHON_SCRIPT .py)/train_node_0.log"
