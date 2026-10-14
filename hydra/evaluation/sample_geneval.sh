#!/bin/bash
# GenEval sampling for the released HYDRA checkpoints (1.5B / 7B).
#
# Prompt file: prompts/geneval_detailed_caption.csv — the *detailed caption*
# prompts (the 'detailed_caption' field of the official geneval_prompt.jsonl,
# exported to CSV with a 0-based index). These are the prompts behind the
# released GenEval score of 0.86. Do NOT switch the prompt file without
# re-measuring: with the rewritten prompts
# (prompts/geneval_meta_data_rewritten.jsonl) the same checkpoints score
# ~0.78-0.80, and with the original short prompts ~0.64.
#
# Notes on inference_t2i.py (current version):
#   * it must be launched with torchrun (DDP); prompt sharding across GPUs is
#     done internally by rank, so one process per GPU is launched here;
#   * it reads `t2i_prompts_file` and `output_dir` (NOT `validation_prompts_file`
#     / `outdir` — the old 8x-independent-python3 launch style no longer works);
#   * sampling settings below are the released GenEval protocol:
#     transport.guidance_scale=5.0, transport.num_inference_steps=50,
#     transport.time_shifting_factor=5.0 (from configs/inference.yaml),
#     n_samples=4 per prompt. Note every sampling knob lives under `transport.`;
#     a bare `guidance_scale=...` is now rejected instead of silently ignored.
#   * for DPG-Bench use transport.num_inference_steps=100; for the 7B use
#     transport.time_shifting_factor=3.0 (and 200 steps for GenEval/WISE).
#
# Usage (from the hydra project root or anywhere):
#   bash evaluation/sample_geneval.sh                # HYDRA 1.5B (default)
#   MODEL=7b bash evaluation/sample_geneval.sh       # HYDRA 7B
# Optional env overrides:
#   PROMPTS_FILE  OUTDIR  NPROC  MASTER_PORT  CKPT_ROOT  QWEN_ROOT  EXTRA_ARGS
#
# After sampling, score with the official GenEval detector pipeline. Each
# prompt folder's metadata.jsonl must carry the full geneval record
# (tag/include/exclude) for the scorer — inference_t2i.py only writes the
# prompt itself; see the scoring helper in the internal geneval workspace.

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

MODEL=${MODEL:-1.5b}
NPROC=${NPROC:-8}
MASTER_PORT=${MASTER_PORT:-29500}
PROMPTS_FILE=${PROMPTS_FILE:-prompts/geneval_detailed_caption.csv}
OUTDIR=${OUTDIR:-hydra_geneval_samples_${MODEL}}

# Weight locations: ckpt/ ships with the release (two levels above the hydra
# project root); the Qwen2.5 base model must be downloaded separately — set
# QWEN_ROOT to where it lives on your machine.
CKPT_ROOT=${CKPT_ROOT:-../../ckpt}
QWEN_ROOT=${QWEN_ROOT:-Qwen}

HYDRA_TOK=${CKPT_ROOT}/hydra_tok_gan
LATENT_STATS=${CKPT_ROOT}/hydra_tok_latent_stats.pt

case ${MODEL} in
  1.5b|1.5B)
    MODEL_PATH=${CKPT_ROOT}/hydra_1.5b
    LLM_PATH=${QWEN_ROOT}/Qwen2.5-1.5B-Instruct/qwen/Qwen2.5-1.5B-Instruct
    LLM_OPTS=""
    ;;
  7b|7B)
    MODEL_PATH=${CKPT_ROOT}/hydra_7b
    LLM_PATH=${QWEN_ROOT}/Qwen2.5-7B-Instruct
    LLM_OPTS="model.llm.hidden_size=3584"
    ;;
  *)
    echo "MODEL must be 1.5b or 7b (got: ${MODEL})"; exit 1
    ;;
esac

echo "Model:      HYDRA ${MODEL}  (${MODEL_PATH})"
echo "Prompts:    ${PROMPTS_FILE}"
echo "Output dir: ${OUTDIR}"
echo "GPUs:       ${NPROC} (master_port ${MASTER_PORT})"

PYTHONPATH=. torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
    evaluation/inference_t2i.py \
    config=configs/inference.yaml \
    model.hydra_tok_model.pretrained_model_path=${HYDRA_TOK} \
    model.hydra_tok_model.latent_stats_path=${LATENT_STATS} \
    model.llm.llm_model_path=${LLM_PATH} \
    model_path=${MODEL_PATH} \
    t2i_prompts_file=${PROMPTS_FILE} \
    output_dir=${OUTDIR} \
    transport.guidance_scale=5.0 transport.num_inference_steps=50 \
    batch_size=4 n_samples=4 \
    skip_grid=true \
    ${LLM_OPTS} ${EXTRA_ARGS}
