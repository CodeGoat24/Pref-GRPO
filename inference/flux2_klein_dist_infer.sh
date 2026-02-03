#!/usr/bin/env bash
set -euo pipefail

GPU_NUM=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL_PATH=black-forest-labs/FLUX.2-klein-base-9B
LORA_PATH=""

OUTPUT_DIR='flux_klein_output'

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19010 \
    inference/py/flux2_klein_multi_node_inference.py \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/unigenbench_test_data.csv" \
    --model_path ${MODEL_PATH} \
    --guidance_scale 4.0 \
    --num_inference_steps 50 \
    ${LORA_PATH:+--lora_path "$LORA_PATH"}
