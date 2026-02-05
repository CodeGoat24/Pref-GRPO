#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="data/Image_Edit_data"
OUTPUT_DIR="data/flux2_klein_edit_embeddings"
MODEL_PATH="black-forest-labs/FLUX.2-klein-base-9B"

USE_CN="${USE_CN:-0}"
GPU_NUM=8
BATCH_SIZE="${BATCH_SIZE:-1}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ "${USE_CN}" == "1" ]]; then
  EXTRA_ARGS+=(--use_instruction_cn)
fi

torchrun --nproc_per_node=$GPU_NUM --master_port 19017 \
  fastvideo/data_preprocess/py/preprocess_flux2_klein_edit_embedding.py \
  --input_path "${INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --dataloader_num_workers 16 \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"
