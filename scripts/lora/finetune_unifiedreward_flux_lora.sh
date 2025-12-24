#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

wandb_online ""

export EXP_NAME="unifiedreward_flux_lora"

API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --use_lora
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev
  --vae_model_path black-forest-labs/FLUX.1-dev
  --data_json_path data/unigenbench_train_data/rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --dataloader_num_workers 1
  --learning_rate 1e-5
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 25
  --eta 0.7
  --num_generations 16
  --reward_spec '{"unifiedreward_alignment": 1.4, "unifiedreward_style": 0.7}'
  --api_url "${API_URL}"
  --lora_alpha 128
  --lora_rank 64
)

torchrun --nnodes=8 --nproc_per_node=8 --node_rank="${INDEX}" --master_addr="${CHIEF_IP}" --master_port=8081 \
  fastvideo/train_flux.py \
  "${TRAIN_ARGS[@]}"
