#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

wandb_online ""

export EXP_NAME="pref_qwenimage_lora"

API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --use_lora
  --pretrained_model_name_or_path Qwen/Qwen-Image
  --vae_model_path Qwen/Qwen-Image
  --data_json_path data/unigenbench_train_data_qwenimage/rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --train_batch_size 1
  --dataloader_num_workers 4
  --learning_rate 1e-4
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 20
  --eta 0.7
  --num_generations 8
  --reward_spec '{"unifiedreward_think": 0.4, "clip": 0.6}'
  --api_url "${API_URL}"
  --selective_checkpointing 0.5
  --lora_alpha 128
  --lora_rank 64
)

torchrun --nnodes=8 --nproc_per_node=8 --node_rank="${INDEX}" --master_addr="${CHIEF_IP}" --master_port=8081 \
  fastvideo/train_qwenimage.py \
  "${TRAIN_ARGS[@]}"
