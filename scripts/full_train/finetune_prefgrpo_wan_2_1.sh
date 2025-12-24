#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

wandb_online ""


export EXP_NAME="pref_wan2_1"

API_URL="http://localhost:8080"
OUTPUT_DIR="data/outputs/grpo"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --vae_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --data_json_path data/train_data_wan2.1/rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --train_batch_size 1
  --dataloader_num_workers 4
  --learning_rate 5e-6
  --output_dir "${OUTPUT_DIR}"
  --h 240
  --w 416
  --t 33
  --sampling_steps 20
  --eta 0.7
  --gradient_accumulation_steps 2
  --num_generations 6
  --cfg_infer 5.0
  --reward_spec '{"unifiedreward_think": 0.4, "clip": 0.6}'
  --kl_beta 0.004
  --api_url "${API_URL}"
  --checkpointing_steps 20
)

torchrun --nnodes=4 --nproc_per_node=8 --node_rank="${INDEX}" --master_addr="${CHIEF_IP}" --master_port=8081 \
  fastvideo/train_wan_2_1.py \
  "${TRAIN_ARGS[@]}"
