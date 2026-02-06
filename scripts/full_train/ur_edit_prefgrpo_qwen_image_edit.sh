#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

export EXP_NAME="qwen-image-edit-grpo"

export VLLM_MAX_WORKERS=32
export VLLM_LOG_STATS=0

wandb_offline ""
API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path Qwen/Qwen-Image-Edit
  --vae_model_path Qwen/Qwen-Image-Edit
  --data_json_path data/qwen_image_edit_embeddings/edit_data.json
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --train_guidance_scale 4.0
  --dataloader_num_workers 4
  --learning_rate 1e-5
  --output_dir "${OUTPUT_DIR}"
  --h 720
  --w 720
  --t 1
  --sampling_steps 15
  --eta 0.7
  --num_generations 9
  --gradient_accumulation_steps 3
  --api_url "${API_URL}"
  --reward_spec '{"unifiedreward_edit_pairwise": 1.0}'
  --skip_path_check
  --eval_guidance_scale 4.0
  # --use_lora
  # --lora_rank 32
  # --lora_alpha 64
)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=8081 \
  fastvideo/train_qwen_image_edit.py \
  "${TRAIN_ARGS[@]}"
