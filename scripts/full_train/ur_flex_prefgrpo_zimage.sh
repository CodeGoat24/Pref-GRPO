#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

wandb_offline ""

export EXP_NAME="unifiedreward_flex_zimage"

API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path Tongyi-MAI/Z-Image
  --vae_model_path Tongyi-MAI/Z-Image
  --data_json_path data/unigenbench_train_data_zimage/rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --train_batch_size 1
  --dataloader_num_workers 4
  --learning_rate 1e-5
  --gradient_accumulation_steps 3
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 20
  --eta 0.7
  --rollout_guidance_scale 3.0
  --num_generations 9
  --reward_spec '{"unifiedreward_flex": 0.7, "clip": 0.3}'
  --api_url "${API_URL}"
  --selective_checkpointing 0.5
)

torchrun --nnodes=4 --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=8081 \
  fastvideo/train_zimage.py \
  "${TRAIN_ARGS[@]}"
