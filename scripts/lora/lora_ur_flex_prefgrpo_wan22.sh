#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

wandb_offline ""

export OVERALL_WEIGHT=0.3
export DIM_WEIGHT=0.7
export CATEGORY_WEIGHTS=1.0,1.0,1.0

export EXP_NAME="unifiedreward_flex_wan22_14b"

API_URL="http://localhost:8080"
OUTPUT_DIR=outputs/$EXP_NAME

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --use_lora
  --pretrained_model_name_or_path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --vae_model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --data_json_path data/train_data_wan2.2/rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --train_batch_size 1
  --dataloader_num_workers 4
  --learning_rate 4e-5
  --output_dir "${OUTPUT_DIR}"
  --h 240
  --w 416
  --t 33
  --sampling_steps 20
  --eta 0.7
  --num_generations 6
  --gradient_accumulation_steps 2
  --cfg_infer 4.0
  --api_url "${API_URL}"
  --checkpointing_steps 10
  --timestep_fraction 0.8
  --lora_rank 64
  --lora_alpha 128
  --reward_spec '{"unifiedreward_flex": 0.7, "clip": 0.3}'
  --boundary_ratio 0.875
  --cfg_infer_2 3.0
  # KL
  --kl_beta 0.004
  --kl_adaptive
  --kl_target 0.0025
  --kl_horizon 80
  --kl_ema_alpha 0.2
  --kl_beta_min 0.004
  --kl_beta_max 0.006
  --eval_every_steps 10
  --eval_num_prompts 64
)
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

torchrun --nnodes=4 --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=8081 \
  fastvideo/train_wan22.py \
  "${TRAIN_ARGS[@]}"
