#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

export EXP_NAME="unifiedreward_flex_pair_flux1_kontext_edit"

export VLLM_MAX_WORKERS=32
export VLLM_LOG_STATS=0

wandb_offline ""
API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-Kontext-dev
  --data_json_path data/flux1_kontext_edit_embeddings/edit_data.json
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --train_guidance_scale 2.5
  --dataloader_num_workers 1
  --learning_rate 3e-5
  --output_dir "${OUTPUT_DIR}"
  --h 512
  --w 512
  --condition_h 512
  --condition_w 512
  --t 1
  --sampling_steps 15
  --eta 0.7
  --num_generations 9
  --gradient_accumulation_steps 3
  --api_url "${API_URL}"
  --reward_spec '{"unifiedreward_edit_pairwise": 1.0}'
  --rollout_store_device cpu
  --eval_every_steps 10
  --eval_num_prompts 32
  --eval_guidance_scale 2.5
  --eval_num_inference_steps 15
  --skip_path_check
  --use_lora
  --lora_rank 64
  --lora_alpha 128
)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=8081 \
  fastvideo/train_flux1_kontext_edit.py \
  "${TRAIN_ARGS[@]}"
