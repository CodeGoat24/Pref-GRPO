#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

export EXP_NAME="unifiedreward_flex_flux2_klein_9b"

export VLLM_MAX_WORKERS=32
export VLLM_LOG_STATS=0

export OVERALL_WEIGHT=0.3
export DIM_WEIGHT=0.7
export CATEGORY_WEIGHTS=1.0,0.7,0.7

wandb_offline ""
API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-9B
  --data_json_path data/flux2_klein_rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --train_guidance_scale 4.0
  --dataloader_num_workers 1
  --learning_rate 5e-5
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 25
  --eta 0.7
  --num_generations 9
  --gradient_accumulation_steps 3
  --api_url "${API_URL}"
  --reward_spec '{"unifiedreward_flex": 0.7, "clip": 0.3}'
  --timestep_fraction 0.6
  --rollout_store_device cpu
  --eval_every_steps 10
  --eval_num_prompts 128
  --eval_guidance_scale 4.0
  --eval_num_inference_steps 25
  --apply_gdpo
  --use_lora
  --lora_rank 64
  --lora_alpha 128
)



torchrun --nnodes=4 --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=8081 \
  fastvideo/train_flux2_klein.py \
  "${TRAIN_ARGS[@]}"
