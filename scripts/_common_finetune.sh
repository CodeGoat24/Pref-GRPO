#!/usr/bin/env bash
set -euo pipefail

COMMON_TRAIN_ARGS=(
  --seed 42
  --cache_dir data/.cache
  --num_train_epochs 3
  --gradient_checkpointing
  --num_latent_t 1
  --sp_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 4
  --lr_scheduler constant
  --mixed_precision bf16
  --checkpointing_steps 20
  --allow_tf32
  --cfg 0.0
  --lr_warmup_ratio 0
  --sampler_seed 1223627
  --max_grad_norm 1.0
  --weight_decay 0.0001
  --shift 3
  --use_group
  --ignore_last
  --timestep_fraction 0.6
  --clip_range 1e-4
  --adv_clip_max 5.0
  --kl_beta 0
  --init_same_noise
  --grpo_step_mode flow
  --rationorm
  --eval_every_steps 10
  --eval_num_prompts 64
  # --apply_gdpo
  # --use_ema
  # --ema_update_interval 1
  # --ema_decay 0.99
  # --ema_use_in_checkpoint
)

wandb_online() {
  local api_key="${1:-}"
  export WANDB_DISABLED=false
  export WANDB_BASE_URL="https://api.wandb.ai"
  export WANDB_MODE=online
  export WANDB_API_KEY="${api_key}"
}

wandb_offline() {
  local api_key="${1:-}"
  export WANDB_DISABLED=false
  export WANDB_BASE_URL="https://api.wandb.ai"
  export WANDB_MODE=offline
  export WANDB_API_KEY="${api_key}"
}

wandb_disabled() {
  export WANDB_DISABLED=true
  export WANDB_BASE_URL="https://api.wandb.ai"
  export WANDB_MODE=disabled
  export WANDB_API_KEY=""
}
