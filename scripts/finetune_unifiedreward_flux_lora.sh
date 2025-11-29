
export WANDB_DISABLED=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY=""

export EXP_NAME="UnifiedReward_flux"

export VLLM_SERVER_IP=localhost

API_URL=http://${VLLM_SERVER_IP}:8080

OUTPUT_DIR=outputs/${EXP_NAME}

torchrun --nnodes=8 --nproc_per_node=8 --node_rank=$INDEX --master_addr=${CHIEF_IP} --master_port=8081 \
    fastvideo/train_flux_unifiedreward_lora.py \
    --seed 42 \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --vae_model_path black-forest-labs/FLUX.1-dev \
    --cache_dir data/.cache \
    --data_json_path data/unigenbench_train_data/rl_embeddings/videos2caption.json \
    --exp_name ${EXP_NAME}\
    --num_train_epochs 3\
    --num_sample 100000\
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler constant \
    --mixed_precision bf16 \
    --checkpointing_steps 20 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir ${OUTPUT_DIR} \
    --t 1 \
    --sampling_steps 25 \
    --eta 0.7 \
    --lr_warmup_ratio 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 16 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --init_same_noise \
    --use_unifiedreward \
    --grpo_step_mode flow \
    --api_url ${API_URL} \
    --lora_alpha 256 \
    --lora_rank 128 \
    # --resume_from_lora_checkpoint /path/to/lora/dir
