
export WANDB_DISABLED=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY=""


export EXP_NAME="pref_wan2_1"

export VLLM_SERVER_IP=localhost

API_URL=http://${VLLM_SERVER_IP}:8080

OUTPUT_DIR=outputs/${EXP_NAME}

torchrun --nnodes=2 --nproc_per_node=8 --master_port=8081 \
    fastvideo/train_wan_2_1_pref_grpo_lora.py \
    --seed 42 \
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --vae_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir data/.cache \
    --data_json_path data/train_data_wan2.1/rl_embeddings/videos2caption.json \
    --exp_name ${EXP_NAME}\
    --num_train_epochs 3\
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 100 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo \
    --h 240 \
    --w 416 \
    --t 33 \
    --sampling_steps 50 \
    --eta 0.3 \
    --lr_warmup_ratio 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --cfg_infer 5.0 \
    --use_unifiedreward_think \
    --use_clip \
    --grpo_step_mode flow \
    --api_url ${API_URL} \
    --lora_rank 16 \
    --lora_alpha 32 \