export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1

export TOKENIZERS_PARALLELISM=false

export WANDB_DISABLED=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY=""

export EXP_NAME="Pref-GRPO_QwenImage"

export VLLM_SERVER_IP=${LOCAL_IP}

API_URL=http://${VLLM_SERVER_IP}:8080

OUTPUT_DIR=outputs/${EXP_NAME}

torchrun --nnodes=8 --nproc_per_node=8 --node_rank=$INDEX --master_addr=${CHIEF_IP} --master_port=8081 \
    fastvideo/train_grpo_qwenimage_pref_grpo.py \
    --seed 42 \
    --pretrained_model_name_or_path Qwen/Qwen-Image \
    --vae_model_path Qwen/Qwen-Image \
    --cache_dir data/.cache \
    --data_json_path data/unigenbench_train_data_qwenimage/rl_embeddings/videos2caption.json \
    --exp_name ${EXP_NAME}\
    --num_train_epochs 3\
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 20 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir ${OUTPUT_DIR} \
    --t 1 \
    --sampling_steps 20 \
    --eta 0.7 \
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
    --use_unifiedreward_think \
    --use_clip \
    --grpo_step_mode flow \
    --api_url ${API_URL} \
    --selective_checkpointing 0.5 