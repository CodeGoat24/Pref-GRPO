
GPU_NUM=8

MODEL_PATH=Wan-AI/Wan2.1-T2V-14B-Diffusers

# transformer ckpt path
TRANSFORMER_PATH=

OUTPUT_DIR=./Wan21_eval

# lora path
LORA_PATH=


HEIGHT=480
WIDTH=832

mkdir -p ${OUTPUT_DIR}

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 inference/py/wan_evaluation.py \
    --model_path ${MODEL_PATH} \
    --output_dir $OUTPUT_DIR \
    --prompt_dir ./data/vbench.txt \
    --num_frames 33 \
    --num_inference_steps 30 \
    --base_seed 42 \
    --enable_tf32 \
    --enable_sdpa \
    --compile_transformer \
    --compile_mode reduce-overhead \
    --vae_dtype fp32 \
    --batch_size 2 \
    --height $HEIGHT \
    --width $WIDTH \
    ${LORA_PATH:+--lora_ckpt_path "$LORA_PATH"} \
    ${TRANSFORMER_PATH:+--pretrained_model_name_or_path "$TRANSFORMER_PATH"}
