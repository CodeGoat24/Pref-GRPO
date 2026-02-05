
GPU_NUM=8

MODEL_PATH=Wan-AI/Wan2.2-T2V-A14B-Diffusers

OUTPUT_DIR=./Wan22_eval

LORA_PATH=

HEIGHT=480
WIDTH=832
GUIDANCE_SCALE=4.0
GUIDANCE_SCALE_2=3.0

mkdir -p ${OUTPUT_DIR}

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 inference/py/wan22_evaluation.py \
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
    --guidance_scale $GUIDANCE_SCALE \
    --guidance_scale_2 $GUIDANCE_SCALE_2 \
    ${LORA_PATH:+--lora_ckpt_path "$LORA_PATH"} \
    ${TRANSFORMER_PATH:+--pretrained_model_name_or_path "$TRANSFORMER_PATH"}
