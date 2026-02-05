GPU_NUM=8


MODEL_PATH=Wan-AI/Wan2.2-T2V-A14B-Diffusers
OUTPUT_DIR='wan22_test_output'
GUIDANCE_SCALE=4.0
GUIDANCE_SCALE_2=3.0
LORA_PATH=""

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 \
    inference/py/wan22_multi_node_inference.py \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/video_prompts.txt" \
    --model_path ${MODEL_PATH} \
    --guidance_scale $GUIDANCE_SCALE \
    --guidance_scale_2 $GUIDANCE_SCALE_2 \
    --batch_size 1 \
    --dataloader_num_workers 8 \
    ${LORA_PATH:+--lora_ckpt_path "$LORA_PATH"}
