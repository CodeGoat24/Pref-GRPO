GPU_NUM=8 

MODEL_PATH=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
OUTPUT_DIR='wan_output'

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 \
    inference/wan_multi_node_inference.py \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/video_prompts.txt" \
    --model_path ${MODEL_PATH} \
    --batch_size 1 \
    --dataloader_num_workers 8