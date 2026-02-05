GPU_NUM=8 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
OUTPUT_DIR="data/train_data_wan2.2/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/py/preprocess_wan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./data/video_prompts.txt"