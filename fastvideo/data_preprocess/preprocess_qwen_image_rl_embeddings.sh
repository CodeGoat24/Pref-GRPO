GPU_NUM=8 
MODEL_PATH="Qwen/Qwen-Image"
OUTPUT_DIR="data/unigenbench_train_data_qwenimage/rl_embeddings"

# pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/py/preprocess_qwenimage_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/unigenbench_train_data.txt"

