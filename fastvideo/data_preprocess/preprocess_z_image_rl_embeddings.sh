GPU_NUM=8
MODEL_PATH="Tongyi-MAI/Z-Image"
OUTPUT_DIR="data/unigenbench_train_data_zimage/rl_embeddings"

# pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_zimage_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_sequence_length 512 \
    --prompt_dir "data/unigenbench_train_data.txt"
