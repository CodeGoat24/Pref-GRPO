GPU_NUM=8 
MODEL_PATH="black-forest-labs/FLUX.1-dev"
OUTPUT_DIR="data/unigenbench_train_data/rl_embeddings"

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19007 \
    fastvideo/data_preprocess/py/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/unigenbench_train_data.txt"