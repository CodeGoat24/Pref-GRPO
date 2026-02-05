GPU_NUM=8
MODEL_PATH="FLUX.2-klein-base-9B"
OUTPUT_DIR="data/flux2_klein_rl_embeddings"

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19017 \
    fastvideo/data_preprocess/py/preprocess_flux2_klein_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/unigenbench_train_data.txt"
