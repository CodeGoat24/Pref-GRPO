GPU_NUM=8 

MODEL_PATH=Qwen/Qwen-Image
OUTPUT_DIR='qwenimage_output'
LORA_DIR=

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 \
    inference/py/qwen_image_multi_node_inference.py \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/unigenbench_test_data.csv" \
    --model_path ${MODEL_PATH} \
    ${LORA_DIR:+--lora_dir "$LORA_DIR"}
