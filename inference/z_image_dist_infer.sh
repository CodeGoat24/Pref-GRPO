GPU_NUM=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL_PATH="Tongyi-MAI/Z-Image"
CKPT_PATH=""
LORA_PATH=""

OUTPUT_DIR='outputs/z_image_infer'

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 \
    inference/py/z_image_multi_node_inference.py \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/unigenbench_test_data.csv" \
    --model_path ${MODEL_PATH} \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"} \
    ${LORA_PATH:+--lora_path "$LORA_PATH"}
