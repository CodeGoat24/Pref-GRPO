GPU_NUM=${GPU_NUM:-8}

# SUPPORTED_REWARDS = {
#     "aesthetic",
#     "clip",
#     "hpsv2",
#     "hpsv3",
#     "pickscore",
#     "unifiedreward_alignment",
#     "unifiedreward_style",
#     "unifiedreward_coherence",
# }

torchrun --nproc_per_node=$GPU_NUM --master_port 29501 \
  tools/image_reward_eval.py \
  --image_dir /path/to/unigenbench/generated/images \
  --prompt_csv data/unigenbench_test_data.csv \
  --reward_spec "clip,aesthetic,pickscore" \
  --api_url http://localhost:8080 \
  --output_json tools/reward_scores.json


