
vllm serve CodeGoat24/UnifiedReward-2.0-qwen3vl-8b \
    --host localhost \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.5 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 16 \
    --port 8080 \
    --enable-prefix-caching \
    --disable-log-requests \
    --mm_processor_cache_gb=500
