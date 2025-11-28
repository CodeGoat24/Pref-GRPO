export http_proxy=
echo ${LOCAL_IP}

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve CodeGoat24/UnifiedReward-Think-qwen-7b \
    --host ${LOCAL_IP} \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 2 \
    --port 8080