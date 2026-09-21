#!/bin/bash

# ================= Mooncake KV Pool (SSD offload) =================
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=$((10 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((80 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((80 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
export MOONCAKE_CONFIG_PATH="/workspace/mooncake.json"

# ================= 基础环境 =================
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_INTRA_ROCE_ENABLE=1

export PYTHONHASHSEED=0 # 确保KV键值的唯一性（P/D/Store三方必须一致）
export VLLM_LOGGING_LEVEL=DEBUG # 验证 save_decode_cache put/lookup/tracker 链路必须 DEBUG
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ================= Prefill: DP1 x TP4 =================
(    
    cd /vllm-workspace/vllm
    export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:${PYTHONPATH:-}"

    exec /usr/local/python3.12.13/bin/python \
        -m vllm.entrypoints.cli.main \
    serve "/data1/weight/GLM-5.2-w4a8" \
    --host 0.0.0.0 \
    --port 8195 \
    --max-model-len 10240 \
    --max-num-batched-tokens 10240 \
    --served-model-name glm5.2 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 10 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --enable-expert-parallel \
    --async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": false, "ascend_compilation_config": {"fuse_muls_add": true}, "scheduler_config": {"recompute_scheduler_enable": false}}' \
    --kv-transfer-config \
    '{
        
         "kv_connector": "AscendStoreConnector",
         "kv_role": "kv_producer",
         "kv_connector_extra_config": {
                "lookup_rpc_port": "35001",
                "backend": "mooncake",
		"consumer_is_to_put": true,
		"save_decode_cache": true
         }
    }'
)  > /workspace/vllm_prefill.log 2>&1 &
