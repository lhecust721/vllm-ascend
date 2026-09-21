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
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_INTRA_ROCE_ENABLE=1

export PYTHONHASHSEED=0 # 确保KV键值的唯一性（P/D/Store三方必须一致）
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# ================= Prefill: DP1 x TP4 =================
(
    cd /vllm-workspace/vllm
    export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:${PYTHONPATH:-}"

    exec /usr/local/python3.12.13/bin/python \
        -m vllm.entrypoints.cli.main \
    serve /data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8 \
    --host 0.0.0.0 \
    --port 8195 \
    --max-model-len 20480 \
    --max-num-batched-tokens 20480 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code \
    --block-size 32 \
    --no-disable-hybrid-kv-cache-manager \
    --enable-prefix-caching \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 32}' \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true, "enable_dsa_cp": true}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeHybridConnector",
                    "kv_role": "kv_producer",
                    "kv_port": "31000",
                    "kv_connector_extra_config": {
                        "prefill": { "dp_size": 1, "tp_size": 4 },
                        "decode":  { "dp_size": 2, "tp_size": 2 }
                    }
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "35001",
                        "backend": "mooncake"
                    }
                }
            ]
        }
    }'
) > /workspace/vllm_prefill.log 2>&1 &
