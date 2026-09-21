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
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

# ================= Decode: DP1 x TP4 =================
(
    cd /vllm-workspace/vllm
    export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:${PYTHONPATH:-}"

    exec /usr/local/python3.12.13/bin/python \
        -m vllm.entrypoints.cli.main \
    serve /data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8 \
    --host 0.0.0.0 \
    --port 8196 \
    --max-model-len 20460 \
    --max-num-batched-tokens 20460 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 16 \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code \
    --block-size 32 \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --async-scheduling \
    --safetensors-load-strategy prefetch \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 32}' \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "enable_shared_expert_dp": true,
        "enable_dsa_cp": true,
        "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable": true
    }' \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeHybridConnector",
                    "kv_role": "kv_consumer",
                    "kv_port": "31100",
                    "kv_connector_extra_config": {
                        "prefill": { "dp_size": 1, "tp_size": 4 },
                        "decode":  { "dp_size": 2, "tp_size": 2 }
                    }
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "35002",
                        "backend": "mooncake",
                        "consumer_is_to_put": true,
                        "save_decode_cache": true
                    }
                }
            ]
        }
    }' 
) > /workspace/vllm_decode.log 2>&1 &
