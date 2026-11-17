# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two live instances validate Mooncake deployment isolation, without SSD.

Run manually on A2 with two visible NPUs and a matching Mooncake Master/SDK.
Directory placement alone does not select an A2 runner in PR CI.
"""

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.cache_prefix_probe import capture_request, validate_observation
from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMooncakeManager
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
PROMPT_TOKENS = 1536
pytestmark = pytest.mark.e2e_model(MODEL)


class LoggedPrefixServer(RemoteOpenAIServer):
    def __init__(self, *args, log_path: Path, **kwargs):
        self.log_path = log_path
        super().__init__(*args, **kwargs)

    def _start_server(self, model, server_cmd, env_dict):
        env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn", **(env_dict or {})}
        with self.log_path.open("w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(server_cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


@contextmanager
def prefix_server(model, pool, log_path, device, prefix, use_layerwise, image="prefix-e2e-build-v1"):
    port = get_open_port()
    transfer = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "cache_prefix": prefix,
            "use_layerwise": use_layerwise,
            "load_async": False,
            "lookup_rpc_port": str(port),
        },
    }
    args = [
        "--served-model-name",
        "kv-prefix-e2e",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.7",
        "--seed",
        "42",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        json.dumps(transfer),
        "--additional-config",
        json.dumps({"kv_pool_range_debug": use_layerwise}),
    ]
    env = {**pool.server_envs, "ASCEND_RT_VISIBLE_DEVICES": device, "PYTHONHASHSEED": "0", "VLLM_IMAGE_TAG": image}
    with LoggedPrefixServer(
        model, args, server_host="127.0.0.1", server_port=port, auto_port=False, env_dict=env, log_path=log_path
    ) as server:
        yield server


@pytest.mark.parametrize("use_layerwise", [False, True], ids=["bulk", "layerwise"])
@pytest.mark.parametrize("prefix", ["", "release-v1", "auto"], ids=["empty", "literal", "auto"])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mooncake_cross_instance_namespace(tmp_path, use_layerwise, prefix):
    pytest.importorskip("mooncake.store")
    # Missing dependencies may skip; an installed but incompatible SDK must
    # fail startup rather than silently turn this into a passing/empty test.
    if shutil.which("mooncake_master") is None:
        pytest.skip("matching mooncake_master is required")
    if torch.npu.device_count() < 2:
        pytest.skip("two visible NPUs are required")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else ["0", "1"]
    assert len(devices) >= 2, "Expose two NPUs for a live writer and a fresh reader"
    model = maybe_model_redirect(MODEL)
    unit = AutoTokenizer.from_pretrained(model).encode(
        "Explain why reliable cache isolation matters. ", add_special_tokens=False
    )
    assert unit
    tokens = (unit * (PROMPT_TOKENS // len(unit) + 1))[:PROMPT_TOKENS]
    config = MooncakeKVPoolConfig(
        master_port=get_open_port(),
        metrics_port=get_open_port(),
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "device_name": "",
            "global_segment_size": "1GB",
            "preferred_segment": False,
            "prefer_alloc_in_same_node": True,
        },
    )
    mode = "layerwise" if use_layerwise else "bulk"
    with (
        SingleNodeMooncakeManager(config, tmp_path.name) as pool,
        prefix_server(model, pool, tmp_path / "writer.log", devices[0], prefix, use_layerwise) as writer,
    ):
        cold = capture_request(writer.url_root, tokens, mode=mode, log_path=writer.log_path, expect="miss")
        validate_observation(cold, "miss")
        (tmp_path / "writer.json").write_text(json.dumps(cold, indent=2), encoding="utf-8")
        # Keep the writer alive throughout fresh-reader startup and loading.
        # Never retry a request on the reader: a first miss can warm its own
        # Store entries and turn a retry into a false cross-instance hit.
        with prefix_server(model, pool, tmp_path / "same.log", devices[1], prefix, use_layerwise) as reader:
            hit = capture_request(reader.url_root, tokens, mode=mode, log_path=reader.log_path)
            validate_observation(hit, "hit", cold)
            (tmp_path / "same.json").write_text(json.dumps(hit, indent=2), encoding="utf-8")
        other = prefix if prefix == "auto" else "release-v2"
        image = "prefix-e2e-build-v2" if prefix == "auto" else "prefix-e2e-build-v1"
        with prefix_server(model, pool, tmp_path / "other.log", devices[1], other, use_layerwise, image) as reader:
            miss = capture_request(reader.url_root, tokens, mode=mode, log_path=reader.log_path, expect="miss")
            validate_observation(miss, "miss", cold)
            (tmp_path / "other.json").write_text(json.dumps(miss, indent=2), encoding="utf-8")
