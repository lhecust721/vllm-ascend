# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deployment namespaces for AscendStore's Mooncake keys."""

from copy import copy

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.utils import hash_factors
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs


def apply_cache_prefix(key: str, cache_prefix: str) -> str:
    return f"{cache_prefix}@{key}" if cache_prefix else key


def resolve_cache_prefix(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None) -> str:
    """Resolve once before Store setup; empty namespaces preserve existing keys."""
    transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    extra_config = transfer_config.kv_connector_extra_config if transfer_config is not None else {}
    # Keep upstream MooncakeStore's coercion, including None -> "None".
    prefix = str(extra_config.get("cache_prefix", ""))
    if not prefix:
        return ""
    if extra_config.get("backend", "mooncake").lower() != "mooncake":
        raise ValueError("AscendStore cache_prefix is supported only by the Mooncake backend")
    if prefix != "auto":
        return prefix

    image_tag = envs.VLLM_IMAGE_TAG.strip()
    build_commit = envs.VLLM_BUILD_COMMIT.strip()
    build = (image_tag if image_tag != "local/vllm-openai:dev" else "") or (
        build_commit if build_commit != "unknown" else ""
    )
    if not build:
        raise ValueError("AscendStore cache_prefix='auto' requires VLLM_IMAGE_TAG or VLLM_BUILD_COMMIT")

    # EngineCore rewrites block_size after worker configs have been copied.
    # Hash the same resolved group sizes on scheduler and worker, without
    # modifying either runtime config or including scheduling capacities.
    cache = copy(vllm_config.cache_config)
    if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
        block_sizes = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
            block_sizes.extend(inner.block_size for inner in specs)
        cache.block_size = min(block_sizes)

    model = vllm_config.model_config
    prefix = "auto-" + hash_factors(
        {
            "build": build,
            "model": model.compute_hash(),
            "hf_commit": getattr(model.hf_config, "_commit_hash", None),
            "cache": cache.compute_hash(),
        }
    )
    logger.info("AscendStore resolved cache_prefix: %s", prefix)
    return prefix
