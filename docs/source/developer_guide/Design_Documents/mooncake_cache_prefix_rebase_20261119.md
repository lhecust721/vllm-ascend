# Mooncake cache_prefix：2026-11-19 main rebase 记录

## 1. 版本与保留内容

| 项目 | 版本 |
| --- | --- |
| 上一次 upstream 基线 | `3e9e1869a6c13b943eecdb9b18df9072bc4414dd` |
| 本次获取的 upstream/main | `f3fd3001ddb43093d6b6478175f78a13b86e21a4` |
| rebase 前功能分支 | `3f1deadf97c04d9d2cb928f805d846fa076c99ae` |
| rebase 前本地备份 | `codex/ascend-store-cache-prefix-pre-rebase-3f1deadf9` |
| 配对 vLLM | `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`，未改变 |

本次上游增加 42 个提交；全部 12 个功能分支提交已重放，包括三处类型标注修复、原 A2 报告及新增 DSV4.1+DSpark 报告。历史实机结果继续绑定报告原记录的代码与环境，不能自动视为对本次新 main 的实机复测。

## 2. 冲突与接口适配

上游 `860b5259f` / PR #21314 将 layerwise KV Pool 改为 `BlockKeyDataPlane` / `GvaDataPlane` 策略，并删除旧元数据类型。

发生一处文本冲突：`layerwise/metadata.py` 的导入区。处理方式是保留 `apply_cache_prefix`，接受上游删除不再使用的 `normalize_block_ids_by_group` 导入。不恢复已废弃的 `LayerMultiBlockReqMeta`、torch 或 Any 导入。

另有一处不会产生 Git 冲突的接口变化：上游把 `MemcacheBackend.make_partial_key()` 移到了 `LayerwiseKeyBuilder.make_partial_key()`。已将 `test_cache_prefix.py` 中对应断言改为调用绑定的 builder；断言的 key 字符串不变。

`git range-diff` 中 10 个重放补丁为 `=`，2 个补丁仅有上述导入上下文差异。测试接口适配与本文在后续提交单独记录。

## 3. 代码路径复核

- bulk 仍由 `KeyMetadata`、`PoolKey` 和 `ChunkedTokenDatabase` 生成带前缀的最终 key。
- 新 `BlockKeyDataPlane` 的普通及 hybrid 路径仍调用 worker 的 `_make_layerwise_full_key()`，经绑定的 `LayerwiseKeyBuilder.make_full_key()` 加入前缀；lookup 经同一 builder 生成候选 key。
- `make_partial_key()` 属于 GVA 的请求坐标对象路径，维持上游原格式，不向 Memcache/Yuanrong 增加命名空间能力。
- helper 仍位于已有 `metadata.py`；没有恢复独立 `cache_prefix.py`。
- Memcache/Yuanrong 默认或空前缀保持原行为，非空前缀继续在启动阶段报错。
- bulk reset 仍调用不按 cache_prefix 过滤的全量清理；layerwise 外部 reset 的原支持边界不变。

## 4. 本地验证

| 检查 | 结果 |
| --- | --- |
| 核心 8 文件 UT，Windows mock bootstrap | 270 passed、191 subtests passed、4 skipped、1 deselected |
| Mooncake layerwise/hybrid 既有 UT | 34 passed、6 subtests passed |
| 功能差异中的 15 个 Python 文件 | Ruff check、Ruff format、Codespell、Typos 通过；全部编译检查通过 |
| 本次更新的 Markdown | Markdownlint 0.45.0 / Node 24.22.1、Codespell、Typos 通过 |
| 前缀解析与拼接 helper | `apply_cache_prefix`、`resolve_cache_prefix` 的 AST 与 rebase 前完全一致 |
| 原始 `bash format.sh ci` | 退出 3；Windows nodeenv 解析预发布 Node 版本触发 `ValueError: invalid literal for int() with base 10: '0-alpha'` |

核心组与上轮相比少 1 个通过项，是上游删除 `TestLayerMultiBlockReqMeta.test_fields`，没有新增测试失败。4 个 skip 是缺少真实依赖时的真实配置测试；1 个 deselect 是既有 mock Mamba dataclass 兼容问题。均不能视为真实配置或 NPU 验证已通过。

核心组复现参数与前次相同：

```bash
python /path/to/local/windows-mock-bootstrap.py \
  tests/ut/distributed/ascend_store/test_cache_prefix.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_config.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_probe.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_pair.py \
  tests/ut/distributed/ascend_store/test_mooncake_pipeline.py \
  tests/ut/distributed/ascend_store/test_backend.py \
  tests/ut/distributed/ascend_store/test_layerwise_keys.py \
  tests/ut/distributed/ascend_store/test_metadata.py \
  -k 'not sharded_mamba_groups_exclude_tp_replicas' -q
```

这里的 bootstrap 是开发机外部辅助脚本，不是 A2 验收入口。A2 应在真实配对依赖环境用标准 pytest 执行，不能照搬 mock 专用排除条件。layerwise/hybrid 组使用同一 bootstrap 执行 `test_mooncake_layerwise.py` 和 `test_mooncake_hybrid.py`。

## 5. A2 后续验证

requirements、配对 vLLM SHA 与 lint 配置没有变化；上游本次包含 `csrc` 和 CMake 改动，需要重新构建安装 vllm-ascend，不能只依赖旧 editable 安装中的已编译扩展。

分支经过 rebase，旧本地历史不能直接 `git pull --ff-only`。保存本地修改后，从远端新建验证分支：

```bash
git fetch origin codex/ascend-store-cache-prefix
git switch -c verify-cache-prefix-rebase-20261119 origin/codex/ascend-store-cache-prefix
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

以上假设 origin 指向 `lhecust721/vllm-ascend`，分支名尚未使用，并已进入原服务 Python 环境和源码目录。

在标准静态检查、三组 mypy、真实配置 UT 之外，重点复验受 PR #21314 影响的 Mooncake layerwise 同前缀加载和异前缀隔离，保留 writer 正向命中及 range 字节证据。bulk 与 DSV4.1+DSpark 实机结果可作为历史对照，仍建议在最终拟合入版本复验。

验证脚本继续使用 `tests/e2e/common/kv_pool/cache_prefix_probe.py`；参数见 [A2 验证手册](../contribution/mooncake_cache_prefix_validation.md)。
