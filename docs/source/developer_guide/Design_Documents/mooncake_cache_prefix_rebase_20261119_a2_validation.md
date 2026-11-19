# Mooncake cache_prefix：2026-11-19 rebase 后 A2 复验报告

- 报告日期：2026-11-19
- 被测代码：`lhecust721/vllm-ascend` 分支 `codex/ascend-store-cache-prefix`，HEAD `7f75916c5b4227488d195e3f8d466a42497d7db1`
- rebase 基线：upstream/main `f3fd3001ddb43093d6b6478175f78a13b86e21a4`；配对 vLLM `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`（未变）
- 关联：设计/测试建议 `mooncake_cache_prefix.md`；评审清单 `mooncake_cache_prefix_validation_followup.md`；rebase 记录 `mooncake_cache_prefix_rebase_20261119.md`

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 代码更新与重装 | 已 fetch/checkout `7f75916c5`，重新构建安装 `vllm_ascend 0.1.dev6787+g7f75916c5`（清理旧 dist-info） |
| 三处 var-annotated 修复 | 已存在；定向 mypy 3.12/3.11/3.10 均 0 error、0 var-annotated |
| 普通路径 E2E（4 卡） | T1 同前缀命中、T2 异前缀隔离 均通过 |
| layerwise E2E（PR#21314 复验，4 卡） | LW-T1 同前缀 range 加载、LW-T2 异前缀隔离 均通过（含 writer 正向 range 对照） |
| UT 回归 | 真实配置 4 passed（含注册 shim）；回归组 271 passed；layerwise/hybrid 34 passed |
| `format.sh ci` | 除 gitleaks 外全部 Passed（gitleaks 为 ARM64 二进制环境问题） |
| 全仓 mypy 矩阵（3.10/3.11/3.12） | 均 exit=1，但 2274 行 error 中 2259 行在生成的 `_cann_ops_custom`；cache_prefix 三文件 0 error |

## 2. 被测代码与环境

| 项 | 值 |
|---|---|
| 源码 | `/root/work/cache-prefix-arm64/vllm-ascend`，HEAD `7f75916c5`，工作树代码干净 |
| vllm-ascend | `0.1.dev6787+g7f75916c5`（editable，`import` 指向源码；已删除残留的旧 dist-info/pth） |
| 配对 vLLM | `0.1.dev1+gba15685ab.empty`（= `ba15685ab`） |
| Python | 3.12.16 |
| torch / torch_npu / transformers | 2.10.0+cpu / 2.10.0.post7 / 5.19.0 |
| CANN / driver | 25.5.1 |
| 硬件 | A2 单机 8× Ascend 910B3 |
| Mooncake | SDK 0.3.14（session/range，支持 layerwise）；master HA（`mooncake-master-0/1`）+ 3× etcd（`mooncake-ha`） |

重装命令：`python -m pip install --no-deps --no-build-isolation -e .`（上游含 `csrc`/CMake 改动，已重新构建）。日志：`/workspace/reinstall_latest.log`。

## 3. var-annotated 修复与三组 mypy

修复提交 `bfe8c9e81`，三处已存在并通过定向检查（`--follow-imports skip --check-untyped-defs`，`MYPYPATH` 指向配对 vLLM 源码）：

| python 目标版本 | exit | var-annotated | error 总数 |
|---|---|---|---|
| 3.12 | 0 | 0 | 0 |
| 3.11 | 0 | 0 | 0 |
| 3.10 | 0 | 0 | 0 |

文件：`metadata.py`、`test_cache_prefix_config.py`、`test_cache_prefix.py`。日志：`/workspace/mypy_target_3.1{0,1,2}.log`。

全仓矩阵 `python .github/workflows/scripts/run_mypy_parallel.py`（3.10/3.11/3.12 并发）：
- 三版本均 exit=1；共约 2274 行 error。
- **2259 行位于生成的 `_cann_ops_custom/...`（CANN TBE 构建产物）**；`import-untyped 1071`、`arg-type/index/call-overload/type-var/assignment` 等集中于此。
- cache_prefix 三文件 **0 error**；`ascend_store` 非生成代码 0 error；`var-annotated` 15 处全部在生成目录。
- 结论：功能改动本身在 3 个目标版本上类型干净；全仓失败由构建生成的 vendor 代码主导，属既有/生成物问题，非本次 cache_prefix 引入。日志：`/workspace/run_mypy_parallel.log`。

## 4. 普通路径 E2E（Qwen3-30B-A3B，4 卡：A=NPU0,1 / B=NPU2,3）

配置：`kv_both`、`use_layerwise:false`、`--no-enable-prefix-caching`、A 端口 8195 / B 端口 8196、`lookup_rpc_port` 35001/35003。探针 `tests/e2e/common/kv_pool/cache_prefix_probe.py`（writer 存活，成对）。

| 用例 | 配置 | 证据 | 结果 |
|---|---|---|---|
| T1 同前缀共享 | 两端 `model-v1` | writer miss(`hits=0`)；reader `external_hits=1535, bulk_get_keys=24, local_hits=0`；输出一致 | passed |
| T2 异前缀隔离 | A=`model-v1`, B=`model-v2` | writer A `external_hits=1535, bulk_get_keys=24`（正向对照）；reader B `external_hits=0, bulk_get_keys=0`；输出一致 | passed |

产物：`rb-bulk-hit.json`、`rb-bulk-isolation.json`、`qwen-rb-payload.json`、`rb-A-cold.json`、`rb-A-selfhit.json`。

## 5. layerwise E2E（PR #21314 数据平面重构复验，4 卡）

配置：两端 `use_layerwise:true`、`--additional-config '{"weight_nz_mode": 2, "kv_pool_range_debug": true, ...}'`、`--no-enable-prefix-caching`、`kv_both`。探针加 `--mode layerwise --writer-log/--reader-log`，以实时日志 `[KVPOOL_RANGE_DEBUG]` 统计 range 加载字节（须等于请求字节）。

| 用例 | 配置 | 证据 | 结果 |
|---|---|---|---|
| LW-T1 同前缀 range 加载 | 两端 `model-v1-layerwise` | writer miss；reader `external_hits=1535, local_hits=0, range_load_bytes=150994944`，输出一致 | passed |
| LW-T2 异前缀隔离 | A=`model-v1-layerwise`, B=`model-v2-layerwise` | writer A `external_hits=1535, range_load_bytes=150994944`（正向 range 对照）；reader B `external_hits=0, range_load_bytes=0` | passed（重跑后） |

产物：`rb-lw-hit.json`、`rb-lw-isolation.json`。

## 6. UT 回归

命令带 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（真实依赖环境）：

| 组 | 命令要点 | 结果 |
|---|---|---|
| 真实配置单测 | `test_cache_prefix_config.py`（需 connector 注册；见 8.3） | **4 passed** |
| 回归组 | `test_cache_prefix.py, test_cache_prefix_probe.py, test_cache_prefix_pair.py, test_mooncake_pipeline.py, test_backend.py, test_layerwise_keys.py, test_metadata.py` | **271 passed** |
| layerwise/hybrid | `test_mooncake_layerwise.py, test_mooncake_hybrid.py` | **34 passed** |

与 rebase 记录（Windows mock：270/271 passed、34 passed）一致；本次为 A2 真实依赖执行。

## 7. CI

### 7.1 `bash format.sh ci`

| 钩子 | 结果 |
|---|---|
| ruff check / ruff format | Passed |
| codespell / typos / clang-format | Passed |
| markdownlint | 首轮 `--fix` 修改两份报告 md（files modified）→ 复跑 Passed |
| actionlint（workflow） | Passed |
| shellcheck / png-lint / 文件名 / `__init__` / logger / forbidden-imports / boolean-ctx / long-functions / symbolic-meta / suggestion | Passed |
| gitleaks-offline-scan | **Failed**：仓库脚本下载的 ARM64 gitleaks 二进制运行即 `fatal error: lfstack.push`（环境/二进制崩溃） |

日志：`/workspace/format_ci_latest.log`。`--fix` 仅改动 `mooncake_cache_prefix_test_report.md` 与 `static_checks_mypy_ut_validation.md`（文档），代码树保持不变（HEAD `7f75916c5`）。

### 7.2 gitleaks 说明
- OBS 预编译 gitleaks 在本机（aarch64）崩溃；GitHub Release 的 arm64 包在本环境下载极慢且多次截断，未能获得可用二进制。
- 因此 `gitleaks detect --log-opts=f3fd3001..HEAD` 未能在本地完成；建议在具备可用 ARM64 gitleaks 的 CI 环境执行提交范围扫描。

## 8. 观察与边界

1. **对象短时淘汰 / 隔离前需重新预热 writer**：T2 与 LW-T2 首次尝试 writer 均瞬时 miss（对象在 reader 重启的数分钟内被淘汰），本次 writer 请求即重算并重新写入；随后重跑即通过，并保留 `--writer-expect hit` 的有效正向对照（bulk 用 GET 指标、layerwise 用 range 字节）。与既有“数分钟后不能命中”观察一致，原因尚未确定（客户端退出/租约/容量回收），不能直接归因为 LRU。
2. **真实配置单测的 connector 注册**：`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 裸跑仍报 `Unsupported connector type: AscendStoreConnector`，需外部 `sitecustomize` 注入 `vllm_ascend.register_connector()`（followup §3.3 记录的待入库夹具项仍未实现）。注意不要把导入 torch/vLLM 的 shim 套到会派生子进程的 pair 探针上，否则会超时。
3. **全仓 mypy 噪声**：失败集中在构建生成的 `_cann_ops_custom`，非功能源码；cache_prefix 三文件 0 error。
4. 本报告为 bulk/layerwise 的“新 rebase 代码 + SSD/Mooncake 既有状态”实机结果；耗时不作 TTFT，不证明物理 SSD 直接读取/DMA。

## 9. 产物（容器 `/workspace/`）
- 代码/安装：`reinstall_latest.log`
- mypy：`mypy_target_3.10.log`、`mypy_target_3.11.log`、`mypy_target_3.12.log`、`run_mypy_parallel.log`
- CI：`format_ci_latest.log`
- E2E（普通）：`rb-bulk-hit.json`、`rb-bulk-isolation.json`、`qwen-rb-payload.json`、`rb-A-cold.json`、`rb-A-selfhit.json`
- E2E（layerwise）：`rb-lw-hit.json`、`rb-lw-isolation.json`
- 脚本：`qwen3_30B_A3B_mix.sh`、`qwen3_30B_A3B_mix2.sh`、`qwen_mkpayload.py`、`kill_port.py`

复验时状态：A（8195，`model-v1-layerwise`）/B（8196，`model-v2-layerwise`）运行中，`kv_both`+`use_layerwise:true`+`kv_pool_range_debug:true`，`/health` 均 200。
