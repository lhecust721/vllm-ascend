# Mooncake cache_prefix：A2 报告评审与后续验证清单

更新日期：2026-11-18。评审对象为 `b48591145` 的 A2 测试及随后上传的三份报告；本次代码修复建立在 `7d26fd264` 上。

2026-11-19 补充：远端 `3f1deadf9` 已增加 DSV4.1+DSpark bulk 实机结果，见 [功能报告](mooncake_cache_prefix_test_report.md) 第 8 节。
下文为 2026-11-18 时点的待办；新的上游数据平面重构尚需 A2 复验，见 [最新 rebase 记录](mooncake_cache_prefix_rebase_20261119.md)。

关联记录：

- [A2 功能测试报告](mooncake_cache_prefix_test_report.md)
- [静态检查、mypy 与真实配置单测报告](static_checks_mypy_ut_validation.md)
- [三处类型标注问题与修复](mooncake_cache_prefix_mypy_var_annotated.md)
- [设计与测试建议](mooncake_cache_prefix.md)
- [A2 安装及验证方法](../contribution/mooncake_cache_prefix_validation.md)

## 1. 已完成的代码修复

| 文件 | 修改 | 目的 |
| --- | --- | --- |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py` | `block_sizes: list[int] = []` | 明确 auto 前缀归一化所用的 block size 类型 |
| `tests/ut/distributed/ascend_store/test_cache_prefix_config.py` | `values: tuple[object, ...]` 后再遍历 | 明确混合类型及空容器测试输入的类型上下文 |
| `tests/ut/distributed/ascend_store/test_cache_prefix.py` | `self.calls: dict[str, list[str]]` | 明确 CPU 存储替身记录的操作名和对象 key 类型 |

本次只补充类型信息，不改变前缀、hash 因子、存储调用或 reset 范围。继续保留上游 `str(...)` 转换语义，包括 `None`、数字及容器输入。mypy 可以从部分空容器的后续使用推断类型；这里添加显式标注，是为消除当前代码实际触发的三处 `var-annotated`。

## 2. 现有报告能支持的结论

A2 报告记录了 Qwen3-30B-A3B、TP=2+EP、双实例 `kv_both` 下的 bulk 同前缀共享、异前缀隔离、空前缀、auto 身份变化，以及 layerwise 手动前缀共享和隔离。关闭本地 prefix cache、保留 writer 命中正向对照、检查真实 GET 或 range 加载证据，均有助于避免误判。

这些是报告记录的硬件结果；本次 Windows 类型修复没有重新执行 A2 E2E。报告中的耗时不是 TTFT 基准，也不能证明物理 SSD 读取或直接 DMA。

需要保留以下边界：

- 两个新版本实例配置空前缀并互通，尚不能单独证明旧版本写入、新版本读取兼容。
- Memcache/Yuanrong 的配置保护和 key 单测通过，不等于两种后端的完整集成回归。
- 改变 build identity 验证了 auto 命名空间变化，尚不是两套实际权重的滚动升级验证。
- bulk reset 对两个前缀同时生效，结合 `remove_all(force=True)` 代码可支持其全量语义；不应将它解释为跨 tenant 全局清理。layerwise 外部 reset 仍不支持。
- 数分钟后对象不能命中的原因尚未确定，应区分客户端退出导致副本失效、租约、容量回收及其他负载；不能直接归因为 LRU。
- Mooncake + DSpark、其他模型及分布式拓扑仍未取得本次硬件覆盖证据。

## 3. 优先完成：检查与测试可复现

### 3.1 固定被测代码和环境

在实际运行 vLLM 的 Python 环境中记录：

```bash
git rev-parse HEAD
git status --short
git diff --binary > "$WORK_ROOT/cache-prefix-worktree.patch"
python --version
python -m pip freeze > "$WORK_ROOT/packages-after-typing-fix.txt"
```

`WORK_ROOT` 应事先设置为已存在的验证目录。报告中的 17 个未提交 logger 调试修改必须单独保存或提交；仅记录 HEAD 不足以标识这些修改后的被测代码。优先使用干净检出复跑，将调试修改与功能代码的检查结果区分。

同时记录镜像 digest、操作系统/架构、驱动/CANN、torch/torch-npu、vLLM 实际源码 SHA、Mooncake SDK/Master 版本，以及两端启动配置。`pip freeze` 不能覆盖全部系统环境信息。

### 3.2 运行标准静态检查与三组 mypy

```bash
bash format.sh ci
python .github/workflows/scripts/run_mypy_parallel.py
```

逐条记录原始日志和退出码。当前社区 CI 确实执行 Python 3.10、3.11、3.12 三组 mypy 规则；仅运行 3.12 不等于完成矩阵。这里的版本参数是 mypy 目标规则，不要求安装三个 Python 解释器。

验收要求：三处 `var-annotated` 消失；全仓检查是否通过需另行报告。若仍有错误，先修复配对 vLLM 源码解析和依赖问题，再在同一环境对比 upstream 基线 `3e9e1869a` 与功能提交。不得批量将 `attr-defined`、`method-assign` 等归为环境噪声，也不要通过全局禁用错误码掩盖问题。

A2 上需提供可运行的 ARM64 gitleaks。仓库脚本优先使用 PATH 中的版本；若仅修复二进制后在空暂存区执行本地检查，本地 staged 扫描通过仍不能证明功能提交范围已扫描。应另外扫描相对基线的提交范围，或保留 GitHub PR CI 的扫描结果。例如在安装了兼容 `detect` 子命令的版本后：

```bash
gitleaks detect --redact --config .gitleaks.toml \
  --log-opts="3e9e1869a..HEAD"
```

### 3.3 使真实配置单测独立于外部 shim

待办：将必要的 connector 注册纳入测试准备过程，同时覆盖测试派生的 Python 子进程。当前报告的 `sitecustomize` 是外部验证适配，不是仓库已经自带的夹具；本次类型修复没有实现这一项。

完成后，应能在配对的真实依赖环境直接执行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/distributed/ascend_store/test_cache_prefix_config.py
```

要求 4 例实际执行并通过，不接受因缺失真实依赖而全部 skip。不要将导入 torch/vLLM 的 shim 套在所有探针子进程上，否则可能引入报告中已观察到的超时。

### 3.4 相关回归测试

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/distributed/ascend_store/test_cache_prefix.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_probe.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_pair.py \
  tests/ut/distributed/ascend_store/test_mooncake_pipeline.py \
  tests/ut/distributed/ascend_store/test_backend.py \
  tests/ut/distributed/ascend_store/test_layerwise_keys.py \
  tests/ut/distributed/ascend_store/test_metadata.py
```

按失败情况补跑相关 worker、scheduler 和 shared-source 测试。记录 collected、passed、skipped、failed 及 skip 原因。Windows mock 结果不能替代真实配置、真实 tensor 或 NPU 验证；不要把本地 mock 环境专用的排除条件复制到 A2 验收命令中。

## 4. 补充功能证据

| 优先级 | 待补项目 | 通过条件 |
| --- | --- | --- |
| P0 | 空前缀兼容 | 增加旧/新生成 key 完全一致的断言；若声称版本互通，再增加旧版本写、新版本读 |
| P0 | layerwise + auto | 同身份跨实例命中，scheduler/worker 前缀一致，实际 range 返回字节等于请求字节 |
| P0 | 结果归档 | 保存探针 JSON、完整启动参数、worker 日志、reset 客户端、提交 SHA 和工作区 diff |
| P1 | Mooncake + DSpark | 在实际 target/draft 配置下验证同前缀共享和不同前缀隔离，保留正向命中对照 |
| P1 | 滚动升级场景 | 两套权重各自使用独立前缀并与自身无外部缓存基线对比；不要求不同权重文本一致 |
| P1 | 对象持久性调查 | 使用隔离池或明确其他负载，收集对象副本/租约和客户端退出日志，区分失效原因 |
| P2 | SSD 与性能长稳 | 分别取得物理 SSD 读取证据及同环境性能对照；不由本次功能命中结果推导 |

双实例 bulk/layerwise 验证继续使用 `tests/e2e/common/kv_pool/cache_prefix_probe.py`，具体参数见 [A2 验证手册](../contribution/mooncake_cache_prefix_validation.md)。用户自行部署模型。隔离检查前可重新预热 writer，但最终必须以 `--writer-expect hit` 建立有效正向对照；reader 为 `kv_both` 时，第一次 miss 后会写入自身命名空间，不应重复同一输入并仍强求 miss。

## 5. 本次类型修复的验证记录

以下是本次 Windows 开发环境实际执行的结果，不是新增 A2 硬件结果：

| 检查 | 结果及边界 |
| --- | --- |
| 三个修改的 Python 文件，mypy 1.11.1 / Python 3.12 规则 | 三处 `var-annotated` 均消失；仍有 17 项错误，包括缺失 torch/vLLM 导入和 NumPy 类型错误，整体退出码为 1，不宣称通过 |
| 同三个文件，Python 3.10 / 3.11 规则 | 均退出 2；环境解析到的 NumPy stub 包含 Python 3.12 `type` 语法，检查提前终止，不能据此确认这两组检查通过 |
| `test_cache_prefix.py` 与 `test_cache_prefix_config.py` | 外部 Windows mock bootstrap 下 16 passed、27 subtests passed、4 skipped；4 个真实配置用例因缺少真实依赖而跳过 |
| Ruff check / format | 修改的三个 Python 文件通过 |
| Codespell / Typos | 本次修改的 Python 和 Markdown 文件通过 |
| Markdownlint 0.45.0 / Node 24.22.1 | 本次修改的两份 Markdown 通过；同时补齐历史 mypy 报告的代码块语言和列表空行 |
| Python 编译检查 / `git diff --check` | 三个 Python 文件编译和 diff 空白检查通过 |
| 原始 `bash format.sh ci` | 退出 3；nodeenv 解析 Node 预发布版本时出现 `ValueError: invalid literal for int() with base 10: '0-alpha'`，未完成全仓检查 |

针对性 mypy 使用下列命令，以 3.12 为例；3.10/3.11 使用相同参数更换目标版本：

```bash
python -m mypy --follow-imports skip --check-untyped-defs --python-version 3.12 \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_config.py \
  tests/ut/distributed/ascend_store/test_cache_prefix.py
```

完整 Linux CI、真实配置单测和新增 NPU 用例，仍按第 3、4 节执行；历史 A2 报告不能自动视为对后续提交的重新测试。
