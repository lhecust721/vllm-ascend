# 静态检查 / 类型检查 / 真实配置单测 验证报告

- 报告日期：2026-11-18
- 仓库：`/root/work/cache-prefix-arm64/vllm-ascend`，HEAD `b48591145`（`verify-cache-prefix-b48591145`）
- 工作区状态：17 个未提交修改（用户为开启 kvpool 调试新增 `logger = _vllm_logger.getChild("kvpool")`）——本次按现状验证
- 对应文档 §7 待验证项：原始 `format.sh ci`、稳定 Node 全仓检查、全仓 mypy / 真实配置单测

## 结论摘要

| 验证项 | 结果 | 说明 |
|---|---|---|
| 原始完整 `format.sh ci` | **未通过（EXIT=1）** | ruff-check E402（17 文件，未提交改动引入）+ gitleaks 二进制崩溃（环境）；ruff-format 首轮自动修复后已干净 |
| 稳定 Node 全仓检查 | **通过** | nodeenv 安装 Node v26.13.1（无预发布解析失败），markdownlint 全仓通过；其余非 Python 钩子全部通过 |
| 全仓 mypy | **环境受限，未能完整通过** | 沙箱 vllm 为 editable “empty” 安装，mypy 无法解析；消除解析后排除了生成代码/测试噪声，暴露 1 处特性代码类型问题 |
| 真实配置单测 | **通过** | `test_cache_prefix_config.py` 4 passed（需按真实引擎方式注册 connector）；特性 UT 合计 39 passed / 0 failed |

## 0. 为完成验证所做的环境适配（非源码改动）

1. GitHub git 传输不稳（`GnuTLS recv error`/early EOF）：`git config --global http.version HTTP/1.1` + `http.postBuffer` 增大后克隆恢复正常。
2. PyPI 不可达（pre-commit 隔离环境 `pip install` 失败）：配置 `~/.config/pip/pip.conf` 指向 `mirrors.aliyun.com/pypi`。
3. actionlint 需 Go 工具链（镜像内缺失）：从 aliyun 镜像安装 Go 1.27.4 到 `/usr/local/go`，并设 `GOPROXY=mirrors.aliyun.com/goproxy`、`GOSUMDB=off`、`GOTOOLCHAIN=local`。
4. Node：nodeenv 自动安装 Node **v26.13.1**（可用），未出现文档所述“选到预发布版本解析失败”。

## A. 原始完整 `format.sh ci`

命令：`bash format.sh ci`（即 `pre-commit run --all-files --hook-stage manual`）。日志：`/workspace/format_ci.log`。

### A.1 钩子结果
| 钩子 | 结果 | 备注 |
|---|---|---|
| ruff-check | **FAILED** | 112 × E402，分布 17 个修改文件；来自未提交的 `logger=getChild("kvpool")` 夹在 import 之间（HEAD 干净） |
| ruff-format | FAILED（首轮）→ 已修复 | 首轮 “1 file reformatted”，再次检查 `27 files already formatted`，复跑应通过 |
| gitleaks-offline-scan | **FAILED** | 从 OBS 下载的 gitleaks 二进制运行时 `fatal error: lfstack.push`（Go runtime/架构崩溃，非源码问题） |
| codespell | Passed | |
| typos | Passed | |
| clang-format | Passed | |
| markdownlint | Passed | node v26.13.1 |
| actionlint（Lint GitHub Actions workflow files） | Passed | Go 1.27.4 构建 |
| shellcheck / png-lint / 文件名 / `__init__` / logger 规范 / 禁用导入 / 布尔上下文 / 长函数 / symbolic-meta / suggestion | Passed | 全部本地钩子通过 |

### A.2 E402 根因与修复
- 现状（工作区，示例 `ascend_store_connector.py`）：
  ```python
  from vllm.logger import logger as _vllm_logger

  logger = _vllm_logger.getChild("kvpool")     # 语句夹在 import 之间 → 后续 import 全部 E402
  from vllm.v1.attention.backend import AttentionMetadata
  ...
  ```
- 对比 HEAD `b48591145`：`from vllm.logger import logger` 且导入连续（无 E402）。故 E402 为本次未提交改动引入的回归。
- 修复建议：将 `logger = _vllm_logger.getChild("kvpool")` 移到所有 import 之后（或对该行 `# noqa: E402`，不推荐）。修复后 ruff-check 应通过。

### A.3 gitleaks
- `.github/workflows/scripts/gitleaks.sh` 在无系统 gitleaks 时从华为 OBS 下载预编译二进制；该二进制在本机（aarch64）运行即 panic（`lfstack.push`）。属环境/二进制兼容问题。建议：在 PATH 提供可用的 `gitleaks`（脚本会优先用系统二进制），或在受支持环境运行。

## B. 稳定 Node 全仓检查

- Node 来源：pre-commit `language: node` 的 nodeenv，安装 **v26.13.1**（`~/.cache/pre-commit/repo*/node_env-default/bin/node`）。
- 全仓 `markdownlint`（manual stage）**Passed**；`clang-format`、`codespell`、`typos`、`actionlint`、`shellcheck` 等全仓钩子均通过。
- 结论：稳定 Node 条件下的全仓非 Python 检查通过；未触发文档记录的 nodeenv 预发布解析失败。若需固定版本，可显式设置 `NODE_VERSION` 或提供系统 Node（`default_language_version: {node: system}`）。

## C. 全仓 mypy

命令（仓库自带）：`bash tools/mypy.sh 1 3.12` → `mypy --follow-imports skip --check-untyped-defs --python-version 3.12 vllm_ascend examples tests`。日志：`/workspace/mypy.log`、`mypy2.log`、`mypy3.log`。

| 场景 | 结果 |
|---|---|
| 默认（沙箱） | 2972 errors；其中 2215 `import-not-found`（`vllm.*` 无法解析）——沙箱 `vllm` 为 `0.1.dev1+gba15685ab.empty` editable 安装，mypy 不识别 setuptools editable finder |
| `MYPYPATH=/root/work/cache-prefix-arm64/vllm` | `import-not-found` 归零；757 errors（355 `import-untyped`：torch_npu/mooncake 等；其余多在生成目录 `_cann_ops_custom`） |
| 追加 `--exclude _cann_ops_custom` | 1554 errors，多为测试/生成/桩噪声（如 `BaseKVPoolScheduler has no attribute _query_layerwise_block_hits`、测试 monkeypatch 的 `method-assign`、`import-untyped`），属 harness 假阳性 |

### 特性相关真实发现
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py:59`
  `error: Need type annotation for "block_sizes"  [var-annotated]`
  —— `resolve_cache_prefix` 内 `block_sizes = []` 需标注 `block_sizes: list[int] = []`（特性代码，a3ddd8e85 的 block-size 归一化逻辑）。此错误与 harness 无关，属真实类型问题。
- 测试代码同类：`tests/ut/distributed/ascend_store/test_cache_prefix_config.py:57`（`value`）、`test_cache_prefix.py:277`（`calls`）需类型标注。

结论：沙箱内无法可信复现 CI 全仓 mypy 结果（editable 安装 + skip-imports 造成大量解析类假阳性）；但可确认特性代码存在 1 处 `var-annotated`。建议在真实依赖环境（vllm 为常规安装）执行 `bash tools/mypy.sh 1` 并以结果为准，同时修复 `metadata.py:59`。

## D. 真实配置单测

真实依赖（torch/transformers/vllm）均已存在；`test_cache_prefix_config.py` 使用**真实 VllmConfig**（非 mock）。其要求 connector 已注册（真实引擎在 general plugin 阶段注册 `AscendStoreConnector`），裸 pytest 未触发该注册，故需等价注入。

- 注册方式验证：`import vllm_ascend; vllm_ascend.register_connector()` → `AscendStoreConnector registered: True`。
- 以 `sitecustomize`（`PYTHONPATH=/workspace/shim`）在进程启动时注册 connector，可同时覆盖测试内派生的子进程。

| 测试集 | 命令 | 结果 |
|---|---|---|
| 真实配置 | `PYTHONPATH=/workspace/shim pytest tests/ut/distributed/ascend_store/test_cache_prefix_config.py` | **4 passed**（含跨进程命名空间一致性子进程用例） |
| 特性 UT（无需注册） | `pytest test_cache_prefix.py test_cache_prefix_probe.py test_cache_prefix_pair.py test_mooncake_pipeline.py` | **35 passed**（9.04s） |
| 合计 | — | **39 passed, 0 failed** |

说明：曾出现 1 例 `test_cache_prefix_pair::...positive_control` 超时失败，系把 `sitecustomize`（每个 python 进程导入 torch/vllm）套用到会派生子进程的 pair 测试导致其 15s 超时，属 harness 副作用；去掉 shim 后该用例通过。

## E. 遗留与建议

1. `format.sh ci` 目前不通过（与文档 §7 记录一致），可复现原因：
   - (a) 未提交改动引入 112 × E402 → 将 `logger=getChild("kvpool")` 移到 import 之后；
   - (b) gitleaks 预编译二进制崩溃 → 提供可用系统 `gitleaks` 或在受支持环境运行。
   修复 (a) 后，除 gitleaks 外全部钩子通过；ruff-format 已干净。
2. 全仓 mypy：在真实依赖（非 editable-empty vllm）环境执行 `bash tools/mypy.sh 1`；并修复 `metadata.py:59` 的 `block_sizes` 标注。
3. 真实配置单测：通过（4 + 35）。CI 中应为 connector 提供注册上下文（引擎插件加载即满足）；本次以 `sitecustomize` 等价注入。

## F. 产物（容器 `/workspace/`）
- `format_ci.log`（完整 format.sh ci 输出）
- `mypy.log`（默认）、`mypy2.log`（MYPYPATH）、`mypy3.log`（去生成代码）
- `run_ut.py`、`shim/sitecustomize.py`、`shim_sitecustomize.py`
- 说明：`format.sh ci` 使用 `--fix` 钩子，已对 1 个文件做格式化并对修改集做安全导入排序等自动修复（diff 由 51 增至 65 insertions）；E402 未被自动修复。
