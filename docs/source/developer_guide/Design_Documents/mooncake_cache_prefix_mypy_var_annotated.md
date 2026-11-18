# Mooncake cache_prefix：mypy `var-annotated` 说明（背景 / 触发原因 / 修复建议）

- 记录日期：2026-11-18
- 基线：分支 `codex/ascend-store-cache-prefix`，提交 `b48591145`（`docs(kvpool): record main rebase and validation`）
- 触发场景：在 A2 Linux 真实依赖环境执行全仓 mypy（`tools/mypy.sh`，mypy 1.11.1，Python 3.12）
- 结论：全仓 mypy 报告 3 处 `var-annotated`（`Need type annotation`），**均由 cache_prefix 功能新增代码/测试引入**，属**静态类型标注缺失**，不影响运行时行为与功能正确性。

## 1. 背景

功能分支相对 rebase 基线 `3e9e1869a`（upstream/main）新增了 7 个提交：

```
d74225459 feat(kvpool): add Mooncake deployment cache prefixes
6cd7bc507 test(kvpool): strengthen Mooncake cache prefix validation
3121db50e test(kvpool): verify operator-deployed cache pool services
5fc43aaca docs(kvpool): record cache prefix design and acceptance
6012762a9 docs(kvpool): explain kv_both layerwise verification
a3ddd8e85 refactor(kvpool): keep cache prefix helpers in metadata
b48591145 docs(kvpool): record main rebase and validation
```

文档此前记录全仓 mypy 为“待真实依赖环境执行”。本次在 A2 上执行后，除大量**环境/桩噪声**外，确认 3 处真实且可一行修复的 `var-annotated`。`git blame` 归因如下：

| 位置 | 引入提交 | 归属 |
|---|---|---|
| `vllm_ascend/.../ascend_store/metadata.py:57` `block_sizes = []` | `a3ddd8e85` | 功能**生产代码**（auto 前缀的 block-size 归一化，重构并入 `metadata.py`） |
| `tests/.../test_cache_prefix_config.py:57` `for value in (...)` | `6cd7bc507` | 功能**测试代码** |
| `tests/.../test_cache_prefix.py:277` `self.calls = {...}` | `6cd7bc507` | 功能**测试代码** |

基线 `3e9e1869a` 的 `metadata.py` 中 `resolve_cache_prefix` 出现次数为 0，且上述两个测试文件在基线中不存在——因此这三处均为本功能新增，而非上游既有问题。

## 2. 触发原因

`var-annotated`（`Need type annotation for "..."`）是 mypy 的**类型标注完备性**检查：当变量由**空容器字面量**（`[]` / `{}`）或**异构字面量集合**初始化、且 mypy 无法从上下文稳定推断其（元素）类型时，会要求显式标注。

触发条件叠加：
- 仓库 `tools/mypy.sh` 使用 `--check-untyped-defs`；
- `mypy.ini` 仅 `disable_error_code = annotation-unchecked`，**未禁用** `var-annotated`；
- 因此全仓 mypy 会对上述三处报错。这与运行时无关——代码实际执行完全正常。

## 3. 三处详情与修复

### 3.1 `metadata.py:57` — `block_sizes`

```python
cache = copy(vllm_config.cache_config)
if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
    block_sizes = []                      # <-- mypy: Need type annotation for "block_sizes"
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        block_sizes.extend(inner.block_size for inner in specs)   # 元素为 int
    cache.block_size = min(block_sizes)
```

- 原因：`block_sizes = []` 为空列表字面量，元素类型需从后续 `.extend(int)` / `min(...)` 推断，mypy 在未标注时要求显式声明。
- 修复（推荐）：
  ```python
  block_sizes: list[int] = []
  ```
- 影响：仅类型标注；`list[int]` 与运行时一致，无行为变化。

### 3.2 `test_cache_prefix_config.py:57` — `value`

```python
def test_upstream_parity_for_literal_conversion_and_auto(self):
    for value in ("", "custom", "AUTO", None, False, 0, [], {}, "auto"):   # <-- mypy: Need type annotation for "value"
        with self.subTest(value=value):
            config = make_config(self.model_path, value)
            self.assertEqual(resolve_cache_prefix(config), _resolve_cache_prefix(config))
```

- 原因：被迭代的元组是**异构字面量**（`str/None/bool/int/list/dict`），循环变量 `value` 无法稳定归一为单一类型，mypy 要求声明。
- 修复（推荐，保留测试意图）：
  ```python
  values: tuple[object, ...] = ("", "custom", "AUTO", None, False, 0, [], {}, "auto")
  for value in values:
      ...
  ```
  也可在循环前声明 `value: object`，或对该行加 `# type: ignore[var-annotated]`（不推荐，掩盖而非澄清）。

### 3.3 `test_cache_prefix.py:277` — `self.calls`

```python
def __init__(self):
    self.objects = {}
    self.calls = {"exists": [], "put": [], "get": []}   # <-- mypy: Need type annotation for "calls"

# 使用处（元素为对象 key 字符串）
self.calls["exists"].extend(keys)   # keys: list[str]
self.calls["put"].extend(keys)
self.calls["get"].extend(keys)
```

- 原因：`self.calls` 由嵌套空列表的 dict 字面量初始化，mypy 无法推断值类型。
- 修复（推荐）：
  ```python
  self.calls: dict[str, list[str]] = {"exists": [], "put": [], "get": []}
  ```
  若 key 元素类型并非 `str`，按实际类型标注（如 `list[Any]`）。

## 4. 复现与验证

复现（A2 真实依赖环境）：

```bash
# 注意：本沙箱中 vllm 为 editable 安装，mypy 需借助 MYPYPATH 解析；CI 中为常规安装则无需
MYPYPATH=<vllm 源码根目录> \
mypy --follow-imports skip --check-untyped-defs --python-version 3.12 \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_config.py \
  tests/ut/distributed/ascend_store/test_cache_prefix.py \
  | grep -E "var-annotated|Need type annotation"
```

期望当前输出（`b48591145`）：

```
metadata.py:57: error: Need type annotation for "block_sizes" ... [var-annotated]
test_cache_prefix_config.py:57: error: Need type annotation for "value" [var-annotated]
test_cache_prefix.py:277: error: Need type annotation for "calls" [var-annotated]
```

按第 3 节修改后，上述三行应消失；随后以 `bash tools/mypy.sh 1` 在常规依赖环境复核。

## 5. 说明与边界

- **非运行时缺陷**：三处均为类型标注完备性问题；cache_prefix 功能的 E2E（普通路径 P0、layerwise 同/异前缀、auto 身份改变隔离、全量 reset）与真实配置/特性单测（39 passed）均已通过，与本问题无关。
- **与“环境噪声”区分**：全仓 mypy 在沙箱中还产生大量 `import-not-found` / `import-untyped` / 测试 monkeypatch 的 `method-assign` 等，属 editable 安装与 `--follow-imports skip` 造成的假阳性，以及生成目录 `_cann_ops_custom` 的产物错误；评估全仓 mypy 时应排除这些，聚焦真实源码问题。
- **是否阻塞 CI**：取决于仓库是否在 CI 运行严格全仓 mypy；若运行，修复上述三处即可消除这类 `var-annotated`。

## 6. 建议的最小改动汇总

```diff
--- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
-        block_sizes = []
+        block_sizes: list[int] = []

--- tests/ut/distributed/ascend_store/test_cache_prefix_config.py
-        for value in ("", "custom", "AUTO", None, False, 0, [], {}, "auto"):
+        values: tuple[object, ...] = ("", "custom", "AUTO", None, False, 0, [], {}, "auto")
+        for value in values:

--- tests/ut/distributed/ascend_store/test_cache_prefix.py
-        self.calls = {"exists": [], "put": [], "get": []}
+        self.calls: dict[str, list[str]] = {"exists": [], "put": [], "get": []}
```
