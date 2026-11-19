# Mooncake cache_prefix：实现思路、修改点与测试建议

记录日期：2026-11-17；最近 rebase 更新：2026-11-19。
本文说明 `lhecust721/vllm-ascend` 的 `codex/ascend-store-cache-prefix` 功能分支，
覆盖当前实现，包括两端 kv_both、layerwise 验证及共享函数归并。
精确版本以包含本文的 Git 提交为准；历史 A2 结果见 [实机测试报告](mooncake_cache_prefix_test_report.md)，
最新 main 的冲突处理和验证边界见 [2026-11-19 rebase 记录](mooncake_cache_prefix_rebase_20261119.md)。

## 1. 目标、范围与版本

在现有 `AscendStoreConnector` 中增加部署级 `cache_prefix`，使同一 Mooncake 测试池内：

- 相同前缀、相同请求且载荷兼容的实例可以复用 KV。
- 不同前缀的实例查找不同的对象 key，可用于新旧权重并存及滚动升级。
- 未配置或空字符串保持旧 key 格式。
- 命名空间共用原来的内存、SSD 和带宽资源，不新增租户配额或访问控制机制。

范围包括 Mooncake 普通与 layerwise 路径，以及它们原已支持的 hybrid、PP、TP、PCP/DCP、DSpark 场景。
这不是新增模型或拓扑支持；尤其不能把 CPU 的 draft 层测试写成 DSV4.1 + DSpark 实机验证通过。
Memcache GVA 专用辅助对象不属于本次 Mooncake 扩展范围。

本功能继续使用 `AscendStoreConnector`，不切换到其他新 connector。
Memcache/Yuanrong 后端的存取实现保持原样：未配置或空前缀保持原行为，非空前缀明确报仅支持 Mooncake。

| 项目 | 版本依据 |
|---|---|
| 原始功能分支起点 | `b53996aad5d959f0b4511b4b87aadb2eb425b045`，保留作历史记录 |
| 当前 rebase 基线 | `f3fd3001ddb43093d6b6478175f78a13b86e21a4`，2026-11-19 拉取的 upstream/main，`[Doc] Remove incorrect ASCEND_VISIBLE_DEVICES from LMCache guide (#22092)` |
| 初版功能 | `75d383213451284368a90e62b190040470218723` |
| 第一轮验证补强 | `f4914cd0953e6968ee7edc780e801f117995997e` |
| 已部署服务验证模式 | `9d1a306a27b902841739d6b63d06bc58e54be564`；正向对照修正随本文提交 |
| 配对上游 vLLM | `.github/vllm-main-verified.commit`：`ba15685ab7f4f5fbed8630bbb7441f8837e76fbe` |
| 上游语义参考 | PR #45767 手动前缀、PR #65160 自动前缀；#65610 不属于此功能 |

2026-11-17 rebase 到 `3e9e1869a` 时无冲突，原有六个功能提交全部重放，`git range-diff` 对应补丁均为 `=`。
表中初版/补强/客户端的旧提交号为 rebase 前历史标识；该次重放对应提交依次为
`d74225459`、`6cd7bc507`、`3121db50e`、`5fc43aaca`、`6012762a9`、`a3ddd8e85`。
本地保留备份分支 `codex/ascend-store-cache-prefix-pre-rebase-a41e7e4e0`。
2026-11-19 再次重放了全部 12 个分支提交，解决 layerwise 导入区一处冲突，并适配上游移动后的 partial-key 测试接口。
上游 PR #21314 的数据平面重构完整保留；不将 Memcache 专用 GVA 能力扩展为 Mooncake 新能力。
配对 vLLM SHA 和 requirements.txt 未因这两次 rebase 改变；本次上游包含原生算子改动，A2 更新需重新构建安装 vllm-ascend。

`cache_prefix` 的 key 隔离逻辑在 vLLM/connector 侧实现；Mooncake 按最终 key 存储和查询。
前缀本身不要求修改 Mooncake SDK。现有 layerwise 数据通道仍要求匹配的 session/range SDK API，
其依赖要求与命名空间功能是两件事。

## 2. 配置和 key 设计

### 2.1 配置契约

读取 `kv_transfer_config.kv_connector_extra_config.cache_prefix`，保持上游的字符串转换：

```python
prefix = str(extra_config.get("cache_prefix", ""))
```

| 输入 | 行为 |
|---|---|
| 未配置、`""` | 使用旧 key |
| `"release-v1"` | 字面命名空间 |
| `"auto"` | 计算自动前缀 |
| `"AUTO"` | 字面字符串，不触发自动模式 |
| JSON `null`、`false`、`0` | 分别转换为 `"None"`、`"False"`、`"0"`，都是非空前缀 |

不额外增加 Ascend 独有的“只接受字符串”校验；推荐用户显式配置字符串，避免误选命名空间。
Memcache/Yuanrong 在转换后检查是否非空，因此上述非空转换结果同样被拒绝。
这是与上游一致的 connector 专用配置通道，使用既有 kv_connector_extra_config；
不新增 Ascend 环境变量或重复的 additional-config 开关。

### 2.2 自动前缀

已有 `metadata.py` 中的 `resolve_cache_prefix()` 复用上游维护的配置哈希，
不扫描任意环境变量，也不读取完整权重文件：

1. 优先选择有效的 `VLLM_IMAGE_TAG`，否则使用 `VLLM_BUILD_COMMIT`。
2. 排除默认开发身份 `local/vllm-openai:dev` 和 `unknown`；均不可用时在 Store setup 前失败。
3. 对构建身份、`model.compute_hash()`、HF `_commit_hash` 和 `cache.compute_hash()` 做 SHA-256。
4. 结果为 `auto-<digest>`。

Ascend 的 scheduler 可能在 worker 配置复制后重写 block size。
实现对 cache 配置做浅拷贝，使用已解析 cache groups 的最小 block size 归一化后再计算哈希，
并展开 `UniformTypeKVCacheSpecs`；不修改原运行时配置。
这一适配需要真实 vLLM 配置测试及 A2 跨实例命中共同验证，不能仅靠替身配置断言。

自动模式不是完整的 KV 兼容性校验：它不直接覆盖实时权重文件、全部环境开关、所有 draft 配置，
也不独立识别 Ascend 源码变化。身份不变但权重、draft 或实际布局变化时，应更新部署身份或手动前缀。
相同前缀也不能让原本不兼容的 TP/布局自动复用。

### 2.3 最终 key

```text
空前缀：旧 key
非空前缀：<cache_prefix>@<旧 key>
```

旧 key 中的模型、group、cache role/family 和现有布局/拓扑命名空间继续保留。
普通路径同时处理对象形式与字符串快速路径；layerwise 在绑定好的完整 key 上添加一次前缀。
backend 普通 get/put/exists 接收最终 key，不再追加第二遍。

手动前缀可以包含类似 `@pp_rank:0` 的文本。
PP/TP 字段替换时必须跳过前缀长度，只修改旧 key 的结构字段，避免破坏用户命名空间。

### 2.4 全量 reset

保留现有普通路径全量 reset，不根据前缀缩小清理范围：

```text
停止流量并协调共享域内的所有 writer
    -> 等待排队 PUT 完成
    -> remove_all(force=True)
```

清理会影响客户端有效 tenant 域内的所有前缀和模型；具体租户边界以部署的 Mooncake 版本和配置为准。
只清本地 prefix cache 不等于清外部池；必须明确请求 connector reset。
layerwise 外部 reset 仍不支持。滚动升级使用新前缀可以避免全量 reset，旧对象由既有淘汰/清理机制回收。

## 3. 核心代码修改点

以下路径均相对仓库根目录，核心源码公共前缀为
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`。

| 文件 | 类/函数 | 修改目的 |
|---|---|---|
| `metadata.py`（已有文件） | `resolve_cache_prefix`、`apply_cache_prefix` | 统一配置转换、backend 限制、自动身份解析及单次拼接 |
| `pool_worker.py` | `BaseKVPoolWorker.__init__`、`_init_metadata` | 初始化解析前缀并传入每个 group 的 metadata |
| `pool_worker.py` | `_replace_key_field` | PP/TP 等字段重写避开前缀内容 |
| `pool_scheduler.py` | `BaseKVPoolScheduler.__init__` | 初始化阶段验证配置，尽早拒绝不支持的前缀配置 |
| `metadata.py` | `KeyMetadata`、`PoolKey.__hash__`、`PoolKey.to_string` | 前缀成为 key 身份的一部分；默认保持旧格式 |
| `metadata.py` | `ChunkedTokenDatabase._get_key_prefix`、`_make_key_by_hash` | 字符串快速路径和对象路径生成一致的最终 key |
| `metadata.py` | `decode_adaptor_prefill_pp` | Prefill PP 分区替换不误改用户前缀 |
| `layerwise/metadata.py` | `LayerPoolKey.__hash__`、`to_string` | 层级 key 保留命名空间 |
| `backend/layerwise_keys.py` | `LayerwiseKeyBuilder.make_full_key`、`make_hit_check_keys` | writer key 与 scheduler 查询 key 共用同一前缀机制 |
| `backend/mooncake_backend.py` | `MooncakeBackend.bind_layerwise_keys` | 将解析出的前缀绑定到单组/hybrid layerwise builder |

前缀相关函数已合并到已有 `metadata.py`，独立的 `cache_prefix.py` 已删除。
worker、scheduler、普通 key 和 layerwise builder 均从该共享模块访问相关函数。
Ascend 的多个组件都需解析/拼接前缀，合并到已有 metadata 层可避免 scheduler/backend 反向导入 worker。
本次归并只调整函数位置和导入路径，保留原函数体、key 格式、自动哈希因素及 reset 范围。
测试中的 patch 目标与真实配置测试的子进程导入同步改为 metadata；客户端验证脚本的调用参数不变。

没有新增 monkey patch、model runner 行为、NPU 算子或 Ascend 环境变量。
已有 `VLLM_IMAGE_TAG`/`VLLM_BUILD_COMMIT` 通过 `vllm.envs` 访问。
初始化阶段计算自动哈希，热路径主要增加字符串处理；没有新增 tensor.item() 或 NPU 同步操作，
但尚无实机性能对比，不能宣称“零开销”或给出性能收益数字。

## 4. 测试和文档修改点

| 文件 | 覆盖内容 |
|---|---|
| `tests/ut/distributed/ascend_store/test_cache_prefix.py` | 配置分支、上游转换语义、普通/层级 key、特殊前缀、后端保护、真实 connector/backend 代码的 CPU 字节回环和全量 reset |
| `tests/ut/distributed/ascend_store/test_cache_prefix_config.py` | 真实 vLLM 配置、与上游 resolver 对照、跨进程一致性、block size 归一化及不修改配置 |
| `tests/ut/distributed/ascend_store/test_mooncake_pipeline.py` | layerwise 的 group/head/PP/draft 数据回环及不同前缀查询 |
| `tests/ut/distributed/ascend_store/test_cache_prefix_probe.py` | 指标缺失、旧日志、部分/失败 range GET、命中但无加载证据等误判保护 |
| `tests/ut/distributed/ascend_store/test_cache_prefix_pair.py` | 模拟 HTTP 双服务，验证 CLI 请求顺序、同一输入、命中/未命中、报告及正向对照要求 |
| `tests/ut/distributed/ascend_store/_mock_deps.py`、`test_pool_worker.py` | 扩展现有 mock 模块和 worker 测试夹具 |
| `tests/e2e/common/kv_pool/cache_prefix_probe.py` | 独立 HTTP 验证客户端；支持单实例探测及已部署 writer/reader 的配对验证 |
| `tests/e2e/pull_request/two_card/test_mooncake_cache_prefix.py` | 可选的自动启动双实例 E2E；用户自行部署时无需运行它 |
| `docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md` | 参数、兼容性、滚动升级和 reset 影响 |
| `docs/source/developer_guide/contribution/mooncake_cache_prefix_validation.md` | A2 源码更新、安装、CI 和验证命令 |
| `release_notes/fragments/0.yaml` | 用户可见功能说明；获得正式 PR 编号后需更新文件名和 pr 字段 |

CPU 字节回环使用 SDK 边界替身，并运行实际 key/connector/backend 代码；不代表执行了真实 Mooncake 网络传输。
模拟 HTTP 服务测试只证明验证脚本控制流程，不能代替真实模型推理或 NPU E2E。

## 5. 已部署模型的验证方法

当前由用户部署模型。客户端不启动或停止服务，不部署权重，不修改服务的 cache_prefix。
验证脚本已提交在
[`tests/e2e/common/kv_pool/cache_prefix_probe.py`](../../../../tests/e2e/common/kv_pool/cache_prefix_probe.py)。
配对客户端回归位于
[`tests/ut/distributed/ascend_store/test_cache_prefix_pair.py`](../../../../tests/ut/distributed/ascend_store/test_cache_prefix_pair.py)。
下述命令在仓库根目录执行；也可单独复制验证脚本，仅安装其两个客户端依赖后执行。

### 5.1 服务约束

| 实例 | 要求 |
|---|---|
| writer（实例 A） | 本文配置 kv_both，允许查询、加载和写入；测试期间保持存活 |
| reader（实例 B） | 同样配置 kv_both，允许查询、加载和写入 |
| 两端 | 使用本分支的 Mooncake backend；共享 master/tenant；关闭本地 prefix caching；没有其他请求干扰 |

writer/reader 是客户端“先请求 A、再请求 B”的顺序标签，不是必须使用的服务角色。
两个实例都可为 PD 混部节点。kv_both 不需要通过 consumer_is_to_load 打开查询；
若另选 kv_consumer 才需 consumer_is_to_load=true，并可关闭 consumer_is_to_put/save_decode_cache。

普通路径中，实例 A 的 kv-transfer-config 示例：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "lookup_rpc_port": "0",
    "use_layerwise": false,
    "cache_prefix": "model-weights-v1"
  }
}
```

实例 B 同样使用 kv_both，将 lookup_rpc_port 改为 "1"；共享测试保持相同前缀，隔离测试改为新前缀。
两个实例均具有查询能力，不等于每次请求都发生外部 GET；验收以外部命中和实际加载证据为准。

两个实例不一定需要两台物理服务器，但都必须正常接受推理请求并同时存活。
writer 退出可能使其贡献内存上的对象消失，导致错误归因。

### 5.2 正向共享测试

先给两端配置相同前缀，并保持模型权重及缓存布局兼容。以下只包含客户端命令：

```bash
python -m pip install requests prometheus-client
python tests/e2e/common/kv_pool/cache_prefix_probe.py \
  --writer-url http://127.0.0.1:8100 \
  --reader-url http://127.0.0.1:8200 \
  --writer-model writer-served-name \
  --reader-model reader-served-name \
  --expect hit --out prefix-hit.json
```

模型参数是 API 的 served-model-name，不是权重路径。URL 支持根地址和 /v1 后缀。
默认通过两端 /tokenize 校验一致性，构造带本次唯一标识的 1536 个 token IDs，
分别发起一次推理：writer 应冷 miss，reader 应 hit；默认生成 16 tokens 并比较相同权重的输出。
客户端只需 requests/prometheus-client，不需安装 torch、vLLM、NPU 或本地 tokenizer。

### 5.3 反向隔离测试必须带正向对照

保留 writer 和上一步对象，由用户给 reader 配置不同前缀，再执行：

```bash
python tests/e2e/common/kv_pool/cache_prefix_probe.py \
  --writer-url http://127.0.0.1:8100 \
  --reader-url http://127.0.0.1:8200 \
  --writer-model writer-served-name \
  --reader-model reader-served-name \
  --payload prefix-hit.json --writer-expect hit \
  --expect miss --out prefix-miss.json
```

这里复用先前报告中的同一组 token IDs。先要求 writer 仍命中并实际加载，证明对象存在；
再要求 reader 外部未命中且无外部加载，才能排除“writer 没写成功”的替代解释。
配对 miss 验证若没有 `--writer-expect hit`，脚本在推理前拒绝执行。
实例 B 是 kv_both：首次未命中后会将结果写入自己的命名空间，下一次相同输入可能正常命中。
重复隔离测试应使用尚未缓存该输入的新 reader 前缀；只重启 B 不会自动删除 Mooncake 中已有对象。

### 5.4 输入、加载证据与结果

- `--payload` 接受 token ID 数组或先前配对报告；不调用 /tokenize。
- `--prompt-file`/`--prompt` 接受至少 512 tokens 的自定义文本，不自动重复/截断。
- `--prompt-tokens` 调整默认生成输入长度；输入和输出之和必须适配两端 max_model_len。
- `--wait-seconds` 为 writer 推理后的 0..60 秒等待，默认 5 秒；它不是 PUT 完成屏障。
- 普通模式同时要求外部命中、GET key 指标和请求成功；服务应配置 kv_load_failure_policy=fail。
- layerwise 额外传 `--mode layerwise --writer-log <路径> --reader-log <路径>`，
  并开启 range debug；脚本必须能读取实时日志，检查本次 GET 返回字节数与请求一致。
- 不自动重试 reader，以免把 reader 自己写入的数据当成跨实例命中。
- 故意使用不同权重时可 `--skip-output-compare`，但应分别验证各模型自身的冷算正确性。

终端显示生成文本、请求总耗时、指标增量和结论；总耗时不是单独 TTFT。
JSON 保存 token IDs、输入指纹、模型名、原始响应、预期结果和失败阶段。
退出码 0 表示满足本次条件，1 表示请求/校验失败，2 表示参数错误。
缺失指标、旧日志、本地命中、失败/部分加载不能作为有效命中。
这些证据仍不能证明发生了物理 SSD 读取或直接 DMA。

### 5.5 两端 kv_both 的 layerwise 验证

前提是安装匹配的 Mooncake Ascend SDK/Master，并具备 session/range API；
当前分支文档要求 layerwise 使用 Mooncake >=0.3.14。仅有 0.3.11.post1 的普通路径 SDK 不足以完成此验收。
保持既有模型、TP/DP/PP 与其他兼容配置不变，实例 A 的 kv-transfer-config 改为：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "lookup_rpc_port": "0",
    "use_layerwise": true,
    "cache_prefix": "model-v1-layerwise"
  }
}
```

实例 B 同样为 kv_both/use_layerwise=true，lookup_rpc_port 使用 "1"。
两端的附加配置和本地缓存选项如下；已有 additional-config 时合并字段，不覆盖其他设置：

```text
--additional-config '{"kv_pool_range_debug": true}'
--no-enable-prefix-caching
```

日志必须包含实际 worker/rank 输出，并在脚本所在机器上实时可读。
共享测试使用相同前缀，执行：

```bash
python tests/e2e/common/kv_pool/cache_prefix_probe.py \
  --writer-url http://127.0.0.1:8100 \
  --reader-url http://127.0.0.1:8200 \
  --writer-model writer-served-name \
  --reader-model reader-served-name \
  --mode layerwise \
  --writer-log /logs/writer.log \
  --reader-log /logs/reader.log \
  --expect hit --out layerwise-hit.json
```

writer 首次 miss 后写入；reader 必须外部 hit、本地命中为零，且本次请求出现
`[KVPOOL_RANGE_DEBUG]` 的 direction=load 记录。脚本要求每个 range 返回字节数与请求一致，
并比较两端同权重输出。默认 1536 tokens 若不足以覆盖模型的有效缓存传输粒度，
应调整 --prompt-tokens，同时满足两端 max_model_len。

隔离测试保留 A，将 B 的前缀改为未使用的 model-v2-layerwise 并由用户重启 B；
复用上一步报告并先确认 A 的对象仍可命中：

```bash
python tests/e2e/common/kv_pool/cache_prefix_probe.py \
  --writer-url http://127.0.0.1:8100 \
  --reader-url http://127.0.0.1:8200 \
  --writer-model writer-served-name \
  --reader-model reader-served-name \
  --mode layerwise \
  --writer-log /logs/writer.log \
  --reader-log /logs/reader-v2.log \
  --payload layerwise-hit.json --writer-expect hit \
  --expect miss --out layerwise-isolation.json
```

本次要求 A 外部 hit 且完成 range 加载，B 外部 miss 且没有 range 加载。
B 首次 miss 后可以正常向自己的前缀写入，脚本不会自动重试或清理该命名空间。
完整输出、失败阶段与退出码见 5.4；不要只根据输出文本相同或耗时下降判断缓存命中。

## 6. 测试建议与优先级

| 优先级 | 用例 | 验收重点 |
|---|---|---|
| P0 | 普通路径手动同前缀 | writer 冷 miss，reader 外部 hit，实际 GET 完成，输出正确 |
| P0 | 普通路径不同前缀 | 复用同一输入，writer hit 正向对照，reader miss 且无 GET |
| P0 | layerwise 同/异前缀 | 同上，并检查当前请求 range 结果，无 session/revoke 错误 |
| P0 | 未配置和空字符串 | 保持旧 key；可在独立测试域验证旧版写、新版读 |
| P0 | auto 相同身份 | 真实配置副本、scheduler/worker 和跨实例前缀一致并命中 |
| P0 | auto 身份改变或缺失 | 构建身份变化隔离；身份缺失在 Store setup 前失败 |
| P0 | Memcache/Yuanrong 回归 | 默认/空值不变；非空值明确报错，不以缺 SDK 的 ImportError 代替 |
| P1 | 全量 reset | 两个前缀都先有对象，协调所有 writer、排空 PUT 后清理，二者均失效 |
| P1 | DSV4.1 + DSpark | 保持既有 target/draft/TP/DP/PP；验证实际 KV 读取、draft 执行和输出 |
| P1 | 真实权重滚动升级 | 新旧前缀并存；每个版本与自己的冷算基线比较；记录重叠容量需求 |
| P2 | 性能与长稳 | 同环境、同负载比较空/手动/auto；记录 key 生成 CPU 开销、端到端延迟及内存影响 |
| 扩展 | SSD | 内存层验证后单独验证 SSD 副本选择与物理读取，不用 lookup hit 代替 SSD 命中证据 |

先完成小模型普通路径，再覆盖 layerwise；大型 DSpark 场景作为专项/nightly，避免每个 PR 重复启动重型模型。
社区 two_card 目录当前默认路由 A3，不能把目录名或 A3 CI 通过当作 A2 证据。
A2 的 SDK、Master、驱动、CANN、模型和相关配置必须随结果记录。

## 7. 已完成验证与待验证项

| 验证层次 | 结果和边界 |
|---|---|
| Windows 核心 CPU/mock 回归（第一轮补强） | 267 passed、189 subtests passed；4 skipped、1 deselected |
| 共享函数归并后的 Windows CPU/mock 回归 | 271 passed、191 subtests passed；4 skipped、1 deselected；含配对客户端用例 |
| Rebase 到 3e9e1869a 后的同组回归 | 再次得到 271 passed、191 subtests passed；4 skipped、1 deselected |
| 新 main 的 hook 分发控制流检查 | 6 passed、10 subtests passed、1 deselected；补齐本地 torch.ones 构造替身，不检验真实张量数据 |
| 函数归并等价检查 | 移动前后的 apply_cache_prefix/resolve_cache_prefix AST 一致，独立文件已删除，无旧模块导入残留 |
| 4 个 skip | 真实 vLLM 配置测试；Windows 无依赖时明确跳过，不能计为配置兼容性通过 |
| 1 个 deselect | 原有 FakeMamba 非 dataclass 限制；A2 真实依赖环境不要沿用排除条件 |
| 最新配对客户端及探针回归 | 8 passed、4 subtests passed，模拟 HTTP 双服务，不是 NPU 推理 |
| 定向静态检查 | 按本次改动运行 Ruff、Codespell、Typos、Markdownlint、语法和相关仓库检查；提交前复核结果 |
| 定向类型检查 | mypy 1.11.1；最新客户端及配对测试的 Python 3.10 规则已通过 |
| 原始完整 format.sh ci | 未通过完整验证：nodeenv 选择 Node 预发布版本后解析失败 |
| 稳定 Node 临时兼容配置的全仓检查 | 因资源和耗时停止，没有完整成功退出结果；未修改仓库 CI 配置 |
| 全仓 mypy/真实配置单测 | Windows 缺少配对 torch/vLLM；待真实依赖环境执行 |
| A2 NPU E2E/DSpark/SSD | 待用户部署与执行，未声明通过 |

新 main 的完整 shared-source GVA 数据回环需要真实 CPU torch；本地无 torch 的 bootstrap 不适用。
首次运行部分新 hook 用例时发现本地替身缺少 torch.ones，补齐构造替身后仅验证无张量内容断言的 hook 分发路径；
完整 GVA 数据回环及被排除的 full-sparse-sequence 用例仍需在真实依赖环境执行。
文件删除已检查 CODEOWNERS：没有指向已删除独立模块的规则，kv_pool 父目录所有者规则继续适用。

完整格式检查应在准备好的 A2 Linux 环境执行 `bash format.sh ci`，并单独执行 mypy 和相关 UT。
ARM64 应准备对应架构的 ShellCheck/Gitleaks、稳定 Node.js >=20 与 npm，避免误用 x86 二进制。
venv 用于 Python 依赖隔离，不是 NPU 或功能要求；专用开发容器可直接使用容器 Python。

## 8. 后续提交与复现记录

PR 应按模板记录两仓库 SHA、实际导入路径、包版本、硬件、模型、命令、指标、日志及失败/跳过项。
补全正式 PR 编号对应的 release note，保留 DCO sign-off。
在取得 A2 证据前，可继续准备代码和评审材料，但不能把 Windows mock/静态检查写成 NPU E2E 通过。

参考：

- [社区开发指导](https://github.com/vllm-project/vllm-ascend/blob/main/AGENTS.md)
- [社区 PR 测试要求](../contribution/testing.md#pr-testing-requirements)
- [A2 安装和验证步骤](../contribution/mooncake_cache_prefix_validation.md)
- [用户配置说明](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#deployment-cache-namespaces)
- [上游手动前缀 PR #45767](https://github.com/vllm-project/vllm/pull/45767)
- [上游自动前缀 PR #65160](https://github.com/vllm-project/vllm/pull/65160)
