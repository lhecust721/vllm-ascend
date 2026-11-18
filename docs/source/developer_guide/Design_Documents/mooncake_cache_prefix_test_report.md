# Mooncake cache_prefix 特性验证测试报告

- 报告日期：2026-11-18
- 被测特性：`lhecust721/vllm-ascend` `codex/ascend-store-cache-prefix` 部署级 `cache_prefix`（Mooncake 普通路径 + layerwise 路径）
- 参考文档：`docs/source/developer_guide/Design_Documents/mooncake_cache_prefix.md`（b48591145）及 Mooncake `kv-cache-sharing-and-isolation.md`
- 测试范围：普通（bulk）路径 P0、layerwise 同/异前缀、auto 身份改变隔离、全量 reset、DSV4.1+DSpark（经 SSD 持久化的跨进程验证，见第 8 节）。SSD 物理读取/DMA、性能与长稳 未纳入本次（见第 6 节）

## 1. 测试目标与结论摘要

验证在共享 Mooncake 池内，`cache_prefix` 能实现「同前缀复用、不同前缀隔离、空/auto 语义正确、auto 身份改变隔离、非 Mooncake 后端受保护、全量 reset 清理」，并在 layerwise 数据平面下同样成立。

结论：普通路径 P0、layerwise 同/异前缀、auto 身份改变隔离、全量 reset、DSV4.1+DSpark 场景均通过。A2 实机、真实 Qwen3-30B-A3B（双实例同 Pod）与 DeepSeek-V4.1-Flash（DP2×TP4+EP8 单机）推理下，跨实例外部命中与加载、命名空间隔离、空/auto 语义、身份缺失失败路径、身份改变隔离、全量清理、以及经 SSD 持久化的跨进程复用均符合文档预期。

| 用例 | 结果 |
|---|---|
| 安装版本一致性 | 通过 |
| T1 正向共享（同前缀 hit，bulk） | 通过 |
| T2 反向隔离（异前缀 miss + 正向对照，bulk） | 通过 |
| T3a 空前缀（旧 key 兼容，bulk） | 通过 |
| T3b auto 同身份跨实例一致（bulk） | 通过 |
| T3c auto 身份缺失（负向，bulk） | 通过 |
| T4 Memcache/Yuanrong 回归 | 通过（UT 16 passed） |
| LW-T1 layerwise 同前缀共享 | 通过 |
| LW-T2 layerwise 异前缀隔离（含正向对照） | 通过（附过程注意点 4A.3） |
| T5 auto 身份改变隔离（bulk） | 通过（附过程观察 4B.2） |
| T6 全量 reset（bulk） | 通过 |
| DSV4.1 + DSpark（同/异前缀，SSD 持久化跨进程） | 通过（见第 8 节） |

## 2. 环境与版本

| 项 | 值 |
|---|---|
| 集群/命名空间 | Kubernetes / `mooncake-vllm` |
| Pod / 容器 | `lihong-mooncakeha-vllm` / `vllm` |
| 镜像 | `docker.io/library/vllm-ascend-mooncake-etcd:v0.30.0` |
| 节点 | `syn-141`（单节点，8× Ascend 910B3） |
| 模型 | `/data1/weight/Qwen3-30B-A3B-w8a8`（ascend 量化 w8a8，MoE，TP=2+EP） |
| vllm-ascend 源码 | `/root/work/cache-prefix-arm64/vllm-ascend`，HEAD `b4859114546cc3b36fec1853d631721879c9dfcd`（分支 `verify-cache-prefix-b48591145`，rebase 基线 `3e9e1869a`） |
| vllm-ascend 安装版本 | `0.1.dev6739+gb48591145`（editable，单一 dist-info，`import` 解析到源码） |
| 配对 vLLM | `0.1.dev1+gba15685ab.empty`；`.github/vllm-main-verified.commit` = `ba15685ab`（一致） |
| Mooncake 池 | `MC_STORE_CLUSTER_ID=mooncake-ha-prod`，etcd 元数据服务，SSD offload 开启 |
| Mooncake SDK | 0.3.14（含 session/range API，支持 layerwise） |

安装一致性核查（重装后）：
- `import vllm_ascend` → `/root/work/cache-prefix-arm64/vllm-ascend/vllm_ascend/__init__.py`
- `importlib.metadata.version("vllm_ascend")` → `0.1.dev6739+gb48591145`
- `from ...ascend_store.metadata import resolve_cache_prefix` → OK（重构后 helper 已并入 `metadata.py`，`cache_prefix.py` 已移除）

## 3. 部署与配置

| 实例 | 端口 | NPU | Mooncake 配置 | lookup_rpc_port |
|---|---|---|---|---|
| A（writer） | 8195 | 0,1 | `mooncake_prefill.json` | 35001 |
| B（reader） | 8196 | 2,3 | `mooncake_decode.json` | 35003 |

共同参数：`--served-model-name qwen3`、`--no-enable-prefix-caching`、`VLLM_LOGGING_LEVEL=DEBUG`、`--max-model-len 40960`、`--max-num-batched-tokens 16384`。两端统一 `kv_both` + `kv_load_failure_policy:"fail"`。原 `kv_consumer` 脚本备份为 `*.bak.kvconsumer`。

layerwise 回环配置（用户按文档 §5.5 启用）：`use_layerwise:true` + `--additional-config '{"kv_pool_range_debug": true}'` + Mooncake SDK 0.3.14。

探针：仓库最新版 `tests/e2e/common/kv_pool/cache_prefix_probe.py`（成对 writer/reader、`--writer-expect` 正向对照、`--payload` 复用、`--mode layerwise --writer-log --reader-log`）。全量 reset 使用自备 ZMQ 客户端（协议见 4B.3）。

## 4. 测试用例与证据（普通路径 bulk）

判定口径：以 Prometheus 指标增量与输出一致性为准；要求 `external_queries>0`、`local_hits==0`；hit 需 `external_hits>0` 且有加载证据（bulk 看 `ascend_store_load_get_keys`）；miss 需 `external_hits==0` 且无加载；同权重比对文本。总耗时不作 TTFT。

### T1 正向共享（两端 `model-weights-v1`）— passed
- writer 冷：`external_hits=0, local_hits=0, bulk_get_keys=0`，21.53s
- reader：`external_hits=1535, local_hits=0, bulk_get_keys=24`，0.70s；文本一致

### T2 反向隔离（A=v1，B=v2；复用 payload）— passed
- writer A（对照）：`external_hits=1535, bulk_get_keys=24`，0.28s
- reader B：`external_hits=0, bulk_get_keys=0`；文本一致

### T3a 空前缀（两端 `""`）— passed
- writer miss 无加载；reader `external_hits=1535, bulk_get_keys=24`；旧 key 兼容成立。

### T3b auto 同身份（两端 `auto` + 同 `VLLM_BUILD_COMMIT`）— passed
- 解析前缀：A、B 及 scheduler/worker 均为 `auto-d237af6b022389937be8ac9f3abe92f7eca045d0c97fb5d72085bcededc6dba9`
- reader `external_hits=1535, bulk_get_keys=24`；文本一致。

### T3c auto 身份缺失（B 去 identity，负向）— passed
- EngineCore 启动失败，`health=000`
- `metadata.py:50`：`ValueError: AscendStore cache_prefix='auto' requires VLLM_IMAGE_TAG or VLLM_BUILD_COMMIT`

### T4 Memcache/Yuanrong 回归（UT）— 16 passed
- 非 mooncake 后端 + 非空前缀 → `ValueError "only ... Mooncake"`；空/默认保持原行为；key 生成、特殊前缀、reset 等。
- 备注：`test_cache_prefix_config.py` 4 例裸 pytest 失败（`Unsupported connector type: AscendStoreConnector`，夹具未注册 connector 的环境问题，非功能缺陷）。

## 4A. 测试用例与证据（layerwise）

layerwise 加载证据改为实时日志 `[KVPOOL_RANGE_DEBUG]` `direction=load` 事件统计（每 range 请求字节数须等于返回字节数），`bulk_get_keys` 不作 layerwise 判据。探针附 `--mode layerwise --writer-log /workspace/vllm.log --reader-log /workspace/vllm2.log`。

### LW-T1 layerwise 同前缀共享 — passed
- writer 冷 miss；日志 `direction=save` 98 条 + `commit` 成功（02:46:10）
- reader：`external_hits=1535, local_hits=0, range_load_bytes=150994944`（~144MiB，字节一致）；文本一致

### LW-T2 layerwise 异前缀隔离 — passed（重跑后）
- writer A(v1) 对照：`external_hits=1535, range_load_bytes=150994944`，`direction=load` 覆盖到 layer 47（02:54:23）
- reader B(v2)：`external_hits=0, range_load_bytes=0`
- B miss 后按 kv_both 写入自身 v2：`direction=save` 96 条（02:54:44）；无 session/revoke 错误
- 结论：layerwise 隔离成立。

### 4A.3 过程注意点
- 首轮「T1 后改 B 前缀并 SIGKILL 重启 B，再复用 payload 要求 writer still hit」失败：A `external_hits=0`。
- 不影响读者时 A 单实例复测可命中（对象可持久）；A 重新产生对象后再做对照即通过。
- 详见 4B.2 的对象持久性观察（同一现象）。

## 4B. auto 身份改变隔离 与 全量 reset（bulk，identity X/Y）

切回 bulk（`use_layerwise:false`），两端 `cache_prefix:auto`；identity X = `cache-prefix-test-b48591145`，identity Y = `cache-prefix-test-b48591145-alt`。

### 4B.1 auto 身份改变隔离 — passed
- 同身份 X（预热共享）：writer miss；reader `external_hits=1535, bulk_get_keys=24`；A/B 均解析 `auto-d237af6b...`
- 改 B 身份为 Y 重启：A=`auto-d237af6b...`，B=`auto-2a6c12bd97c3e0755642c3592e2b689154aa2e80d4bdfb0f7045e9497160064c`（不同命名空间）
- 隔离（复用同一 payload）：
  - writer A(auto-X)：`external_hits=1535, bulk_get_keys=24`（正向对照，自身命名空间命中并加载）
  - reader B(auto-Y)：`external_hits=0, bulk_get_keys=0`（miss 无加载）
  - 文本一致
- 结论：auto 身份变化产生不同命名空间，实现跨实例隔离。

### 4B.2 过程观察（对象持久性 / 正向对照）
- 首次「B 重启后立即复用 payload 要求 A still hit」时，A 返回 `external_hits=0`；同 payload 对 A 单实例即时复测可命中。对象在共享池中数分钟（约 5 分钟窗口）后可能难以再次命中（推断：共享池 LRU/容量回收，或读者重启前后的占用/租约变化；本次环境存在其它工作负载，无法仅凭本证据区分）。
- 处理：隔离断言前先对 writer 用同一 payload **重新预热**（本次预热点火后 A 立即命中，再执行 `--writer-expect hit` 的隔离即可稳定通过）。
- 建议：验收规程将「隔离前重新预热 writer」写入步骤；生产避免依赖长窗口对象持久性。

### 4B.3 全量 reset — passed
- 机制：`LookupKeyServer`（ZeroMQ REP，`ipc:///tmp/lookup_rpc_port_{port}_dp_rank0`）收到 `RESET_MSG`（`b"reset"`）→ `pool_worker.reset_store()`（join 发送队列）→ `m_store.remove_all()`。layerwise 不支持 reset（`reset_cache` 显式报错），故在 bulk 下验证。
- 前置：两端均已有对象（A=auto-X、B=auto-Y 均持有同一 payload 的 KV），保持空闲（无在途请求）。
- 执行：向 A 的 RPC（`ipc:///tmp/lookup_rpc_port_35001_dp_rank0`）发送 `b"reset"` → 返回 `RESP_OK (b"\x01")`
- 日志：`lookup_key.py:173  AscendStore reset via remove_all succeeded.`（04:01:30）
- 校验（reset 后立即单实例 miss 探测，同一 payload）：
  - A(auto-X)：`external_hits=0, bulk_get_keys=0`（失效）
  - B(auto-Y)：`external_hits=0, bulk_get_keys=0`（失效）
- 影响面：仅在 A 侧触发一次 remove_all，即清空**共享租户域内的全部前缀**（auto-X 与 auto-Y 同时失效），符合文档「清理影响客户端有效 tenant 域内所有前缀/模型」的描述。
- 结论：全量 reset 可按预期使全部前缀对象失效。

## 5. 关键日志摘录
- `AscendStore resolved cache_prefix: auto-d237af6b...` / `auto-2a6c12bd...`（`metadata.py:73`，scheduler 与 worker 一致）
- layerwise：`[KVPOOL_RANGE_DEBUG] {"event":"range","direction":"save|load"...}` + `{"event":"commit","results":[0,...]}`
- T3c 负向：`ValueError: AscendStore cache_prefix='auto' requires VLLM_IMAGE_TAG or VLLM_BUILD_COMMIT`
- reset：`LookupKeyServer` 收 `b"reset"` → `AscendStore reset via remove_all succeeded.`
- 启动参数：`kv_role='kv_both' ... kv_load_failure_policy='fail'`（layerwise 用例为 `use_layerwise=True`）

## 6. 遗留项与后续

| 项目 | 状态 | 前置/说明 |
|---|---|---|
| 普通路径 P0（T1-T4） | 已完成 | 见第 4 节 |
| layerwise 同/异前缀（LW-T1/T2） | 已完成 | 见 4A |
| auto 身份改变隔离（T5） | 已完成 | 见 4B |
| 全量 reset（T6） | 已完成 | layerwise 外部 reset 仍不支持 |
| SSD 副本 / 物理读取 | 扩展 | 内存层通过后单独验证；本次证据不能证明物理 SSD 读取或 DMA |
| DSV4.1 + DSpark（同/异前缀） | 已完成 | 见第 8 节；单机需顺序执行 + SSD 落盘 |
| 性能与长稳 | P2 | 同环境对比空/手动/auto，记录 key 生成 CPU 开销与延迟 |

## 7. 产物清单（容器 `/workspace/`）

- 普通路径：`prefix-hit.json`、`prefix-miss.json`、`prefix-empty-hit.json`、`prefix-auto-hit.json`、`final-auto-hit.json`
- layerwise：`layerwise-hit.json`、`layerwise-A-selfhit.json`、`layerwise-isolation.json`、`payload-t1.json`
- auto/reset：`auto-id-hit.json`、`auto-A-selfhit.json`、`auto-id-isolation.json`、`auto-p1.json`、`reset-A-miss.json`、`reset-B-miss.json`、`reset_client.py`、`mk_payload.py`、`mk_p2.py`
- DSV4.1+DSpark：`dsv4-cold.json`、`dsv4-ssd-hit.json`、`dsv4-v2-miss.json`、`dsv4-restored-hit.json`、`dsv4-repeat.json`、`dsv4-payload.json`、`dsv41_mix_v2.sh`、`dsv4_filler.py`、`dsv4_mkpayload.py`
- 脚本备份：`qwen3_30B_A3B_mix.sh.bak.kvconsumer`、`qwen3_30B_A3B_mix2.sh.bak.kvconsumer`
- 日志：`/workspace/vllm.log`、`/workspace/vllm2.log`（DEBUG，含 `[KVPOOL_RANGE_DEBUG]`）

报告生成时状态：A（8195）/B（8196）均为 bulk、`kv_both`、`cache_prefix:auto`（A=X，B=Y），`/health` 均 200；已执行一次全量 reset 并验证两命名空间失效。（DSV4.1+DSpark 段时的状态见第 8 节。）

## 8. DSV4.1 + DSpark 场景验证（实机）

环境：DeepSeek-V4.1-Flash-w4a8-Ascend，DP2×TP4（EP8）8×910B3 单机；DSpark `--speculative-config '{"method":"dspark","num_speculative_tokens":3,"enforce_eager":true}'`；`kv_both`、bulk（`use_layerwise:false`）、`cache_prefix:"dsv41-target-r2-dspark-r2"`（隔离用 `...-alt`）。Mooncake master HA（`mooncake-master-0/1` + etcd），SSD 卸载开启（client `rank_*` @ `/data1/mooncakenvme`）。

约束：该模型需 EP8（8 卡）才放得下；拆成 2×(TP4,EP4) 触发 `NPU out of memory`，故同机无法 A/B 双实例并存。采用**单实例顺序 + SSD 持久化**验证。

步骤与证据（清 SSD → 重启 → 打流量落盘 → 跨进程/隔离）：

| 步骤 | 操作 | 证据 |
|---|---|---|
| 0 | 清 `/data1/mooncakenvme` | 13G→0 |
| 1 | 冷写目标 P（1536 token，uuid） | miss（hits=0）；DSpark `drafts=4 / draft_tokens=12 / accepted=12` |
| 2 | 灌流量（约 74 次 ×8000 token 各不相同的 prompt）触发 SSD 批量落盘 | `rank_*` 0→184.9MB→1478.8MB（过 256MB/500key 阈值，P 随批落盘） |
| 3 | 停写者 → 重启同前缀 v1 → 请求 P | 跨进程 HIT：`external_hits=1408, bulk_get_keys=44, local_hits=0`，输出=冷算，DSpark `drafts=4/accepted=12` |
| 4 | 重启异前缀 v2（`...-alt`）→ 请求 P | 隔离 MISS：`external_hits=0, bulk_get_keys=0`；`rank_*=177MB/rank` 仍在（v1 数据在 SSD，作正向对照） |
| 5 | 恢复 v1 → 请求 P | 再次 HIT（`external_hits=1408, load=44`），可复现 |

结论：
- 同前缀：DSV4.1+DSpark 目标 KV 写入 Mooncake store，经 SSD 持久化后，**写实例删除/重启仍可被新实例跨进程外部命中并加载**；DSpark draft 在冷算与命中路径均执行，输出一致。
- 异前缀：key 命名空间隔离成立（miss 且无加载），并具备正向对照（v1 数据仍在 SSD）。
- 环境要点：热 KV 需流量达到 256MB/500key 才批量落 SSD；小流量仅存 DRAM，实例删除即丢失（先前的顺序 miss 即由此产生）；模型需 EP8，故同机顺序执行 + SSD 方案是可行路径。

补充证据：同实例内跨 DP 引擎外部命中（`dsv4-repeat.json`，`external_hits=1408`、`bulk_get_keys=44`、`local_hits=0`）。

DSV4 段结束时状态：原始 v1 实例（8700，`dsv41_mix.sh`）运行中，`/health` 200；SSD `rank_*` 各约 177MB 持久保留。
