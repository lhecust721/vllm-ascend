# DeepSeek-V4 PD 分离跨轮 KV 复用问题分析与修复 — 背景信息（2026-09-13）

> 用途：opencode 后续分析及三方（vllm-ascend 维护者/应用层接入方）参考。所有结论均在本机实测/代码验证过，标注证据来源。
> 前置背景见 `dsv4-flash-analysis-background.md`（standalone 部署、模板机制、4096 粒度结论）。本文覆盖其后的 PD 分离跨轮复用排查全过程。

## 0. TL;DR

- **问题**：PD 分离下，decode 第一轮保存的 KV 在第二轮 prefill 的 AscendStore lookup 恒 0 命中（思考/非思考模式均然）。
- **根因共 4 个**（2 个真凶 + 2 个伴生）：
  1. **put 触发时机 1-token hash 滞后**（核心真凶，vllm-ascend）：put 在 tracker.token_len 跨 4096 的调度步触发，但该步的边界 token 尚未被采样/哈希（async 调度固有），`block_hashes` 差 1 个 → 整数除法截断每组最后一个 chunk；**c128 组（g1）的唯一 4096-token chunk 因 2047//2048=0 整个消失**，g0 恒 31/32，SWA 组恒 3/4、15/16。
  2. **回填 token 的 BPE 重编码边界歧义**（应用层）：chat content 回填走 detokenize→retokenize，同一段文本存在多种合法切分（如 `[允许][多层]` vs `[允][许多][层]`），分叉点 < 4096 时命中归 0。内容相关、概率性。
  3. tracker 种子不含 P→D 外部加载块（伴生，鲁棒性缺陷，非 31/0 的真凶）。
  4. put 侧组级 0-key 静默跳过（伴生，日志盲区，排查最大障碍）。
- **修复后**：两模式端到端验证均通过——6 组 100% 命中，`vllm:external_prefix_cache_hits_total` 差值 +4096，P 侧 `Scheduled to load 4096 tokens` 真实加载。
- **命中条件**（非思考）：assistant content 原样回填即可（受 BPE 运气影响）；（思考）：**三条件缺一不可**：`thinking=true` + `drop_thinking=false` + assistant 带 `reasoning` 字段（`reasoning_content` 在线等效，离线不等效）。
- **根治 BPE 风险**：会话层缓存 `token_ids`（`return_token_ids` 可用），回填走 `/v1/completions` token-ids 路径。

## 1. 部署拓扑（本排查涉及）

```
client → 代理(8700, load_balance_proxy_server_example.py)
          ├→ P(8195, 卡0-3, kv_producer, MultiConnector=[MooncakeHybrid(producer,31000), AscendStore(producer, lookup_rpc 35001)])
          └→ D(8196, 卡4-7, kv_consumer, MultiConnector=[MooncakeHybrid(consumer,31100), AscendStore(consumer, consumer_is_to_put+save_decode_cache, lookup_rpc 35002)])
共享: mooncake master 71.10.29.141:12001 (SSD offload), key 前缀同 model_name → P/D 共享同一 store
```

- 代理对 P 侧改写 `max_tokens=1/min_tokens=1` + 注入 kv_transfer_params；D 侧透传原 payload，响应由 D 经代理回传（→ **chat 响应中的 prompt_token_ids/return_* 字段即 D 侧地面真值**）。
- P 侧 `_truncate_request_for_prefill`（mooncake_hybrid_connector.py:1290）会 pop 掉最后一个 prompt token（D 侧重算末 token），不影响跨轮 hash（hash 链从头算）。
- 双方 `--no-enable-prefix-caching`：本地前缀缓存关闭，跨轮复用**只能**走 AscendStore（MooncakeHybrid 只管每请求 P→D 搬运，不跨轮）。
- P 侧 put 条件：本请求总量 ≥ 4096 才存（chunk_boundary 逻辑）→ **短 prompt + 长生成的第一轮只有 D 会存**，这是聊天续问的典型形态。

## 2. 问题现象（修复前）

- 第二轮请求 P 侧 lookup：g0 `exists_chunks` 19/32 或 31/32（视当轮 token 分叉位置），**g1 恒 0/1**，SWA 组 0/4、3/4，最终 `hit=0`；`/metrics vllm:prefix_cache_hits_total` 差值 0。
- D 侧 round-1 put：g0 每卡 8/8/8/**7**=31 个 key，**g1 从无 key 产出**。
- decode 实例 `External prefix cache hit rate: 99.9%` 是 P→D 每请求搬运（do_remote_prefill），与跨轮复用无关，勿误读。

## 3. 根因分析（按最终定位顺序，含误判修正）

### 3.1 核心真凶：put 触发时 1-token hash 滞后（vllm-ascend 缺陷）

- **链路**：D 侧 `save_decode_cache` 的 put 由 `_process_running_cached_request` 在 tracker.token_len 首次 ≥ 4096 的调度步构造 ReqMeta；async 调度下该步的第 4096 个 token 尚未被采样 → `Request.block_hashes`（hash_block_size=2，即 2 token/hash）只有 **2047** 个（覆盖 4095 tok）。
- **后果**：put 的 `_iter_token_chunks` 用 `min(len(grouped_hashes), cdiv(token_len, effective))` 截断：
  - g0 (c4, chunk=128tok, scale=64): 2047//64 = **31** → chunk31（token 3968-4095）丢；
  - g2/g3 (SWA, chunk=32, scale=16): 2047//16=127 → 窗口尾 chunk127 丢；
  - g5 (chunk=8, scale=4): 2047//4=511 → 丢 1；
  - **g1 (c128, chunk=4096, scale=2048): 2047//2048 = 0 → 唯一 chunk 全灭**。
- **防护为何失效**：`from_request_tracker` 的 `boundary_without_hash` 比较 `full_block_count(1) > len(block_hashes)(2047)` 恒假——它假设 block_hashes 按 4096 粒度计数，而实际 hash_block_size = gcd(组块大小 [32,32,32,32,2,8]) = **2**。
- **误判修正（记录避免重蹈）**：曾据 g0=31 推断"chunk0 因外部块缺失被 offset 跳过"——后用 put/lookup 的 sample_keys 对比证明 **TP0 存了 chunk0**（`aa8dd10a` 两轮一致），缺的是 chunk31；该误判源于把 logprobs 时代 gen_ids 不含 EOS 的惯例套到了 return_token_ids（其 token_ids **含**收尾 EOS）上。

### 3.2 回填 token 的 BPE 重编码边界歧义（应用层，内容相关）

- decode 保存序列 = prompt + 真实生成 token；第二轮 prompt = 模板渲染(prompt + 回填 content)。content 是生成的 detokenize，渲染后再整体 retokenize——**不保证闭环**。
- 实测例证（非思考模式 [4b](3)）：生成 `[允许][多层]`，重编码 `[允][许多][层]`（"反向传播允许多层神经网络"），LCP 2456/4608，分叉点 2510 < 4096 → 命中 0。
- 表现：端到端命中 0 时，g0 exists = floor(分叉点/128) 个 chunk——用 lookup 的 `exists_chunks` 可反推分叉位置。
- 根治：客户端缓存 `token_ids` 回填（/v1/completions token-ids 路径）；或接受概率性命中。

### 3.3 tracker 种子缺外部加载块（伴生，已修但非真凶）

`_process_new_request` 原种子取 `NewRequestData.block_ids`（仅首步 allocate_slots 返回的新块）；P→D 远程加载块走 `allocate_new_computed_blocks` 内部路径不在返回值中。实测本栈 D 侧首步新块恰好覆盖全表（种子日志 groups_len=[1,0,2,2,27,7] 即全量表），故非 31/0 真凶；修复保留为语义正确性增强。

### 3.4 put 静默跳过（伴生，日志盲区）

`kv_transfer.py` put 循环原 `if not keys: continue` 无任何输出——g1 从不保存这一事实在日志中完全不可见，是排查耗时最长的原因。

### 3.5 指标误读（观测层）

PD 下 `vllm:prefix_cache_hits_total` 恒 0（本地前缀缓存关闭）；AscendStore 跨轮命中记在 **P 实例**的 `vllm:external_prefix_cache_hits_total`；D 实例的 external 指标是 P→D 搬运量。

## 4. 关键原理速查

### 4.1 六个 KV cache 组（DSV4_BLOCK_SIZES[32]=[mla 32, swa 32, c4_state 2, c128_state 8]）

| 组 | 内容 | block | family | key 粒度(tok) | store/lookup mask |
|---|---|---|---|---|---|
| g0 | c4 压缩注意力 KV | 32 | c4 | 128 | 全放行 |
| g1 | c128 压缩注意力 KV | 32 | c128 | **4096**（=传输粒度） | 全放行 |
| g2/g3 | SWA 注意力 KV | 32 | c1 | 32 | reachable：仅 4096 段尾窗口 4 块 |
| g4 | c4 compressor state | 2 | c1 | 2 | 窗口 8(=2×4)：段尾 4 块 |
| g5 | c128 compressor state | 8 | c1 | 8 | 窗口 128(=1×128)：段尾 16 块 |

### 4.2 key 与 hash 链

- key：`{model}@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@group:{g}@cache_role:kv@cache_family:{cN}@{chunk_hash}`
- chunk_hash = sha256(域串 "vllm-ascend-grouped-block-hash-v1" + 连续 base hash)；base hash = H(parent_hash, 2-token)（**纯 token 函数**，无 salt；P/D 对同序列算出同 key，store 共享已实证）。
- **跨轮命中要求 6 组交集**且对齐 4096 传输粒度 → g1 的 1 个 key 缺失即整体归 0；g0 缺 chunk31 同理把 4096 掐成 3968→floor 0。

### 4.3 put/lookup 流程要点

- put（D，save_decode_cache）：tracker.token_len 跨 4096 触发；分片 put_step=4（MLA num_kv_head=1，4 卡各存 1/4 chunk，key 中 head_or_tp_rank 恒 0）；先 exists 再存缺失。
- lookup（P）：`get_num_new_matched_tokens` → floor(4096) → ZMQ 到 P rank0 worker → coordinator 6 组交集；SWA 组只查段尾窗口。
- P→D 搬运块数：压缩组 `prompt_len // compress_ratio`（**floor**，prompt<128 时 g1 传 0 块——语义合理，信息在 state 组，但决定了 D 侧 g1 依赖 decode 期自建块，块在 token 128 处由 `cdiv` 上取整分配并经 new_block_ids 进 tracker，已实证）。

## 5. vllm-ascend 修改清单（editable 安装，路径 /vllm-workspace/vllm-ascend）

| 文件 | 位置 | 修改 | 作用 |
|---|---|---|---|
| `kv_pool/ascend_store/config_data.py` | `from_request_tracker` (~L990, L1000-1013) | 新增 `hash_block_size` 参数；`save_target_len = min(target, len(block_hashes)*hash_block_size)` 封顶保存长度 | **核心修复**：put 推迟到 hash 覆盖齐备的下一步触发，消除每组末 chunk 截断与 g1 全灭 |
| `kv_pool/ascend_store/pool_scheduler.py` | `update_state_after_alloc` (~L600) | 无条件快照全量块表（原仅 num_external>0） | MultiConnector 下外部 token 记在 Mooncake 子连接器、本侧恒 0，原存恒空表 |
| 同上 | `_process_new_request` (~L738, L762) | 种子优先全量表（含外部块），空表回退；种子快照日志 | 外部块进 tracker；`tracker seed req=... source=... groups_len=...` 可观测 |
| 同上 | `_process_running_cached_request` (~L886) | 新块到达时 tracker 快照日志 | 实证 g1 块在 token 128 进入 tracker（排除分配链路嫌疑的关键证据） |
| 同上 | `_build_req_meta`(~L715)/`_process_async_load_request`(~L946) | 传 `hash_block_size=self.hash_block_size` | 核心修复接线 |
| `kv_pool/ascend_store/kv_transfer.py` | imports(L32) + put 循环 (~L765-815) | 三级诊断：0-key **WARNING**（有 chunk 但块表空）/ partial coverage DEBUG / sharding+already-exist DEBUG（含 expected_chunks、len(block_ids)） | 消除静默盲区；本次即靠 `expected=1,len=1,0 keys` + `g0=31` 锁定真凶 |

未改动但被证伪/证实的假设：`_compute_transfer_block_ids` 的 floor 语义（P→D 0 块）**不是** g1 不保存的原因；g2-g5 的 1/N 缺失是 hash 截断的下游结果而非 SWA mask bug。

## 6. 验证脚本与测试过程

### 6.1 脚本

- `verify_dsv4_multiturn_kv_reuse.py`（非思考，v3）：新增 [4b] Path-B 链路五检查（(0) D 侧渲染对照 (1) reasoning 切分 (2) content 字符串保真 (3) retokenize 闭环 LCP (4) 完整链 LCP 定位分叉 token）；[5] 改 external 指标 + round-3 同 prompt 重发（确定性 +4096）。Path-B 请求带 `return_token_ids/return_prompt_text` 取 D 侧地面真值。
- `verify_dsv4_thinking_multiturn_kv_reuse.py`（思考，v3 重写）：v2 直连单实例 → PD 拓扑（/v1 走代理、/tokenize@P、metrics P+D）；期望序列改用 return_token_ids 地面真值（注意 **token_ids 含收尾 EOS**，勿重复追加）；指标改 external；加 turn-3 重发；长度标定（prompt≈3465-3468，min_tokens=800 保保存 ≥4245 且对照组分叉点<4096）。

### 6.2 测试轮次（每轮含 P/D 重启，模型加载约 9-10 分钟）

1. 修复 1-3 后：g0 31/32、g1 0/1、SWA 3/4——所有组恰缺触及 4096 边界的最后一个 chunk + tracker 日志显示 g1 块正常 → 锁定 hash 滞后。
2. 修复 4 后：D 侧 put g0=8/8/8/8、**g1 keys=1（TP0，首次）**；lookup 6 组 100% → `hit=4096`；P 侧 `Scheduled to load 4096 tokens`。
3. 最终轮（非思考，当轮 token 全匹配）：`ext@P` round-2 +4096、round-3 +4096。
4. 思考模式 v3：V-B（三条件齐）LCP **4334/4334 完整匹配**、turn-2B `ext@P` +4096、turn-2A（默认丢弃）0、turn-3B +4096；[3b] 证明 reasoning_content 在线归一化生效（prompt_text == V-B 离线渲染）。

### 6.3 运维备忘

- 重启需带 `VLLM_LOGGING_LEVEL=DEBUG`（部署脚本未含，旧实例带、新起默认 INFO 会丢全部 put/lookup/tracker 日志）：`export VLLM_LOGGING_LEVEL=DEBUG && bash dsv4_prefill.sh && bash dsv4_decode.sh`；脚本 `>` 截断日志，先备份。
- 日志关键 grep：`coordinator lookup group|final`（P，命中明细）、`KV pool put|put skipped|already in store`（D，存盘明细）、`tracker seed|tracker update`（D，块表轨迹）、`kvpool hit tokens|Scheduled to load`（P，命中与加载）。
- 排查跨轮 miss 的对照法：P lookup `exists_chunks=N/32` × 128 = 分叉 token 位置；与 D put `sample_keys` 首 key（=chunk0）比对可辨 chunk0 是否存过。

## 7. 最终结论

1. **两模式跨轮复用均成立**（修复后）：命中 4096/轮（粒度上限，floor(保存量/4096)）。
2. 思考模式三条件：`thinking=true` + `drop_thinking=false` + assistant 带 `reasoning`（在线 `reasoning_content` 等效、离线不等效）；任何缺失 → 分叉点在 `<｜Assistant｜>` 后（≈prompt 末）→ 命中 0。
3. 剩余风险：BPE 重编码歧义（内容相关，token-ids 回填可根治）；`return_token_ids`/`return_prompt_text` 为 vLLM 扩展，代理透传无损。
4. 建议上游合入：config_data.py 的 hash 封顶为核心修复；put 诊断日志与 tracker 快照建议常态化。

## 8. 文件索引

- 本文：`/workspace/opencode/dsv4-pd-kv-reuse-analysis-background.md`
- 前置背景：`/workspace/opencode/dsv4-flash-analysis-background.md`
- 验证脚本：`/workspace/verify_dsv4_multiturn_kv_reuse.py`（非思考 v3）、`/workspace/verify_dsv4_thinking_multiturn_kv_reuse.py`（思考 v3）
- 修复源码：`/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/{config_data,pool_scheduler,kv_transfer}.py`
- 运行日志（修复后 DEBUG 全量）：`/workspace/vllm_prefill.log`、`/workspace/vllm_decode.log`（历史备份 `*.bak0913_*`）
- 验证输出存档：`/tmp/opencode/verify_after_fix3.log`、`/tmp/opencode/verify_final.log`、`/tmp/opencode/verify_thinking_v3_final.log`（临时目录，重要结论已录本文）
