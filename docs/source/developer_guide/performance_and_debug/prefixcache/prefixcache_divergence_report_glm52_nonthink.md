# GLM-5.2 (w4a8) standalone 非思考模式 — decode 保存 KV → prefill 复用 BPE 分叉验证报告

> 运行目录: `outputs/perf_glm52_prefixcache_multiturn_nonthink/20260921_005845` (2026-09-21 00:59–01:45, ~46 min)
> 分析工具: `analyze_prefixcache_divergence.py --granularity 128` (离线 token 级重放, GLM tokenizer)

## 0. 结论摘要

| 验证项 | 结果 |
|---|---|
| **BPE 重编码分叉** | **0/54 转移** (23,022 个生成 token, 58,617 个 prompt token) — 未检出 |
| token 级完整 round-trip | **52/54 (96%)**, 与 DSV4 非思考模式持平 (96/100) |
| 2 个非完整匹配转移 | 均为 **max_out_len=1024 截断 + 模板 content.strip() 丢弃尾随空白** (仅损失末尾 1 token, 对齐命中不变), 非 BPE 分叉 |
| 粒度对齐吞没 | 1/54 (自然答案仅 9 token, LCP=108 < 128 粒度, 完整匹配但命中归零) |
| ≥128 对齐命中 | 53/54 (98%); 潜在 external 命中率 90.3% |
| 服务端实际 ext 命中率 | **30.2%** (17,920/59,291) — 瓶颈是 mooncake 传输失败 (3,050 次 Transfer failed), **不是 token 分叉** |

**回答核心问题: GLM-5.2 非思考模式下 decode 保存 KV → 下一轮 prefill 复用, token 级无 BPE 分叉;**
回填链路 (content 文本回填 → 服务端模板渲染) 在本数据集上 token 级保真。
唯一系统性损耗在 KV store 传输层 (混部 HCCL 冲突, 见 §C), 与分叉无关。

## A. 部署与测试配置

| 项 | 值 |
|---|---|
| 服务 | GLM-5.2-w4a8 standalone TP8, 端口 8195, `--no-enable-prefix-caching` (复用只走 AscendStore external 路径) |
| KV 保存 | `save_decode_cache=true` + `consumer_is_to_put=true`, 复用粒度 128 token (glm_moe_dsa 单组 hash_block_size=128) |
| 数据集 | `ShareGPT_prefixcache_glm.json`: 20 会话 × 2-4 轮 (74 请求, 54 个相邻轮次转移) |
| 生成参数 | 非思考 (`enable_thinking=false`), temperature=0.01, ignore_eos=false, max_out_len=1024, **无 min_tokens** (自然停止, EOS 不进响应文本, 无 DSV4 的 EOS 分叉机制) |
| 采集 | `return_token_ids=true`: 流式首块顶层 `prompt_token_ids` + `choices[i].token_ids` (vLLM 0.23.0 布局), details.jsonl 0 缺失 |
| 已知污染 | 运行前 probe 使用了数据集前 3 个 prompt (同参数重生成), 对应会话 turn-1 存在少量前置 store 命中 |

- 输入: `/workspace/outputs/perf_glm52_prefixcache_multiturn_nonthink/20260921_005845/performances/glm52-w4a8-prefixcache-nonthink/sharegpt-prefixcache_details.jsonl`
- 会话数: 20  (缺 token ids 跳过的请求数: 0; 需 return_token_ids=True 采集)
- 对齐粒度: 128 (glm_moe_dsa 单组 hash_block_size (GLM-5.2), LCP 不足 128 对齐归零)

## 1. 总览

| 指标 | 值 |
|---|---|
| 相邻轮次对 (转移) 数 | 54 |
| token 级完整匹配 | 52/54 (96%) |
| 达到 128 对齐命中 | 53/54 (98%) |
| 潜在命中 token 合计 (对齐后) | 50688 |
| turn≥1 的 prompt token 合计 | 56142 |
| 潜在 external 命中率 (pot/tot_prompt) | 90.3% |

## 2. 分叉位置分布

| 转移 | 数量 | LCP 中位 | LCP 最小/最大 | 分叉在上一轮生成内的位置(中位) | 对齐命中≥128 |
|---|---|---|---|---|---|
| turn0→turn1 | 20 | 1038 | 108/1391 | 1023 | 19/20 |
| turn1→turn2 | 19 | 1053 | 166/1950 | - | 19/19 |
| turn2→turn3 | 15 | 1087 | 171/2446 | - | 15/15 |

```
LCP [    0,  512):  15 ################
LCP [  512, 1024):   8 ########
LCP [ 1024, 2048):  29 ################################
LCP [ 2048, 4096):   2 ##
LCP [ 4096, 8192):   0 
LCP [ 8192,  inf):   0 
```

## 3. 分叉归类

- 完整匹配: 52/54
- 上一轮生成内容内: 2/54

- 其中 token 级完整匹配但 LCP < 128 (粒度对齐吞没, 非分叉): 1/54

说明: '上一轮生成内容内' 分为两类 —— **BPE 重编码分叉** (decode 生成 token 与下一轮 re-tokenize 在文本中段边界不一致) 与 **生成末尾分叉** (max_out_len 截断或EOS 收尾的尾随空白被模板 content.strip() 丢弃); 后者只损失末尾 1-2 token, 对齐后命中通常不变。'上一轮 prompt 内' = 模板/渲染差异, 需另行排查。

## 4. 典型分叉样例 (Top 2, BPE 重编码分叉优先, 共 0 个 BPE 分叉 / 2 个生成末尾分叉 / 0 个 prompt 内分叉)

### 样例 1/2: 5664bb49 turn0→turn1 [生成末尾 (截断尾随空白被模板 strip / EOS 边界)] response_id=chatcmpl-a0d520bbc4194f1e
- LCP=1044 (上一轮 prompt 21 + 生成 1024), 生成内容内偏移 1023
- 对齐后命中: 1024
- 服务端 lookup 日志: hbm=0 queried=1024 hit=256 miss_at=256 g0_chunk=2 (req 前缀匹配 chatcmpl-a0d520bbc4194f1e*)
- **分叉点前公共文本** (token 1004..1044, 两侧相同):
  > … of the page you want to attach the file to. (You can get this by calling `GET /rest/api/content?title=YourPageTitle`).⏎⏎#### Using `curl` (Bash
- **decode 侧原始 token** @ 分叉点 1044 (起 10 个):
  - ids: `[692]`
  - token 串: ')ĊĊ'
  - 边界文本: )⏎⏎
  - 后续文本 (token 1044..1045):
  > )⏎⏎
- **prefill 重编码 token** @ 分叉点 1044 (起 10 个):
  - ids: `[8, 154827, 4814, 498, 2968, 752, 264, 10130, 729, 369]`
  - token 串: ')' | '<|user|>' | 'can' | 'Ġyou' | 'Ġgive' | 'Ġme' | 'Ġa' | 'Ġpython' | 'Ġfunction' | 'Ġfor'
  - 边界文本: )‖<|user|>‖can‖ you‖ give‖ me‖ a‖ python‖ function‖ for
  - 后续文本 (token 1044..1062):
  > )<|user|>can you give me a python function for this creating a page?<|assistant|><think></think>
  - 注: prefill 侧分叉点紧跟模板 token (如 `<|user|>`) 且 decode 侧末 token含尾随空白 → 生成在 max_out_len 截断/EOS 收尾后, 尾随空白被模板 `content.strip()` 丢弃; 仅损失末尾 token, 对齐命中通常不变

### 样例 2/2: 27bc2356 turn0→turn1 [生成末尾 (截断尾随空白被模板 strip / EOS 边界)] response_id=chatcmpl-b49882796cd2173f
- LCP=1204 (上一轮 prompt 181 + 生成 1024), 生成内容内偏移 1023
- 对齐后命中: 1152
- 服务端 lookup 日志: hbm=0 queried=1152 hit=768 miss_at=768 g0_chunk=6 (req 前缀匹配 chatcmpl-b49882796cd2173f*)
- **分叉点前公共文本** (token 1164..1204, 两侧相同):
  > …```typescript⏎const newNote = await prisma.note.create({⏎  data: {⏎    title: "My Note",⏎    content: "This is the note content.",⏎    userId: "userId",
- **decode 侧原始 token** @ 分叉点 1204 (起 10 个):
  - ids: `[981]`
  - token 串: 'ĠĠĠĠĠĠ'
  - 边界文本:       
  - 后续文本 (token 1204..1205):
  >       
- **prefill 重编码 token** @ 分叉点 1204 (起 10 个):
  - ids: `[154827, 44713, 154828, 154841, 154842]`
  - token 串: '<|user|>' | 'Nice' | '<|assistant|>' | '<think>' | '</think>'
  - 边界文本: <|user|>‖Nice‖<|assistant|>‖<think>‖</think>
  - 后续文本 (token 1204..1209):
  > <|user|>Nice<|assistant|><think></think>
  - 注: prefill 侧分叉点紧跟模板 token (如 `<|user|>`) 且 decode 侧末 token含尾随空白 → 生成在 max_out_len 截断/EOS 收尾后, 尾随空白被模板 `content.strip()` 丢弃; 仅损失末尾 token, 对齐命中通常不变

## 5. 命中损失估算

- 若全部 round-trip 精确: 潜在命中 50688 token (90.3% of turn≥1 prompt)
- 实际达成 128 命中的转移: 53/54
- 因分叉导致 128 命中归零的转移: 0/54
- token 级完整匹配但自然答案过短 (LCP < 128) 归零的转移: 1/54 (粒度对齐吞没, 非分叉)

> 治本方案: 客户端按 token_ids 回填 (ais_bench infer_every 当前用 output.content 文本回填, 经 detokenize→re-tokenize 不可逆); 或服务端接受 token-ids 形式的 assistant 消息。

## C. 服务端 AscendStore 链路实测 (运行级)

### C.1 /metrics 差值 (运行前后)

| 指标 | 差值 |
|---|---|
| `prefix_cache_queries_total` / `hits_total` (本地) | 0 / 0 (no-enable-prefix-caching, 符合预期) |
| `external_prefix_cache_queries_total` | +59,291 token |
| `external_prefix_cache_hits_total` | +17,920 token (**30.2%**) |
| 离线重放潜在命中 (LCP 对齐) | 50,688 / 56,142 (**90.3%**) |

### C.2 命中差距根因: mooncake 传输失败, 非 token 分叉

- 日志切片 (运行窗口) 内 `Transfer failed for key` **3,050 次** (目标 71.10.29.141:2016x/2054x, `TRANSFER_FAIL`),
  与 §glm52-background 0.3 节记录的同主机 PD 混部 HCCL 冲突现象一致 (vLLM 8 卡进程占满 NPU 后 mooncake 传输引擎建链失败);
- decode 侧 put 按 **chunk_index % tp_size 分片**到 8 个 TP rank 并行上传
  (`KV pool put chunks map=[(0,…)] @TP0 / [(128,…)] @TP1 / …`, chunk key `head_or_tp_rank:0` 证实 MLA TP 无关命名),
  传输失败的 chunk 未真正落库;
- 转移级对照 (47/54 转移关联到 MISS 日志): 多数会话服务端 hit 停留在前几轮的部分 chunk
  (如 64256262: t0→t1 hit=0, t1→t2 hit=896 = t0 轮预期, t2→t3 仍 896 —— store 中只有早期成功传输的 chunk);
  小会话 (chunk 少) 反而能 full-hit (如 2534c599 t1→t2 / t2→t3);
- 两个分叉样例的服务端 miss_at (256 / 768) 与离线 LCP 粒度对齐值 (1024 / 1152) 的差距同样来自 chunk 缺失,
  与分叉点位置 (1044 / 1204) 无关。

### C.3 治理建议

1. **分叉侧 (client)**: 无需治理 (0 BPE 分叉)。若追求极致, 可按 token_ids 回填;
   max_out_len 截断场景可在回填前 rstrip 内容, 消除 strip 型末尾分叉。
2. **传输侧 (server, 收益最大)**: 解决混部 HCCL 冲突 (mooncake transfer engine 独立网卡/NPU 资源、
   或 store 后端走 SSD-only 路径), 可将 ext 命中率从 30.2% 提升到接近潜在 90.3%。
3. 对比 DSV4 PD 分离部署 (专用 P/D 实例): ext 命中率 70.7%, 同样远低于其潜在值但无传输失败 ——
   印证 standalone 混部是本 run 传输失败的主因。

## D. 与 DSV4 非思考模式验证对照

| 指标 | DSV4-Flash (PD 分离, TP4/DP2×TP2) | GLM-5.2 (standalone TP8) |
|---|---|---|
| token 级完整匹配 | 96/100 | **52/54 (96%)** |
| BPE 重编码分叉 | 4/100 | **0/54** |
| 其他分叉 | 0 | 2 (截断+strip, 可避免) |
| 复用粒度 | 4096 (c128 压缩层) | 128 (单组, 无压缩) |
| 强制长生成 (min_tokens) | 必须 (4096, 消 EOS 分叉) | 不需要 (自然停止即无 EOS 分叉) |
| ext 命中率 实际/潜在 | 70.7% / — | 30.2% / 90.3% (传输受限) |

---
