# DeepSeek-V4 PD 分离多轮 prefix-cache 分叉分析报告（TP 不对称：P TP4 / D DP2×TP2）

- 数据源：`vllm_prefill.log`（P 侧 102 次 lookup + 97 条新增 MISS 诊断日志）、
  `vllm_decode.log`（D 侧 1666 条新增 put chunks 全量 map 日志）
- 负载：ais_bench ShareGPT 50 会话 × 3 轮 = 150 请求，`ignore_eos=True`，
  每轮真实模型输出回填下一轮（infer_mode="every"），max_out_len=4096
- vllm-ascend：含 09-13 存储修复（hash 覆盖度封顶等）+ 09-14 三处诊断日志

## 1. lookup 结果分布（P 侧，102 次查询）

| 查询 | 次数 | hit=0 | hit=4096 | hit=8192 |
|---|---|---|---|---|
| turn-1（tl=4096） | 51 | **48** | 3 | — |
| turn-2（tl=8192） | 51 | — | 49 | **2** |

- turn-1：48/51（94%）完全未命中——上一轮 decode 保存的 KV **一个 chunk 都没复用上**
- turn-2：49/51 的 hit=4096 **全部来自本地 HBM**（turn-1 自己的 prefill KV），
  external 增量 = 0；仅 2 个请求 hit=8192（真正吃到 decode 保存的生成段 KV）

## 2. 分叉位置（g0/c4 组，128 token/chunk）

- turn-1 查询（对照 turn-0 保存的 KV）：分叉前匹配 chunk 中位 = **11 ≈ 1408 token**，
  即重编码后的 turn-0 输出在 ~1408 token 处与 decode 实际生成的 token 序列断裂
- turn-2 查询：store 在 4096-8192 区间可找到 4-32 个零散 chunk（重编码 turn-1 输出与
  decode 实际 token 的部分匹配），但 < 4096 → **4096 对齐后全部归零**
- g1/c128 组（整 4096-token 一个 chunk）：turn-1 查询 48/51 整块 miss

## 3. BPE 重编码影响的量化

| 指标 | turn-1 | turn-2 | 合计 |
|---|---|---|---|
| 实际命中 token | 12,288 | 217,088 | 229,376 |
| 潜在命中 token（无分叉） | 208,896 | 417,792 | 626,688 |
| **损失** | **94%** | **48%** | **63.4%** |

按 **external（decode→prefill 跨节点）口径**单独算：
- 潜在 external 命中 = 51×4096 + 51×4096 ≈ 417,792 token
- 实际 external 命中 ≈ 12,288 + 8,192 = 20,480 token（3 个 turn-1 + 2 个 turn-2 请求）
- **external 损失 ≈ 95%**——与用户观测的 external hit rate 3~5% 完全吻合
- 本地 26.7% 是 turn-2 命中 turn-1 prefill 的自复用（结构性上限，与 decode 无关）

## 4. 根因链（每轮都复现）

1. decode 生成的 token 序列（真实采样）被 detokenize 成文本回填到下一轮消息
2. 下一轮 prompt 经 chat 模板渲染后**整体 re-tokenize**
3. BPE 同段文本存在多种合法切分（实例：`[网络]` vs `[网][络]`、`[允许][多层]` vs
   `[允][许多][层]`），分叉点中位 ~1408 token（首个 4096 窗口内）
4. hash 链自分叉点断裂 → 该 4096 窗口内所有 chunk 无法命中 → 4096 对齐把部分匹配归零

## 5. 新增诊断日志的验证结论

- `KV pool lookup MISS`（P，97 条）：miss_at 分布 {0: 48, 4096: 49}、g0_chunk、
  g0_hash、分叉点 token 上下文——每 miss 请求一行，直接可读
- `KV pool put chunks`（D，1666 条）：全量 chunk→hash 映射 + 请求 token head；
  与 MISS 的 g0_hash 交叉比对可确认分叉 chunk 的 hash 不一致
- `hbm_preset` 字段：厘清 exists_chunks 中本地预置与 store 命中的构成
  （turn-2 的 exists=51/32 = hbm 32 + store 19）

## 6. 治本建议

1. **客户端 token-ids 回填**：ais_bench 已具备采集能力（return_token_ids），
   infer_every 回填改为把上一轮 token ids 作为 assistant 消息传递（需服务端支持
   token-ids 形式的 assistant 消息，或走 /v1/completions token-ids 路径）
2. 或推理服务端在 chat 模板渲染时对 assistant content 做**与前文一致的贪心重切分**
   （保证同一文本在同上下文中 tokenization 唯一）
3. benchmark 侧也可对比"回填文本 vs 回填 token ids"的端到端命中差异作为验收
