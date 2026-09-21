# PD 分离多轮 prefix-cache 分叉离线重放分析报告

- 输入: `/workspace/benchmark/outputs/perf_dsv4_prefixcache_multiturn/20260915_000401/performances/dsv4-w4a8-prefixcache/sharegpt-prefixcache_details.jsonl`
- 会话数: 50  (缺 token ids 跳过的请求数: 0; 需 return_token_ids=True 采集)
- 对齐粒度: 4096 (c128 压缩层逻辑块, LCP 不足 4096 对齐归零)

## 1. 总览

| 指标 | 值 |
|---|---|
| 相邻轮次对 (转移) 数 | 100 |
| token 级完整匹配 | 99/100 (99%) |
| 达到 4096 对齐命中 | 7/100 (7%) |
| 潜在命中 token 合计 (对齐后) | 32768 |
| turn≥1 的 prompt token 合计 | 248335 |
| 潜在 external 命中率 (pot/tot_prompt) | 13.2% |

## 2. 分叉位置分布

| 转移 | 数量 | LCP 中位 | LCP 最小/最大 | 分叉在上一轮生成内的位置(中位) | 对齐命中≥4096 |
|---|---|---|---|---|---|
| turn0→turn1 | 50 | 1579 | 132/5365 | - | 1/50 |
| turn1→turn2 | 50 | 2745 | 594/9468 | 1032 | 6/50 |

```
LCP [    0,  512):   6 ###
LCP [  512, 1024):  14 ########
LCP [ 1024, 2048):  33 ###################
LCP [ 2048, 4096):  40 ########################
LCP [ 4096, 8192):   6 ###
LCP [ 8192,  inf):   1 
```

## 3. 分叉归类

- 完整匹配: 99/100
- 上一轮生成内容内: 1/100

说明: '上一轮生成内容内' = BPE 重编码分叉 (decode 生成的 token id 与下一轮渲染 re-tokenize 结果不一致), 是 external 命中损失的直接原因; '上一轮 prompt 内' = 模板/渲染差异, 需另行排查。

## 4. 典型分叉样例 (Top 1, BPE 重编码分叉优先, 共 1 个 BPE 分叉 / 0 个 prompt 内分叉)

### 样例 1/1: 66354222 turn1→turn2 [上一轮生成内容内] response_id=chatcmpl-a5beb4d9-5d93-49ce-900f-34e227c0716c
- LCP=1497 (上一轮 prompt 465 + 生成 1351), 生成内容内偏移 1032
- 对齐后命中: 0
- **分叉点前公共文本** (token 1457..1497, 两侧相同):
  > …,*  ⏎*His steede was a Tesla, swift and hye.*  ⏎*He spak of "synergies" and "scalable growth,"*  ⏎*And of "disruption
- **decode 侧原始 token** @ 分叉点 1497 (起 10 个):
  - ids: `[4, 965, 9336, 515, 793, 277, 1805, 14, 383, 89]`
  - token 串: '"' | 'âĢĶ' | 'that' | 'Ġwas' | 'Ġhis' | 'Ġo' | 'the' | ',' | 'Ġy' | 'w'
  - 边界文本: "‖—‖that‖ was‖ his‖ o‖the‖,‖ y‖w
  - 后续文本 (token 1497..1557):
  > "—that was his othe, ywys.*  ⏎*He lovede noght the olde, but the newe,*  ⏎*And helde a "vision quest" for his crewe.*⏎⏎---⏎⏎**Then I'd continue with the rest of the "pilgrims":**⏎⏎
- **prefill 重编码 token** @ 分叉点 1497 (起 10 个):
  - ids: `[109023, 9336, 515, 793, 277, 1805, 14, 383, 89, 922]`
  - token 串: '"âĢĶ' | 'that' | 'Ġwas' | 'Ġhis' | 'Ġo' | 'the' | ',' | 'Ġy' | 'w' | 'ys'
  - 边界文本: "—‖that‖ was‖ his‖ o‖the‖,‖ y‖w‖ys
  - 后续文本 (token 1497..1557):
  > "—that was his othe, ywys.*  ⏎*He lovede noght the olde, but the newe,*  ⏎*And helde a "vision quest" for his crewe.*⏎⏎---⏎⏎**Then I'd continue with the rest of the "pilgrims":**⏎⏎-
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

## 5. 命中损失估算

- 若全部 round-trip 精确: 潜在命中 32768 token (13.2% of turn≥1 prompt)
- 实际达成 4096 命中的转移: 7/100
- 被分叉吞噬的转移: 93/100 (93% 的转移 external 命中为 0)

> 治本方案: 客户端按 token_ids 回填 (ais_bench infer_every 当前用 output.content 文本回填, 经 detokenize→re-tokenize 不可逆); 或服务端接受 token-ids 形式的 assistant 消息。
