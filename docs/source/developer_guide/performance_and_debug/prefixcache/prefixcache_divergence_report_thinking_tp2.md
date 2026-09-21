# PD 分离多轮 prefix-cache 分叉离线重放分析报告

- 输入: `/workspace/benchmark/outputs/perf_dsv4_prefixcache_multiturn/20260915_100015/performances/dsv4-w4a8-prefixcache/sharegpt-prefixcache_details.jsonl`
- 会话数: 50  (缺 token ids 跳过的请求数: 0; 需 return_token_ids=True 采集)
- 对齐粒度: 4096 (c128 压缩层逻辑块, LCP 不足 4096 对齐归零)

## 1. 总览

| 指标 | 值 |
|---|---|
| 相邻轮次对 (转移) 数 | 100 |
| token 级完整匹配 | 99/100 (99%) |
| 达到 4096 对齐命中 | 99/100 (99%) |
| 潜在命中 token 合计 (对齐后) | 610304 |
| turn≥1 的 prompt token 合计 | 730136 |
| 潜在 external 命中率 (pot/tot_prompt) | 83.6% |

## 2. 分叉位置分布

| 转移 | 数量 | LCP 中位 | LCP 最小/最大 | 分叉在上一轮生成内的位置(中位) | 对齐命中≥4096 |
|---|---|---|---|---|---|
| turn0→turn1 | 50 | 4796 | 1747/5738 | 742 | 49/50 |
| turn1→turn2 | 50 | 9256 | 8233/9847 | - | 50/50 |

```
LCP [    0,  512):   0 
LCP [  512, 1024):   0 
LCP [ 1024, 2048):   1 
LCP [ 2048, 4096):   0 
LCP [ 4096, 8192):  49 #############################
LCP [ 8192,  inf):  50 ##############################
```

## 3. 分叉归类

- 完整匹配: 99/100
- 上一轮生成内容内: 1/100

说明: '上一轮生成内容内' = BPE 重编码分叉 (decode 生成的 token id 与下一轮渲染 re-tokenize 结果不一致), 是 external 命中损失的直接原因; '上一轮 prompt 内' = 模板/渲染差异, 需另行排查。

## 4. 典型分叉样例 (Top 1, BPE 重编码分叉优先, 共 1 个 BPE 分叉 / 0 个 prompt 内分叉)

### 样例 1/1: 10b6e07d turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-6b9a0819-b583-4d38-857a-531e85866380
- LCP=1747 (上一轮 prompt 1005 + 生成 4096), 生成内容内偏移 742
- 对齐后命中: 0
- **分叉点前公共文本** (token 1707..1747, 两侧相同):
  > …acht. Anna hätte nicht gewollt, dass du dein Leben aufgibst."⏎⏎C sah, wie hens Schultern sich spannten. "Woher willst du das wissen?"⏎⏎
- **decode 侧原始 token** @ 分叉点 1747 (起 10 个):
  - ids: `[4, 2581, 321, 16363, 10561, 47038, 73988, 45425, 16, 19330]`
  - token 串: '"' | 'We' | 'il' | 'Ġich' | 'Ġsie' | 'Ġgek' | 'annt' | 'Ġhabe' | '.' | 'ĠUnd'
  - 边界文本: "‖We‖il‖ ich‖ sie‖ gek‖annt‖ habe‖.‖ Und
  - 后续文本 (token 1747..1807):
  > "Weil ich sie gekannt habe. Und weil ich dich kenne." Marta legte ihre runzlige Hand auf hens Arm. "Du musst nicht gehen, um zu trauern. Und du musst nicht bleiben, um zu vergessen. Aber du solltest nicht aus Angst
- **prefill 重编码 token** @ 分叉点 1747 (起 10 个):
  - ids: `[26520, 321, 16363, 10561, 47038, 73988, 45425, 16, 19330, 62085]`
  - token 串: '"We' | 'il' | 'Ġich' | 'Ġsie' | 'Ġgek' | 'annt' | 'Ġhabe' | '.' | 'ĠUnd' | 'Ġweil'
  - 边界文本: "We‖il‖ ich‖ sie‖ gek‖annt‖ habe‖.‖ Und‖ weil
  - 后续文本 (token 1747..1807):
  > "Weil ich sie gekannt habe. Und weil ich dich kenne." Marta legte ihre runzlige Hand auf hens Arm. "Du musst nicht gehen, um zu trauern. Und du musst nicht bleiben, um zu vergessen. Aber du solltest nicht aus Angst vor
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

## 5. 命中损失估算

- 若全部 round-trip 精确: 潜在命中 610304 token (83.6% of turn≥1 prompt)
- 实际达成 4096 命中的转移: 99/100
- 被分叉吞噬的转移: 1/100 (1% 的转移 external 命中为 0)

> 治本方案: 客户端按 token_ids 回填 (ais_bench infer_every 当前用 output.content 文本回填, 经 detokenize→re-tokenize 不可逆); 或服务端接受 token-ids 形式的 assistant 消息。
