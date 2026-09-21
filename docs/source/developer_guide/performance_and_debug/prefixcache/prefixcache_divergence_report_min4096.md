# PD 分离多轮 prefix-cache 分叉离线重放分析报告

- 输入: `/workspace/benchmark/outputs/perf_dsv4_prefixcache_multiturn/20260915_011226/performances/dsv4-w4a8-prefixcache/sharegpt-prefixcache_details.jsonl`
- 会话数: 50  (缺 token ids 跳过的请求数: 0; 需 return_token_ids=True 采集)
- 对齐粒度: 4096 (c128 压缩层逻辑块, LCP 不足 4096 对齐归零)

## 1. 总览

| 指标 | 值 |
|---|---|
| 相邻轮次对 (转移) 数 | 100 |
| token 级完整匹配 | 96/100 (96%) |
| 达到 4096 对齐命中 | 97/100 (97%) |
| 潜在命中 token 合计 (对齐后) | 602112 |
| turn≥1 的 prompt token 合计 | 730131 |
| 潜在 external 命中率 (pot/tot_prompt) | 82.5% |

## 2. 分叉位置分布

| 转移 | 数量 | LCP 中位 | LCP 最小/最大 | 分叉在上一轮生成内的位置(中位) | 对齐命中≥4096 |
|---|---|---|---|---|---|
| turn0→turn1 | 50 | 4720 | 1211/5738 | 1032 | 47/50 |
| turn1→turn2 | 50 | 9256 | 8232/9847 | 4095 | 50/50 |

```
LCP [    0,  512):   0 
LCP [  512, 1024):   0 
LCP [ 1024, 2048):   2 #
LCP [ 2048, 4096):   1 
LCP [ 4096, 8192):  47 ############################
LCP [ 8192,  inf):  50 ##############################
```

## 3. 分叉归类

- 完整匹配: 96/100
- 上一轮生成内容内: 4/100

说明: '上一轮生成内容内' = BPE 重编码分叉 (decode 生成的 token id 与下一轮渲染 re-tokenize 结果不一致), 是 external 命中损失的直接原因; '上一轮 prompt 内' = 模板/渲染差异, 需另行排查。

## 4. 典型分叉样例 (Top 4, BPE 重编码分叉优先, 共 4 个 BPE 分叉 / 0 个 prompt 内分叉)

### 样例 1/4: 9bc5edce turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-2a86bbb3-2885-48b8-84ac-563a8ffb0adc
- LCP=1211 (上一轮 prompt 1071 + 生成 4096), 生成内容内偏移 140
- 对齐后命中: 0
- **分叉点前公共文本** (token 1171..1211, 两侧相同):
  > … "Coincidences" and the Flow of Ideas (The 36-Hour Mark)**⏎What you experienced regarding the ideas, the drawings, and the obsidian is a classic phenomenon known as **
- **decode 侧原始 token** @ 分叉点 1211 (起 10 个):
  - ids: `[27854, 64965, 601, 666, 343, 1805, 2883, 24514, 304, 34056]`
  - token 串: 'Ap' | 'ophen' | 'ia' | '**' | 'Ġ(' | 'the' | 'Ġhuman' | 'Ġtendency' | 'Ġto' | 'Ġperceive'
  - 边界文本: Ap‖ophen‖ia‖**‖ (‖the‖ human‖ tendency‖ to‖ perceive
  - 后续文本 (token 1211..1271):
  > Apophenia** (the human tendency to perceive meaningful connections between unrelated things) combined with **Priming**. ⏎⏎- **Priming:** Even if you don't consciously remember reading or seeing something, your brain absorbs an incredible am…
- **prefill 重编码 token** @ 分叉点 1211 (起 10 个):
  - ids: `[35, 20859, 690, 601, 666, 343, 1805, 2883, 24514, 304]`
  - token 串: 'A' | 'pop' | 'hen' | 'ia' | '**' | 'Ġ(' | 'the' | 'Ġhuman' | 'Ġtendency' | 'Ġto'
  - 边界文本: A‖pop‖hen‖ia‖**‖ (‖the‖ human‖ tendency‖ to
  - 后续文本 (token 1211..1271):
  > Apophenia** (the human tendency to perceive meaningful connections between unrelated things) combined with **Priming**. ⏎⏎- **Priming:** Even if you don't consciously remember reading or seeing something, your brain absorbs an incredible am…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 2/4: 5a065ef3 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-4886b0e0-ca23-49ff-830e-8e282681a3ec
- LCP=1421 (上一轮 prompt 389 + 生成 4096), 生成内容内偏移 1032
- 对齐后命中: 0
- **分叉点前公共文本** (token 1381..1421, 两侧相同):
  > …- 诺基亚从木材到手机再到衰落，是没能完成第二次进化。⏎- 所以，评估一家公司时，要问：“如果它的主营业务明天被技术清零，它的组织能力
- **decode 侧原始 token** @ 分叉点 1421 (起 10 个):
  - ids: `[659, 59025, 9789, 9013, 1043, 1360, 824, 3467, 4663, 666]`
  - token 串: 'èĥ½' | 'è®©å®ĥ' | 'å¿«éĢŁ' | 'æī¾åĪ°' | 'æĸ°' | 'æ´»' | 'æ³ķ' | 'åĲĹ' | 'ï¼ŁâĢĿĊĊ' | '**'
  - 边界文本: 能‖让它‖快速‖找到‖新‖活‖法‖吗‖？”⏎⏎‖**
  - 后续文本 (token 1421..1481):
  > 能让它快速找到新活法吗？”⏎⏎**3. 从“长期持有”到“动态持有”**⏎“长期持有”本身没有错，但前提是**“长期”必须建立在“业务逻辑持续有效”的基础上**。芒格的意思是：当你发现一家公司的“生态
- **prefill 重编码 token** @ 分叉点 1421 (起 10 个):
  - ids: `[53422, 1877, 9789, 9013, 1043, 1360, 824, 3467, 4663, 666]`
  - token 串: 'èĥ½è®©' | 'å®ĥ' | 'å¿«éĢŁ' | 'æī¾åĪ°' | 'æĸ°' | 'æ´»' | 'æ³ķ' | 'åĲĹ' | 'ï¼ŁâĢĿĊĊ' | '**'
  - 边界文本: 能让‖它‖快速‖找到‖新‖活‖法‖吗‖？”⏎⏎‖**
  - 后续文本 (token 1421..1481):
  > 能让它快速找到新活法吗？”⏎⏎**3. 从“长期持有”到“动态持有”**⏎“长期持有”本身没有错，但前提是**“长期”必须建立在“业务逻辑持续有效”的基础上**。芒格的意思是：当你发现一家公司的“生态
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 3/4: a276efad turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-3302f355-c585-4406-961f-85ccd38181a7
- LCP=3074 (上一轮 prompt 1090 + 生成 4096), 生成内容内偏移 1984
- 对齐后命中: 0
- **分叉点前公共文本** (token 3034..3074, 两侧相同):
  > …公正、透明地进行，以还原事实真相，维护社会正义。同时，我们也提醒公众，在信息爆炸的时代，保持独立思考和批判性思维尤为重要，不要轻易被任何单一来源或
- **decode 侧原始 token** @ 分叉点 3074 (起 10 个):
  - ids: `[34930, 898, 7671, 320, 8563, 303, 4602, 2490, 24870, 15087]`
  - token 串: 'ç«ĭåľº' | 'æīĢ' | 'å·¦åı³' | 'ãĢĤ' | 'æľĢç»Ī' | 'ï¼Į' | 'åıªæľī' | 'éĢļè¿ĩ' | 'çĲĨæĢ§' | 'å¯¹è¯Ŀ'
  - 边界文本: 立场‖所‖左右‖。‖最终‖，‖只有‖通过‖理性‖对话
  - 后续文本 (token 3074..3134):
  > 立场所左右。最终，只有通过理性对话和依法办事，我们才能共同构建一个更加公正、和谐的社会。对于任何争议，我们都应坚持“以事实为依据，以法律为准绳”的原则，避免情绪化、标签化和政治化的解读。同时，也应尊重每个人的合法权益
- **prefill 重编码 token** @ 分叉点 3074 (起 10 个):
  - ids: `[1661, 18790, 7671, 320, 8563, 303, 4602, 2490, 24870, 15087]`
  - token 串: 'ç«ĭ' | 'åľºæīĢ' | 'å·¦åı³' | 'ãĢĤ' | 'æľĢç»Ī' | 'ï¼Į' | 'åıªæľī' | 'éĢļè¿ĩ' | 'çĲĨæĢ§' | 'å¯¹è¯Ŀ'
  - 边界文本: 立‖场所‖左右‖。‖最终‖，‖只有‖通过‖理性‖对话
  - 后续文本 (token 3074..3134):
  > 立场所左右。最终，只有通过理性对话和依法办事，我们才能共同构建一个更加公正、和谐的社会。对于任何争议，我们都应坚持“以事实为依据，以法律为准绳”的原则，避免情绪化、标签化和政治化的解读。同时，也应尊重每个人的合法权益
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 4/4: 3904b99c turn1→turn2 [上一轮生成内容内] response_id=chatcmpl-5083f095-24a5-44c5-aad6-57f5e7838fda
- LCP=8232 (上一轮 prompt 4137 + 生成 4096), 生成内容内偏移 4095
- 对齐后命中: 8192
- **分叉点前公共文本** (token 8192..8232, 两侧相同):
  > …---⏎⏎**Really.** ⏎⏎---⏎⏎**I promise.** ⏎⏎---⏎⏎**Okay, bye for real this time.** ⏎⏎---⏎⏎**But I'm still here.** ⏎⏎---⏎⏎**Just in case.** ⏎⏎---⏎⏎**
- **decode 侧原始 token** @ 分叉点 8232 (起 10 个):
  - ids: `[28927]`
  - token 串: 'ðŁĺ'
  - 边界文本: 
  - 后续文本 (token 8232..8233):
  > �
- **prefill 重编码 token** @ 分叉点 8232 (起 10 个):
  - ids: `[1, 128803, 633, 14, 342, 4571, 2887, 304, 25867, 270]`
  - token 串: '<｜end▁of▁sentence｜>' | '<｜User｜>' | 'ok' | ',' | 'ĠI' | "'m" | 'Ġgoing' | 'Ġto' | 'Ġpaste' | 'Ġthe'
  - 边界文本: <｜end▁of▁sentence｜>‖<｜User｜>‖ok‖,‖ I‖'m‖ going‖ to‖ paste‖ the
  - 后续文本 (token 8232..8292):
  > <｜end▁of▁sentence｜><｜User｜>ok, I'm going to paste the text of an article from today's Guardian newspaper. Please rewrite in the style of the Canterbury tales and make it about 500-800 words:⏎⏎Boris Johnson faces a battle for his future in p…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

## 5. 命中损失估算

- 若全部 round-trip 精确: 潜在命中 602112 token (82.5% of turn≥1 prompt)
- 实际达成 4096 命中的转移: 97/100
- 被分叉吞噬的转移: 3/100 (3% 的转移 external 命中为 0)

> 治本方案: 客户端按 token_ids 回填 (ais_bench infer_every 当前用 output.content 文本回填, 经 detokenize→re-tokenize 不可逆); 或服务端接受 token-ids 形式的 assistant 消息。
