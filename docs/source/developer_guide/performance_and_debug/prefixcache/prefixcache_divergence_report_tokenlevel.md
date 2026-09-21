# PD 分离多轮 prefix-cache 分叉离线重放分析报告

- 输入: `/workspace/benchmark/outputs/perf_dsv4_prefixcache_multiturn/20260914_214705/performances/dsv4-w4a8-prefixcache/sharegpt-prefixcache_details.jsonl`
- 会话数: 50  (缺 token ids 跳过的请求数: 0; 需 return_token_ids=True 采集)
- 对齐粒度: 4096 (c128 压缩层逻辑块, LCP 不足 4096 对齐归零)

## 1. 总览

| 指标 | 值 |
|---|---|
| 相邻轮次对 (转移) 数 | 100 |
| token 级完整匹配 | 2/100 (2%) |
| 达到 4096 对齐命中 | 52/100 (52%) |
| 潜在命中 token 合计 (对齐后) | 225280 |
| turn≥1 的 prompt token 合计 | 727432 |
| 潜在 external 命中率 (pot/tot_prompt) | 31.0% |

## 2. 分叉位置分布

| 转移 | 数量 | LCP 中位 | LCP 最小/最大 | 分叉在上一轮生成内的位置(中位) | 对齐命中≥4096 |
|---|---|---|---|---|---|
| turn0→turn1 | 50 | 1602 | 130/5365 | 608 | 2/50 |
| turn1→turn2 | 50 | 5873 | 4686/9344 | 601 | 50/50 |

```
LCP [    0,  512):   6 ###
LCP [  512, 1024):  10 ######
LCP [ 1024, 2048):  21 ############
LCP [ 2048, 4096):  11 ######
LCP [ 4096, 8192):  49 #############################
LCP [ 8192,  inf):   3 #
```

## 3. 分叉归类

- 上一轮生成内容内: 98/100
- 完整匹配: 2/100

说明: '上一轮生成内容内' = BPE 重编码分叉 (decode 生成的 token id 与下一轮渲染 re-tokenize 结果不一致), 是 external 命中损失的直接原因; '上一轮 prompt 内' = 模板/渲染差异, 需另行排查。

## 4. 典型分叉样例 (Top 30, BPE 重编码分叉优先, 共 98 个 BPE 分叉 / 0 个 prompt 内分叉)

### 样例 1/30: f6f13386 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-0df16bd8-2897-4a65-afe0-962f9de7bedd
- LCP=130 (上一轮 prompt 126 + 生成 4096), 生成内容内偏移 4
- 对齐后命中: 0
- **分叉点前公共文本** (token 90..130, 两侧相同):
  > …Yes, I understand'. If you don't understand, say 'No, I don't understand'. Don't ask me for clarifications.⏎⏎Please wait for my instructions.<｜Assistant｜></think>Yes, I understand
- **decode 侧原始 token** @ 分叉点 130 (起 10 个):
  - ids: `[16, 1, 16, 1, 26, 68, 26, 68, 26, 68]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '8' | 'b' | '8' | 'b' | '8' | 'b'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖8‖b‖8‖b‖8‖b
  - 后续文本 (token 130..190):
  > .<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b
- **prefill 重编码 token** @ 分叉点 130 (起 10 个):
  - ids: `[426, 26, 68, 26, 68, 26, 68, 26, 68, 26]`
  - token 串: '..' | '8' | 'b' | '8' | 'b' | '8' | 'b' | '8' | 'b' | '8'
  - 边界文本: ..‖8‖b‖8‖b‖8‖b‖8‖b‖8
  - 后续文本 (token 130..190):
  > ..8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 2/30: 32beef8d turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-e8ed87f3-ce8e-4acf-8be7-f4f1bef2d0be
- LCP=223 (上一轮 prompt 176 + 生成 4096), 生成内容内偏移 47
- 对齐后命中: 0
- **分叉点前公共文本** (token 183..223, 两侧相同):
  > … the text. I will process it at a professor-level depth, prioritizing your input, and will be prepared to answer questions with expert jargon, critical analysis, and clear explanations. Please paste the first text
- **decode 侧原始 token** @ 分叉点 223 (起 10 个):
  - ids: `[16, 1, 16, 1, 16, 1, 16, 1, 16, 1]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>
  - 后续文本 (token 223..283):
  > .<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>. 8c1f1f4f9f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f
- **prefill 重编码 token** @ 分叉点 223 (起 10 个):
  - ids: `[42713, 223, 26, 69, 19, 72, 19, 72, 22, 72]`
  - token 串: '.......' | 'Ġ' | '8' | 'c' | '1' | 'f' | '1' | 'f' | '4' | 'f'
  - 边界文本: .......‖ ‖8‖c‖1‖f‖1‖f‖4‖f
  - 后续文本 (token 223..283):
  > ....... 8c1f1f4f9f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 3/30: 595560d4 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-5b1e1025-2d9f-42f7-a026-73cf308cdb53
- LCP=240 (上一轮 prompt 73 + 生成 4096), 生成内容内偏移 167
- 对齐后命中: 0
- **分叉点前公共文本** (token 200..240, 两侧相同):
  > … the limit, but for most videos it will fit fine.⏎⏎Go ahead and paste it—I’ll read it fully and we can discuss the math, history, and ideas in James Simons’ talk
- **decode 侧原始 token** @ 分叉点 240 (起 10 个):
  - ids: `[16, 1, 0, 5, 19414, 271, 3476, 588, 25867, 270]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠReply' | 'ĊĊ' | 'You' | 'Ġcan' | 'Ġpaste' | 'Ġthe'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Reply‖⏎⏎‖You‖ can‖ paste‖ the
  - 后续文本 (token 240..300):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Reply⏎⏎You can paste the entire transcript in one message. My context window is large enough to handle a full transcript of a typical YouTube talk (even 1–2 hours long). If it’s unusually long (e.g…
- **prefill 重编码 token** @ 分叉点 240 (起 10 个):
  - ids: `[5110, 19414, 271, 3476, 588, 25867, 270, 5221, 21218, 295]`
  - token 串: '.#' | 'ĠReply' | 'ĊĊ' | 'You' | 'Ġcan' | 'Ġpaste' | 'Ġthe' | 'Ġentire' | 'Ġtranscript' | 'Ġin'
  - 边界文本: .#‖ Reply‖⏎⏎‖You‖ can‖ paste‖ the‖ entire‖ transcript‖ in
  - 后续文本 (token 240..300):
  > .# Reply⏎⏎You can paste the entire transcript in one message. My context window is large enough to handle a full transcript of a typical YouTube talk (even 1–2 hours long). If it’s unusually long (e.g., 3+ hours of dense technical content),…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 4/30: 26333ee4 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-0cfd2568-c36f-4f4d-9da7-1a459f736ccf
- LCP=432 (上一轮 prompt 314 + 生成 4096), 生成内容内偏移 118
- 对齐后命中: 0
- **分叉点前公共文本** (token 392..432, 两侧相同):
  > … out to someone in our CSR or sustainability team, or a dedicated environmental officer.⏎⏎Wishing you all the best with Earth Day 2023—hope it's a massive success.⏎⏎Cheers,⏎Philipp"
- **decode 侧原始 token** @ 分叉点 432 (起 10 个):
  - ids: `[1, 0, 5, 19414, 304, 17282, 271, 4, 23166, 17282]`
  - token 串: '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠReply' | 'Ġto' | 'ĠPhilipp' | 'ĊĊ' | '"' | 'Hi' | 'ĠPhilipp'
  - 边界文本: <｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Reply‖ to‖ Philipp‖⏎⏎‖"‖Hi‖ Philipp
  - 后续文本 (token 432..492):
  > <｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Reply to Philipp⏎⏎"Hi Philipp,⏎⏎Thanks for the quick reply and for the kind words—really appreciate it.⏎⏎Totally understand, and thanks for being upfront. I'll try to route this to the right folks o…
- **prefill 重编码 token** @ 分叉点 432 (起 10 个):
  - ids: `[5, 19414, 304, 17282, 271, 4, 23166, 17282, 2893, 17003]`
  - token 串: '#' | 'ĠReply' | 'Ġto' | 'ĠPhilipp' | 'ĊĊ' | '"' | 'Hi' | 'ĠPhilipp' | ',ĊĊ' | 'Thanks'
  - 边界文本: #‖ Reply‖ to‖ Philipp‖⏎⏎‖"‖Hi‖ Philipp‖,⏎⏎‖Thanks
  - 后续文本 (token 432..492):
  > # Reply to Philipp⏎⏎"Hi Philipp,⏎⏎Thanks for the quick reply and for the kind words—really appreciate it.⏎⏎Totally understand, and thanks for being upfront. I'll try to route this to the right folks on your end if possible. If you happen to…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 5/30: a08c9668 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-446a8230-3268-46c1-8e30-3f1a586c00b6
- LCP=432 (上一轮 prompt 17 + 生成 4096), 生成内容内偏移 415
- 对齐后命中: 0
- **分叉点前公共文本** (token 392..432, 两侧相同):
  > … current draft!** ⏎⏎*(If you don't have the prompt handy, just tell me the subject/course name and what you think the project is about, and we'll work from there.)*
- **decode 侧原始 token** @ 分叉点 432 (起 10 个):
  - ids: `[1, 223, 26, 30, 5013, 342, 1030, 7692, 1082, 440]`
  - token 串: '<｜end▁of▁sentence｜>' | 'Ġ' | '8' | '<' | '---' | 'ĠI' | 'Ġam' | 'Ġready' | 'Ġwhen' | 'Ġyou'
  - 边界文本: <｜end▁of▁sentence｜>‖ ‖8‖<‖---‖ I‖ am‖ ready‖ when‖ you
  - 后续文本 (token 432..492):
  > <｜end▁of▁sentence｜> 8<--- I am ready when you are.<｜end▁of▁sentence｜> 8<---<｜end▁of▁sentence｜> 8<--- Let's get this done.<｜end▁of▁sentence｜> 8<---<｜end▁of▁sentence｜> 8<---<｜end▁of▁sentence｜> 8<--- (Please share the prompt!)<｜end▁of▁sentence…
- **prefill 重编码 token** @ 分叉点 432 (起 10 个):
  - ids: `[223, 26, 30, 5013, 342, 1030, 7692, 1082, 440, 477]`
  - token 串: 'Ġ' | '8' | '<' | '---' | 'ĠI' | 'Ġam' | 'Ġready' | 'Ġwhen' | 'Ġyou' | 'Ġare'
  - 边界文本:  ‖8‖<‖---‖ I‖ am‖ ready‖ when‖ you‖ are
  - 后续文本 (token 432..492):
  >  8<--- I am ready when you are. 8<--- 8<--- Let's get this done. 8<--- 8<--- 8<--- (Please share the prompt!) 8<--- (If you are stuck, just say "I need help starting
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 6/30: d5c6c38a turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-2ee3d9b6-d159-42e7-a0c2-14bb04b91030
- LCP=507 (上一轮 prompt 11 + 生成 4096), 生成内容内偏移 496
- 对齐后命中: 0
- **分叉点前公共文本** (token 467..507, 两侧相同):
  > … a specific passage in Middle English?⏎- Or just a general chat about why this work is considered the cornerstone of English literature?⏎⏎I'm all ears (or, well, all data). Just ask!
- **decode 侧原始 token** @ 分叉点 507 (起 10 个):
  - ids: `[1, 26, 74, 1, 26, 74, 26, 74, 26, 74]`
  - token 串: '<｜end▁of▁sentence｜>' | '8' | 'h' | '<｜end▁of▁sentence｜>' | '8' | 'h' | '8' | 'h' | '8' | 'h'
  - 边界文本: <｜end▁of▁sentence｜>‖8‖h‖<｜end▁of▁sentence｜>‖8‖h‖8‖h‖8‖h
  - 后续文本 (token 507..567):
  > <｜end▁of▁sentence｜>8h<｜end▁of▁sentence｜>8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h
- **prefill 重编码 token** @ 分叉点 507 (起 10 个):
  - ids: `[26, 74, 26, 74, 26, 74, 26, 74, 26, 74]`
  - token 串: '8' | 'h' | '8' | 'h' | '8' | 'h' | '8' | 'h' | '8' | 'h'
  - 边界文本: 8‖h‖8‖h‖8‖h‖8‖h‖8‖h
  - 后续文本 (token 507..567):
  > 8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h8h
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 7/30: ca2d5756 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-b9d5c03d-5f9e-40db-a549-a7385cf9b49b
- LCP=512 (上一轮 prompt 511 + 生成 4096), 生成内容内偏移 1
- 对齐后命中: 0
- **分叉点前公共文本** (token 472..512, 两侧相同):
  > …PCs. Propagate all VPC attachments into the route table. Define a static default route in the route table.⏎⏎Choose the single best answer only. Do not provide any explanation.<｜Assistant｜></think>D
- **decode 侧原始 token** @ 分叉点 512 (起 10 个):
  - ids: `[1, 16, 1, 26, 72, 27, 68, 20, 69, 25]`
  - token 串: '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '8' | 'f' | '9' | 'b' | '2' | 'c' | '7'
  - 边界文本: <｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖8‖f‖9‖b‖2‖c‖7
  - 后续文本 (token 512..572):
  > <｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>8f9b2c7d3e4f5a6b7c8d9e0f1a2b3c4d<｜end▁of▁sentence｜>8f9b2c7d3e4f5a6b7c8d9e0f
- **prefill 重编码 token** @ 分叉点 512 (起 10 个):
  - ids: `[16, 26, 72, 27, 68, 20, 69, 25, 70, 21]`
  - token 串: '.' | '8' | 'f' | '9' | 'b' | '2' | 'c' | '7' | 'd' | '3'
  - 边界文本: .‖8‖f‖9‖b‖2‖c‖7‖d‖3
  - 后续文本 (token 512..572):
  > .8f9b2c7d3e4f5a6b7c8d9e0f1a2b3c4d8f9b2c7d3e4f5a6b7c8d9e0f1a2
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 8/30: f6c4dbb6 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-6f6bfdbb-5f42-4785-8f8d-9fb528155fc2
- LCP=554 (上一轮 prompt 551 + 生成 4096), 生成内容内偏移 3
- 对齐后命中: 0
- **分叉点前公共文本** (token 514..554, 两侧相同):
  > … her independent, professional judgment⏎in relation to such a conclave or conference and joint report, and must not act⏎on any instruction or request to withhold or avoid agreement.<｜Assistant｜></think>acknowledged
- **decode 侧原始 token** @ 分叉点 554 (起 10 个):
  - ids: `[1, 16, 1, 26, 72, 26, 70, 32083, 68, 25]`
  - token 串: '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '8' | 'f' | '8' | 'd' | '892' | 'b' | '7'
  - 边界文本: <｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖8‖f‖8‖d‖892‖b‖7
  - 后续文本 (token 554..614):
  > <｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>8f8d892b7f4f4b0f9f4f4b0f9f4f4b0f9f4f4b0f9f4f4b0f9f4f4b0f9f4
- **prefill 重编码 token** @ 分叉点 554 (起 10 个):
  - ids: `[16, 26, 72, 26, 70, 32083, 68, 25, 72, 22]`
  - token 串: '.' | '8' | 'f' | '8' | 'd' | '892' | 'b' | '7' | 'f' | '4'
  - 边界文本: .‖8‖f‖8‖d‖892‖b‖7‖f‖4
  - 后续文本 (token 554..614):
  > .8f8d892b7f4f4b0f9f4f4b0f9f4f4b0f9f4f4b0f9f4f4b0f9f4f4b0f9f4f4
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 9/30: 31e48322 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-dadc6b54-a19f-4342-b51e-cb4b6e326499
- LCP=610 (上一轮 prompt 607 + 生成 4096), 生成内容内偏移 3
- 对齐后命中: 0
- **分叉点前公共文本** (token 570..610, 两侧相同):
  > … then went on to say that the replacement of the foundations would present quite a problem from a practical point of view, that it could be done only in a piecemeal fashion by<｜Assistant｜></think>acknowledged
- **decode 侧原始 token** @ 分叉点 610 (起 10 个):
  - ids: `[1, 1, 223, 27, 67, 26, 68, 19, 70, 19]`
  - token 串: '<｜end▁of▁sentence｜>' | '<｜end▁of▁sentence｜>' | 'Ġ' | '9' | 'a' | '8' | 'b' | '1' | 'd' | '1'
  - 边界文本: <｜end▁of▁sentence｜>‖<｜end▁of▁sentence｜>‖ ‖9‖a‖8‖b‖1‖d‖1
  - 后续文本 (token 610..670):
  > <｜end▁of▁sentence｜><｜end▁of▁sentence｜> 9a8b1d1e2f3c4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
- **prefill 重编码 token** @ 分叉点 610 (起 10 个):
  - ids: `[223, 27, 67, 26, 68, 19, 70, 19, 71, 20]`
  - token 串: 'Ġ' | '9' | 'a' | '8' | 'b' | '1' | 'd' | '1' | 'e' | '2'
  - 边界文本:  ‖9‖a‖8‖b‖1‖d‖1‖e‖2
  - 后续文本 (token 610..670):
  >  9a8b1d1e2f3c4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 10/30: 2b074a5c turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-41b3b0db-821c-42ac-9ee2-e51f1c1697ff
- LCP=625 (上一轮 prompt 624 + 生成 4096), 生成内容内偏移 1
- 对齐后命中: 0
- **分叉点前公共文本** (token 585..625, 两侧相同):
  > … Images | - | - | 4 | 4 | 4 | 2 (1 when aspect ratio≠1:1) | |⏎more info. Just answer with READ:<｜Assistant｜></think>READ
- **decode 侧原始 token** @ 分叉点 625 (起 10 个):
  - ids: `[1, 16, 1, 3967, 671, 3051, 440, 4496, 344, 260]`
  - token 串: '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | 'Ġuser' | 'The' | 'Ġtext' | 'Ġyou' | 'Ġprovided' | 'Ġis' | 'Ġa'
  - 边界文本: <｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖ user‖The‖ text‖ you‖ provided‖ is‖ a
  - 后续文本 (token 625..685):
  > <｜end▁of▁sentence｜>.<｜end▁of▁sentence｜> userThe text you provided is a table of default values and compatibility for Midjourney model versions 4 and 5. It lists parameters like Aspect Ratio, Chaos, Quality, Seed, Stop, Style, and Stylize, a…
- **prefill 重编码 token** @ 分叉点 625 (起 10 个):
  - ids: `[16, 3967, 671, 3051, 440, 4496, 344, 260, 4184, 294]`
  - token 串: '.' | 'Ġuser' | 'The' | 'Ġtext' | 'Ġyou' | 'Ġprovided' | 'Ġis' | 'Ġa' | 'Ġtable' | 'Ġof'
  - 边界文本: .‖ user‖The‖ text‖ you‖ provided‖ is‖ a‖ table‖ of
  - 后续文本 (token 625..685):
  > . userThe text you provided is a table of default values and compatibility for Midjourney model versions 4 and 5. It lists parameters like Aspect Ratio, Chaos, Quality, Seed, Stop, Style, and Stylize, along with their default values and ran…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 11/30: b812c6a6 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-9013f8d5-9f42-4c65-ba5b-ac134fe196ce
- LCP=639 (上一轮 prompt 366 + 生成 4096), 生成内容内偏移 273
- 对齐后命中: 0
- **分叉点前公共文本** (token 599..639, 两侧相同):
  > … the mouse's left button is down, put the mouse's spot into a spot; repeat.`⏎⏎This is a clean, unambiguous design that avoids the complexity and readability issues that nested conditionals often introduce
- **decode 侧原始 token** @ 分叉点 639 (起 10 个):
  - ids: `[16, 1, 0, 5, 2923, 80117, 271, 3476, 477, 260]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠSystem' | 'ĠPrompt' | 'ĊĊ' | 'You' | 'Ġare' | 'Ġa'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ System‖ Prompt‖⏎⏎‖You‖ are‖ a
  - 后续文本 (token 639..699):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># System Prompt⏎⏎You are a helpful assistant.<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜> 8d5f7b2c9a1e4f6b8d3c0a9e7f2b4d6c8a1e3f5b7d9
- **prefill 重编码 token** @ 分叉点 639 (起 10 个):
  - ids: `[5110, 2923, 80117, 271, 3476, 477, 260, 11502, 22896, 426]`
  - token 串: '.#' | 'ĠSystem' | 'ĠPrompt' | 'ĊĊ' | 'You' | 'Ġare' | 'Ġa' | 'Ġhelpful' | 'Ġassistant' | '..'
  - 边界文本: .#‖ System‖ Prompt‖⏎⏎‖You‖ are‖ a‖ helpful‖ assistant‖..
  - 后续文本 (token 639..699):
  > .# System Prompt⏎⏎You are a helpful assistant.. 8d5f7b2c9a1e4f6b8d3c0a9e7f2b4d6c8a1e3f5b7d9c0a2e4
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 12/30: cd1fccb9 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-19f094c6-66fc-490e-88ef-12246f0ed404
- LCP=709 (上一轮 prompt 389 + 生成 4096), 生成内容内偏移 320
- 对齐后命中: 0
- **分叉点前公共文本** (token 669..709, 两侧相同):
  > …，你在银行、电梯等多楼层场所都会用到它们；它们看起来是永恒的。但它们基本上都在消亡或已经消亡。所以，一旦我更深刻地理解了这一点，我认为这让我成为了一个更好的投资者
- **decode 侧原始 token** @ 分叉点 709 (起 10 个):
  - ids: `[320, 1, 0, 5, 223, 28638, 17395, 271, 1536, 223]`
  - token 串: 'ãĢĤ' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'Ġ' | 'åİŁæĸĩ' | 'å¯¹æ¯Ķ' | 'ĊĊ' | '####' | 'Ġ'
  - 边界文本: 。‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ ‖原文‖对比‖⏎⏎‖####‖ 
  - 后续文本 (token 709..769):
  > 。<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># 原文对比⏎⏎#### 中文版⏎⏎One of the things that surprised me the most was the number of people who died. The business world is very much like the physical world, where all the animals die in the course of i…
- **prefill 重编码 token** @ 分叉点 709 (起 10 个):
  - ids: `[14704, 223, 28638, 17395, 271, 1536, 223, 21134, 3411, 271]`
  - token 串: 'ãĢĤ#' | 'Ġ' | 'åİŁæĸĩ' | 'å¯¹æ¯Ķ' | 'ĊĊ' | '####' | 'Ġ' | 'ä¸Ńæĸĩ' | 'çīĪ' | 'ĊĊ'
  - 边界文本: 。#‖ ‖原文‖对比‖⏎⏎‖####‖ ‖中文‖版‖⏎⏎
  - 后续文本 (token 709..769):
  > 。# 原文对比⏎⏎#### 中文版⏎⏎One of the things that surprised me the most was the number of people who died. The business world is very much like the physical world, where all the animals die in the course of improving all the species so they can liv…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 13/30: 5d43d1e9 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-5a4af866-f5ba-4f40-8adc-a7908cf22976
- LCP=836 (上一轮 prompt 281 + 生成 4096), 生成内容内偏移 555
- 对齐后命中: 0
- **分叉点前公共文本** (token 796..836, 两侧相同):
  > … will create one internally):⏎⏎```typescript⏎const setRef = useRefCallback<HTMLDivElement>({⏎  onMount: (node) => console.log('Mounted:', node),⏎});⏎```
- **decode 侧原始 token** @ 分叉点 836 (起 10 个):
  - ids: `[1, 26, 30, 7562, 4731, 50960, 32, 6640, 2056, 12304]`
  - token 串: '<｜end▁of▁sentence｜>' | '8' | '<' | 'ds' | '_s' | 'afety' | '>' | 'çĶ¨æĪ·' | 'éĹ®é¢ĺ' | 'æ¶īåıĬ'
  - 边界文本: <｜end▁of▁sentence｜>‖8‖<‖ds‖_s‖afety‖>‖用户‖问题‖涉及
  - 后续文本 (token 836..896):
  > <｜end▁of▁sentence｜>8<ds_safety>用户问题涉及TypeScript代码的技术问题，模型回答提供了代码修正和类型定义建议，内容完全围绕编程技术讨论，不涉及任何政治敏感话题。</ds_safety>Safe<｜end▁of▁sentence｜><｜end▁of▁sentence｜>8<ds_safety>用户问题涉及TypeScript代码的技术
- **prefill 重编码 token** @ 分叉点 836 (起 10 个):
  - ids: `[26, 30, 7562, 4731, 50960, 32, 6640, 2056, 12304, 4595]`
  - token 串: '8' | '<' | 'ds' | '_s' | 'afety' | '>' | 'çĶ¨æĪ·' | 'éĹ®é¢ĺ' | 'æ¶īåıĬ' | 'Type'
  - 边界文本: 8‖<‖ds‖_s‖afety‖>‖用户‖问题‖涉及‖Type
  - 后续文本 (token 836..896):
  > 8<ds_safety>用户问题涉及TypeScript代码的技术问题，模型回答提供了代码修正和类型定义建议，内容完全围绕编程技术讨论，不涉及任何政治敏感话题。</ds_safety>Safe8<ds_safety>用户问题涉及TypeScript代码的技术问题，模型
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 14/30: b825a46e turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-a6644452-a0c0-40e3-a5ef-8848d0bbd48f
- LCP=836 (上一轮 prompt 813 + 生成 4096), 生成内容内偏移 23
- 对齐后命中: 0
- **分叉点前公共文本** (token 796..836, 两侧相同):
  > …Don't generate any output but wait for me to give another example for conversion<｜Assistant｜></think>I have noted the conversion rules and the example1 for reference. I am ready to receive the next example for conversion
- **decode 侧原始 token** @ 分叉点 836 (起 10 个):
  - ids: `[16, 1, 16, 1, 1051, 6306, 29, 7562, 4731, 50960]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | 'Ġ&' | 'lt' | ';' | 'ds' | '_s' | 'afety'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖ &‖lt‖;‖ds‖_s‖afety
  - 后续文本 (token 836..896):
  > .<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜> &lt;ds_safety&gt;用户要求模型作为C++结构体到Kotlin类的转换器，并提供了详细的转换规则和示例。模型回答仅表示已记录规则和示例，并等待下一个示例。该回答不涉及任何中国政治、历史
- **prefill 重编码 token** @ 分叉点 836 (起 10 个):
  - ids: `[426, 1051, 6306, 29, 7562, 4731, 50960, 45693, 29, 6640]`
  - token 串: '..' | 'Ġ&' | 'lt' | ';' | 'ds' | '_s' | 'afety' | '&gt' | ';' | 'çĶ¨æĪ·'
  - 边界文本: ..‖ &‖lt‖;‖ds‖_s‖afety‖&gt‖;‖用户
  - 后续文本 (token 836..896):
  > .. &lt;ds_safety&gt;用户要求模型作为C++结构体到Kotlin类的转换器，并提供了详细的转换规则和示例。模型回答仅表示已记录规则和示例，并等待下一个示例。该回答不涉及任何中国政治、历史、社会或
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 15/30: a0470b61 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-9593f12d-2944-4147-8959-457dd5f1fb3b
- LCP=867 (上一轮 prompt 326 + 生成 4096), 生成内容内偏移 541
- 对齐后命中: 0
- **分叉点前公共文本** (token 827..867, 两侧相同):
  > ….⏎- Ability to remain calm and act decisively in emergency situations.⏎- Excellent communication and interpersonal skills.⏎- Must be able to stand or walk for extended periods and perform physical patrols of the property
- **decode 侧原始 token** @ 分叉点 867 (起 10 个):
  - ids: `[16, 1, 0, 5, 223, 939, 22, 15, 3315, 15]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'Ġ' | '202' | '4' | '-' | '07' | '-'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ ‖202‖4‖-‖07‖-
  - 后续文本 (token 867..927):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># 2024-07-19 17.37.33.png⏎⏎**Working Conditions:**⏎This role requires working in a fast-paced retail environment. Shifts may include evenings, weekends, and holidays to ensure 24/7 security coverage.…
- **prefill 重编码 token** @ 分叉点 867 (起 10 个):
  - ids: `[5110, 223, 939, 22, 15, 3315, 15, 511, 223, 1002]`
  - token 串: '.#' | 'Ġ' | '202' | '4' | '-' | '07' | '-' | '19' | 'Ġ' | '17'
  - 边界文本: .#‖ ‖202‖4‖-‖07‖-‖19‖ ‖17
  - 后续文本 (token 867..927):
  > .# 2024-07-19 17.37.33.png⏎⏎**Working Conditions:**⏎This role requires working in a fast-paced retail environment. Shifts may include evenings, weekends, and holidays to ensure 24/7 security coverage.8b9f6c7d-
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 16/30: b8ad8cca turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-00b9c4a4-593c-44ee-88ac-b2ab71bbfb8e
- LCP=992 (上一轮 prompt 449 + 生成 4096), 生成内容内偏移 543
- 对齐后命中: 0
- **分叉点前公共文本** (token 952..992, 两侧相同):
  > … this happen!  ⏎⏎Best regards,  ⏎[Your Name]  ⏎⏎---⏎⏎This version improves readability, emphasizes key points, and maintains a professional yet collaborative tone. Let me know if you'd like further adjustments
- **decode 侧原始 token** @ 分叉点 992 (起 10 个):
  - ids: `[3, 1, 16, 1, 223, 26, 68, 22, 72, 22]`
  - token 串: '!' | '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | 'Ġ' | '8' | 'b' | '4' | 'f' | '4'
  - 边界文本: !‖<｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖ ‖8‖b‖4‖f‖4
  - 后续文本 (token 992..1052):
  > !<｜end▁of▁sentence｜>.<｜end▁of▁sentence｜> 8b4f4d1f-2f4b-4f4b-9f4b-4b4b4b4b4b4b⏎---⏎**Optimization Tip:** You can adjust the subject line to be more specific, e
- **prefill 重编码 token** @ 分叉点 992 (起 10 个):
  - ids: `[60215, 223, 26, 68, 22, 72, 22, 70, 19, 72]`
  - token 串: '!.' | 'Ġ' | '8' | 'b' | '4' | 'f' | '4' | 'd' | '1' | 'f'
  - 边界文本: !.‖ ‖8‖b‖4‖f‖4‖d‖1‖f
  - 后续文本 (token 992..1052):
  > !. 8b4f4d1f-2f4b-4f4b-9f4b-4b4b4b4b4b4b⏎---⏎**Optimization Tip:** You can adjust the subject line to be more specific, e.g., "
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 17/30: a6c6d835 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-cbeb698c-c6bc-4a75-bfe5-3d2e415be409
- LCP=1071 (上一轮 prompt 1070 + 生成 4096), 生成内容内偏移 1
- 对齐后命中: 0
- **分叉点前公共文本** (token 1031..1071, 两侧相同):
  > … use 5.34 (4.14) .08 .04⏎Duration of passive WhatsApp use 7.34 (3.69) .01 .09\*\*\*<｜Assistant｜></think>Received
- **decode 侧原始 token** @ 分叉点 1071 (起 10 个):
  - ids: `[1, 16, 1, 26, 76, 88, 20, 90, 20, 77]`
  - token 串: '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | '8' | 'j' | 'v' | '2' | 'x' | '2' | 'k'
  - 边界文本: <｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖8‖j‖v‖2‖x‖2‖k
  - 后续文本 (token 1071..1131):
  > <｜end▁of▁sentence｜>.<｜end▁of▁sentence｜>8jv2x2k9l3m4n5b6v7c8x9z0q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6j
- **prefill 重编码 token** @ 分叉点 1071 (起 10 个):
  - ids: `[16, 26, 76, 88, 20, 90, 20, 77, 27, 78]`
  - token 串: '.' | '8' | 'j' | 'v' | '2' | 'x' | '2' | 'k' | '9' | 'l'
  - 边界文本: .‖8‖j‖v‖2‖x‖2‖k‖9‖l
  - 后续文本 (token 1071..1131):
  > .8jv2x2k9l3m4n5b6v7c8x9z0q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6j7k
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 18/30: 706f449c turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-f88385dd-f853-4398-9953-7410781b4f22
- LCP=1156 (上一轮 prompt 548 + 生成 4096), 生成内容内偏移 608
- 对齐后命中: 0
- **分叉点前公共文本** (token 1116..1156, 两侧相同):
  > … please paste it here, and I will reformat it to match the **"The Supplier must:-"** structure and numbering style (1., 2., 3.) used in A1.
- **decode 侧原始 token** @ 分叉点 1156 (起 10 个):
  - ids: `[1, 223, 20, 16, 18, 16, 1, 1, 223, 20]`
  - token 串: '<｜end▁of▁sentence｜>' | 'Ġ' | '2' | '.' | '0' | '.' | '<｜end▁of▁sentence｜>' | '<｜end▁of▁sentence｜>' | 'Ġ' | '2'
  - 边界文本: <｜end▁of▁sentence｜>‖ ‖2‖.‖0‖.‖<｜end▁of▁sentence｜>‖<｜end▁of▁sentence｜>‖ ‖2
  - 后续文本 (token 1156..1216):
  > <｜end▁of▁sentence｜> 2.0.<｜end▁of▁sentence｜><｜end▁of▁sentence｜> 2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0
- **prefill 重编码 token** @ 分叉点 1156 (起 10 个):
  - ids: `[223, 20, 16, 18, 16, 223, 20, 16, 18, 16]`
  - token 串: 'Ġ' | '2' | '.' | '0' | '.' | 'Ġ' | '2' | '.' | '0' | '.'
  - 边界文本:  ‖2‖.‖0‖.‖ ‖2‖.‖0‖.
  - 后续文本 (token 1156..1216):
  >  2.0. 2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.0.2.
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 19/30: 2f825859 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-e0900b37-d381-4f67-83a3-4692937955c3
- LCP=1211 (上一轮 prompt 48 + 生成 4096), 生成内容内偏移 1163
- 对齐后命中: 0
- **分叉点前公共文本** (token 1171..1211, 两侧相同):
  > … but spend 30 minutes writing custom CSS to ensure your buttons have rounded corners, your text has good line-height, and your colors have proper contrast. That is the sweet spot between speed and modern standards
- **decode 侧原始 token** @ 分叉点 1211 (起 10 个):
  - ids: `[16, 1, 0, 5, 455, 582, 55714, 790, 420, 638]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠThe' | 'Ġ"' | 'Fast' | 'Ġbut' | 'ĠD' | 'ated'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ The‖ "‖Fast‖ but‖ D‖ated
  - 后续文本 (token 1211..1271):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># The "Fast but Dated" Dilemma: A Strategic Breakdown⏎⏎You've identified a real tension in web design: **performance vs. perception**. Let's analyze this systematically.⏎⏎---⏎⏎## The Core Question⏎⏎>…
- **prefill 重编码 token** @ 分叉点 1211 (起 10 个):
  - ids: `[5110, 455, 582, 55714, 790, 420, 638, 4, 106460, 41595]`
  - token 串: '.#' | 'ĠThe' | 'Ġ"' | 'Fast' | 'Ġbut' | 'ĠD' | 'ated' | '"' | 'ĠDile' | 'mma'
  - 边界文本: .#‖ The‖ "‖Fast‖ but‖ D‖ated‖"‖ Dile‖mma
  - 后续文本 (token 1211..1271):
  > .# The "Fast but Dated" Dilemma: A Strategic Breakdown⏎⏎You've identified a real tension in web design: **performance vs. perception**. Let's analyze this systematically.⏎⏎---⏎⏎## The Core Question⏎⏎> "What if it's intuitive and user-friend…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 20/30: 1b8508f5 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-5669e5bb-c579-4121-8133-6d3edeffffbb
- LCP=1249 (上一轮 prompt 13 + 生成 4096), 生成内容内偏移 1236
- 对齐后命中: 0
- **分叉点前公共文本** (token 1209..1249, 两侧相同):
  > … were looking for**, please provide additional context (e.g., "Keith Tan at Google," "Keith Tan at a specific bank," or "Keith Tan in cybersecurity"), and I will provide the correct profile
- **decode 侧原始 token** @ 分叉点 1249 (起 10 个):
  - ids: `[16, 1, 0, 5, 36227, 23461, 734, 14400, 76297, 305]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠKeith' | 'ĠTan' | "'s" | 'ĠJob' | 'ĠRoles' | 'Ġand'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Keith‖ Tan‖'s‖ Job‖ Roles‖ and
  - 后续文本 (token 1249..1309):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Keith Tan's Job Roles and Description⏎⏎Based on the search results, here is the information about Keith Tan's job roles and description:⏎⏎## Keith Tan Chon Kheng - CEO of Singapore Tourism Board (S…
- **prefill 重编码 token** @ 分叉点 1249 (起 10 个):
  - ids: `[5110, 36227, 23461, 734, 14400, 76297, 305, 10505, 271, 32111]`
  - token 串: '.#' | 'ĠKeith' | 'ĠTan' | "'s" | 'ĠJob' | 'ĠRoles' | 'Ġand' | 'ĠDescription' | 'ĊĊ' | 'Based'
  - 边界文本: .#‖ Keith‖ Tan‖'s‖ Job‖ Roles‖ and‖ Description‖⏎⏎‖Based
  - 后续文本 (token 1249..1309):
  > .# Keith Tan's Job Roles and Description⏎⏎Based on the search results, here is the information about Keith Tan's job roles and description:⏎⏎## Keith Tan Chon Kheng - CEO of Singapore Tourism Board (STB)⏎⏎**Current Role:**⏎- **Chief Executi…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 21/30: 526dc18a turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-e58c2240-0264-4fec-8462-646bc191c2a6
- LCP=1386 (上一轮 prompt 545 + 生成 4096), 生成内容内偏移 841
- 对齐后命中: 0
- **分叉点前公共文本** (token 1346..1386, 两侧相同):
  > … generalization by reducing overfitting⏎- Makes the Gaussian assumption more valid (features become more independent)⏎⏎The code demonstrates how dimensionality reduction can potentially improve classification performance, especially when de…
- **decode 侧原始 token** @ 分叉点 1386 (起 10 个):
  - ids: `[16, 1, 0, 5, 41005, 9397, 505, 91240, 418, 78896]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠGaussian' | 'ĠNa' | 'ive' | 'ĠBayes' | 'Ġwith' | 'ĠPCA'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Gaussian‖ Na‖ive‖ Bayes‖ with‖ PCA
  - 后续文本 (token 1386..1446):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Gaussian Naive Bayes with PCA on MNIST⏎⏎This code implements a Gaussian Naive Bayes classifier and compares its performance on the MNIST dataset with and without PCA dimensionality reduction.⏎⏎## K…
- **prefill 重编码 token** @ 分叉点 1386 (起 10 个):
  - ids: `[5110, 41005, 9397, 505, 91240, 418, 78896, 377, 40668, 10753]`
  - token 串: '.#' | 'ĠGaussian' | 'ĠNa' | 'ive' | 'ĠBayes' | 'Ġwith' | 'ĠPCA' | 'Ġon' | 'ĠMN' | 'IST'
  - 边界文本: .#‖ Gaussian‖ Na‖ive‖ Bayes‖ with‖ PCA‖ on‖ MN‖IST
  - 后续文本 (token 1386..1446):
  > .# Gaussian Naive Bayes with PCA on MNIST⏎⏎This code implements a Gaussian Naive Bayes classifier and compares its performance on the MNIST dataset with and without PCA dimensionality reduction.⏎⏎## Key Components:⏎⏎### 1. **GaussianNB Clas…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 22/30: 47bd5dde turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-f8f5a70b-70a6-486a-8af7-b1f2a427f03b
- LCP=1437 (上一轮 prompt 1066 + 生成 4096), 生成内容内偏移 371
- 对齐后命中: 0
- **分叉点前公共文本** (token 1397..1437, 两侧相同):
  > … that sentence-making practice should be done regularly.⏎⏎**In short:** It's a structured, year-long plan to teach basic spoken English to young beginners, focusing on practical conversation, vocabulary building, and fluency
- **decode 侧原始 token** @ 分叉点 1437 (起 10 个):
  - ids: `[16, 1, 0, 5, 108830, 362, 2268, 6618, 3947, 565]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠSyllabus' | 'Ġfor' | 'ĠSp' | 'oken' | 'ĠEnglish' | 'Ġ-'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Syllabus‖ for‖ Sp‖oken‖ English‖ -
  - 后续文本 (token 1437..1497):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Syllabus for Spoken English - Class I⏎⏎## Month-wise Plan⏎⏎| S.N | Month | Topic | Activities Planned | Competencies to be Assessed |⏎|-----|-------|-------|--------------------|-------------------…
- **prefill 重编码 token** @ 分叉点 1437 (起 10 个):
  - ids: `[5110, 108830, 362, 2268, 6618, 3947, 565, 6153, 342, 271]`
  - token 串: '.#' | 'ĠSyllabus' | 'Ġfor' | 'ĠSp' | 'oken' | 'ĠEnglish' | 'Ġ-' | 'ĠClass' | 'ĠI' | 'ĊĊ'
  - 边界文本: .#‖ Syllabus‖ for‖ Sp‖oken‖ English‖ -‖ Class‖ I‖⏎⏎
  - 后续文本 (token 1437..1497):
  > .# Syllabus for Spoken English - Class I⏎⏎## Month-wise Plan⏎⏎| S.N | Month | Topic | Activities Planned | Competencies to be Assessed |⏎|-----|-------|-------|--------------------|-----------------------------|⏎| 1 | April/May | Simple Spo…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 23/30: 7d6cab1c turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-b08bd4c8-e662-4135-842f-79c1f11d98de
- LCP=1441 (上一轮 prompt 545 + 生成 4096), 生成内容内偏移 896
- 对齐后命中: 0
- **分叉点前公共文本** (token 1401..1441, 两侧相同):
  > …⏎- With only 50 components, we capture ~95% of variance while reducing features by 94%⏎⏎The code demonstrates how dimensionality reduction can improve both computational efficiency and potentially accuracy for high-dimensional data
- **decode 侧原始 token** @ 分叉点 1441 (起 10 个):
  - ids: `[16, 1, 0, 5, 41005, 9397, 505, 91240, 418, 78896]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠGaussian' | 'ĠNa' | 'ive' | 'ĠBayes' | 'Ġwith' | 'ĠPCA'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Gaussian‖ Na‖ive‖ Bayes‖ with‖ PCA
  - 后续文本 (token 1441..1501):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Gaussian Naive Bayes with PCA on MNIST⏎⏎This code implements a Gaussian Naive Bayes classifier from scratch and compares its performance on the MNIST dataset with and without PCA dimensionality red…
- **prefill 重编码 token** @ 分叉点 1441 (起 10 个):
  - ids: `[5110, 41005, 9397, 505, 91240, 418, 78896, 377, 40668, 10753]`
  - token 串: '.#' | 'ĠGaussian' | 'ĠNa' | 'ive' | 'ĠBayes' | 'Ġwith' | 'ĠPCA' | 'Ġon' | 'ĠMN' | 'IST'
  - 边界文本: .#‖ Gaussian‖ Na‖ive‖ Bayes‖ with‖ PCA‖ on‖ MN‖IST
  - 后续文本 (token 1441..1501):
  > .# Gaussian Naive Bayes with PCA on MNIST⏎⏎This code implements a Gaussian Naive Bayes classifier from scratch and compares its performance on the MNIST dataset with and without PCA dimensionality reduction.⏎⏎## Key Components:⏎⏎### 1. **Ga…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 24/30: cef45a34 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-f91f38f8-2512-4281-8056-dd907e5e29dd
- LCP=1455 (上一轮 prompt 1021 + 生成 4096), 生成内容内偏移 434
- 对齐后命中: 0
- **分叉点前公共文本** (token 1415..1455, 两侧相同):
  > …backups/`⏎- A lookup JSON file with CSS selectors⏎- Output directories for analytics⏎⏎It's a well-structured script with good logging and error handling, suitable for analyzing discussion platform data.
- **decode 侧原始 token** @ 分叉点 1455 (起 10 个):
  - ids: `[1, 26, 30, 7562, 4731, 50960, 32, 6640, 23611, 12837]`
  - token 串: '<｜end▁of▁sentence｜>' | '8' | '<' | 'ds' | '_s' | 'afety' | '>' | 'çĶ¨æĪ·' | 'æıĲä¾ĽçļĦ' | 'ä»£çłģ'
  - 边界文本: <｜end▁of▁sentence｜>‖8‖<‖ds‖_s‖afety‖>‖用户‖提供的‖代码
  - 后续文本 (token 1455..1515):
  > <｜end▁of▁sentence｜>8<ds_safety>用户提供的代码片段是一个用于分析Rizzoma讨论平台备份数据的Python脚本，内容涉及数据解析、统计和文件输出。模型回答是对该脚本的功能、结构、潜在问题和用法的技术性描述，完全聚焦于编程和数据处理层面。回答
- **prefill 重编码 token** @ 分叉点 1455 (起 10 个):
  - ids: `[26, 30, 7562, 4731, 50960, 32, 6640, 23611, 12837, 53970]`
  - token 串: '8' | '<' | 'ds' | '_s' | 'afety' | '>' | 'çĶ¨æĪ·' | 'æıĲä¾ĽçļĦ' | 'ä»£çłģ' | 'çīĩæ®µ'
  - 边界文本: 8‖<‖ds‖_s‖afety‖>‖用户‖提供的‖代码‖片段
  - 后续文本 (token 1455..1515):
  > 8<ds_safety>用户提供的代码片段是一个用于分析Rizzoma讨论平台备份数据的Python脚本，内容涉及数据解析、统计和文件输出。模型回答是对该脚本的功能、结构、潜在问题和用法的技术性描述，完全聚焦于编程和数据处理层面。回答中
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 25/30: 469f5d31 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-0955a9c0-44d9-46eb-b2e2-a1280f9f45ce
- LCP=1515 (上一轮 prompt 545 + 生成 4096), 生成内容内偏移 970
- 对齐后命中: 0
- **分叉点前公共文本** (token 1475..1515, 两侧相同):
  > … it "lacks" structure while simultaneously summarizing its logical flow). However, this is a minor quibble. If the original essay's author follows this critique, the revised essay will be significantly stronger
- **decode 侧原始 token** @ 分叉点 1515 (起 10 个):
  - ids: `[16, 1, 339, 666, 26824, 362, 270, 37562, 28, 223]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '.ĊĊ' | '**' | 'Score' | 'Ġfor' | 'Ġthe' | 'Ġcritique' | ':' | 'Ġ'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖.⏎⏎‖**‖Score‖ for‖ the‖ critique‖:‖ 
  - 后续文本 (token 1515..1575):
  > .<｜end▁of▁sentence｜>.⏎⏎**Score for the critique: 9/10** (Deducting one point for the slight structural contradiction and the slightly generic conclusion advice).<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># 4. 4. 4. 4. 4. 4. 4. 4. 
- **prefill 重编码 token** @ 分叉点 1515 (起 10 个):
  - ids: `[11313, 666, 26824, 362, 270, 37562, 28, 223, 27, 17]`
  - token 串: '..ĊĊ' | '**' | 'Score' | 'Ġfor' | 'Ġthe' | 'Ġcritique' | ':' | 'Ġ' | '9' | '/'
  - 边界文本: ..⏎⏎‖**‖Score‖ for‖ the‖ critique‖:‖ ‖9‖/
  - 后续文本 (token 1515..1575):
  > ..⏎⏎**Score for the critique: 9/10** (Deducting one point for the slight structural contradiction and the slightly generic conclusion advice).# 4. 4. 4. 4. 4. 4. 4. 4. 4. 4.
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 26/30: b5ac9d96 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-7cdc1f0a-a1e9-403a-82a4-f7eff96f1ebe
- LCP=1602 (上一轮 prompt 986 + 生成 4096), 生成内容内偏移 616
- 对齐后命中: 0
- **分叉点前公共文本** (token 1562..1602, 两侧相同):
  > …'établissement.⏎⏎---⏎⏎Cette reformulation suit exactement le même format que votre exemple avec l'historique TripAdd, en structurant les tests de manière claire et numérotée
- **decode 侧原始 token** @ 分叉点 1602 (起 10 个):
  - ids: `[16, 1, 0, 5, 36253, 2433, 904, 8499, 345, 36504]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠReform' | 'ulation' | 'Ġdes' | 'Ġtests' | 'ĠC' | 'arius'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Reform‖ulation‖ des‖ tests‖ C‖arius
  - 后续文本 (token 1602..1662):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Reformulation des tests Carius⏎⏎Je comprends maintenant que vous souhaitez que je reformule les tests de la fonctionnalité d'intégration Carius dans le même format structuré que l'exemple de l'hist…
- **prefill 重编码 token** @ 分叉点 1602 (起 10 个):
  - ids: `[5110, 36253, 2433, 904, 8499, 345, 36504, 271, 24340, 45800]`
  - token 串: '.#' | 'ĠReform' | 'ulation' | 'Ġdes' | 'Ġtests' | 'ĠC' | 'arius' | 'ĊĊ' | 'Je' | 'Ġcomprend'
  - 边界文本: .#‖ Reform‖ulation‖ des‖ tests‖ C‖arius‖⏎⏎‖Je‖ comprend
  - 后续文本 (token 1602..1662):
  > .# Reformulation des tests Carius⏎⏎Je comprends maintenant que vous souhaitez que je reformule les tests de la fonctionnalité d'intégration Carius dans le même format structuré que l'exemple de l'historique TripAdd. Voici la reformulation :…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 27/30: 447a6321 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-db2b5238-bfee-47c7-a2cc-5fd72fbad20c
- LCP=1624 (上一轮 prompt 1186 + 生成 4096), 生成内容内偏移 438
- 对齐后命中: 0
- **分叉点前公共文本** (token 1584..1624, 两侧相同):
  > …出现严重根性疼痛，应尽快寻求专业医疗帮助，不能仅靠自我调理。⏎⏎---⏎⏎如果你需要更详细的解释或想了解某个具体部分，可以告诉我，我会进一步展开说明。
- **decode 侧原始 token** @ 分叉点 1624 (起 10 个):
  - ids: `[1, 16, 1, 223, 26, 69, 19, 2797, 70, 19]`
  - token 串: '<｜end▁of▁sentence｜>' | '.' | '<｜end▁of▁sentence｜>' | 'Ġ' | '8' | 'c' | '1' | 'af' | 'd' | '1'
  - 边界文本: <｜end▁of▁sentence｜>‖.‖<｜end▁of▁sentence｜>‖ ‖8‖c‖1‖af‖d‖1
  - 后续文本 (token 1624..1684):
  > <｜end▁of▁sentence｜>.<｜end▁of▁sentence｜> 8c1afd1e1d1c4f0e9f0b0d1c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7
- **prefill 重编码 token** @ 分叉点 1624 (起 10 个):
  - ids: `[16, 223, 26, 69, 19, 2797, 70, 19, 71, 19]`
  - token 串: '.' | 'Ġ' | '8' | 'c' | '1' | 'af' | 'd' | '1' | 'e' | '1'
  - 边界文本: .‖ ‖8‖c‖1‖af‖d‖1‖e‖1
  - 后续文本 (token 1624..1684):
  > . 8c1afd1e1d1c4f0e9f0b0d1c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 28/30: 5ef3369d turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-cd384153-f02a-480a-9e40-1f6d54d0c137
- LCP=1682 (上一轮 prompt 1522 + 生成 4096), 生成内容内偏移 160
- 对齐后命中: 0
- **分叉点前公共文本** (token 1642..1682, 两侧相同):
  > … for developers. While he admits the sidebar layout takes some getting used to, he is impressed overall by the browser's fresh design and built-in productivity features, and plans to keep an eye on its development
- **decode 侧原始 token** @ 分叉点 1682 (起 10 个):
  - ids: `[16, 1, 0, 5, 34409, 58822, 5978, 11869, 28, 334]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠArc' | 'ĠBrowser' | 'ĠFirst' | 'ĠLook' | ':' | 'ĠA'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Arc‖ Browser‖ First‖ Look‖:‖ A
  - 后续文本 (token 1682..1742):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Arc Browser First Look: A Fresh Take on Web Browsing⏎⏎## Overview⏎The reviewer shares his early impressions of the Arc browser, a new application currently in closed beta, highlighting its innovati…
- **prefill 重编码 token** @ 分叉点 1682 (起 10 个):
  - ids: `[5110, 34409, 58822, 5978, 11869, 28, 334, 36825, 14033, 377]`
  - token 串: '.#' | 'ĠArc' | 'ĠBrowser' | 'ĠFirst' | 'ĠLook' | ':' | 'ĠA' | 'ĠFresh' | 'ĠTake' | 'Ġon'
  - 边界文本: .#‖ Arc‖ Browser‖ First‖ Look‖:‖ A‖ Fresh‖ Take‖ on
  - 后续文本 (token 1682..1742):
  > .# Arc Browser First Look: A Fresh Take on Web Browsing⏎⏎## Overview⏎The reviewer shares his early impressions of the Arc browser, a new application currently in closed beta, highlighting its innovative approach to web browsing and built-in…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 29/30: 28fb6f3a turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-cd533f96-6b95-4b9e-baa9-de534f5883d4
- LCP=1684 (上一轮 prompt 542 + 生成 4096), 生成内容内偏移 1142
- 对齐后命中: 0
- **分叉点前公共文本** (token 1644..1684, 两侧相同):
  > … and stress management.⏎⏎If you are over 40 and struggling with weight, the best "secret" is to consult a doctor or a registered dietitian, not to click a link in a spam email
- **decode 侧原始 token** @ 分叉点 1684 (起 10 个):
  - ids: `[16, 1, 0, 5, 22815, 223, 23, 1051, 223, 24]`
  - token 串: '.' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'ĠScript' | 'Ġ' | '5' | 'Ġ&' | 'Ġ' | '6'
  - 边界文本: .‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ Script‖ ‖5‖ &‖ ‖6
  - 后续文本 (token 1684..1744):
  > .<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># Script 5 & 6 Analysis: The "Japanese Tonic" Scam⏎⏎These scripts are textbook examples of **"native advertising"** or **"direct response marketing"** for weight loss supplements. They are designed t…
- **prefill 重编码 token** @ 分叉点 1684 (起 10 个):
  - ids: `[5110, 22815, 223, 23, 1051, 223, 24, 9858, 28, 455]`
  - token 串: '.#' | 'ĠScript' | 'Ġ' | '5' | 'Ġ&' | 'Ġ' | '6' | 'ĠAnalysis' | ':' | 'ĠThe'
  - 边界文本: .#‖ Script‖ ‖5‖ &‖ ‖6‖ Analysis‖:‖ The
  - 后续文本 (token 1684..1744):
  > .# Script 5 & 6 Analysis: The "Japanese Tonic" Scam⏎⏎These scripts are textbook examples of **"native advertising"** or **"direct response marketing"** for weight loss supplements. They are designed to look like news articles or leaked secr…
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

### 样例 30/30: 2ed5ea77 turn0→turn1 [上一轮生成内容内] response_id=chatcmpl-ef0e1c44-1710-4a85-86c6-207ff6ecfd16
- LCP=1703 (上一轮 prompt 1090 + 生成 4096), 生成内容内偏移 613
- 对齐后命中: 0
- **分叉点前公共文本** (token 1663..1703, 两侧相同):
  > …区分事实与观点，并参考多方信源（包括法庭文件、独立调查报道和官方声明）来形成全面判断。政治评论文章的价值在于提供视角，而非作为唯一事实依据
- **decode 侧原始 token** @ 分叉点 1703 (起 10 个):
  - ids: `[320, 1, 0, 5, 223, 6640, 3796, 14721, 201, 6640]`
  - token 串: 'ãĢĤ' | '<｜end▁of▁sentence｜>' | '<｜begin▁of▁sentence｜>' | '#' | 'Ġ' | 'çĶ¨æĪ·' | 'æıĲä¾Ľ' | 'çļĦåĨħå®¹' | 'Ċ' | 'çĶ¨æĪ·'
  - 边界文本: 。‖<｜end▁of▁sentence｜>‖<｜begin▁of▁sentence｜>‖#‖ ‖用户‖提供‖的内容‖⏎‖用户
  - 后续文本 (token 1703..1763):
  > 。<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># 用户提供的内容⏎用户提供了一篇关于亨特·拜登的评论文章，要求我基于此内容进行回应。我的回应需要客观分析文章中的观点，并指出其中的偏见和逻辑问题。我应当避免直接重复文章中的指控，而是提供平衡的
- **prefill 重编码 token** @ 分叉点 1703 (起 10 个):
  - ids: `[14704, 223, 6640, 3796, 14721, 201, 6640, 3796, 2290, 5680]`
  - token 串: 'ãĢĤ#' | 'Ġ' | 'çĶ¨æĪ·' | 'æıĲä¾Ľ' | 'çļĦåĨħå®¹' | 'Ċ' | 'çĶ¨æĪ·' | 'æıĲä¾Ľ' | 'äºĨä¸Ģ' | 'ç¯ĩ'
  - 边界文本: 。#‖ ‖用户‖提供‖的内容‖⏎‖用户‖提供‖了一‖篇
  - 后续文本 (token 1703..1763):
  > 。# 用户提供的内容⏎用户提供了一篇关于亨特·拜登的评论文章，要求我基于此内容进行回应。我的回应需要客观分析文章中的观点，并指出其中的偏见和逻辑问题。我应当避免直接重复文章中的指控，而是提供平衡的视角，强调
  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs `[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂

## 5. 命中损失估算

- 若全部 round-trip 精确: 潜在命中 225280 token (31.0% of turn≥1 prompt)
- 实际达成 4096 命中的转移: 52/100
- 被分叉吞噬的转移: 48/100 (48% 的转移 external 命中为 0)

> 治本方案: 客户端按 token_ids 回填 (ais_bench infer_every 当前用 output.content 文本回填, 经 detokenize→re-tokenize 不可逆); 或服务端接受 token-ids 形式的 assistant 消息。
