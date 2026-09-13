# DeepSeek-V4-Flash w4a8 部署与推理分析 — 背景信息（截至 2026-09-12）

> 用途：opencode 后续分析的背景知识。所有结论均在本机实测/代码验证过，标注了证据来源。

## 1. 硬件与环境

- 硬件：8 × Ascend 910B3（每卡 HBM 64GB = 65536MiB），aarch64，192 CPU 核，内存 ~2TB，/dev/shm 512G
- **卡 0–3 长期被其他容器进程占用（~53.5GiB HBM/卡）**；本容器 `ps` 看不到这些 PID（跨容器共享 NPU）。分析资源时先看 `npu-smi info`，不要以为 ps 为空就是空闲
- 软件栈：vllm 0.23.0+empty（/vllm-workspace/vllm，editable）、vllm-ascend 0.23.0（/vllm-workspace/vllm-ascend，`_build_info.__device_type__='A2'`）、torch 2.10.0+cpu、torch_npu 2.10.0.post4、CANN 9.1.0、mooncake-transfer-engine-npu 0.3.11.post1、triton_ascend 3.2.2
- vllm-ascend 注册点速查：
  - `--quantization ascend`：vllm_ascend/platform.py:190-202
  - w4a8(modelslim) 路径：vllm_ascend/quantization/modelslim_config.py（quant_model_description.json + quant_model_weights.safetensors.index.json，vLLM 权重加载兼容）
  - block-size 约束：vllm_ascend/utils.py:1386 附近 `refresh_block_size`，deepseek_v4 允许 32/64/128
  - patch：vllm_ascend/patch/platform/patch_async_swa_kv_lifetime.py（SWA/ChunkedLocal 块生命周期，改 allocate_slots/remove_skipped_blocks）；SchedulerDynamicBatch 仅在 ascend_config.SLO_limits_for_dynamic_batch != -1 时启用

## 2. 模型信息

- 路径：`/data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8`，盘上 159GB
- config：architectures=DeepseekV4ForCausalLM，model_type=deepseek_v4，43 层，256 routed + 1 shared experts（EP 下每卡 64/256），expert_dtype=fp4，head_dim=512，hidden 4096，vocab 129280，bos_id=0，eos_id=1，sliding_window=128，compress_ratios=[0,0,4,128,...,0]（混合压缩注意力 + indexer），max_position_embeddings=1M（yarn factor 16，基座 65536），num_nextn_predict_layers=1（MTP，未启用 speculative）
- 量化：modelslim w4a8（quant_model_description.json + 43 个 quant_model_weights-*.safetensors 分片 + index）
- tokenizer_config.json **无 chat_template、无 auto_map**（不需要 trust-remote-code）；`--tokenizer-mode deepseek_v4` 用 vllm 内置实现：`vllm/tokenizers/deepseek_v4.py`（DeepseekV4Tokenizer 包装）+ `vllm/tokenizers/deepseek_v4_encoding.py`（encode_messages，模板核心）
- 官方 best_practice yaml（DeepSeek-V4-Flash-DSpark_best_practice.yaml）verified_tags = **Atlas_A3**；本机为 910B3(A2)，未官方验证但实测可用

## 3. 部署脚本分析结论

三个脚本：`/workspace/dsv4.sh`（standalone TP4 卡4-7 port8196）、`dsv4_prefill.sh`（卡0-3 port8196）、`dsv4_decode.sh`（卡4-7 port8197，PD 分离 + MultiConnector[MooncakeHybridConnector + AscendStoreConnector]）。standalone 与 PD 是互斥方案（端口/卡冲突，不能同时跑）。

- dsv4.sh 语法/参数全部合法（bash -n 通过；--quantization ascend、tokenizer/tool/reasoning-parser "deepseek_v4" 均已注册、--no-disable-hybrid-kv-cache-manager 为 BooleanOptionalAction 合法形式）
- **standalone 部署已成功运行**：http://0.0.0.0:8196，served name `dsv4`，日志 `/workspace/vllm_standalone.log`
- 显存（每卡 4/5/6/7 一致，来源 log:261-275）：
  - 可用 60.96 GiB（64G 标称减驱动保留），0.90 预算 = 54.86 GiB
  - **权重 37.685 GiB/卡**（4 卡合计 ~150.7 GiB）+ 峰值激活 1.77 GiB + non-torch 0.45 GiB + NPU graph 0（enforce-eager）+ **KV cache 14.96 GiB**
  - npu-smi 实际：进程 55063MiB/卡，HBM 总用 58438/65536 MiB
  - KV pool：4 卡共 **58,115 tokens**（bf16，block 32，混合注意力），~277KiB/token/卡；32k 满长并发仅 1.77x
  - 扩容选项（log:272 给出）：`--kv-cache-memory 22055902208`（20.54GiB/卡）→ ~8 万 tokens

## 4. 多轮对话 KV 复用研究（非思考模式）— 核心结论

验证脚本：`/workspace/verify_dsv4_multiturn_kv_reuse.py`（Path-A token 级精确匹配 + Path-B 端到端 /metrics 验证，可重复跑；依赖 vllm.tokenizers.deepseek_v4 离线渲染 == 服务端渲染，已证相等）

1. **渲染结果**（非思考模式 thinking_mode="chat"，默认）：每个 user 消息后渲染 `<｜User｜>{content}<｜Assistant｜></think>` — **`<｜Assistant｜>` 之后只有 `</think>`**（无 `<think>`、无换行）。证据：deepseek_v4_encoding.py render_message L390-398 else 分支
2. **第 2 轮 assistant1 渲染** = `</think>{content}<｜end▁of▁sentence｜>`（assistant_msg_template L51，chat 模式无 reasoning 段，模板自动补 EOS；drop_thinking 默认 True 但 chat 模式本就不渲染 reasoning）
3. **模型输出**：非思考模式直接输出正文，不以 `<think>` 开头；reasoning parser 走 IdentityReasoningParser（content 原样透传，reasoning_content=None）
4. **匹配结论**：assistant1 content 原样回填时，第 2 轮渲染前缀（system+user1+assistant1）与第 1 轮 decode 序列（prompt+生成+EOS）**token 级完整匹配，实测 312/312**，字符串级 startswith=True。唯一例外：turn-1 被 max_tokens 截断（finish=length）时 decode 无 EOS，仅末尾差 1 个 token
5. **最重要发现：实际 KV 复用粒度 = 4096 tokens**。vllm-ascend `CompressAttentionManager`（vllm_ascend/core/single_type_kv_cache_manager.py L195-240）对 compress_ratio=128 的压缩注意力层，命中粒度 = `block_size × compress_ratio = 32×128 = 4096 token` 的逻辑块，多 KV group 固定点取 min → **跨轮命中只可能发生在 4096 的整数倍边界；前缀 < 4096 时命中恒为 0（结构性全量重算），与模板是否匹配无关**
6. 实测数据（/metrics vllm:prefix_cache_hits_total 差值）：
   - 相同 prompt ×2：45/763/311/3763 token 前缀 → 0 命中；5364 token → **+4096**
   - 真实两轮对话：turn1 (prompt 3899 + completion 420 = 4319 保存)，turn2 prompt 4330 → **命中 4096**
   - 服务周期日志 `Prefix cache hit rate: 0.0%`（上下文 <4096 时恒为 0 是预期行为，不是故障）
7. 方法论教训：
   - temperature=0 下两次独立生成在 NPU 批量下**非逐位确定**（首 ~120 token 后分歧）→ 对照实验必须用同一次生成的文本回填，不能跨请求比文本
   - 该版本 usage.prompt_tokens_details.cached_tokens 返回 None → 用 /metrics 计数器差值验证
   - `/tokenize` 路由在根路径（http://host:8196/tokenize），不在 /v1 下
   - standalone 走 async scheduling（vllm 0.23 默认开，log "Asynchronous scheduling is enabled"）

## 4b. 思考模式多轮对话 KV 复用（thinking=true）— 已实测

脚本：`/workspace/verify_dsv4_thinking_multiturn_kv_reuse.py`（v2：[1]渲染 [2]raw生成 [2b]chat响应message格式 [3]五变体LCP [3b]chat归一化实证 [4]完整渲染串 [5]跨4096真实命中）。链路：raw 输出 = `reasoning</think>content`（`<think>` 标签是模板注入的，不在输出里）。

1. **turn-1 渲染**以 `<｜Assistant｜><think>` 结尾（`<think>` 开标签来自最后一条 user 的 transition token）；decode 保存 KV = `<think>` + 思考 + `</think>` + 正文 + EOS
2. **响应侧属性名是 `reasoning`**（vLLM 0.23 ChatMessage/DeltaMessage，protocol.py:67；`reasoning_content` 在响应中不存在）。思考被 max_tokens 截断时 content=null、reasoning 有值（finish=length）
3. **请求侧两个字段都生效**：真实 `/v1/chat/completions` 的请求模型有 before-validator（protocol.py:433-449）把 `reasoning_content` **归一化为 `reasoning`**；但 offline `apply_chat_template` 与 `/tokenize` 路径**没有**该归一化（V-D 在这两条路径下渲染同 V-C）
4. **默认（drop_thinking=True）不拼接思考**：`_drop_thinking_messages`（encode_messages L559-605）剥掉 last-user 之前 assistant 的 reasoning；历史 user transition 渲染 `</think>`。实测 LCP=51/906（分叉在 `<｜Assistant｜>` 后：decode `<think>` vs 渲染 `</think>`）→ 只能复用 turn-1 输入
5. **drop_thinking=False + 回传思考（reasoning 或 reasoning_content）→ 完整匹配**：渲染形态 assistant1 = `{reasoning}</think>{content}<EOS>`，实测 LCP=906/906 含 EOS；且 [3b] 用 return_prompt_text 实证 chat API 实际渲染与离线 V-B 逐字相同
6. drop_thinking=False 但不回传思考 → `<think></think>{content}` 空思考，LCP=52/906（多匹配 1 个 `<think>`）→ 要复用必须三条件同时满足：thinking=true + drop_thinking=false + assistant 消息带回思考内容
7. drop_thinking 生效点：encode_messages L554-557（带 tools 强制 False）；render_message L350-354（assistant 在最后一条 user 之后/prefill 场景即使 True 也渲染）；reasoning_effort="none" 强制 chat 模式
8. 拼接责任：渲染由服务端模板负责；客户端 = assistant 消息带思考内容（字段名 reasoning 或 reasoning_content 均可走 chat API）+ `chat_template_kwargs: {"thinking": true, "drop_thinking": false}`；自行拼进 content 无法对齐
9. **跨 4096 端到端实测**（[5]）：长输入 prompt 3467 + min_tokens=450 强制 → 保存 4258 tokens；turn-2 回传思考+drop_thinking=false → **hits +4096**；turn-2 默认丢弃 → **hits +0**（分叉点 ~3467+51 < 4096）。结论：回传思考的完整匹配在真实缓存中兑现为 4096 粒度命中
10. LCP 含义：与 decode 保存序列的最长公共前缀；49/781 = 前 49 token（到 `<｜Assistant｜>`）一致、第 50 个 token 分叉；实际复用还要过 32-token 分块与压缩层 4096 逻辑块两道粒度
11. **prefill/续写场景（请求以 assistant 结尾）**：例外 1 在此生效——即使默认 drop_thinking=True，该 assistant 的 reasoning 也渲染（`index > last_user_idx`），实测在线 chat API prompt_text 含 `<think>思考</think>`。触发方式：`continue_final_message=true` 请求参数或直接把 assistant 放最后；用途：输出前缀引导、截断续传、网关重建会话
12. **wo_eos 在线不可用（实测）**：消息级 `wo_eos=true` 只在离线 `apply_chat_template` 生效；`/tokenize` 与真实 chat API 的解析链路（`_parse_chat_message_content` 只保留 role/content/tool_calls/reasoning）会过滤掉它，`continue_final_message` 也只服务 HF jinja 模板、deepseek_v4 python wrapper 不读 → **在线 API 无法抑制结尾 EOS，真正的无缝续写只能走 /v1/completions token-ids 或离线 engine**；带 EOS 时模型从 EOS 后"开新一轮"而非续写

## 5. 已知无害告警（不用排查）

- `rope_parameters's factor field must be a float`（类型提示）
- `Auto-initialization of reasoning token IDs failed`（reasoning parser 未实现 reasoning_start/end_str，Identity parser 下无影响）
- CANN 自带 gelu_grad_v2.py SyntaxWarning；NPUCachingAllocator 32-padding 提示
- block-size "should be 32, 64 or 128" warning（本就传 32，二段配置时触发，无影响）

## 6. 工作区文件索引

- `/workspace/dsv4.sh` / `dsv4_prefill.sh` / `dsv4_decode.sh`：部署脚本（standalone / P / D）
- `/workspace/vllm_standalone.log`：standalone 运行日志
- `/workspace/verify_dsv4_multiturn_kv_reuse.py`：非思考模式多轮 KV 复用验证脚本（本文第 4 节结论的复现工具）
- `/workspace/verify_dsv4_thinking_multiturn_kv_reuse.py`：思考模式多轮 KV 复用验证脚本（5 变体对照，本文第 4b 节结论的复现工具）
- `/workspace/verify_qwen3_multiturn_kv_reuse.py`：早期 Qwen3 版本（风格参考）
- `/workspace/mooncake.json`：mooncake 配置（P2PHANDSHAKE，ascend 协议，10GB segment，SSD offload /home/mooncakenvme；PD 场景用）
- `/workspace/Mooncake-a3-ssd/`、`load_balance_proxy_server_example.py`：mooncake/负载均衡相关
- `/workspace/opencode/`：本背景目录
- `/workspace/opencode/dsv4-pd-kv-reuse-analysis-background.md`：**PD 分离跨轮 KV 复用问题分析与修复**（2026-09-13，含 4 个根因、vllm-ascend 三文件修复、思考/非思考两模式端到端 +4096 验证全记录；§4b.6/§4b.9 的旧结论在其 §7 有 PD 拓扑下的修正版）
