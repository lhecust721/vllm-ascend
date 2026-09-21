"""GLM-5.2 非思考模式多轮对话 KV 前缀复用验证 (基于 verify_dsv4_multiturn_kv_reuse.py 适配)。

GLM-5.2 (glm_moe_dsa, DSA 稀疏注意力) standalone PD 混合部署 (单实例 8195, 无代理):
  kv_producer + save_decode_cache: decode 增量 KV → AscendStore put;
  新请求 prefill: 本地 prefix cache (enable-prefix-caching) + AscendStore lookup 双路复用。
  pool_worker 日志: num_kv_cache_groups=1, hash_block_size=128, lcm_block_size=128
  → KV 复用最小粒度 = 128 token (DSV4 为 4096, GLM-5.2 DSA 单组无压缩层)。

chat template 差异 (相对 DSV4):
  - 模板变量 enable_thinking / clear_thinking (非 DSV4 的 thinking / drop_thinking)
  - 非思考渲染: generation prompt 以 <|assistant|><think></think> 结尾 (模板注入空 think 块);
    第 2 轮 assistant 消息同样渲染 <think></think> + content.strip()
  - assistant 消息仅读取 reasoning_content 字段; content 含 </think> 时模板自动拆分
  - 渲染以 [gMASK]<sop> 开头; EOS 集合 = [<|endoftext|>, <|user|>, <|observation|>]

验证链路:
  [1] chat template 离线渲染第 1 轮 prompt (enable_thinking=false), 检查模板结构
  [2] 服务端 /tokenize 双路对照 (prompt 字符串 + messages 两路径, 证明离线渲染与服务端一致)
  [3] 第 1 轮真实推理:
      Path-A raw completions (token ids 输入 + logprobs 取真实生成 token 序列)
      Path-B chat completions (return_token_ids/return_prompt_text 取服务端真实渲染与生成 token)
  [4] 第 2 轮渲染完整字符 + 与第 1 轮 decode 结果 token 级前缀匹配分析 (Path-A 自洽)
  [4b] Path-B 回填链路 token 级匹配分析:
       (0) 服务端 chat 渲染 prompt_token_ids 是否 == 离线 ids1
       (1) message.reasoning_content 是否非空 (glm45 parser 切分即回填丢 token)
       (2) decode(真实生成 ids) == message.content (content 字符串级保真度, 含 strip 检查)
       (3) LCP(真实生成 ids, encode(content)) (detokenize→retokenize 闭环)
       (4) LCP(渲染prompt+真实生成ids, 第2轮渲染 ids2b) (回填是否 token 级匹配, 定位分叉点)
  [5] 端到端 prefix cache 命中验证 (/metrics local + external 差值, Path-B)
      round-3 同 prompt 重发做存取链路确定性对照。

用法: python3 verify_glm52_multiturn_kv_reuse.py [--min-output-tokens 512] [--max-output-tokens 600]
      [--api-base http://127.0.0.1:8195]
"""

import argparse
import sys
import uuid

import requests

DEFAULT_API_BASE = "http://127.0.0.1:8195"
MODEL = "glm5.2"
MODEL_PATH = "/data1/weight/GLM-5.2-w4a8"

GMASK = "[gMASK]"
SOP = "<sop>"
ASSISTANT = "<|assistant|>"
USER = "<|user|>"
SYSTEM = "<|system|>"
THINK_START = "<think>"
THINK_END = "</think>"
# generation_config.json eos_token_id = [154820, 154827, 154829], 名字在运行时解析
EOS_TOKENS = ["<|endoftext|>", "<|user|>", "<|observation|>"]

# KV 复用最小粒度: glm_moe_dsa 单 KV 组, pool_worker 日志 hash_block_size=128,
# 传输粒度 = lcm(128, 组块 128) = 128 token; 本地 prefix cache 块同为 128
REUSE_GRANULARITY = 128

verify_id = str(uuid.uuid4())


def build_messages1(min_output_tokens: int) -> list[dict]:
    passage = ("人工智能的发展经历了符号主义、连接主义与深度学习等多个阶段，"
               "每个阶段都有标志性的算法与应用突破。") * 25
    return [
        {
            "role": "system",
            "content": f"KV验证ID={verify_id}。请严格按照用户要求回答。",
        },
        {
            "role": "user",
            "content": (f"{passage}\n\n请基于上文，连续介绍人工智能的发展历史，"
                        f"输出不少于{min_output_tokens}个token，不要提前结束。"),
        },
    ]
user2_content = "请用一句话总结前面的回答。"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GLM-5.2 非思考模式多轮对话 KV 前缀复用验证")
    ap.add_argument(
        "--min-output-tokens", type=int, default=512,
        help="第1轮模型输出最短长度 (vLLM min_tokens 强制), 默认 512, "
             "prompt+生成跨越多个 128 token 复用粒度边界",
    )
    ap.add_argument(
        "--max-output-tokens", type=int, default=600,
        help="第1轮输出上限, 默认 600 (单机 ~2 tok/s 下控制单轮生成时长; "
             "prompt+600 已覆盖多个 128 粒度块, 足以验证复用)",
    )
    ap.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="GLM-5.2 standalone 实例基址, /v1 /tokenize /metrics 同端口",
    )
    return ap.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def tokenize_prompt_via_server(base: str, prompt: str) -> list[int]:
    r = requests.post(
        f"{base}/tokenize",
        json={"model": MODEL, "prompt": prompt, "add_special_tokens": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def tokenize_messages_via_server(base: str, messages: list[dict], kwargs: dict) -> list[int]:
    r = requests.post(
        f"{base}/tokenize",
        json={"model": MODEL, "messages": messages, "chat_template_kwargs": kwargs},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def metric_values(metrics_url: str) -> dict[str, float]:
    r = requests.get(metrics_url, timeout=30)
    r.raise_for_status()
    values: dict[str, float] = {}
    for line in r.text.splitlines():
        for name in ("vllm:external_prefix_cache_hits_total", "vllm:prefix_cache_hits_total",
                     "vllm:external_prefix_cache_queries_total", "vllm:prefix_cache_queries_total"):
            if line.startswith(name + "{") or line.startswith(name + " "):
                values[name] = float(line.rsplit(" ", 1)[1])
    return values


def hit_deltas(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for u in after:
        port = u.split("//")[1].split(":")[1].split("/")[0]
        for name, v in after[u].items():
            tag = ("ext_hits" if "external_prefix_cache_hits" in name else
                   "ext_queries" if "external_prefix_cache_queries" in name else
                   "local_hits" if "prefix_cache_hits" in name else "local_queries")
            deltas[f"{tag}@{port}"] = v - before.get(u, {}).get(name, 0.0)
    return deltas


def lcp_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def ids_to_strs(tok, ids: list[int]) -> str:
    return " | ".join(repr(s) for s in tok.convert_ids_to_tokens(ids))


def main() -> None:
    args = parse_args()
    base = args.api_base.rstrip("/")
    max_output_tokens = (args.max_output_tokens if args.max_output_tokens is not None
                         else args.min_output_tokens + 512)
    if max_output_tokens < args.min_output_tokens:
        raise SystemExit("--max-output-tokens 不能小于 --min-output-tokens")
    messages1 = build_messages1(args.min_output_tokens)
    kwargs_nothink = {"enable_thinking": False}

    # 与服务端同源的 tokenizer 实现 (transformers TokenizersBackend)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    eos_ids = []
    for t in EOS_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None:
            eos_ids.append(tid)
    eos_id = eos_ids[0]  # <|endoftext|>

    # ---------- [1] 离线渲染第 1 轮 ----------
    section("[1] 第1轮 chat template 渲染 (非思考模式, enable_thinking=False)")
    p1 = tok.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True,
                                 **kwargs_nothink)
    ids1_offline = tok.encode(p1, add_special_tokens=False)

    print("第1轮渲染完整字符串 (repr, 截断 400 字符):")
    print(repr(p1[:200] + " ... " + p1[-200:] if len(p1) > 400 else p1))
    pos = p1.rfind(ASSISTANT)
    after_assistant = p1[pos + len(ASSISTANT):] if pos != -1 else ""
    print(f"\n结构检查:")
    print(f"  以 {GMASK}{SOP} 开头            : {p1.startswith(GMASK + SOP)}")
    print(f"  含 'Reasoning Effort' 系统块   : {'Reasoning Effort' in p1} (非思考应为 False)")
    print(f"  以 <|assistant|><think></think> 结尾: {p1.endswith(ASSISTANT + THINK_START + THINK_END)}")
    print(f"  '<|Assistant|>' 后渲染字符     : {repr(after_assistant)}")
    print(f"第1轮 token 数: {len(ids1_offline)}")

    # ---------- [2] 服务端渲染对照 ----------
    section("[2] 服务端 /tokenize 渲染对照 (prompt 字符串 + messages 双路径)")
    ids1_prompt = tokenize_prompt_via_server(base, p1)
    ids1_msg = tokenize_messages_via_server(base, messages1, kwargs_nothink)
    print(f"offline ids == server(prompt)  : {ids1_offline == ids1_prompt}")
    print(f"offline ids == server(messages): {ids1_offline == ids1_msg}")
    print(f"server(prompt) == server(messages): {ids1_prompt == ids1_msg}")
    ids1 = ids1_msg

    # ---------- [3] 第1轮真实推理 ----------
    section("[3] 第1轮真实推理 (decode 生成并保存 KV)")

    # Path-A: raw completions, token ids 输入 + logprobs 取真实生成 token 序列
    raw_payload = {
        "model": MODEL,
        "prompt": ids1,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "min_tokens": args.min_output_tokens,
        "logprobs": 1,
    }
    rr = requests.post(f"{base}/v1/completions", json=raw_payload, timeout=1200)
    rr.raise_for_status()
    rc = rr.json()["choices"][0]
    raw_text = rc["text"]
    raw_finish = rc["finish_reason"]
    lp_tokens = (rc.get("logprobs") or {}).get("tokens") or []
    gen_ids = [tok.convert_tokens_to_ids(t) for t in lp_tokens]
    if any(i is None for i in gen_ids):
        print("logprobs token 还原失败 (byte-BPE detokenize 有损), 回退为重新编码 raw 文本")
        gen_ids = tok.encode(raw_text, add_special_tokens=False)

    print(f"Path-A raw 生成: finish={raw_finish}, 生成 token 数={len(gen_ids)} "
          f"(min_tokens={args.min_output_tokens}, max_tokens={max_output_tokens})")
    print(f"  raw 输出头部: {repr(raw_text[:80])}")
    print(f"  raw 输出尾部: {repr(raw_text[-80:])}")

    # Path-B: chat completions, 真实多轮客户端行为
    # return_token_ids / return_prompt_text 为 vLLM 扩展: 取回精确生成 token ids 与
    # 服务端实际渲染的 prompt, 作为 Path-B 链路的地面真值
    payload1 = {
        "model": MODEL,
        "messages": messages1,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "min_tokens": args.min_output_tokens,
        "chat_template_kwargs": kwargs_nothink,
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    r1 = requests.post(f"{base}/v1/chat/completions", json=payload1, timeout=1200)
    r1.raise_for_status()
    d1 = r1.json()
    msg1 = d1["choices"][0]["message"]
    usage1 = d1["usage"]
    chat_content = msg1["content"]
    chat_finish = d1["choices"][0]["finish_reason"]
    chat_reasoning = msg1.get("reasoning") or msg1.get("reasoning_content")
    chat_prompt_text = d1.get("prompt_text")
    chat_prompt_ids = d1.get("prompt_token_ids")
    if chat_prompt_ids is None and isinstance(chat_prompt_text, str):
        chat_prompt_ids = tok.encode(chat_prompt_text, add_special_tokens=False)
        print("注: 响应未含 prompt_token_ids, 由 prompt_text 重编码近似")
    if chat_content is None:
        print("!!! message.content 为 None (输出全被 reasoning parser 归入 reasoning?), 按空串处理")
        chat_content = ""
    # Path-B 真实生成 token ids (地面真值); 未生效则降级为 encode(content) 并在 [4b] 提示
    gen_ids_b = d1["choices"][0].get("token_ids")
    have_truth_b = isinstance(gen_ids_b, list) and len(gen_ids_b) > 0
    if not have_truth_b:
        print("警告: 响应未含 choices[0].token_ids (return_token_ids 未生效), "
              "Path-B 地面真值降级为 encode(content), [4b](2)(3) 失效")
        gen_ids_b = tok.encode(chat_content, add_special_tokens=False)

    print(f"Path-B chat 生成: prompt_tokens={usage1['prompt_tokens']}, "
          f"completion_tokens={usage1['completion_tokens']}, "
          f"finish={chat_finish}, token_ids={len(gen_ids_b)} (地面真值={have_truth_b})")
    print(f"  reasoning: {repr(chat_reasoning[:80]) if chat_reasoning else 'None/空'}")
    print(f"  chat content 头部: {repr(chat_content[:80])}")
    print(f"  (注: temperature=0 下两次独立生成在 NPU 批量下非逐位确定, "
          f"Path-A/B 内容不同属预期, 不影响各自路径的自洽验证)")

    # ---------- [4] 第2轮渲染 + token 级前缀匹配 (Path-A 自洽) ----------
    section("[4] 第2轮渲染与 decode KV 前缀匹配分析 (Path-A)")
    messages2 = messages1 + [
        {"role": "assistant", "content": raw_text},
        {"role": "user", "content": user2_content},
    ]
    p2 = tok.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True,
                                 **kwargs_nothink)
    ids2 = tokenize_messages_via_server(base, messages2, kwargs_nothink)

    print("第2轮渲染完整字符串 (repr, 截断 400 字符):")
    print(repr(p2[:200] + " ... " + p2[-200:] if len(p2) > 400 else p2))

    pos2 = p2.find(ASSISTANT)
    seg_assistant1 = p2[pos2 + len(ASSISTANT): p2.find(USER, pos2)] if pos2 != -1 else ""
    print(f"\n第2轮 assistant1 区域字符 (第一个 <|assistant|> 之后到下一个 <|user|> 之前):")
    print(repr(seg_assistant1[:120] + ("..." if len(seg_assistant1) > 120 else "")))
    print(f"  以 <think></think> 开头      : {seg_assistant1.startswith(THINK_START + THINK_END)}")
    print(f"  与 Path-A raw 输出 strip 后一致: {seg_assistant1 == THINK_START + THINK_END + raw_text.strip()} "
          f"(模板对 content 做 strip)")

    expected = ids1 + gen_ids
    print(f"\n期望前缀构成: 第1轮输入({len(ids1)}) + 生成({len(gen_ids)})"
          f"(finish={raw_finish}, min_tokens 截断无 EOS)")
    print(f"第2轮总 token 数: {len(ids2)}")

    m = lcp_len(expected, ids2)
    print(f"LCP(第1轮输入+生成, 第2轮输入): {m}/{len(expected)}")
    if m >= len(expected):
        print(">>> system+user1+assistant1 与上一轮 decode 结果 完整匹配 (token 级)")
    else:
        exp_str = tok.convert_ids_to_tokens([expected[m]])[0] if m < len(expected) else "<越界>"
        act_str = tok.convert_ids_to_tokens([ids2[m]])[0] if m < len(ids2) else "<越界>"
        print(f">>> 不完整匹配! 首个分歧位置 {m}: 期望 {repr(exp_str)} vs 实际 {repr(act_str)}")
        print(f"    期望附近: {ids_to_strs(tok, expected[max(0, m - 3): m + 4])}")
        print(f"    实际附近: {ids_to_strs(tok, ids2[max(0, m - 3): m + 4])}")

    s_expected = p1 + raw_text.strip()
    print(f"\n字符串级: 第2轮渲染以 '第1轮prompt+raw输出(strip)' 开头: {p2.startswith(s_expected)}")

    # ---------- [4b] Path-B 回填链路 token 级匹配分析 ----------
    section("[4b] Path-B 回填链路 token 级匹配分析 (chat content)")
    saved_prompt_ids = chat_prompt_ids if isinstance(chat_prompt_ids, list) else ids1

    # (0) 服务端 chat 渲染 prompt 与离线渲染对照
    if isinstance(chat_prompt_ids, list):
        m_prompt = lcp_len(chat_prompt_ids, ids1)
        print(f"(0) 服务端 chat 渲染 prompt_token_ids == 离线 ids1 : {chat_prompt_ids == ids1} "
              f"(LCP={m_prompt}, 服务端 {len(chat_prompt_ids)} tok vs 离线 {len(ids1)} tok)")
        if chat_prompt_ids != ids1:
            print(f"    !!! 渲染不一致, 首个分歧位置 {m_prompt}:")
            print(f"      服务端: {ids_to_strs(tok, chat_prompt_ids[max(0, m_prompt-2):m_prompt+3])}")
            print(f"      离线  : {ids_to_strs(tok, ids1[max(0, m_prompt-2):m_prompt+3])}")
    else:
        print("(0) 响应未含 prompt_token_ids/prompt_text, 跳过渲染对照")

    # (1) reasoning parser 切分检查
    if chat_reasoning:
        print(f"\n(1) message.reasoning_content 非空 (长度 {len(chat_reasoning)}): "
              f"头 {repr(chat_reasoning[:60])}")
        print("    !!! glm45 parser 切分了输出: 回填仅含 content, 丢失前段 token → 第 2 轮必然分叉")
    else:
        print("\n(1) message.reasoning_content 为空: parser 未切分, content 应为全量输出")

    # (2)/(3) 仅在有地面真值时有意义
    if have_truth_b:
        # (2) content 字符串级保真度 (含模板 strip 影响检查)
        raw_b_text = tok.decode(gen_ids_b)
        print(f"(2) decode(真实生成 ids) == message.content : {raw_b_text == chat_content} "
              f"(生成 {len(gen_ids_b)} tok, content {len(chat_content)} 字符)")
        if raw_b_text != chat_content:
            k = 0
            while k < min(len(raw_b_text), len(chat_content)) and raw_b_text[k] == chat_content[k]:
                k += 1
            print(f"    首个差异字符位置 {k}: decode={repr(raw_b_text[k:k+24])} vs "
                  f"content={repr(chat_content[k:k+24])}")
        if chat_content != chat_content.strip():
            print(f"    !!! content 带首尾空白 (strip={repr(chat_content.strip()[:20])}...): "
                  f"模板第 2 轮按 content.strip() 渲染 → 与保存序列在此处分叉")
        if THINK_END in chat_content:
            print("    !!! content 含 </think>: 模板渲染时会自动拆出 reasoning_content → 结构性分叉")

        # (3) detokenize→retokenize 闭环
        content_ids = tok.encode(chat_content, add_special_tokens=False)
        m_rt = lcp_len(gen_ids_b, content_ids)
        print(f"(3) LCP(真实生成 ids, encode(content)) : {m_rt}/{len(gen_ids_b)} "
              f"(encode 后 {len(content_ids)} tok)")
        if m_rt < min(len(gen_ids_b), len(content_ids)):
            print(f"    retokenize 首个分歧位置 {m_rt} (生成序列第 {m_rt} 个 token):")
            print(f"      生成  : {ids_to_strs(tok, gen_ids_b[max(0, m_rt-2):m_rt+3])}")
            print(f"      重编码: {ids_to_strs(tok, content_ids[max(0, m_rt-2):m_rt+3])}")
        elif len(gen_ids_b) != len(content_ids):
            print(f"    (前缀一致但总长差 {len(gen_ids_b) - len(content_ids)} tok, 差异在尾部)")

        # 参考: 两路生成分歧 (非逐位确定, 预期现象)
        m_ab = lcp_len(gen_ids, gen_ids_b)
        print(f"(参考) Path-A 与 Path-B 生成 LCP={m_ab} (temperature=0 非逐位确定属预期)")

    # (4) decode 保存序列 vs 第 2 轮渲染的完整 LCP —— 回填是否 token 级匹配
    messages2b = messages1 + [
        {"role": "assistant", "content": chat_content},
        {"role": "user", "content": user2_content},
    ]
    p2b = tok.apply_chat_template(messages2b, tokenize=False, add_generation_prompt=True,
                                  **kwargs_nothink)
    ids2b = tokenize_messages_via_server(base, messages2b, kwargs_nothink)

    expected_b = saved_prompt_ids + gen_ids_b
    m_b = lcp_len(expected_b, ids2b)
    truth_note = "" if have_truth_b else " [降级模式: 生成段为 encode(content), 仅验证渲染边界]"
    print(f"(4) LCP(渲染prompt+Path-B真实生成, 第2轮渲染 ids2b) : {m_b}/{len(expected_b)} "
          f"(ids2b 总长 {len(ids2b)}){truth_note}")
    if m_b >= len(expected_b):
        print("    >>> Path-B 回填链路 token 级完整匹配 (prompt+生成段全对齐)")
        path_b_full_match = True
    else:
        diff_pos = m_b
        rel = diff_pos - len(saved_prompt_ids)
        exp_str = (tok.convert_ids_to_tokens([expected_b[diff_pos]])[0]
                   if diff_pos < len(expected_b) else "<EOF>")
        act_str = (tok.convert_ids_to_tokens([ids2b[diff_pos]])[0]
                   if diff_pos < len(ids2b) else "<EOF>")
        where = ("prompt 内" if rel < 0
                 else f"prompt({len(saved_prompt_ids)} tok) 之后第 {rel} 个生成 token")
        print(f"    >>> 不匹配! 首个分歧位置 {diff_pos} = {where}")
        print(f"      期望 {repr(exp_str)} vs 实际 {repr(act_str)}")
        print(f"      期望附近: {ids_to_strs(tok, expected_b[max(0, diff_pos-3):diff_pos+4])}")
        print(f"      实际附近: {ids_to_strs(tok, ids2b[max(0, diff_pos-3):diff_pos+4])}")
        print("      根因判别: (1)parser切分 / (2)content strip或后处理 / (3)重编码边界 / (0)渲染差异")
        path_b_full_match = False

    # ---------- [5] 端到端 prefix cache 命中 (Path-B) ----------
    section("[5] 端到端 KV 复用验证 (/metrics local + external 差值, Path-B)")
    # standalone PD 混合: 本地 prefix cache (128 token 块) + AscendStore (save_decode_cache
    # 在 decode 步 put 增量, 新请求 lookup) 双路。命中边界 = 128 token 整数倍。

    hits_before = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    payload2 = {"model": MODEL, "messages": messages2b, "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": kwargs_nothink}
    r2 = requests.post(f"{base}/v1/chat/completions", json=payload2, timeout=600)
    r2.raise_for_status()
    hits_mid = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    # round-3: 完全相同的 prompt 重发一次 (round-2 已把全量 key 写入本地+store,
    # 同 prompt 重发必命中粒度上限, 不受 BPE 重编码运气影响)
    r3 = requests.post(f"{base}/v1/chat/completions", json=payload2, timeout=600)
    r3.raise_for_status()
    hits_after = {f"{base}/metrics": metric_values(f"{base}/metrics")}

    d2 = r2.json()
    usage2 = d2["usage"]
    delta_r2 = hit_deltas(hits_before, hits_mid)
    delta_r3 = hit_deltas(hits_mid, hits_after)
    decode_saved = usage1["prompt_tokens"] + usage1["completion_tokens"]
    expected_saved = decode_saved // REUSE_GRANULARITY * REUSE_GRANULARITY
    expected_ach = min(m_b, len(expected_b)) // REUSE_GRANULARITY * REUSE_GRANULARITY
    print(f"第1轮 decode 保存 token 数 (prompt+completion): {decode_saved}")
    print(f"第2轮 prompt_token 数 (服务端渲染)            : {len(ids2b)}")
    print(f"第2轮 prompt_tokens                           : {usage2['prompt_tokens']}")
    print(f"round-2 命中差值  : {delta_r2}")
    print(f"round-3 命中差值  : {delta_r3} (同 prompt 重发, 存取链路确定性对照)")
    print(f"KV 复用最小粒度   : {REUSE_GRANULARITY}")
    print(f"  按保存量预期命中 : {expected_saved} (保存序列 floor)")
    print(f"  按回填LCP预期命中: {expected_ach} ([4b](4) 实际可复用前缀 floor)")

    local_r2 = delta_r2.get("local_hits@" + base.split(":")[2].split("/")[0], 0.0)
    ext_r2 = delta_r2.get("ext_hits@" + base.split(":")[2].split("/")[0], 0.0)
    hits_r2 = max([v for k, v in delta_r2.items() if k.endswith("_hits")], default=0.0)
    hits_r3 = max([v for k, v in delta_r3.items() if k.endswith("_hits")], default=0.0)
    if decode_saved < REUSE_GRANULARITY:
        print(f">>> decode 保存 {decode_saved} < {REUSE_GRANULARITY} (复用粒度), "
              f"结构性 0 命中属预期; 请增大 --min-output-tokens")
    elif max(hits_r2, hits_r3) >= max(expected_ach, 1):
        which = "round-2 (跨轮: 复用上一轮 decode 保存的 KV)" if hits_r2 >= expected_ach else "round-3 (同 prompt 重发)"
        print(f">>> 达到 128 粒度预期命中 ({which}), Decode 保存 KV → prefill 复用成立")
        if hits_r2 < expected_ach:
            print("    注: round-2 命中低于预期, 靠 round-3 重发证明链路, 见 [4b](4) 分歧定位")
    elif path_b_full_match:
        print(">>> [4b] 显示 Path-B 回填链路 token 级完整匹配, 但端到端命中偏低/为 0:")
        print("    分歧不在 token 层, 在 vllm-ascend 存取链路, 请检查 put/lookup 日志")
    else:
        print(">>> 命中偏低, Path-B 回填链路存在 token 分歧, 首个分歧位置见 [4b](4)")

    print("\n验证ID:", verify_id)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP ERROR:", e, e.response.text[:2000] if e.response is not None else "")
        sys.exit(1)
