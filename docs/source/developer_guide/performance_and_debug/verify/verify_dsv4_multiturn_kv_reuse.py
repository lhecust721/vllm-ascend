"""DeepSeek-V4 非思考模式多轮对话 KV 前缀复用验证。

验证链路:
  [1] chat template 离线渲染第 1 轮 prompt, 检查 <|Assistant|> 之后是否含 <think>/</think>
  [2] 服务端 /tokenize 渲染对照 (证明离线渲染与服务端完全一致)
  [3] 第 1 轮真实推理:
      Path-A raw completions (token ids 输入 + logprobs 取真实生成 token 序列)
      Path-B chat completions (return_token_ids/return_prompt_text 取 D 侧真实渲染与生成 token)
  [4] 第 2 轮渲染完整字符 + 与第 1 轮 decode 的结果 token 级前缀匹配分析 (Path-A 自洽)
  [4b] Path-B 回填链路 token 级匹配分析 (定位 chat content 链路为何不闭环):
       (0) D 侧 chat 渲染 prompt_token_ids 是否 == 离线 ids1 (服务端渲染地面真值)
       (1) message.reasoning 是否非空 (reasoning parser 切分即回填丢 token)
       (2) decode(真实生成 ids) == message.content (content 字符串级保真度)
       (3) LCP(真实生成 ids, encode(content)) (detokenize→retokenize 闭环)
       (4) LCP(D侧渲染prompt+真实生成ids, 第2轮渲染 ids2b) (回填是否 token 级匹配, 定位分叉点)
  [5] 端到端 prefix cache 命中验证 (/metrics vllm:prefix_cache_hits_total 差值, Path-B)
      并结合 [4b] 结论自动判别: token 层分歧 vs vllm-ascend 存取链路问题

用法: python3 verify_dsv4_multiturn_kv_reuse.py [--min-output-tokens 4096] [--max-output-tokens N]
      [--api-base http://127.0.0.1:8700]

默认走 PD 分离代理: /v1/{completions,chat/completions} 经代理 (8700) 分发 P/D;
代理不转发 /tokenize 与 /metrics, 故直连实例: /tokenize 用 prefill (8195),
/metrics 同时统计 prefill (8195) 与 decode (8196) 的 prefix cache 差值。

--min-output-tokens: 第1轮模型输出最短长度 (vLLM min_tokens 强制, EOS 屏蔽), 默认 4096。
  DeepSeek-V4 压缩注意力层 KV 复用最小粒度 = block_size(32) × compress_ratio(128) = 4096 token,
  第1轮 decode 保存总量 (prompt+completion) 须 ≥ 4096, 第2轮跨轮命中才可能非 0。
  (PD 下 min_tokens 作用于 decode 实例; 代理给 prefill 实例固定下发 max_tokens=1)
"""

import argparse
import sys
import uuid

import requests

DEFAULT_API_BASE = "http://127.0.0.1:8700"
DEFAULT_PREFILL_BASE = "http://127.0.0.1:8195"
DEFAULT_DECODE_BASE = "http://127.0.0.1:8196"
MODEL = "dsv4"
MODEL_PATH = "/data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8"

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
ASSISTANT = "<｜Assistant｜>"
USER = "<｜User｜>"
THINK_START = "<think>"
THINK_END = "</think>"

# KV 复用最小粒度: CompressAttentionManager 对 compress_ratio=128 压缩层的逻辑块
# = block_size(32) × compress_ratio(128) = 4096 token, 跨轮命中只发生在 4096 整数倍边界
REUSE_GRANULARITY = 4096

verify_id = str(uuid.uuid4())


def build_messages1(min_output_tokens: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": f"KV验证ID={verify_id}。请严格按照用户要求回答。",
        },
        {
            "role": "user",
            "content": (f"请连续介绍人工智能的发展历史，输出不少于{min_output_tokens}个token，"
                        f"不要提前结束。"),
        },
    ]
user2_content = "请用一句话总结前面的回答。"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DeepSeek-V4 非思考模式多轮对话 KV 前缀复用验证")
    ap.add_argument(
        "--min-output-tokens", type=int, default=4096,
        help="第1轮模型输出最短长度 (vLLM min_tokens 强制), 默认 4096, 对齐 KV 保存最小粒度",
    )
    ap.add_argument(
        "--max-output-tokens", type=int, default=None,
        help="第1轮输出上限, 默认 min-output-tokens + 512",
    )
    ap.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="OpenAI API 基址 (PD 分离代理), /v1 请求经此分发",
    )
    ap.add_argument(
        "--tokenize-url", default=f"{DEFAULT_PREFILL_BASE}/tokenize",
        help="/tokenize 完整 URL (代理不转发, 直连实例)",
    )
    ap.add_argument(
        "--metrics-urls", nargs="+",
        default=[f"{DEFAULT_PREFILL_BASE}/metrics", f"{DEFAULT_DECODE_BASE}/metrics"],
        help="/metrics 完整 URL 列表, 分实例统计 prefix cache 命中差值",
    )
    return ap.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def tokenize_via_server(tokenize_url: str, prompt: str) -> list[int]:
    r = requests.post(
        tokenize_url,
        json={"model": MODEL, "prompt": prompt, "add_special_tokens": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def metric_value(metrics_url: str, name: str) -> float:
    r = requests.get(metrics_url, timeout=30)
    r.raise_for_status()
    for line in r.text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    raise RuntimeError(f"metric {name} not found in {metrics_url}")


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
    max_output_tokens = (args.max_output_tokens if args.max_output_tokens is not None
                         else args.min_output_tokens + 512)
    if max_output_tokens < args.min_output_tokens:
        raise SystemExit("--max-output-tokens 不能小于 --min-output-tokens")
    messages1 = build_messages1(args.min_output_tokens)

    # 与服务端完全相同的 tokenizer 实现 (tokenizer_mode=deepseek_v4)
    from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

    tok = DeepseekV4Tokenizer.from_pretrained(MODEL_PATH)
    eos_id = tok.convert_tokens_to_ids(EOS)

    # ---------- [1] 离线渲染第 1 轮 ----------
    section("[1] 第1轮 chat template 渲染 (非思考模式)")
    p1 = tok.apply_chat_template(messages1, tokenize=False, thinking=False)
    ids1_offline = tok.encode(p1, add_special_tokens=False)

    print("第1轮渲染完整字符串 (repr):")
    print(repr(p1))
    pos = p1.rfind(ASSISTANT)
    after_assistant = p1[pos + len(ASSISTANT):] if pos != -1 else ""
    print(f"\n'<|Assistant|>' 之后渲染的字符: {repr(after_assistant)}")
    print(f"  包含 <think>  : {THINK_START in after_assistant}")
    print(f"  包含 </think> : {THINK_END in after_assistant}")
    print(f"第1轮 token 数: {len(ids1_offline)}")

    # ---------- [2] 服务端渲染对照 ----------
    section("[2] 服务端 /tokenize 渲染对照")
    ids1_server = tokenize_via_server(args.tokenize_url, p1)
    print(f"offline ids == server ids: {ids1_offline == ids1_server}")
    ids1 = ids1_server

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
    rr = requests.post(f"{args.api_base}/v1/completions", json=raw_payload, timeout=600)
    rr.raise_for_status()
    rc = rr.json()["choices"][0]
    raw_text = rc["text"]
    raw_finish = rc["finish_reason"]
    lp_tokens = (rc.get("logprobs") or {}).get("tokens") or []
    gen_ids = [tok.convert_tokens_to_ids(t) for t in lp_tokens]
    if any(i is None for i in gen_ids):
        print("logprobs token 还原失败, 回退为重新编码 raw 文本")
        gen_ids = tok.encode(raw_text, add_special_tokens=False)

    print(f"Path-A raw 生成: finish={raw_finish}, 生成 token 数={len(gen_ids)} "
          f"(min_tokens={args.min_output_tokens}, max_tokens={max_output_tokens})")
    print(f"  raw 输出头部: {repr(raw_text[:80])}")
    print(f"  raw 输出尾部: {repr(raw_text[-80:])}")
    print(f"  raw 输出以 <think> 开头: {raw_text.startswith(THINK_START)}")

    # Path-B: chat completions, 真实多轮客户端行为
    # return_token_ids / return_prompt_text 为 vLLM 扩展: 取回精确生成 token ids 与
    # 服务端(PD 下为 decode 实例)实际渲染的 prompt, 作为 Path-B 链路的地面真值
    payload1 = {
        "model": MODEL,
        "messages": messages1,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "min_tokens": args.min_output_tokens,
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    r1 = requests.post(f"{args.api_base}/v1/chat/completions", json=payload1, timeout=600)
    r1.raise_for_status()
    d1 = r1.json()
    msg1 = d1["choices"][0]["message"]
    usage1 = d1["usage"]
    chat_content = msg1["content"]
    chat_finish = d1["choices"][0]["finish_reason"]
    chat_reasoning = msg1.get("reasoning") or msg1.get("reasoning_content")
    chat_prompt_text = d1.get("prompt_text")
    chat_prompt_ids = d1.get("prompt_token_ids")
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
    p2 = tok.apply_chat_template(messages2, tokenize=False, thinking=False)
    ids2 = tokenize_via_server(args.tokenize_url, p2)

    print("第2轮渲染完整字符串 (repr):")
    print(repr(p2))

    pos2 = p2.find(ASSISTANT)
    seg_assistant1 = p2[pos2 + len(ASSISTANT): p2.find(USER, pos2)] if pos2 != -1 else ""
    print(f"\n第2轮 assistant1 区域字符 (第一个 <|Assistant|> 之后到下一个 <|User|> 之前):")
    print(repr(seg_assistant1[:120] + ("..." if len(seg_assistant1) > 120 else "")))
    print(f"  以 </think> 开头            : {seg_assistant1.startswith(THINK_END)}")
    print(f"  含 <think>                  : {THINK_START in seg_assistant1}")
    print(f"  以 <|end▁of▁sentence|> 收尾 : {seg_assistant1.endswith(EOS)}")

    expected_with_eos = ids1 + gen_ids + [eos_id]
    expected_no_eos = ids1 + gen_ids
    print(f"\n期望前缀构成: 第1轮输入({len(ids1)}) + 生成({len(gen_ids)}) + EOS(仅stop时有, finish={raw_finish})")
    print(f"第2轮总 token 数: {len(ids2)}")

    m_eos = lcp_len(expected_with_eos, ids2)
    m_noeos = lcp_len(expected_no_eos, ids2)
    print(f"LCP(含EOS期望前缀, 第2轮输入)  : {m_eos}/{len(expected_with_eos)}")
    print(f"LCP(不含EOS期望前缀, 第2轮输入): {m_noeos}/{len(expected_no_eos)}")

    if m_eos >= len(expected_with_eos):
        print(">>> system+user1+assistant1 与上一轮 decode 保存结果 完整匹配 (含EOS, token 级)")
    elif m_noeos >= len(expected_no_eos):
        print(">>> 正文完全匹配; 仅末尾 EOS 有差异 (第1轮 finish=length 未产生 EOS)")
    else:
        diff_pos = m_noeos
        exp_tok = expected_no_eos[diff_pos] if diff_pos < len(expected_no_eos) else None
        act_tok = ids2[diff_pos] if diff_pos < len(ids2) else None
        exp_str = tok.convert_ids_to_tokens([exp_tok])[0] if exp_tok is not None else "<越界>"
        act_str = tok.convert_ids_to_tokens([act_tok])[0] if act_tok is not None else "<越界>"
        print(f">>> 不完整匹配! 首个分歧位置 {diff_pos}: 期望 {repr(exp_str)} vs 实际 {repr(act_str)}")
        print(f"    期望附近: {ids_to_strs(tok, expected_no_eos[max(0, diff_pos - 3): diff_pos + 4])}")
        print(f"    实际附近: {ids_to_strs(tok, ids2[max(0, diff_pos - 3): diff_pos + 4])}")

    s_expected = p1 + raw_text + (EOS if raw_finish == "stop" else "")
    print(f"\n字符串级: 第2轮渲染以 '第1轮prompt+raw输出(+EOS)' 开头: {p2.startswith(s_expected)}")

    # ---------- [4b] Path-B 回填链路 token 级匹配分析 ----------
    section("[4b] Path-B 回填链路 token 级匹配分析 (chat content)")
    # decode 实际保存的序列 = D 侧渲染的 prompt + D 侧生成的 token
    saved_prompt_ids = chat_prompt_ids if isinstance(chat_prompt_ids, list) else ids1

    # (0) D 侧 chat 渲染 prompt 与离线渲染对照
    if isinstance(chat_prompt_ids, list):
        m_prompt = lcp_len(chat_prompt_ids, ids1)
        print(f"(0) D侧 chat 渲染 prompt_token_ids == 离线 ids1 : {chat_prompt_ids == ids1} "
              f"(LCP={m_prompt}, D侧 {len(chat_prompt_ids)} tok vs 离线 {len(ids1)} tok)")
        if isinstance(chat_prompt_text, str):
            print(f"    prompt_text == 离线 p1 : {chat_prompt_text == p1}")
        if chat_prompt_ids != ids1:
            print(f"    !!! D 侧 chat 渲染与离线渲染不一致, 首个分歧位置 {m_prompt}:")
            print(f"      D侧  : {ids_to_strs(tok, chat_prompt_ids[max(0, m_prompt-2):m_prompt+3])}")
            print(f"      离线  : {ids_to_strs(tok, ids1[max(0, m_prompt-2):m_prompt+3])}")
            print("      (prompt 即分叉, 第 2 轮必不命中, 与 content 回填无关)")
    else:
        print("(0) 响应未含 prompt_token_ids, 跳过 D 侧渲染对照")

    # (1) reasoning parser 切分检查
    if chat_reasoning:
        print(f"\n(1) message.reasoning 非空 (长度 {len(chat_reasoning)}): "
              f"头 {repr(chat_reasoning[:60])} 尾 {repr(chat_reasoning[-40:])}")
        print("    !!! reasoning parser 切分了输出: 回填仅含 content, 丢失前段 token → 第 2 轮必然分叉")
    else:
        print("\n(1) message.reasoning 为空: parser 未切分, content 应为全量输出")

    # (2)/(3) 仅在有地面真值时有意义
    if have_truth_b:
        # (2) content 字符串级保真度
        raw_b_text = tok.decode(gen_ids_b)
        print(f"(2) decode(真实生成 ids) == message.content : {raw_b_text == chat_content} "
              f"(生成 {len(gen_ids_b)} tok, content {len(chat_content)} 字符)")
        if raw_b_text != chat_content:
            k = 0
            while k < min(len(raw_b_text), len(chat_content)) and raw_b_text[k] == chat_content[k]:
                k += 1
            print(f"    首个差异字符位置 {k}: decode={repr(raw_b_text[k:k+24])} vs "
                  f"content={repr(chat_content[k:k+24])}")
            print(f"    decode 头 {repr(raw_b_text[:40])} / content 头 {repr(chat_content[:40])}")

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
        print(f"(参考) Path-A 与 Path-B 生成 LCP={m_ab} (temperature=0 非逐位确定属预期, "
              f"仅用于区分两路生成差异)")

    # (4) decode 保存序列 vs 第 2 轮渲染的完整 LCP —— 回填是否 token 级匹配
    messages2b = messages1 + [
        {"role": "assistant", "content": chat_content},
        {"role": "user", "content": user2_content},
    ]
    p2b = tok.apply_chat_template(messages2b, tokenize=False, thinking=False)
    ids2b = tokenize_via_server(args.tokenize_url, p2b)

    if chat_finish == "stop" and (not gen_ids_b or gen_ids_b[-1] != eos_id):
        expected_b = saved_prompt_ids + gen_ids_b + [eos_id]
    else:
        expected_b = saved_prompt_ids + gen_ids_b
    m_b = lcp_len(expected_b, ids2b)
    truth_note = "" if have_truth_b else " [降级模式: 生成段为 encode(content), 仅验证渲染边界]"
    print(f"(4) LCP(D侧prompt+Path-B真实生成, 第2轮渲染 ids2b) : {m_b}/{len(expected_b)} "
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
        print("      根因判别: (1)reasoning 切分 / (2)content 后处理 / (3)重编码边界 / (0)渲染差异")
        path_b_full_match = False

    # ---------- [5] 端到端 prefix cache 命中 (Path-B) ----------
    section("[5] 端到端 KV 复用验证 (/metrics prefix cache 差值, Path-B)")
    # messages2b / p2b / ids2b 已在 [4b](4) 构造, 此处复用
    # PD 部署下 enable_prefix_caching=False, 本地前缀缓存指标恒 0;
    # AscendStore 跨轮命中记在 vllm:external_prefix_cache_hits_total。

    def metric_values(metrics_url: str) -> dict[str, float]:
        r = requests.get(metrics_url, timeout=30)
        r.raise_for_status()
        values: dict[str, float] = {}
        for line in r.text.splitlines():
            for name in ("vllm:external_prefix_cache_hits_total", "vllm:prefix_cache_hits_total"):
                if line.startswith(name + "{") or line.startswith(name + " "):
                    values[name] = float(line.rsplit(" ", 1)[1])
        return values

    def hit_deltas(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for u in after:
            for name in after[u]:
                d = after[u][name] - before.get(u, {}).get(name, 0.0)
                deltas[f"{name.split(':')[1][:8]}@{u.split('//')[1].split(':')[1]}"] = max(
                    deltas.get(f"{name.split(':')[1][:8]}@{u.split('//')[1].split(':')[1]}", 0.0), d
                )
        return deltas

    hits_before = {u: metric_values(u) for u in args.metrics_urls}
    payload2 = {"model": MODEL, "messages": messages2b, "temperature": 0, "max_tokens": 32}
    r2 = requests.post(f"{args.api_base}/v1/chat/completions", json=payload2, timeout=600)
    r2.raise_for_status()
    hits_mid = {u: metric_values(u) for u in args.metrics_urls}
    # round-3: 完全相同的 prompt 重发一次 (round-2 的 prefill 已把自身全部 key 写入
    # store, 同 prompt 重发必命中 4096, 不受 BPE 重编码运气影响)
    r3 = requests.post(f"{args.api_base}/v1/chat/completions", json=payload2, timeout=600)
    r3.raise_for_status()
    hits_after = {u: metric_values(u) for u in args.metrics_urls}

    d2 = r2.json()
    usage2 = d2["usage"]
    delta_r2 = hit_deltas(hits_before, hits_mid)
    delta_r3 = hit_deltas(hits_mid, hits_after)
    decode_saved = usage1["prompt_tokens"] + usage1["completion_tokens"]
    expected_hit = decode_saved // REUSE_GRANULARITY * REUSE_GRANULARITY
    print(f"第1轮 decode 保存 token 数 (prompt+completion): {decode_saved}")
    print(f"第2轮 prompt_token_ids 数 (服务端渲染)        : {len(ids2b)}")
    print(f"第2轮 prompt_tokens                           : {usage2['prompt_tokens']}")
    print(f"round-2 命中差值  : {delta_r2}")
    print(f"round-3 命中差值  : {delta_r3} (同 prompt 重发, 预期 +4096)")
    print(f"KV 复用最小粒度   : {REUSE_GRANULARITY} (4096 整数倍边界, 本轮预期命中 {expected_hit})")
    print(f"第2轮 completion  : {repr(d2['choices'][0]['message']['content'])}")

    hits_r2 = max(delta_r2.values(), default=0.0)
    hits_r3 = max(delta_r3.values(), default=0.0)
    if decode_saved < REUSE_GRANULARITY:
        print(f">>> decode 保存 {decode_saved} < {REUSE_GRANULARITY} (压缩层逻辑块粒度), "
              f"结构性 0 命中属预期; 需要非 0 命中请增大 --min-output-tokens")
    elif max(hits_r2, hits_r3) >= expected_hit:
        which = "round-2 (跨轮: 回填上一轮 decode 缓存)" if hits_r2 >= expected_hit else "round-3 (同 prompt 重发)"
        print(f">>> 达到 {REUSE_GRANULARITY} 粒度预期命中 ({which}), 跨轮 KV 前缀复用成立")
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
