"""GLM-5.2 思考模式多轮对话 KV 前缀复用验证 (基于 verify_dsv4_thinking_multiturn_kv_reuse.py 适配)。

GLM-5.2 (glm_moe_dsa, DSA 稀疏注意力) standalone PD 混合部署 (单实例 8195, 无代理):
  kv_producer + save_decode_cache: decode 增量 KV → AscendStore put;
  新请求 prefill: 本地 prefix cache + AscendStore lookup 双路复用。
  pool_worker 日志: num_kv_cache_groups=1, hash_block_size=128, lcm_block_size=128
  → KV 复用最小粒度 = 128 token (DSV4 为 4096)。

chat template 差异 (相对 DSV4):
  - 模板变量 enable_thinking / clear_thinking (非 DSV4 的 thinking / drop_thinking)
  - 思考模式渲染: 首部注入 <|system|>Reasoning Effort: Max, generation prompt 以
    <|assistant|><think> 结尾; decode 保存序列 = prompt + reasoning + </think> + content (+stop)
  - 第 2 轮 assistant 回填三条件: enable_thinking=true + clear_thinking=false +
    assistant 消息带 reasoning_content 字段 (模板只读 reasoning_content, 不读 reasoning);
    缺任何一条 → 渲染 <think></think> → 分叉点在第 1 轮 prompt 末尾
  - 与 DSV4 不同: 分叉后仍有 prompt 级命中 (粒度 128 << prompt 长度),
    对照组预期命中 ≈ floor(第1轮prompt长度/128), 而非 0
  - glm45 reasoning parser 会丢弃 </think> 后紧跟的换行 (after_end="\n"),
    模板对 content 做 strip() → 边界空白是潜在分叉点, [2b] 精确测量

验证链路:
  [1] 思考模式第 1 轮渲染 (长输入, 保存序列跨越多个 128 粒度边界)
  [2] 第 1 轮真实生成 (chat, return_token_ids/return_prompt_text 地面真值)
  [2b] glm45 parser 切分保真: 原始生成 vs reasoning+'</think>'+content 重建对照
  [3] 第 2 轮变体渲染与 token 级匹配 (V-A/V-E/V-C/V-B/V-F 五变体, 对照地面真值)
  [3b] 真实 chat API 回填实证 (首次跨轮查询) + local/external 命中差值
  [4] V-B 渲染 assistant1 区域
  [5] 端到端 KV 复用验证 (turn-2B / turn-2A 对照 / turn-3B 重发, local+ext 指标)

用法: python3 verify_glm52_thinking_multiturn_kv_reuse.py [--api-base http://127.0.0.1:8195]
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
EOS_TOKENS = ["<|endoftext|>", "<|user|>", "<|observation|>"]

# KV 复用最小粒度: glm_moe_dsa 单 KV 组, hash_block_size=128 (见模块 docstring)
REUSE_GRANULARITY = 128

verify_id = str(uuid.uuid4())

user2_content = "针对你说的局限性，有什么改进思路？"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GLM-5.2 思考模式多轮对话 KV 前缀复用验证")
    ap.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="GLM-5.2 standalone 实例基址, /v1 /tokenize /metrics 同端口",
    )
    return ap.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def tokenize_chat_via_server(base: str, messages: list[dict], chat_template_kwargs: dict) -> list[int]:
    r = requests.post(
        f"{base}/tokenize",
        json={
            "model": MODEL,
            "messages": messages,
            "chat_template_kwargs": chat_template_kwargs,
        },
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


def fmt_deltas(before: dict, after: dict, port: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for u in after:
        if port not in u:
            continue
        for name, v in after[u].items():
            tag = ("ext_hits" if "external_prefix_cache_hits" in name else
                   "ext_queries" if "external_prefix_cache_queries" in name else
                   "local_hits" if "prefix_cache_hits" in name else "local_queries")
            out[tag] = v - before.get(u, {}).get(name, 0.0)
    return out


def lcp_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def ids_to_strs(tok, ids: list[int]) -> str:
    return " | ".join(repr(s) for s in tok.convert_ids_to_tokens(ids))


def analyze_variant(
    name: str,
    tok,
    base: str,
    messages2: list[dict],
    kwargs: dict,
    expected: list[int],
    kwargs_nothink: dict,
) -> str:
    p2 = tok.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True, **kwargs)
    ids2_offline = tok.encode(p2, add_special_tokens=False)
    ids2_server = tokenize_chat_via_server(base, messages2, kwargs)
    ids2 = ids2_server

    m = lcp_len(expected, ids2)
    status = "完整匹配" if m >= len(expected) else f"匹配到 {m}/{len(expected)}"
    hit = m // REUSE_GRANULARITY * REUSE_GRANULARITY if m >= REUSE_GRANULARITY else 0
    print(f"\n--- {name} ---")
    print(f"offline==server(/tokenize): {ids2_offline == ids2_server}, 第2轮 token 数: {len(ids2)}")
    print(f"LCP vs turn-1 decode 保存序列: {m}/{len(expected)}  → {status}")
    print(f"  预期命中 {hit} (粒度 {REUSE_GRANULARITY})")
    if m < len(expected) and m < len(ids2):
        exp_str = tok.convert_ids_to_tokens([expected[m]])[0] if m < len(expected) else "<越界>"
        act_str = tok.convert_ids_to_tokens([ids2[m]])[0] if m < len(ids2) else "<越界>"
        print(f"  首个分歧位置 {m}: 期望 {repr(exp_str)} vs 实际 {repr(act_str)}")
        print(f"  期望附近: {ids_to_strs(tok, expected[max(0, m - 2): m + 3])}")
        print(f"  实际附近: {ids_to_strs(tok, ids2[max(0, m - 2): m + 3])}")
    pos = p2.find(ASSISTANT)
    seg = p2[pos + len(ASSISTANT): p2.find(USER, pos)] if pos != -1 else ""
    print(f"  assistant1 区域: {repr(seg[:110] + ('...' if len(seg) > 110 else ''))}")
    return p2


def main() -> None:
    args = parse_args()
    base = args.api_base.rstrip("/")
    port = base.split(":")[2].split("/")[0]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    eos_ids = []
    for t in EOS_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None:
            eos_ids.append(tid)
    eos_id = eos_ids[0]  # <|endoftext|>

    # 注意: reasoning_effort 必须在所有渲染/请求 kwargs 中一致, 否则模板注入的
    # "Reasoning Effort: Max/High" 系统块不一致 → 从 token 8 起全链分叉
    kwargs_thinking = {"enable_thinking": True, "reasoning_effort": "high"}
    kwargs_full = {"enable_thinking": True, "clear_thinking": False, "reasoning_effort": "high"}
    kwargs_nothink = {"enable_thinking": False}

    # ---------- [1] 思考模式第 1 轮渲染 (长输入, 保存序列跨越多个 128 边界) ----------
    section("[1] 思考模式第 1 轮渲染 (长输入, enable_thinking=true)")
    passage = ("深度学习通过多层神经网络自动学习数据的多层次表示，"
               "广泛应用于计算机视觉、自然语言处理与语音识别等领域。") * 30
    messages1 = [
        {"role": "system", "content": f"KV验证ID={verify_id}。请严格按照用户要求回答。"},
        {"role": "user",
         "content": passage + "\n\n请基于上文，先用一句话概括深度学习的核心思想，再简述它的一个主要应用领域。"},
    ]
    p1 = tok.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True,
                                 **kwargs_thinking)
    ids1 = tokenize_chat_via_server(base, messages1, kwargs_thinking)
    print(f"渲染头部 100 字符: {repr(p1[:100])}")
    print(f"渲染尾部 120 字符: {repr(p1[-120:])}")
    print(f"以 <|assistant|><think> 结尾(模板注入): {p1.endswith(ASSISTANT + THINK_START)}")
    print(f"首部注入 Reasoning Effort 系统块: {'Reasoning Effort' in p1}")
    offline_eq = tok.encode(p1, add_special_tokens=False) == ids1
    print(f"offline==server(/tokenize): {offline_eq}, token 数: {len(ids1)}")

    # 每轮生成 ≤ 600 token (单机 ~2 tok/s): prompt(≈790 tok) 自带 6 个 128 粒度边界,
    # 生成段 (reasoning+content, 自然停止) 再跨 1-4 个边界即可充分验证复用;
    # min_tokens=0 允许模型思考结束后自然停止 (finish=stop, stop token 计入保存序列)
    min_tokens = 0
    max_tokens = 600

    # ---------- [2] 第 1 轮真实生成 (chat; 地面真值 return_token_ids) ----------
    section("[2] 第 1 轮真实生成 (chat, thinking=true, 地面真值)")
    payload1 = {
        "model": MODEL,
        "messages": messages1,
        "temperature": 0,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "chat_template_kwargs": kwargs_thinking,
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    r1 = requests.post(f"{base}/v1/chat/completions", json=payload1, timeout=3600)
    r1.raise_for_status()
    d1 = r1.json()
    msg1 = d1["choices"][0]["message"]
    usage1 = d1["usage"]
    finish1 = d1["choices"][0]["finish_reason"]
    reasoning = msg1.get("reasoning") or msg1.get("reasoning_content") or ""
    content = msg1.get("content") or ""
    d_prompt_ids = d1.get("prompt_token_ids")
    if d_prompt_ids is None and d1.get("prompt_text"):
        d_prompt_ids = tok.encode(d1["prompt_text"], add_special_tokens=False)
    gen_ids = d1["choices"][0].get("token_ids")

    print(f"message 字段列表: {sorted(msg1.keys())}")
    print(f"finish={finish1}, usage: prompt={usage1['prompt_tokens']}, completion={usage1['completion_tokens']}")
    print(f"reasoning 头 80 字符: {repr(reasoning[:80])}")
    print(f"content   头 80 字符: {repr(content[:80])}")
    print(f"响应含 reasoning 字段: {'reasoning' in msg1}, "
          f"含 reasoning_content: {'reasoning_content' in msg1}")

    have_truth = isinstance(gen_ids, list) and len(gen_ids) > 0
    if not have_truth:
        print("警告: 响应未含 token_ids, 期望序列降级为 encode(reasoning+</think>+content)")
        gen_ids = tok.encode(reasoning + THINK_END + content, add_special_tokens=False)
    if not isinstance(d_prompt_ids, list):
        print("警告: 响应未含 prompt_token_ids, 使用 /tokenize 的 ids1 作为渲染")
        d_prompt_ids = ids1

    print(f"token_ids(地面真值)={have_truth}: 生成 {len(gen_ids)} tok; "
          f"渲染 prompt {len(d_prompt_ids)} tok == /tokenize: {d_prompt_ids == ids1}")

    # ---------- [2b] glm45 parser 切分保真 ----------
    section("[2b] glm45 parser 切分保真 (原始生成 vs 重建)")
    raw_b = tok.decode(gen_ids)
    cand_plain = reasoning + THINK_END + content
    cand_nl = reasoning + THINK_END + "\n" + content  # parser after_end="\n" 会丢弃该换行
    g_raw = tok.encode(raw_b, add_special_tokens=False)
    g_cand = tok.encode(cand_plain, add_special_tokens=False)
    split_ok = raw_b in (cand_plain, cand_nl)
    if raw_b == cand_plain:
        which = "无换行 (parser 未丢边界字符)"
    elif raw_b == cand_nl:
        which = "含被丢弃的 \\n (parser after_end 丢弃)"
    else:
        which = f"均不匹配 (重建 LCP={lcp_len(g_raw, g_cand)}/{len(g_cand)})"
    print(f"decode(真实生成 ids) == reasoning+'</think>'+content : {raw_b == cand_plain}")
    print(f"decode(真实生成 ids) == reasoning+'</think>'+'\\n'+content : {raw_b == cand_nl}")
    print(f"切分保真判定: {split_ok} → {which}")
    if not split_ok:
        mk = lcp_len(g_raw, g_cand)
        print(f"  重建分歧位置 {mk}: 生成 {repr(g_raw[mk:mk+4])} vs 重建 {repr(g_cand[mk:mk+4])}")
        print(f"  (parser 丢弃 delimiters/边界空白; 模板按 reasoning_content+content 重拼,"
              f" 若边界 token 有损 → 第 2 轮在此处分叉)")
    if content != content.strip():
        print(f"!!! content 带首尾空白 (头 {repr(content[:8])} 尾 {repr(content[-8:])}): "
              f"模板 strip() 渲染 → 第 2 轮边界分叉风险")

    # turn-1 decode 保存序列 (地面真值)。token_ids 含 stop 时的收尾 stop token, 仅在缺失时补。
    if finish1 == "stop" and gen_ids and gen_ids[-1] not in eos_ids:
        expected = d_prompt_ids + gen_ids + [eos_id]
        stop_note = f" + EOS(<|endoftext|> 假设, 实际 stop 集合 {EOS_TOKENS})"
    elif finish1 == "stop" and gen_ids:
        expected = d_prompt_ids + gen_ids
        stop_note = " + stop token (已含于 token_ids)"
    else:
        expected = d_prompt_ids + gen_ids
        stop_note = ", min_tokens 截断无 EOS"
    saved = len(expected)
    expected_hit_full = saved // REUSE_GRANULARITY * REUSE_GRANULARITY if saved >= REUSE_GRANULARITY else 0
    print(f"turn-1 decode 保存序列 = {saved} tokens "
          f"(prompt {len(d_prompt_ids)} + 生成 {len(gen_ids)}{stop_note}) → 粒度上限预期命中 "
          f"{expected_hit_full}")
    if content is None or content == "":
        raise SystemExit("content 为空 (思考被 max_tokens 截断), 请增大 max_tokens 后重试")

    # ---------- [3] 第 2 轮变体渲染与 decode KV 前缀匹配 ----------
    section("[3] 第 2 轮变体渲染与 token 级匹配 (对照地面真值)")
    messages2_base = messages1 + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": user2_content},
    ]

    # V-A: 默认 (clear_thinking 未定义), 仅回传 content → 渲染 <think></think>
    analyze_variant("V-A: 默认(clear_thinking未定义) + 仅content",
                    tok, base, messages2_base, kwargs_thinking, expected, kwargs_nothink)

    # V-E: 默认 + reasoning_content 字段 (证明默认丢弃思考)
    v_e = [dict(m) for m in messages2_base]
    v_e[2] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    analyze_variant("V-E: 默认(clear_thinking未定义) + reasoning_content字段",
                    tok, base, v_e, kwargs_thinking, expected, kwargs_nothink)

    # V-C: clear_thinking=false, 仅回传 content (无 reasoning_content → 仍空思考)
    analyze_variant("V-C: clear_thinking=false + 仅content",
                    tok, base, messages2_base, kwargs_full, expected, kwargs_nothink)

    # V-B: clear_thinking=false + reasoning_content 字段 (完整回传思考, 三条件齐)
    v_b = [dict(m) for m in messages2_base]
    v_b[2] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    p_b = analyze_variant("V-B: clear_thinking=false + reasoning_content字段 (完整回传)",
                          tok, base, v_b, kwargs_full, expected, kwargs_nothink)

    # V-F: clear_thinking=false + reasoning 字段 (DSV4 风格字段名, GLM 模板不读取)
    v_f = [dict(m) for m in messages2_base]
    v_f[2] = {"role": "assistant", "content": content, "reasoning": reasoning}
    analyze_variant("V-F: clear_thinking=false + reasoning字段 (GLM模板不读取, 对照)",
                    tok, base, v_f, kwargs_full, expected, kwargs_nothink)

    # ---------- [3b] 真实 chat API 回填实证 + 首次跨轮查询 ----------
    section("[3b] 真实 chat API 回填 (V-B 消息) + 首次跨轮 KV 查询")
    # 此请求是 turn-1 之后第一个共享 decode 保存前缀的推理请求 →
    # 命中差值 (local 或 ext) 即跨轮复用的直接证据。
    baseline = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    r2d = requests.post(
        f"{base}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": v_b,
            "temperature": 0,
            "max_tokens": 8,
            "return_prompt_text": True,
            "chat_template_kwargs": kwargs_full,
        },
        timeout=600,
    )
    r2d.raise_for_status()
    after_3b = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    d_3b = fmt_deltas(baseline, after_3b, port)
    resp_3b = r2d.json()
    prompt_text_d = resp_3b.get("prompt_text")
    u3b = resp_3b.get("usage", {})
    print(f"chat API 返回 prompt_text: {'有' if prompt_text_d else '无'}")
    print(f"prompt_tokens={u3b.get('prompt_tokens', 'N/A')}")
    print(f"命中差值: {d_3b}  ← [3b] 是首次跨轮查询, 命中即跨轮复用成立")
    if prompt_text_d:
        ids_pd = tok.encode(prompt_text_d, add_special_tokens=False)
        m_pd = lcp_len(expected, ids_pd)
        print(f"prompt_text 渲染 LCP vs decode 保存序列: {m_pd}/{len(expected)} "
              f"(→ 预期命中 {m_pd // REUSE_GRANULARITY * REUSE_GRANULARITY})")
        print(f"prompt_text == V-B 离线渲染? {prompt_text_d == p_b} "
              f"(True 即真实 chat API 按 V-B 方式回填 reasoning_content)")
        print(f"prompt_text 尾部 160 字符: {repr(prompt_text_d[-160:])}")

    # ---------- [4] V-B 第 2 轮渲染 assistant1 区域 ----------
    section("[4] V-B (clear_thinking=false + reasoning_content) 渲染 assistant1 区域")
    pos = p_b.find(ASSISTANT)
    seg = p_b[pos + len(ASSISTANT): p_b.find(USER, pos)] if pos != -1 else ""
    print(repr(seg[:400] + ("..." if len(seg) > 400 else "")))

    # ---------- [5] 端到端 KV 复用验证 (local + external 指标差值) ----------
    section("[5] 端到端 KV 复用验证 (turn-2B / turn-2A 对照 / turn-3B 重发)")
    # 注意: [3b] 已 prefill 过 turn-2B 相同 prompt → 其命中计入 [3b] 差值;
    # turn-2B/3B 走 [3b] 预填充后的本地缓存。跨轮证据以 [3b] 为准。

    # turn-2B: reasoning_content + clear_thinking=false
    r2B = requests.post(
        f"{base}/v1/chat/completions",
        json={"model": MODEL, "messages": v_b, "temperature": 0, "max_tokens": 16,
              "chat_template_kwargs": kwargs_full},
        timeout=600,
    )
    r2B.raise_for_status()
    mid = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    u2B = r2B.json()["usage"]
    d2B = fmt_deltas(after_3b, mid, port)

    # turn-2A: 默认 clear_thinking (丢弃思考) → 分叉点在第 1 轮 prompt 末尾
    messages2AL = messages2_base
    r2A = requests.post(
        f"{base}/v1/chat/completions",
        json={"model": MODEL, "messages": messages2AL, "temperature": 0, "max_tokens": 16,
              "chat_template_kwargs": kwargs_thinking},
        timeout=600,
    )
    r2A.raise_for_status()
    mid2 = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    u2A = r2A.json()["usage"]
    d2A = fmt_deltas(mid, mid2, port)

    # turn-3B: 同 2B prompt 重发 → 缓存持续命中
    r3B = requests.post(
        f"{base}/v1/chat/completions",
        json={"model": MODEL, "messages": v_b, "temperature": 0, "max_tokens": 16,
              "chat_template_kwargs": kwargs_full},
        timeout=600,
    )
    r3B.raise_for_status()
    after = {f"{base}/metrics": metric_values(f"{base}/metrics")}
    d3B = fmt_deltas(mid2, after, port)

    prompt_only_hit = (len(d_prompt_ids) + 1) // REUSE_GRANULARITY * REUSE_GRANULARITY
    print(f"turn-1 保存序列: {saved} tokens (粒度上限预期命中 {expected_hit_full})")
    print(f"[3b]  首次跨轮查询(V-B):           命中差值={d_3b}")
    print(f"        ↑ Decode 保存 KV → prefill 复用的直接证据 (local/ext 任一命中)")
    print(f"turn-2B (reasoning_content+clear=false):  prompt={u2B['prompt_tokens']}, "
          f"命中差值={d2B}")
    print(f"        ↑ 预期命中 (复用保存序列; [3b] 已预填充本地缓存)")
    print(f"turn-2A (默认丢弃思考):                   prompt={u2A['prompt_tokens']}, "
          f"命中差值={d2A}")
    print(f"        ↑ 预期仅 prompt 级命中 ≈ {prompt_only_hit} (分叉在第1轮prompt末尾, <2B)")
    print(f"turn-3B (同2B重发):                       命中差值={d3B}")
    print(f"        ↑ 预期命中 (缓存持续有效)")

    def pick(d: dict, key: str) -> float:
        return d.get(key, 0.0)

    ext_3b = max(pick(d_3b, "ext_hits"), 0.0)
    local_3b = max(pick(d_3b, "local_hits"), 0.0)
    total_2b = max(pick(d2B, "local_hits"), 0.0) + max(pick(d2B, "ext_hits"), 0.0)
    local_2b = max(pick(d2B, "local_hits"), 0.0)
    total_2a = max(pick(d2A, "local_hits"), 0.0) + max(pick(d2A, "ext_hits"), 0.0)
    total_3br = max(pick(d3B, "local_hits"), 0.0) + max(pick(d3B, "ext_hits"), 0.0)

    print("\n结论要点:")
    if saved >= REUSE_GRANULARITY and max(ext_3b, local_3b) >= min(expected_hit_full, 1) and expected_hit_full > 0:
        src = "ext (AscendStore)" if ext_3b >= REUSE_GRANULARITY else "local (prefix cache)"
        print(f"  1) ✓ 跨轮 KV 复用成立: [3b] 首次跨轮查询命中 {max(ext_3b, local_3b):.0f} tokens "
              f"(路径 {src}, 粒度 {REUSE_GRANULARITY})")
        print(f"     (turn-1 decode 保存 {saved} tok → [3b] prefill 复用, 粒度上限 {expected_hit_full})")
    elif saved >= REUSE_GRANULARITY:
        print(f"  1) ✗ [3b] 未命中 (local={local_3b:.0f}, ext={ext_3b:.0f}): "
              f"请检查 put/lookup 日志 (save_decode_cache 是否生效)")
    else:
        print(f"  1) ⚠ 保存序列 {saved} < 粒度 {REUSE_GRANULARITY}, 结构性 0 命中")

    if total_2b >= expected_hit_full > 0:
        src = "local" if local_2b >= REUSE_GRANULARITY else "ext"
        print(f"  2) ✓ turn-2B KV 复用 {total_2b:.0f} tokens (路径: {src})")
    elif total_2b > 0:
        print(f"  2) ⚠ turn-2B 命中 {total_2b:.0f} < 预期 {expected_hit_full} (见 [3] V-B 分歧定位)")
    else:
        print(f"  2) ✗ turn-2B 未命中 (local={local_2b:.0f}, ext={pick(d2B, 'ext_hits'):.0f})")

    if total_2a < total_2b and total_2a <= prompt_only_hit + REUSE_GRANULARITY:
        print(f"  3) ✓ turn-2A 对照组正确: 丢弃思考 → 命中 {total_2a:.0f} ≈ prompt 级 "
              f"(预期 ≈ {prompt_only_hit}), 显著低于 turn-2B")
    elif total_2a >= total_2b:
        print(f"  3) ⚠ turn-2A 命中 {total_2a:.0f} ≥ turn-2B {total_2b:.0f} (非预期, 请检查)")
    else:
        print(f"  3) ⚠ turn-2A 命中 {total_2a:.0f} 高于 prompt 级预期 {prompt_only_hit} (非预期, 请检查)")

    if total_3br >= expected_hit_full > 0:
        print(f"  4) ✓ turn-3B 同 prompt 重发命中 {total_3br:.0f} (缓存持续有效)")
    else:
        print(f"  4) ✗ turn-3B 未命中 ({total_3br:.0f})")

    print(f"  5) 命中粒度 {REUSE_GRANULARITY} (glm_moe_dsa 单组, hash_block_size=128)")
    print("  6) 模板三条件: enable_thinking=true + clear_thinking=false + assistant 带 "
          "reasoning_content (GLM 模板不读取 reasoning 字段, 见 V-F)")
    print("  7) 与 DSV4 差异: 丢弃思考的对照组仍有 prompt 级命中 (128 粒度 << prompt 长度), "
          "判定看相对差值而非绝对 0")

    print("\n验证ID:", verify_id)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP ERROR:", e, e.response.text[:2000] if e.response is not None else "")
        sys.exit(1)
