"""DeepSeek-V4 思考模式多轮对话 KV 前缀复用验证 (v4, 修正指标测量逻辑)。

v4 相对 v3 的修正:
  - [3b] 是 turn-1 之后**第一个**共享 decode 前缀的推理请求 → 其 ext 命中差值
    是 AscendStore 跨轮复用的直接证据。v3 的 `before` 快照取在 [3b] 之后,
    导致 [3b] 的命中被吞入基线, turn-2B/3B 只显示 local 命中, 判定错误。
  - 判定逻辑改为分层: [3b] ext (AscendStore 跨轮) → turn-2B local+ext (KV 复用)
    → turn-2A 全 0 (对照组) → turn-3B local+ext (缓存持续)
  - 同时报告 local 和 ext, 不再只看 ext

v3 相对 v2 的适配 (PD 拓扑 + 地面真值 + external 指标):
  - 拓扑: /v1 请求经 PD 代理 (8700) 分发 (真实服务路径, P prefill / D decode);
    /tokenize 直连 prefill 实例 (8195, 代理不转发); /metrics 同时采集 P (8195) 与 D (8196)
  - 地面真值: 第 1 轮 chat 请求带 return_token_ids + return_prompt_text, 直接取回
    D 侧真实渲染 prompt_token_ids 与精确生成 token ids (v2 用 raw 文本重编码构造
    期望序列, 受 BPE 重编码边界歧义影响, 见非思考模式脚本 [4b](3) 的实测教训)
  - [5] 命中指标: PD 部署 --no-enable-prefix-caching, vllm:prefix_cache_hits_total
    恒 0; 改用 vllm:external_prefix_cache_hits_total (AscendStore 跨轮命中, 记在
    prefill 实例)。并增加 turn-3 同 prompt 重发: turn-2 prefill 已写全量 key,
    重发必 +4096, 不受回填 token 分叉运气影响 (存取链路确定性验证)

思考模式链路 (背景 §4b):
  turn-1 渲染以 <｜Assistant｜><think> 结尾 (<think> 为模板注入, 不在生成输出中);
  decode 保存序列 = prompt(以<think>结尾) + reasoning + </think> + content (+EOS);
  跨轮复用需三条件同时满足: thinking=true + drop_thinking=false + assistant 消息带
  reasoning 字段 (或 reasoning_content, 真实 chat API 有归一化)。
  默认 drop_thinking=True 只渲染 content, 分叉点在 <｜Assistant｜> 之后 → 命中 0。

用法: python3 verify_dsv4_thinking_multiturn_kv_reuse.py
"""

import json
import sys
import uuid

import requests

PROXY_BASE = "http://127.0.0.1:8700"
TOKENIZE_URL = "http://127.0.0.1:8195/tokenize"
METRICS_URLS = ["http://127.0.0.1:8195/metrics", "http://127.0.0.1:8196/metrics"]
MODEL = "dsv4"
MODEL_PATH = "/data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8"

EOS = "<｜end▁of▁sentence｜>"
ASSISTANT = "<｜Assistant｜>"
USER = "<｜User｜>"
THINK_START = "<think>"
THINK_END = "</think>"

# KV 复用最小粒度: c128 压缩层逻辑块 = block_size(32) × compress_ratio(128) = 4096
REUSE_GRANULARITY = 4096

verify_id = str(uuid.uuid4())

user2_content = "针对你说的局限性，有什么改进思路？"


def section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def tokenize_chat_via_server(messages: list[dict], chat_template_kwargs: dict) -> list[int]:
    r = requests.post(
        TOKENIZE_URL,
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
        for name in ("vllm:external_prefix_cache_hits_total", "vllm:prefix_cache_hits_total"):
            if line.startswith(name + "{") or line.startswith(name + " "):
                values[name] = float(line.rsplit(" ", 1)[1])
    return values


def fmt_deltas(before: dict, after: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for u in after:
        port = u.split("//")[1].split(":")[1].split("/")[0]
        for name, v in after[u].items():
            key = f"{'ext' if 'external' in name else 'local'}@{port}"
            out[key] = v - before.get(u, {}).get(name, 0.0)
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
    messages2: list[dict],
    kwargs: dict,
    expected: list[int],
) -> str:
    p2 = tok.apply_chat_template(messages2, tokenize=False, **kwargs)
    ids2_offline = tok.encode(p2, add_special_tokens=False)
    ids2_server = tokenize_chat_via_server(messages2, kwargs)
    ids2 = ids2_server

    m = lcp_len(expected, ids2)
    status = "完整匹配" if m >= len(expected) else f"匹配到 {m}/{len(expected)}"
    hit_pred = (f"分叉点 {m} {'≥' if m >= REUSE_GRANULARITY else '<'} {REUSE_GRANULARITY} → "
                f"预期命中 {m // REUSE_GRANULARITY * REUSE_GRANULARITY if m >= REUSE_GRANULARITY else 0}")
    print(f"\n--- {name} ---")
    print(f"offline==server(/tokenize@P): {ids2_offline == ids2_server}, 第2轮 token 数: {len(ids2)}")
    print(f"LCP vs turn-1 decode 序列: {m}/{len(expected)}  → {status}")
    print(f"  {hit_pred}")
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
    from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

    tok = DeepseekV4Tokenizer.from_pretrained(MODEL_PATH)
    eos_id = tok.convert_tokens_to_ids(EOS)
    kwargs_thinking = {"thinking": True}
    kwargs_drop_false = {"thinking": True, "drop_thinking": False}

    # ---------- [1] 思考模式第 1 轮渲染 (长输入, 保存序列将跨越 4096) ----------
    section("[1] 思考模式第 1 轮渲染 (长输入, thinking=true)")
    passage = "深度学习通过多层神经网络自动学习数据的多层次表示，广泛应用于计算机视觉、自然语言处理与语音识别等领域。" * 155
    messages1 = [
        {"role": "system", "content": f"KV验证ID={verify_id}。请严格按照用户要求回答。"},
        {"role": "user",
         "content": passage + "\n\n请基于上文连续介绍人工智能的发展历史，输出至少400个token，不要提前结束。"},
    ]
    p1 = tok.apply_chat_template(messages1, tokenize=False, **kwargs_thinking)
    ids1 = tokenize_chat_via_server(messages1, kwargs_thinking)
    print(f"渲染尾部 120 字符: {repr(p1[-120:])}")
    print(f"以 <think> 结尾(模板注入, 生成输出不含): {p1.endswith(THINK_START)}")
    print(f"offline==server(/tokenize@P): {tok.encode(p1, add_special_tokens=False) == ids1}, "
          f"token 数: {len(ids1)}")

    # 强制保存序列跨越 4096: min_tokens 保证 prompt+生成 ≥ 4245
    min_tokens = max(800, 4245 - len(ids1))
    max_tokens = min_tokens + 3000

    # ---------- [2] 第 1 轮真实生成 (chat via 代理; 地面真值 return_token_ids) ----------
    section("[2] 第 1 轮真实生成 (chat via PD 代理, thinking=true, 地面真值)")
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
    r1 = requests.post(f"{PROXY_BASE}/v1/chat/completions", json=payload1, timeout=1200)
    r1.raise_for_status()
    d1 = r1.json()
    msg1 = d1["choices"][0]["message"]
    usage1 = d1["usage"]
    finish1 = d1["choices"][0]["finish_reason"]
    reasoning = msg1.get("reasoning") or ""
    content = msg1.get("content") or ""
    d_prompt_ids = d1.get("prompt_token_ids")
    gen_ids = d1["choices"][0].get("token_ids")

    print(f"message 字段列表: {sorted(msg1.keys())}")
    print(f"finish={finish1}, usage: prompt={usage1['prompt_tokens']}, completion={usage1['completion_tokens']}")
    print(f"reasoning 头 80 字符: {repr(reasoning[:80])}")
    print(f"content   头 80 字符: {repr(content[:80])}")
    print(f"响应含 reasoning 字段(响应侧属性名): {'reasoning' in msg1}, "
          f"含 reasoning_content: {'reasoning_content' in msg1}")

    have_truth = isinstance(gen_ids, list) and len(gen_ids) > 0
    if not have_truth:
        print("警告: 响应未含 token_ids, 期望序列降级为 encode(reasoning+</think>+content)")
        gen_ids = tok.encode(reasoning + THINK_END + content, add_special_tokens=False)
    if not isinstance(d_prompt_ids, list):
        print("警告: 响应未含 prompt_token_ids, 使用 /tokenize@P 的 ids1 作为 D 侧渲染")
        d_prompt_ids = ids1

    print(f"token_ids(地面真值)={have_truth}: 生成 {len(gen_ids)} tok; "
          f"D侧渲染 prompt {len(d_prompt_ids)} tok == /tokenize@P: {d_prompt_ids == ids1}")

    # parser 切分保真: 原始生成 = reasoning + </think> + content (+ 尾部 EOS)
    raw_b = tok.decode(gen_ids)
    cand = reasoning + THINK_END + content
    split_ok = raw_b == cand or raw_b == cand + EOS
    print(f"decode(真实生成 ids) == reasoning+'</think>'+content(+EOS) : {split_ok}"
          f"{'' if split_ok else ' (解析差异, 见下方 LCP)'}")

    # turn-1 decode 保存序列 (地面真值)。注意: return_token_ids 的 token_ids
    # 已含 stop 时的收尾 EOS, 仅在缺失时补 (避免重复追加)。
    if finish1 == "stop" and (not gen_ids or gen_ids[-1] != eos_id):
        expected = d_prompt_ids + gen_ids + [eos_id]
    else:
        expected = d_prompt_ids + gen_ids
    saved = len(expected)
    print(f"turn-1 decode 保存序列 = {saved} tokens "
          f"(prompt {len(d_prompt_ids)} + 生成 {len(gen_ids)}"
          f"{' + EOS' if finish1 == 'stop' else ', 截断无EOS'}) → 预期命中 "
          f"{saved // REUSE_GRANULARITY * REUSE_GRANULARITY if saved >= REUSE_GRANULARITY else 0}")
    if content is None or content == "":
        raise SystemExit("content 为空 (思考被 max_tokens 截断), 请增大 max_tokens 后重试")

    # ---------- [3] 第 2 轮变体渲染与 decode KV 前缀匹配 ----------
    section("[3] 第 2 轮变体渲染与 token 级匹配 (对照地面真值)")
    messages2_base = messages1 + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": user2_content},
    ]

    # V-A: 默认 drop_thinking=True, 仅回传 content
    analyze_variant("V-A: 默认(drop_thinking=True) + 仅content",
                    tok, messages2_base, kwargs_thinking, expected)

    # V-E: 默认 drop_thinking=True + reasoning 字段 (证明默认丢弃)
    v_e = [dict(m) for m in messages2_base]
    v_e[2] = {"role": "assistant", "content": content, "reasoning": reasoning}
    analyze_variant("V-E: 默认(drop_thinking=True) + reasoning字段",
                    tok, v_e, kwargs_thinking, expected)

    # V-C: drop_thinking=False, 仅回传 content (空思考)
    p_c = analyze_variant("V-C: drop_thinking=False + 仅content",
                          tok, messages2_base, kwargs_drop_false, expected)

    # V-B: drop_thinking=False + reasoning 字段 (完整回传思考)
    v_b = [dict(m) for m in messages2_base]
    v_b[2] = {"role": "assistant", "content": content, "reasoning": reasoning}
    p_b = analyze_variant("V-B: drop_thinking=False + reasoning字段 (完整回传)",
                          tok, v_b, kwargs_drop_false, expected)

    # V-D: drop_thinking=False + reasoning_content 字段
    # offline 与 /tokenize 路径无归一化 → 预期与 V-C 相同; 真实 chat API 有归一化 → 见 [3b]
    v_d = [dict(m) for m in messages2_base]
    v_d[2] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    p_d = analyze_variant("V-D: drop_thinking=False + reasoning_content字段 (offline//tokenize路径)",
                          tok, v_d, kwargs_drop_false, expected)
    print(f"\nV-D 渲染 == V-C(仅content) 渲染? {p_d == p_c}  (offline//tokenize 无归一化 → True)")
    print(f"V-D 渲染 == V-B(reasoning)  渲染? {p_d == p_b}")

    # ---------- [3b] 真实 chat API 归一化实证 + 首次跨轮 AscendStore 查询 ----------
    section("[3b] 真实 chat API: reasoning_content 归一化 + 首次跨轮 AscendStore 查询")
    # 此请求是 turn-1 之后第一个共享 decode 保存前缀的推理请求。
    # P 本地缓存此时仅有 turn-1 的 prompt（< 4096 → c128 无缓存块）,
    # 因此 P 必须查 AscendStore → ext 命中即证明跨轮复用成立。
    baseline = {u: metric_values(u) for u in METRICS_URLS}
    r2d = requests.post(
        f"{PROXY_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": v_d,
            "temperature": 0,
            "max_tokens": 8,
            "return_prompt_text": True,
            "chat_template_kwargs": kwargs_drop_false,
        },
        timeout=600,
    )
    r2d.raise_for_status()
    after_3b = {u: metric_values(u) for u in METRICS_URLS}
    d_3b = fmt_deltas(baseline, after_3b)
    prompt_text_d = r2d.json().get("prompt_text")
    u3b = r2d.json().get("usage", {})
    print(f"chat API 返回 prompt_text: {'有' if prompt_text_d else '无'}")
    print(f"prompt_tokens={u3b.get('prompt_tokens', 'N/A')}")
    print(f"AscendStore 命中差值: {d_3b}  ← [3b] 是首次跨轮查询, ext 命中即跨轮复用成立")
    if prompt_text_d:
        ids_pd = tok.encode(prompt_text_d, add_special_tokens=False)
        m_pd = lcp_len(expected, ids_pd)
        print(f"prompt_text 渲染 LCP vs decode 序列: {m_pd}/{len(expected)} "
              f"({'≥4096 可命中' if m_pd >= REUSE_GRANULARITY else '<4096 命中0'})")
        print(f"prompt_text == V-B 离线渲染? {prompt_text_d == p_b}  (True 即归一化生效)")
        print(f"prompt_text 头 160 字符: {repr(prompt_text_d[:160])}")

    # ---------- [4] V-B 第 2 轮渲染 assistant1 区域 ----------
    section("[4] V-B (drop_thinking=False + reasoning) 渲染 assistant1 区域")
    pos = p_b.find(ASSISTANT)
    seg = p_b[pos + len(ASSISTANT): p_b.find(USER, pos)] if pos != -1 else ""
    print(repr(seg[:400] + ("..." if len(seg) > 400 else "")))

    # ---------- [5] 端到端 KV 复用验证 (local + external 指标差值) ----------
    section("[5] 端到端 KV 复用验证 (/metrics local + external 指标差值)")
    # 注意: [3b] 已经 prefill 过 turn-2B 的相同 prompt → P 本地缓存已含 4096 tok。
    # turn-2B/turn-3B 的命中将走本地缓存（local）而非 AscendStore（ext）。
    # AscendStore 跨轮复用的判定依据是 [3b] 的 ext 差值（上方已输出）。

    # turn-2B: 回传 reasoning + drop_thinking=False → 本地缓存命中（[3b] 预填充）
    messages2BL = v_b
    r2B = requests.post(
        f"{PROXY_BASE}/v1/chat/completions",
        json={"model": MODEL, "messages": messages2BL, "temperature": 0, "max_tokens": 16,
              "chat_template_kwargs": kwargs_drop_false},
        timeout=600,
    )
    r2B.raise_for_status()
    mid = {u: metric_values(u) for u in METRICS_URLS}
    u2B = r2B.json()["usage"]
    d2B = fmt_deltas(after_3b, mid)

    # turn-2A: 默认 drop_thinking=True (丢弃思考) → 分叉点 < 4096 → 期望全 0
    messages2AL = messages2_base
    r2A = requests.post(
        f"{PROXY_BASE}/v1/chat/completions",
        json={"model": MODEL, "messages": messages2AL, "temperature": 0, "max_tokens": 16,
              "chat_template_kwargs": kwargs_thinking},
        timeout=600,
    )
    r2A.raise_for_status()
    mid2 = {u: metric_values(u) for u in METRICS_URLS}
    u2A = r2A.json()["usage"]
    d2A = fmt_deltas(mid, mid2)

    # turn-3B: 同 2B prompt 重发 → 本地缓存命中（turn-2B prefill 已写入）
    r3B = requests.post(
        f"{PROXY_BASE}/v1/chat/completions",
        json={"model": MODEL, "messages": messages2BL, "temperature": 0, "max_tokens": 16,
              "chat_template_kwargs": kwargs_drop_false},
        timeout=600,
    )
    r3B.raise_for_status()
    after = {u: metric_values(u) for u in METRICS_URLS}
    d3B = fmt_deltas(mid2, after)

    print(f"turn-1 保存序列: {saved} tokens (预期命中 "
          f"{saved // REUSE_GRANULARITY * REUSE_GRANULARITY if saved >= REUSE_GRANULARITY else 0})")
    print(f"[3b]  首次跨轮查询(V-D):                 命中差值={d_3b}")
    print(f"        ↑ 这是 AscendStore 跨轮复用的直接证据 (ext 命中)")
    print(f"turn-2B (回传reasoning+drop=false):       prompt={u2B['prompt_tokens']}, "
          f"命中差值={d2B}")
    print(f"        ↑ 预期 local 命中 ([3b] 已预填充本地缓存)")
    print(f"turn-2A (默认丢弃思考):                   prompt={u2A['prompt_tokens']}, "
          f"命中差值={d2A}")
    print(f"        ↑ 预期全 0 (分叉点 < 4096)")
    print(f"turn-3B (同2B重发):                       命中差值={d3B}")
    print(f"        ↑ 预期 local 命中 (turn-2B prefill 已写入)")

    # 分层判定
    ext_3b = d_3b.get("ext@8195", 0.0)
    local_3b = d_3b.get("local@8195", 0.0)
    ext_2b = d2B.get("ext@8195", 0.0)
    local_2b = d2B.get("local@8195", 0.0)
    total_2b = ext_2b + local_2b
    ext_2a = d2A.get("ext@8195", 0.0)
    local_2a = d2A.get("local@8195", 0.0)
    ext_3br = d3B.get("ext@8195", 0.0)
    local_3br = d3B.get("local@8195", 0.0)

    print("\n结论要点:")
    # 主判定: [3b] 的 AscendStore 命中
    if saved >= REUSE_GRANULARITY and ext_3b >= REUSE_GRANULARITY:
        print(f"  1) ✓ AscendStore 跨轮 KV 复用成立: [3b] 首次跨轮查询 ext 命中 "
              f"{ext_3b:.0f} tokens ({REUSE_GRANULARITY} 粒度)")
        print(f"     (turn-1 decode 保存 → [3b] prefill 从 AscendStore 加载)")
    elif saved >= REUSE_GRANULARITY and local_3b >= REUSE_GRANULARITY:
        print(f"  1) ⚠ [3b] 命中 local {local_3b:.0f} + ext {ext_3b:.0f}: "
              f"本地缓存抢先覆盖了 AscendStore 的查询范围")
        print(f"     (P 开启 prefix-caching 时, [3b] 的本地命中来自 turn-1 的 prompt)")
    else:
        print(f"  1) ✗ [3b] 未命中 (local={local_3b:.0f}, ext={ext_3b:.0f}): "
              f"请检查 P 侧 lookup 日志和 D 侧 put 日志")

    # 辅助判定: turn-2B/2A/3B
    if total_2b >= REUSE_GRANULARITY:
        src = "local" if local_2b >= REUSE_GRANULARITY else "ext"
        print(f"  2) ✓ turn-2B KV 复用 {total_2b:.0f} tokens (路径: {src}, "
              f"{'[3b] 预填充本地缓存' if local_2b >= REUSE_GRANULARITY else 'AscendStore'})")
    else:
        print(f"  2) ✗ turn-2B 未命中 (local={local_2b:.0f}, ext={ext_2b:.0f})")

    if ext_2a == 0 and local_2a == 0:
        print(f"  3) ✓ turn-2A 对照组正确: 默认 drop_thinking 丢弃思考 → 命中 0")
    else:
        print(f"  3) ⚠ turn-2A 命中 local={local_2a:.0f} ext={ext_2a:.0f} (非预期, 请检查)")

    total_3br = ext_3br + local_3br
    if total_3br >= REUSE_GRANULARITY:
        print(f"  4) ✓ turn-3B 同 prompt 重发命中 {total_3br:.0f} (缓存持续有效)")
    else:
        print(f"  4) ✗ turn-3B 未命中 (local={local_3br:.0f}, ext={ext_3br:.0f})")

    print(f"  5) 命中粒度 {REUSE_GRANULARITY} (c128 压缩层逻辑块)")
    print("  6) 响应侧属性名: message.reasoning; 请求侧 reasoning_content 会被真实 chat API 归一化"
          " (见 [3b]), offline//tokenize 路径无归一化 (见 [3] V-D)")

    print("\n验证ID:", verify_id)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP ERROR:", e, e.response.text[:2000] if e.response is not None else "")
        sys.exit(1)
