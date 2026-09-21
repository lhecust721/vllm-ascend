"""DeepSeek-V4 PD 分离多轮 prefix-cache 分叉离线重放分析。

输入: ais_bench details.jsonl (需经 return_token_ids=True 采集, 见
      perf_dsv4_prefixcache_multiturn.py 与 VLLMCustomAPIChat.parse_stream_response
      的改动; 每条请求的 extra_details_data 含 prompt_token_ids / gen_token_ids /
      response_id)。

重放逻辑 (对照跨轮 KV 复用的真实语义):
  decode 侧保存序列 (turn k)  = prompt_token_ids[k] + gen_token_ids[k] (+EOS 如有)
  prefill 侧查询序列 (turn k+1) = prompt_token_ids[k+1]  (服务端真实渲染, 非离线重编码)
  LCP(保存序列, 查询序列) = 跨轮可复用前缀; 实际命中 = LCP // 4096 * 4096
                             (c128 压缩层逻辑块粒度, 不足 4096 对齐归零)

对每个相邻轮次对输出: 分叉位置 (绝对/相对上一轮 prompt 边界)、分叉点前后 detokenize
文本 (decode 侧原始 token 与 prefill 重编码 token 的 ids/串/文本三重对照)、对齐后命中;
汇总输出分布直方图、命中率损失估算、几十个典型分叉样例 (BPE 重编码分叉优先),
并可选关联 P 侧 "KV pool lookup MISS" 日志 (按 response_id = vllm request_id 匹配)。

用法:
  python3 analyze_prefixcache_divergence.py <details.jsonl> \
      [--model-path /data/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8] \
      [--out report.md] [--granularity 4096] [--top 10] \
      [--prefill-log /workspace/vllm_prefill.log]
"""

import argparse
import json
import re
import sys
from collections import defaultdict

MODEL_PATH_DEFAULT = "/data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8"


def lcp_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def toks_strs(tok, ids):
    if tok is None:
        return str(ids)
    return " | ".join(repr(s) for s in tok.convert_ids_to_tokens(ids))


def tok_text(tok, ids, maxlen=240):
    """detokenize 一段 token id 为可读文本 (换行折叠, 截断保护)。"""
    if tok is None or not ids:
        return "<n/a>"
    try:
        s = tok.decode(ids)
    except Exception:  # noqa: BLE001
        return "<decode失败>"
    s = s.replace("\r", "").replace("\n", "⏎")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


_GPT2_BYTE_DECODER = None


def _get_byte_decoder():
    """GPT-2 风格 byte-level BPE 的 字符->字节 反查表 (惰性构建)。"""
    global _GPT2_BYTE_DECODER
    if _GPT2_BYTE_DECODER is None:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _GPT2_BYTE_DECODER = {chr(c): b for b, c in zip(bs, cs)}
    return _GPT2_BYTE_DECODER


def _utf8_prefix(data: bytes):
    """返回 (最长可 UTF-8 解码的前缀文本, 末尾不完整字节)。"""
    for keep in range(len(data), max(len(data) - 4, -1), -1):
        try:
            return data[:keep].decode("utf-8"), data[keep:]
        except UnicodeDecodeError:
            continue
    return "", data


def toks_boundary_text(tok, ids, maxlen=200):
    """按 token 边界显示可读文本: '神经‖网络'。

    - '‖'  = 边界落在完整字符之间 (U+2016, 避免与文本中的字面 '|' 混淆)
    - '¦'  = 边界切在多字节字符中间 (被切字符归入后一段显示)
    - 特殊 token (byte 字母表之外的 added token, 如 <｜end▁of▁sentence｜>)
      以词面原样展示, 天然自帶分隔视觉效果
    """
    if tok is None or not ids:
        return "<n/a>"
    bd = _get_byte_decoder()
    try:
        toks = tok.convert_ids_to_tokens(list(ids))
    except Exception:  # noqa: BLE001
        return "<n/a>"
    out = ""
    sep = ""
    pending = b""
    for idx, t in enumerate(toks):
        if not t:
            continue
        if idx > 0:
            out += sep
        if all(ch in bd for ch in t):
            pending += bytes(bd[ch] for ch in t)
            text, pending = _utf8_prefix(pending)
            out += text
        else:
            text, pending = _utf8_prefix(pending)
            out += text + t
        sep = "¦" if pending else "‖"
    out = out.replace("\r", "").replace("\n", "⏎")
    return out[:maxlen] + ("…" if len(out) > maxlen else "")


def load_details(path):
    convs = defaultdict(list)
    skipped = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        extra = d.get("extra_details_data") or {}
        p_ids = extra.get("prompt_token_ids")
        g_ids = extra.get("gen_token_ids")
        if not isinstance(p_ids, list) or not isinstance(g_ids, list):
            skipped += 1
            continue
        convs[d["uuid"]].append(
            dict(
                turn_id=d["turn_id"],
                prompt_ids=p_ids,
                gen_ids=g_ids,
                response_id=extra.get("response_id"),
                input_tokens=d.get("input_tokens"),
                output_tokens=d.get("output_tokens"),
            )
        )
    for v in convs.values():
        v.sort(key=lambda x: x["turn_id"])
    return convs, skipped


def parse_miss_lines(prefill_log):
    """解析 vllm-ascend 的 'KV pool lookup MISS' 日志行 (P/standalone 侧)。

    vLLM 内部 request_id 可能在 chatcmpl id 后追加唯一后缀
    (如 chatcmpl-xxx-bbe47fde), 因此匹配时按前缀关联。
    """
    out = {}
    if not prefill_log:
        return out
    pat = re.compile(
        r"KV pool lookup MISS req=([^ ]+) prompt=(\d+) hbm=(\d+) queried=(\d+) "
        r"hit=(\d+) miss_at=(\d+) g0_chunk=(-?\d+) g0_hash=(\S+)"
    )
    for raw in open(prefill_log, "rb").read().decode("utf-8", "ignore").splitlines():
        m = pat.search(raw)
        if m:
            out[m.group(1)] = dict(
                prompt=int(m.group(2)),
                hbm=int(m.group(3)),
                queried=int(m.group(4)),
                hit=int(m.group(5)),
                miss_at=int(m.group(6)),
                g0_chunk=int(m.group(7)),
                g0_hash=m.group(8),
            )
    return out


def match_miss(miss_map, response_id):
    """response_id 精确匹配, 失败则按前缀匹配 (vLLM req 后缀)。"""
    if not response_id:
        return {}
    if response_id in miss_map:
        return miss_map[response_id]
    for req, v in miss_map.items():
        if req.startswith(response_id):
            return v
    return {}


def classify_divergence(t):
    """分叉子类型: 区分 BPE 重编码 / 生成末尾 (截断+strip) / prompt 内渲染差异。

    - div_in 位于生成末尾 2 token 内, 且 prefill 侧分叉点紧跟模板 token
      (如 <|user|>) → max_out_len 截断或 EOS 收尾的尾随空白被模板 strip()
    - 其余生成内容内分叉 → BPE 重编码 (边界歧义)
    """
    if t["where"] == "上一轮 prompt 内":
        return "prompt 内 (模板/渲染差异)"
    if t["where"] != "上一轮生成内容内":
        return "完整匹配"
    gen_len = len(t["prev"]["gen_ids"])
    if t["div_in"] is not None and t["div_in"] >= gen_len - 2:
        return "生成末尾 (截断尾随空白被模板 strip / EOS 边界)"
    return "BPE 重编码分叉"


def main():
    ap = argparse.ArgumentParser(description="PD 多轮 prefix-cache 分叉离线重放分析")
    ap.add_argument("details", help="ais_bench details.jsonl 路径")
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT,
                    help="tokenizer 模型路径 (用于分叉 token 上下文显示)")
    ap.add_argument("--out", default=None, help="报告输出 markdown 路径 (缺省打印 stdout)")
    ap.add_argument("--granularity", type=int, default=4096,
                    help="KV 复用对齐粒度 (c128 逻辑块, 默认 4096)")
    ap.add_argument("--top", type=int, default=30,
                    help="典型分叉样例数 (默认 30, BPE 重编码分叉优先)")
    ap.add_argument("--prefill-log", default=None,
                    help="可选: P 侧 vllm_prefill.log, 关联 'KV pool lookup MISS' 日志")
    args = ap.parse_args()

    tok = None
    try:
        from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
        tok = DeepseekV4Tokenizer.from_pretrained(args.model_path)
    except Exception:  # noqa: BLE001
        # 回退: 通用 tokenizer (如 GLM-5.2 TokenizersBackend / 其他 HF 模型)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] tokenizer 加载失败 ({e}), 报告将只显示 token id", file=sys.stderr)

    convs, skipped = load_details(args.details)
    miss_map = parse_miss_lines(args.prefill_log)

    g = args.granularity
    lines = []
    add = lines.append
    add("# PD 分离多轮 prefix-cache 分叉离线重放分析报告")
    add("")
    add(f"- 输入: `{args.details}`")
    add(f"- 会话数: {len(convs)}  (缺 token ids 跳过的请求数: {skipped}; "
        f"需 return_token_ids=True 采集)")
    _g_desc = {
        4096: "c128 压缩层逻辑块 (DSV4)",
        128: "glm_moe_dsa 单组 hash_block_size (GLM-5.2)",
    }.get(g, "复用对齐粒度")
    add(f"- 对齐粒度: {g} ({_g_desc}, LCP 不足 {g} 对齐归零)")
    add("")

    transitions = []
    for uuid, turns in sorted(convs.items()):
        for k in range(len(turns) - 1):
            prev, cur = turns[k], turns[k + 1]
            saved = prev["prompt_ids"] + prev["gen_ids"]
            cur_prompt = cur["prompt_ids"]
            lcp = lcp_len(saved, cur_prompt)
            aligned = lcp // g * g
            prev_plen = len(prev["prompt_ids"])
            if lcp >= len(saved) and len(saved) <= len(cur_prompt):
                where = "完整匹配"
                div_in = None
            elif lcp < prev_plen:
                where = "上一轮 prompt 内"
                div_in = lcp
            else:
                where = "上一轮生成内容内"
                div_in = lcp - prev_plen
            transitions.append(
                dict(uuid=uuid, k=k, prev=prev, cur=cur, saved_len=len(saved),
                     cur_len=len(cur_prompt), lcp=lcp, aligned=aligned,
                     where=where, div_in=div_in, prev_plen=prev_plen)
            )
    for t in transitions:
        t["subtype"] = classify_divergence(t)

    if not transitions:
        add("> 没有可分析的相邻轮次对 (会话均只有 1 轮或缺少 token ids)。")
        print("\n".join(lines))
        return

    # ---- 汇总统计 ----
    add("## 1. 总览")
    add("")
    n = len(transitions)
    full = sum(1 for t in transitions if t["where"] == "完整匹配")
    hit4096 = sum(1 for t in transitions if t["aligned"] >= g)
    pot = sum(t["aligned"] for t in transitions)
    tot_prompt = sum(t["cur_len"] for t in transitions)
    add(f"| 指标 | 值 |")
    add(f"|---|---|")
    add(f"| 相邻轮次对 (转移) 数 | {n} |")
    add(f"| token 级完整匹配 | {full}/{n} ({full/n:.0%}) |")
    add(f"| 达到 {g} 对齐命中 | {hit4096}/{n} ({hit4096/n:.0%}) |")
    add(f"| 潜在命中 token 合计 (对齐后) | {pot} |")
    add(f"| turn≥1 的 prompt token 合计 | {tot_prompt} |")
    add(f"| 潜在 external 命中率 (pot/tot_prompt) | {pot/tot_prompt:.1%} |")
    add("")

    # ---- 按转移类型分布 ----
    add("## 2. 分叉位置分布")
    add("")
    add("| 转移 | 数量 | LCP 中位 | LCP 最小/最大 | 分叉在上一轮生成内的位置(中位) | 对齐命中≥{0} |".format(g))
    add("|---|---|---|---|---|---|")
    by_trans = defaultdict(list)
    for t in transitions:
        by_trans[f"turn{t['k']}→turn{t['k']+1}"].append(t)
    for key in sorted(by_trans, key=lambda x: int(re.search(r"turn(\d+)", x.split("→")[0]).group(1))):
        arr = sorted(by_trans[key], key=lambda t: t["lcp"])
        lcps = sorted(t["lcp"] for t in arr)
        med = lcps[len(lcps)//2]
        divs = sorted(t["div_in"] for t in arr if t["div_in"] is not None)
        div_med = divs[len(divs)//2] if divs else "-"
        hitn = sum(1 for t in arr if t["aligned"] >= g)
        add(f"| {key} | {len(arr)} | {med} | {lcps[0]}/{lcps[-1]} | {div_med} | {hitn}/{len(arr)} |")
    add("")

    # 直方图
    buckets = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30)]
    add("```")
    for lo, hi in buckets:
        cnt = sum(1 for t in transitions if lo <= t["lcp"] < hi)
        bar = "#" * int(cnt * 60 / max(n, 1))
        add(f"LCP [{lo:>5},{hi if hi < 1<<30 else 'inf':>5}): {cnt:>3} {bar}")
    add("```")
    add("")

    # ---- 分叉归类 ----
    add("## 3. 分叉归类")
    add("")
    cnt_where = defaultdict(int)
    for t in transitions:
        cnt_where[t["where"]] += 1
    for w, c in sorted(cnt_where.items(), key=lambda x: -x[1]):
        add(f"- {w}: {c}/{n}")
    add("")
    sub_full = sum(1 for t in transitions if t["subtype"] == "完整匹配"
                   and t["aligned"] < g and t["where"] == "完整匹配")
    if sub_full:
        add(f"- 其中 token 级完整匹配但 LCP < {g} (粒度对齐吞没, 非分叉): {sub_full}/{n}")
    add("")
    add("说明: '上一轮生成内容内' 分为两类 —— **BPE 重编码分叉** (decode 生成 token 与"
        "下一轮 re-tokenize 在文本中段边界不一致) 与 **生成末尾分叉** (max_out_len 截断或"
        "EOS 收尾的尾随空白被模板 content.strip() 丢弃); 后者只损失末尾 1-2 token, "
        "对齐后命中通常不变。'上一轮 prompt 内' = 模板/渲染差异, 需另行排查。")
    add("")

    # ---- 典型分叉样例 (BPE 重编码分叉优先, 其次生成末尾分叉, 按分叉早晚排序) ----
    bpe_div = sorted(
        [t for t in transitions if t["subtype"] == "BPE 重编码分叉"],
        key=lambda t: t["lcp"],
    )
    tail_div = sorted(
        [t for t in transitions if t["subtype"] == "生成末尾 (截断尾随空白被模板 strip / EOS 边界)"],
        key=lambda t: t["lcp"],
    )
    other_div = sorted(
        [t for t in transitions if t["subtype"] == "prompt 内 (模板/渲染差异)"],
        key=lambda t: t["lcp"],
    )
    samples = (bpe_div + tail_div + other_div)[: args.top]
    add(f"## 4. 典型分叉样例 (Top {len(samples)}, BPE 重编码分叉优先, "
        f"共 {len(bpe_div)} 个 BPE 分叉 / {len(tail_div)} 个生成末尾分叉 / "
        f"{len(other_div)} 个 prompt 内分叉)")
    add("")
    for i, t in enumerate(samples, 1):
        prev, cur = t["prev"], t["cur"]
        saved = prev["prompt_ids"] + prev["gen_ids"]
        lcp = t["lcp"]
        srv = match_miss(miss_map, cur["response_id"])
        add(f"### 样例 {i}/{len(samples)}: {t['uuid'][:8]} turn{t['k']}→turn{t['k']+1} "
            f"[{t['subtype']}] response_id={cur['response_id']}")
        add(f"- LCP={lcp} (上一轮 prompt {t['prev_plen']} + 生成 {len(prev['gen_ids'])})"
            + (f", 生成内容内偏移 {t['div_in']}" if t["where"] == "上一轮生成内容内" else ""))
        add(f"- 对齐后命中: {t['aligned']}")
        if srv:
            add(f"- 服务端 lookup 日志: hbm={srv['hbm']} queried={srv['queried']} "
                f"hit={srv['hit']} miss_at={srv['miss_at']} g0_chunk={srv['g0_chunk']}"
                f" (req 前缀匹配 {cur['response_id']}*)")
        # 分叉点前公共文本 (LCP 内两侧 token 完全一致, 取 decode 侧解码)
        pre_lo = max(0, lcp - 40)
        add(f"- **分叉点前公共文本** (token {pre_lo}..{lcp}, 两侧相同):")
        add(f"  > …{tok_text(tok, saved[pre_lo:lcp])}")
        # decode 侧: 原始 token (decode 阶段真实采样)
        d_win = saved[lcp : lcp + 10]
        d_txt_lo, d_txt_hi = lcp, min(len(saved), lcp + 60)
        add(f"- **decode 侧原始 token** @ 分叉点 {lcp} (起 10 个):")
        add(f"  - ids: `{d_win}`")
        add(f"  - token 串: {toks_strs(tok, d_win)}")
        add(f"  - 边界文本: {toks_boundary_text(tok, d_win)}")
        add(f"  - 后续文本 (token {d_txt_lo}..{d_txt_hi}):")
        add(f"  > {tok_text(tok, saved[d_txt_lo:d_txt_hi])}")
        # render 侧: prefill 重新分词后的 token
        r_win = cur["prompt_ids"][lcp : lcp + 10]
        r_txt_lo, r_txt_hi = lcp, min(len(cur["prompt_ids"]), lcp + 60)
        add(f"- **prefill 重编码 token** @ 分叉点 {lcp} (起 10 个):")
        add(f"  - ids: `{r_win}`")
        add(f"  - token 串: {toks_strs(tok, r_win)}")
        add(f"  - 边界文本: {toks_boundary_text(tok, r_win)}")
        add(f"  - 后续文本 (token {r_txt_lo}..{r_txt_hi}):")
        add(f"  > {tok_text(tok, cur['prompt_ids'][r_txt_lo:r_txt_hi])}")
        if t["subtype"] == "BPE 重编码分叉":
            add("  - 注: 两侧**文本内容相同**、token 切分边界不同 (如 `[允许][多层]` vs "
                "`[允][许多][层]`) → BPE 重编码分叉, decode 保存的 KV hash 链自此断裂")
        elif t["subtype"].startswith("生成末尾"):
            add("  - 注: prefill 侧分叉点紧跟模板 token (如 `<|user|>`) 且 decode 侧末 token"
                "含尾随空白 → 生成在 max_out_len 截断/EOS 收尾后, 尾随空白被模板 "
                "`content.strip()` 丢弃; 仅损失末尾 token, 对齐命中通常不变")
        add("")

    # ---- 损失估算 ----
    add("## 5. 命中损失估算")
    add("")
    add(f"- 若全部 round-trip 精确: 潜在命中 {pot} token "
        f"({pot/tot_prompt:.1%} of turn≥1 prompt)")
    add(f"- 实际达成 {g} 命中的转移: {hit4096}/{n}")
    div0 = sum(1 for t in transitions
               if t["where"] != "完整匹配" and t["aligned"] < g)
    gran0 = sum(1 for t in transitions
                if t["where"] == "完整匹配" and t["aligned"] < g)
    add(f"- 因分叉导致 {g} 命中归零的转移: {div0}/{n}")
    if gran0:
        add(f"- token 级完整匹配但自然答案过短 (LCP < {g}) 归零的转移: {gran0}/{n} "
            f"(粒度对齐吞没, 非分叉)")
    add("")
    add("> 治本方案: 客户端按 token_ids 回填 (ais_bench infer_every 当前用"
        " output.content 文本回填, 经 detokenize→re-tokenize 不可逆); "
        "或服务端接受 token-ids 形式的 assistant 消息。")
    add("")

    report = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"报告已写入 {args.out}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
