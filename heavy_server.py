#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
heavy_server.py —— 把「双模型复合流水线」包成一个本地 OpenAI 兼容服务

目的：让 DSH（或任何 OpenAI 兼容客户端）把「本地 Heavy（Bonsai ×3 视角 + minicpm5）」
当成**一个普通模型选项**来选，而不用改 DSH 一行代码。

对外：
  GET  /v1/models              → 列出 dual-heavy
  POST /v1/chat/completions    → 支持 stream: true/false

内部两种模式：
  A) 请求里【带 tools】（= agent 在用）：先跑多视角预分析 → 再带 tools 做最终调用（保住工具调用能力）
  B) 请求里【不带 tools】（= 普通聊天）：跑完整 Heavy 流水线（plan→workers→compress→synth→critique）

用法： python heavy_server.py [--port 8020] [--no-preamble]
"""

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import dual_agent as da  # noqa: E402

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BOSS_URL = f"{da.BOSS}/v1/chat/completions"
MODEL_ID = "dual-heavy"
# 小模型直通路由：对外暴露的 id。用 `minicpm5-2b-32k` 是对齐 Ollama 里那个派生模型名，
# 这样 DSH 侧只要把 baseURL 从 11434 换到 8020，模型 id 一个字都不用改。
MINI_SERVED = ("minicpm5-2b-32k", "mini-guarded")
MINI_UPSTREAM = "minicpm5-2b-32k"
SERVED_MODELS = (MODEL_ID,) + MINI_SERVED
DEFAULT_PORT = 8020
TRACE_LOG = __file__.rsplit("\\", 1)[0] + r"\heavy-server-trace.log"

CONFIG = {"preamble": True, "max_ctx": 24000, "prompt_budget": 24000,
          "preamble_max_tokens": 12000,
          "frequency_penalty": 0.2, "presence_penalty": 0.1,
          "degeneration_guard": True, "chat_max_tokens": 2000}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(TRACE_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def flatten(messages):
    """把消息压成一段纯文本（给多视角 worker 用）。"""
    out = []
    for m in messages or []:
        role = m.get("role", "?")
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text")
        if c:
            out.append(f"[{role}] {c}")
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {})
            out.append(f"[tool_call] {fn.get('name')}({fn.get('arguments')})")
    txt = "\n".join(out)
    return txt[-CONFIG["max_ctx"]:]


def pre_analysis(question, progress=None):
    """带 tools 模式下的「先想再动手」。

    ⚠️ 实测教训：只把候选结论当「参考资料」塞进上下文，模型会**无视**它
    （三个 worker 都说 3.9，最终调用照样答 3.11）。必须走一遍**裁决**角色
    （synthesize + critique），再把定案结果当结论下发。

    progress: 可选回调，用来把多视角过程实时播给客户端看（当成"思考中"）。
    """
    def note(msg):
        if progress:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001
                pass

    trace = []
    note("① 拆解子问题…")
    subs = da.plan(question, trace, "boss")
    note(f"② 多视角并行分析（{len(subs)} 个子问题 / {len(da.LENSES)} 个视角）…")
    cands = da.work(question, subs, trace)
    note("③ 汇总去重（minicpm5）…")
    comp = da.compress(question, cands, trace)
    note("④ 交叉裁决…")
    draft, _ = da.synthesize(question, comp, trace)
    note("⑤ 挑刺复查…")
    final, fixed = da.critique(question, draft, trace)
    log(f"pre-analysis 定案={final!r}（挑刺{'改过' if fixed else '未动'}）")
    return (f"多视角独立分析已定案，结论为：{final}\n"
            f"依据摘要：{comp[:600]}\n"
            f"注意：以上是无工具情况下的推理结论。**如果这个问题需要外部信息或操作**"
            f"（查天气、读文件、跑命令、联网等），不要采信上面任何具体结论，"
            f"直接调用相应的工具去获取。\n"
            f"直接据此行动，不要复述本段。")


def rough_tokens(messages):
    """粗略估 token（不依赖任何分词器，宁可高估）。混合中英按 0.7 token/字符。"""
    n = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        n += len(c or "")
        for tc in (m.get("tool_calls") or []):
            n += len(json.dumps(tc, ensure_ascii=False))
    return int(n * 0.7) + 400


def trim_to_fit(messages, budget_tokens):
    """把请求裁到预算内：保留首条 system，从**最老的**开始丢（连带丢掉孤立的 tool 结果）。

    踩过的坑（两个，都是实测撞出来的）：
    1. DSH 那边算的上下文和 llama-server 实际分词不一致，实测收到 40698 token > 32768
       → 上游 400。与其报错，不如丢掉最老的对话继续干活（就是"失忆"，但任务能往下走）。
    2. **不能把「用户最初的提问」丢掉**：Bonsai 的模板里有
       `{%- if ns.multi_step_tool %}{{ raise_exception('No user query found in messages.') }}`，
       一旦剩下的消息里没有"真正的用户提问"（只剩 assistant / tool），上游直接 500。
       实测日志：`upstream 500 ... line 100 ... multi_step_tool ... raise_exception`。
    """
    msgs = [dict(m) for m in (messages or [])]
    sysmsg = None
    if msgs and msgs[0].get("role") == "system":
        sysmsg = msgs.pop(0)

    def is_real_query(m):
        if m.get("role") != "user":
            return False
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        c = (c or "").strip()
        return bool(c) and not (c.startswith("<tool_response>") and c.endswith("</tool_response>"))

    protected = next((m for m in reversed(msgs) if is_real_query(m)), None)
    base = [sysmsg] if sysmsg else []
    while msgs and rough_tokens(base + msgs) > budget_tokens:
        msgs.pop(0)
        while msgs and msgs[0].get("role") == "tool":
            msgs.pop(0)          # 配对的 tool_call 已经丢了，孤立 tool 结果一并丢
    if protected is not None and not any(m is protected for m in msgs):
        # 提问被裁掉了 → 补回来紧跟 system，保证模板里有"用户提问"
        msgs.insert(0, protected)
    return base + msgs


def degeneration_report(text):
    """统计退化程度：段落数 / 唯一段落数 / 重复占比。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    seen, uniq = set(), []
    for p in paras:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return len(paras), len(uniq), 1 - len(uniq) / max(1, len(paras))


def clean_degeneration(text, threshold=0.5):
    """**复读机清理**：低比特量化模型写长文时容易退化成「同一段循环 N 次」。

    实测案例（2026-10-07《尾巴与耳朵的敏感篇》）：273 段里只有 13 段是新的，
    同一段重复 88 次，非空行重复率 95.4% —— 45 KB 的「文章」真正的内容只有约 1 KB。
    这里按**首次出现顺序**对段落去重，只留真内容。
    返回 (清理后的文本, 是否触发, 重复占比)。
    """
    n_paras, n_uniq, ratio = degeneration_report(text)
    if n_paras < 6 or ratio < threshold:
        return text, False, ratio
    seen, out = set(), []
    for p in [x.strip() for x in re.split(r"\n\s*\n", text or "") if x.strip()]:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out), True, ratio


def clean_tool_call_args(tool_calls):
    """清理**工具调用参数里**的复读机退化，返回清理了几处。

    ⚠️ 为什么必须管这里：实测《尾巴与耳朵的敏感篇》45 KB 的内容不是 content 输出的，
    而是模型用 `write` 工具一次性塞进 arguments 的 —— 日志里只看得到
    "输出0字 工具调用1个"，所以只清理 content 完全抓不到这个退化。
    """
    n = 0
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str) or len(args) < 800:
            continue
        try:
            obj = json.loads(args)
        except Exception:  # noqa: BLE001
            continue
        changed = False
        for k, v in list(obj.items()):
            if isinstance(v, str) and len(v) > 400:
                cl, hit, ratio = clean_degeneration(v)
                if hit:
                    log(f"⚠ 工具参数 {fn.get('name')}.{k} 复读机退化："
                        f"重复率 {ratio:.1%}，{len(v)} → {len(cl)} 字符")
                    obj[k] = cl
                    changed = True
                    n += 1
        if changed:
            fn["arguments"] = json.dumps(obj, ensure_ascii=False)
    return n


# ---------- 工具参数校验 / 自修复 ----------
# 实测报错（用户截图）：
#   Error: invalid arguments: missing required property "new_string"
#   Error: old_string and new_string must differ
#   Error: old_string was not found in "..."
# 根因：本地 27B@2.13bpw 在工具参数上保真度差；而且**空值参数会被上游工具解析丢掉**
# （模型想删除文本时 new_string 本来就是空串 → 键消失 → schema 校验说"缺少必需参数"）。
ARG_ALIASES = {
    "new_string": ["new_str", "newString", "new_content", "new_text", "newText",
                   "replacement", "to", "replace_with"],
    "old_string": ["old_str", "oldString", "old_content", "old_text", "oldText",
                   "search", "from", "find"],
    "file_path": ["path", "filepath", "filePath", "file", "filename"],
    "content": ["text", "body", "data", "contents"],
    "replace_all": ["all", "replaceAll"],
}
EMPTY_OK = {"new_string", "new_str", "new_content", "new_text", "content", "contents"}


def _schemas(tools):
    return {t.get("function", {}).get("name"): t.get("function", {}).get("parameters", {})
            for t in (tools or [])}


def _parse_args(tc):
    """解析工具参数；容忍**二次编码**（arguments 里又是一层 JSON 字符串）。"""
    fn = tc.get("function") or {}
    raw = fn.get("arguments") or "{}"
    for _ in range(3):
        try:
            obj = json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(obj, str):
            raw = obj
            continue
        return obj
    return None


def verify_edit_against_file(tool_name, args):
    """**edit 的 old_string 必须真的在文件里** —— shim 就跑在本机，能直接核对。

    这一步专治"编造原文"：规则或小模型可能把 old_string 补成一句不存在的话，
    结构校验能过，但到 DSH 必然 `old_string was not found`。
    返回 (ok, 说明)。
    """
    if tool_name != "edit":
        return True, ""
    path, old = args.get("file_path"), args.get("old_string")
    if not isinstance(path, str) or not path:
        return False, "缺 file_path"
    if not isinstance(old, str) or not old:
        return False, "old_string 为空"
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return True, f"读不到文件（跳过核对）：{e}"     # 读不到就不拦，交给 DSH
    if old not in txt:
        return False, "old_string 不在文件里（疑似编造，应先 read）"
    return True, ""


def validate_tool_calls(tool_calls, tools):
    """按 DSH 送来的 schema 校验工具调用，返回问题描述列表。"""
    schemas = _schemas(tools)
    problems = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = _parse_args(tc)
        if args is None:
            problems.append(f"{name}: arguments 不是合法 JSON")
            continue
        if not isinstance(args, dict):
            problems.append(f"{name}: arguments 不是对象")
            continue
        missing = [r for r in ((schemas.get(name) or {}).get("required") or [])
                   if r not in args or args[r] is None]
        if missing:
            problems.append(f"{name}: 缺少必需参数 {missing}")
        if name == "edit":
            # DSH 的真实契约：old_string 必须非空，且与 new_string 不同
            if not isinstance(args.get("old_string"), str) or args.get("old_string") == "":
                problems.append("edit: old_string 必须是非空字符串")
            elif args.get("old_string") == args.get("new_string"):
                problems.append("edit: old_string 与 new_string 相同")
    return problems


def repair_tool_calls(tool_calls, tools):
    """规则修复：解包 / 类型校准 / 别名 / 空值补齐。全部**确定性**，不编造内容。

    实测价值：10 个坏参数用例里，规则只能修 2 个；补上「解包 + 类型」之后，
    多包一层、二次编码、`"10"`→`10`、数组→字符串 这些**机械问题**都能免费修掉。
    """
    schemas = _schemas(tools)
    n = 0
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        spec = schemas.get(name) or {}
        props = spec.get("properties") or {}
        args = _parse_args(tc)
        if not isinstance(args, dict):
            continue
        changed = False

        # ① 解包：{"arguments": {...}} / {"args": {...}} 这种多包一层
        for wrapper in ("arguments", "args", "parameters", "input", "payload"):
            if list(args.keys()) == [wrapper] and isinstance(args[wrapper], dict):
                args = args[wrapper]
                changed = True
                n += 1
                log(f"⇄ 工具参数修复：{name} 多包一层 → 已解包 .{wrapper}")
                break

        # ② 类型校准（按 schema 来）
        for k, v in list(args.items()):
            t = (props.get(k) or {}).get("type")
            try:
                if t == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
                    args[k] = int(v)
                elif t == "boolean" and isinstance(v, str) and v.strip().lower() in ("true", "false"):
                    args[k] = v.strip().lower() == "true"
                elif t == "string" and isinstance(v, (list, tuple)):
                    args[k] = "\n".join(str(x) for x in v)
                else:
                    continue
                changed = True
                n += 1
                log(f"⇄ 工具参数修复：{name}.{k} 类型校准为 {t}")
            except Exception:  # noqa: BLE001
                pass

        # ③ 别名纠正
        for canonical, aliases in ARG_ALIASES.items():
            if canonical in args or canonical not in props:
                continue
            for a in aliases:
                if a in args:
                    args[canonical] = args.pop(a)
                    changed = True
                    n += 1
                    log(f"⇄ 工具参数修复：{name}.{a} → {canonical}")
                    break

        # ④ 空值补齐（只对"可以为空"的参数；file_path 这类绝不补）
        for r in (spec.get("required") or []):
            if (r not in args or args[r] is None) and r in EMPTY_OK:
                args[r] = ""
                changed = True
                n += 1
                log(f"＋ 工具参数修复：{name}.{r} 缺失 → 补空串（空值参数会被上游解析丢掉）")

        if changed or not isinstance(fn.get("arguments"), str) or _parse_args(tc) != args:
            fn["arguments"] = json.dumps(args, ensure_ascii=False)
    return n


_LOOP_STATE = {}   # 非法工具调用签名 → 连续失败次数


def note_invalid_call(tool_calls, problems, limit=2):
    """记录一次非法工具调用。同一调用反复失败时返回 True（= 该熔断，别再让它转圈）。

    实测：用户截图里出现过 **连续 9 次** `Error: old_string and new_string must differ` ——
    本地模型不会自己跳出这个坑，DSH 也只是把错误当工具结果喂回去，于是无限重试。
    """
    key = "|".join(sorted(str((tc.get("function") or {}).get("arguments") or "")
                          for tc in (tool_calls or [])))
    key = f"{key}##{'/'.join(sorted(problems or []))}"
    cnt = _LOOP_STATE.get(key, 0) + 1
    _LOOP_STATE[key] = cnt
    if len(_LOOP_STATE) > 200:
        _LOOP_STATE.clear()
        _LOOP_STATE[key] = cnt
    return cnt >= limit


LOOP_BREAK_MESSAGE = (
    "\n\n---\n"
    "⚠️ 本地模型（Dual-Heavy）没能给出合法的 `edit` 调用，连续多次把 `old_string` 和 `new_string` "
    "写成了同一段文字，我已中断这次重试以免无限循环。\n\n"
    "可以这样继续：\n"
    "1. 让它先用 `read` 读取文件确认原文，再做替换；\n"
    "2. 要**删除**某段 → `edit` 的 `new_string` 传空字符串（可以省略内容，但参数名不能少）；\n"
    "3. 要**大段改写** → 直接用 `write` 重写整个文件，比 `edit` 可靠得多；\n"
    "4. 或者把这一步切回云端模型（`deepseek-flash`）来做精确编辑。\n"
)


# ---------- 小模型修参数（实测 10/10，平均 0.2 秒）----------
# 2026-10-07 对照实验（10 个坏参数用例）：
#   我的规则修复 4/10 ｜ minicpm5-2b **10/10（0.2s/次）** ｜ Bonsai-27B 10/10（0.8s/次）
# 结论：**"按 schema 把参数改对"是规格明确的文本改写，正是小模型的强项**，
# 而且比大模型快 4 倍。所以链路定为：规则（免费）→ 小模型（快）→ 大模型重调 → 熔断。
REPAIR_PROMPT = """你是工具调用参数修复器。
工具名：{tool}
工具 schema：{schema}
当前参数（不合法）：{args}
校验错误：{problem}

请输出修复后的参数，**只输出一个 JSON 对象**，不要解释、不要 markdown 代码块。
规则：
1. 参数名必须与 schema 完全一致，类型必须匹配；
2. 删除文字时 new_string 传空字符串 ""（参数不能省略）；
3. old_string 必须是非空字符串，且不能与 new_string 相同；
4. **不要凭空编造原文内容**；如果确实缺原文，就把 old_string 设为空字符串。"""


def repair_with_small_model(tool_name, schema, raw_args, problems, timeout=120):
    """用 minicpm5-2b 修参数。返回修复后的 dict，失败返回 None。"""
    prompt = REPAIR_PROMPT.format(
        tool=tool_name, schema=json.dumps(schema or {}, ensure_ascii=False),
        args=raw_args, problem="；".join(problems))
    payload = {"model": da.MINI_MODEL, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "think": False,
               "options": {"temperature": 0.0, "num_predict": 600}}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{da.MINI_API}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=timeout) as r:
            txt = (json.loads(r.read().decode("utf-8")).get("message") or {}).get("content", "")
    except Exception as e:  # noqa: BLE001
        log(f"小模型修参数失败：{e}")
        return None
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        fixed = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    return fixed if isinstance(fixed, dict) else None


TOOL_SCHEMAS_CACHE = {"tools": []}


def repair_tool_calls_with_model(tool_calls, tools):
    """对规则修不好的调用，交给小模型修。返回修复处数。"""
    schemas = _schemas(tools)
    n = 0
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw = fn.get("arguments") or "{}"
        probs = validate_tool_calls([tc], tools)
        if not probs:
            continue
        fixed = repair_with_small_model(name, schemas.get(name), raw, probs)
        if not fixed:
            continue
        # 小模型也可能编造（例如把 old_string 填成一句不存在的话）——仍然要过**结构 + 语义**两关
        fixed_probs = validate_tool_calls(
            [{"id": "x", "type": "function",
              "function": {"name": name, "arguments": json.dumps(fixed, ensure_ascii=False)}}], tools)
        sem_ok, sem_why = verify_edit_against_file(name, fixed)
        if fixed_probs or not sem_ok:
            log(f"小模型修完仍不合格：{fixed_probs or sem_why}")
            continue
        fn["arguments"] = json.dumps(fixed, ensure_ascii=False)
        n += 1
        log(f"🔧 小模型修复：{name} {probs} → 已通过校验")
    return n


def boss_stream(messages, tools, temperature=0.3, max_tokens=1500):
    """向 Bonsai 发起**流式**请求，逐帧 yield 解析后的 chunk。"""
    payload = {"model": da.BOSS_MODEL, "messages": trim_to_fit(messages, CONFIG["prompt_budget"]),
               "temperature": temperature, "max_tokens": max_tokens, "stream": True,
               "frequency_penalty": CONFIG["frequency_penalty"],
               "presence_penalty": CONFIG["presence_penalty"]}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BOSS_URL, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                yield json.loads(body)
            except Exception:  # noqa: BLE001
                continue


def relay_stream(messages, tools, temperature, max_tokens, emit, chunk_factory):
    """**流式转发**：Bonsai 一吐 token 就转给 DSH，不用等整段生成完。

    工具调用参数单独累积（因为要先校验/修复才能发出去 —— 参数被写坏的情况太多），
    content 则**实时转发**；同时做**流式复读检测**，一发现退化成循环就掐断。
    返回 (content, tool_calls, usage, 是否已掐断)。
    """
    parts, tcs, usage = [], {}, {}
    buf = ""
    loop_cut = False
    for obj in boss_stream(messages, tools, temperature, max_tokens):
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                emit(chunk_factory({"reasoning_content": d["reasoning_content"]}))
            if d.get("content"):
                piece = d["content"]
                parts.append(piece)
                buf += piece
                if not loop_cut and len(buf) % 400 < len(piece):
                    _, hit, ratio = clean_degeneration(buf)
                    if hit:
                        loop_cut = True
                        log(f"⚠ 流式检测到复读机退化（重复率 {ratio:.0%}）→ 掐断这次输出")
                        emit(chunk_factory({"content":
                                            "\n\n[本地 Heavy 检测到重复退化，已中断本次输出；"
                                            "建议改用 write 整体重写或切回云端模型]"}))
                        break
                emit(chunk_factory({"content": piece}))
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = tcs.setdefault(i, {"id": tc.get("id") or f"call_{i}", "type": "function",
                                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
        if loop_cut:
            break
    return "".join(parts), [tcs[k] for k in sorted(tcs)], usage, loop_cut


def heavy_stream(question, emit, chunk_factory):
    """带流式的 Heavy 流水线：前几步照旧（多视角并行），**最后一步实时转发**。

    为什么不能整条流水线都流式：多视角 worker 之间要汇总比较，天然需要先收齐。
    但"最后一锤子"（裁决作答）可以边生成边发 —— 用户就不用等到全部结束才看到字。
    """
    trace = []
    emit(chunk_factory({"reasoning_content": "① 拆解子问题…\n"}))
    subs = da.plan(question, trace, "boss")
    emit(chunk_factory({"reasoning_content":
                        f"② 多视角并行分析（{len(da.LENSES)} 个视角 / {len(subs)} 个子问题）…\n"}))
    cands = da.work(question, subs, trace)
    emit(chunk_factory({"reasoning_content": "③ 汇总去重（minicpm5）…\n"}))
    comp = da.compress(question, cands, trace)
    emit(chunk_factory({"reasoning_content": "④ 裁决定案 —— 开始流式输出正文\n"}))
    sys_s = ("你是最终裁决者。下面几个视角的分析结论可能一致也可能冲突。"
             "用最可靠的那条理由定案，**直接给出完整答案**，不要复述分析过程，不要开场白。")
    content, _tcs, usage, _cut = relay_stream(
        [{"role": "system", "content": sys_s},
         {"role": "user", "content": f"问题：{question}\n\n各视角整理结果：\n{comp}"}],
        None, 0.4, int(CONFIG.get("chat_max_tokens") or 2000), emit, chunk_factory)
    return content, usage


class UpstreamError(Exception):
    """上游（Bonsai）返回的错误，带原始 body，便于上层判断和转述。"""

    def __init__(self, code, body):
        super().__init__(f"upstream {code}: {str(body)[:200]}")
        self.code = code
        self.body = body


def boss_with_tools(messages, tools, temperature=0.3, max_tokens=1500):
    """带 tools 调 Bonsai（OpenAI 兼容），返回 (message_dict, usage)。

    * 自动裁上下文：先按预算裁，若上游仍报"超出上下文"就把预算减半再试（最多 2 次）。
    * **带上重复惩罚**：实测缺了它，模型会写成"同一段循环 88 次"的复读机
      （Bonsai 启动脚本只有 --temp/--top-p/--top-k/--min-p，没有 repeat penalty，
       DSH 也不会主动送 frequency_penalty）。
    """
    budget = CONFIG["prompt_budget"]
    for attempt in range(3):
        msgs = trim_to_fit(messages, budget)
        payload = {"model": da.BOSS_MODEL, "messages": msgs,
                   "temperature": temperature, "max_tokens": max_tokens, "stream": False,
                   "frequency_penalty": CONFIG["frequency_penalty"],
                   "presence_penalty": CONFIG["presence_penalty"]}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(BOSS_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            with _opener.open(req, timeout=900) as r:
                resp = json.loads(r.read().decode("utf-8"))
            msg = ((resp.get("choices") or [{}])[0].get("message") or {})
            return msg, (resp.get("usage") or {})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code == 400 and "exceeds the available context size" in body and attempt < 2:
                budget = int(budget * 0.5)
                log(f"上游说超上下文，裁剪预算降到 {budget} token 重试（丢更老的对话）")
                continue
            raise UpstreamError(e.code, body)
    raise UpstreamError(500, "unreachable")


def last_user_text(messages):
    """取最后一条用户消息的纯文本（预分析只看这一句，不喂整段历史和系统提示）。"""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c
                             if isinstance(p, dict) and p.get("type") == "text")
            return (c or "").strip()
    return ""


def inject_advice(messages, advice):
    """把预分析建议**并进首条 system 消息**，而不是新插一条。

    ⚠️ 踩过的坑（2026-10-07，用户看到 502）：
    Bonsai 的 Jinja 模板里有
      `{%- if not loop.first %}{{ raise_exception('System message must be at the beginning.') }}`
    而 DSH 的请求**第一条本来就是 system**。再插一条 system → 第二条 system 不在首位
    → 上游 500「System message must be at the beginning」→ 我这个 shim 又包成 502，
    用户的会话就被重试 5 次全失败。
    """
    msgs = [dict(m) for m in messages or []]
    head = []
    while msgs and msgs[0].get("role") == "system":
        head.append(msgs.pop(0))

    def as_text(c):
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c
                            if isinstance(p, dict) and p.get("type") == "text")
        return c or ""

    sys_text = "\n\n".join(t for t in (as_text(m.get("content")) for m in head) if t)
    merged = f"{sys_text}\n\n{advice}" if sys_text else advice
    return [{"role": "system", "content": merged}] + msgs


# ---------- 怪语言 / 退化守卫（minicpm5 直通路由）----------
# 实测（2026-10-07，用户截图："你怎么又切回这个诡异语言了"）：
# 2.5B 的 minicpm5 会**整段输出卡纳达文**这类训练里极罕见的文字系统。
# 这不是翻译错误，是模型失去连贯性后从高熵区采样 —— 我们自己复现了 20 次没复现出来，
# 说明触发点在具体上下文里，不在"提示词太长"这种通用因素上。
#
# 所以这个守卫**不试图修好模型**（模型层的问题在 shim 层修不了），只保证一件事：
# **用户永远不会看到那屏垃圾**。做法：
#   ① 流式先扣住开头 GUARD_WINDOW 个字符不发，判定干净才放行（正常回答只慢零点几秒）；
#   ② 判定异常 → 丢弃这次输出，带强约束中文提示**重试一次**；
#   ③ 重试仍异常，或异常出现在已放行之后 → 掐断 + 给出可读说明，绝不原样透出。
SCRIPT_RANGES = (
    (0x0900, 0x097F, "天城文"), (0x0980, 0x09FF, "孟加拉文"), (0x0A00, 0x0A7F, "古木基文"),
    (0x0A80, 0x0AFF, "古吉拉特文"), (0x0B00, 0x0B7F, "奥里亚文"), (0x0B80, 0x0BFF, "泰米尔文"),
    (0x0C00, 0x0C7F, "泰卢固文"), (0x0C80, 0x0CFF, "卡纳达文"), (0x0D00, 0x0D7F, "马拉雅拉姆文"),
    (0x0D80, 0x0DFF, "僧伽罗文"), (0x0E00, 0x0E7F, "泰文"), (0x0E80, 0x0EFF, "老挝文"),
    (0x0F00, 0x0FFF, "藏文"), (0x1000, 0x109F, "缅甸文"), (0x1780, 0x17FF, "高棉文"),
    (0x0530, 0x058F, "亚美尼亚文"), (0x10A0, 0x10FF, "格鲁吉亚文"), (0x1200, 0x137F, "埃塞俄比亚文"),
    (0x0590, 0x05FF, "希伯来文"), (0x0600, 0x06FF, "阿拉伯文"), (0x0700, 0x074F, "叙利亚文"),
)

GUARD_WINDOW = 16      # 流式：先扣住这么多字符，判定干净才放行
# ⚠️ 这个值不能再调大：原来是 48，结果**所有 48 字以内的回答都不流式了**
# （整段卡在 pending 里、直到结束才吐一次）—— 而小模型的回答恰恰多数很短。
# 16 已经足够挡住"一开口就是乱码"的情况：卡纳达文在攒到 ~10 个字母时比例就是 100%，
# 会先被判掉；而放行之后的每一片仍然走滑动窗口 + safe_prefix，中途变怪也拦得住。
GUARD_TAIL = 400       # 放行之后只查尾部，避免长回答把异常占比稀释掉
GUARD_MIN_HITS = 6     # 至少这么多"怪字"才可能判定（少数字符出现在引文里不算）
GUARD_MIN_RATIO = 0.12  # 且要占字母类字符的 12% 以上

GUARD_HINT = (
    "【硬性约束】必须使用**简体中文**回答（必要时可夹英文单词、数字、代码、emoji）。"
    "**严禁**输出卡纳达文、马拉雅拉姆文、泰卢固文、泰米尔文、天城文、孟加拉文、泰文、"
    "阿拉伯文、希伯来文等任何其他文字系统的字符。"
    "如果你不确定答案，就用简体中文说明你不确定，不要输出任何乱码或外语。"
)

GUARD_NOTICE = ("\n\n[本地小模型守卫] 这次输出被判定为「文字系统异常」（疑似模型失去连贯性后"
                "吐出了乱码/罕见文字），已拦下不展示。重试也失败，建议：换个模型重问一次，"
                "或把问题缩小到一句话再问。\n")

# ⚠️ minicpm5 是 thinking 模型，`reasoning` 会先吃 token 预算。实测 max_tokens=20 时
# **整段预算被思考吃光、content 返回空串** —— 看起来像"模型没反应"，极难排查。
# 这个下限只在小模型路由上生效（DSH 自己发的是 4096，不受影响）。
MINI_MIN_MAX_TOKENS = 256


def script_anomaly(text, min_hits=GUARD_MIN_HITS, min_ratio=GUARD_MIN_RATIO):
    """统计「不该出现的文字系统」占比。返回 (命中字数, 占比, 命中的文字系统集合)。

    分母是**字母类字符**（L*），所以中文标点/数字/emoji 不会把比例冲淡。
    少于 8 个字母的文本直接判为正常（避免误伤"OK"这种短回答）。
    """
    letters = [c for c in (text or "") if unicodedata.category(c).startswith("L")]
    if len(letters) < 8:
        return 0, 0.0, set()
    hits, names = 0, set()
    for c in letters:
        cp = ord(c)
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= cp <= hi:
                hits += 1
                names.add(name)
                break
    ratio = hits / len(letters)
    return hits, ratio, names


def script_is_bad(text):
    hits, ratio, names = script_anomaly(text)
    return (hits >= GUARD_MIN_HITS and ratio >= GUARD_MIN_RATIO), hits, ratio, names


def safe_prefix(text):
    """返回「第一个不属于预期文字系统」的位置（找不到就返回 len）。

    用在判定为异常之后：把这一段里**前面干净的部分**放行，从第一个怪字处切断。
    这样即使上游一次给一大块（不是逐 token），也不会把整块脏内容透出去。
    """
    for i, c in enumerate(text or ""):
        if not unicodedata.category(c).startswith("L"):
            continue
        cp = ord(c)
        for lo, hi, _name in SCRIPT_RANGES:
            if lo <= cp <= hi:
                return i
    return len(text or "")


def has_short_period(text, tail_len=600, max_period=80, min_match=0.92):
    """流式用的「复读机」检测：看尾部有没有一个**很短的周期**在重复。

    为什么不用 `clean_degeneration`：那个是按空行分段去重的，而流式退化经常是
    **没有空行的一行刷到底**（实测 `这是一段会重复的话。` ×80 → 段落数只有 1，
    段落去重完全不触发）。这里直接做周期性检测，周期 1~80 字符、匹配率 ≥92% 就算退化。
    返回 (是否退化, 周期)。
    """
    tail = (text or "")[-tail_len:]
    n = len(tail)
    if n < 400:
        return False, 0
    for p in range(2, max_period + 1):
        if n < 4 * p:
            break
        same = 0
        for i in range(n - p):
            if tail[i] == tail[i + p]:
                same += 1
        if same / (n - p) >= min_match:
            return True, p
    return False, 0


def mini_call_stream(messages, tools, temperature, max_tokens, model):
    """向 Ollama 的**原生** /api/chat 发起流式请求，逐帧 yield 成 OpenAI 形状的 chunk。

    ⚠️ 为什么不用 /v1/chat/completions（2026-10-07 实测踩到的坑）：
    minicpm5 是 thinking 模型，而 **OpenAI 兼容口不认 `think: false`**。
    实测同一个问题给 max_tokens=600，它把 600 个 token 全烧在 reasoning 上、
    `content` 返回**空串**（finish_reason=length，reasoning 1088 字）——
    表现就是"模型没反应"，极难排查。原生口认 `think: false`，实测：
      · 非流式：一句话直接给完答案，done_reason=stop，thinking=None
      · 流式：一帧一帧给 content，形状稳定
      · 工具调用：单帧给完整 tool_calls（arguments 是**对象**，需要序列化成字符串）
    小模型干"搬运/工具"这类活本来就不需要思考链，关掉之后又快又不空。
    """
    payload = {"model": model, "messages": trim_to_fit(messages, CONFIG["prompt_budget"]),
               "stream": True, "think": False,
               "options": {"temperature": temperature, "num_predict": max_tokens}}
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{da.MINI_API}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            msg = obj.get("message") or {}
            delta = {}
            if msg.get("content"):
                delta["content"] = msg["content"]
            if msg.get("thinking"):
                delta["reasoning_content"] = msg["thinking"]
            tcs = []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args if args is not None else {}, ensure_ascii=False)
                tcs.append({"index": fn.get("index", i), "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {"name": fn.get("name") or "", "arguments": args}})
            if tcs:
                delta["tool_calls"] = tcs
            if delta:
                yield {"choices": [{"index": 0, "delta": delta}]}
            if obj.get("done"):
                if obj.get("prompt_eval_count") or obj.get("eval_count"):
                    yield {"choices": [], "usage": {
                        "prompt_tokens": obj.get("prompt_eval_count") or 0,
                        "completion_tokens": obj.get("eval_count") or 0,
                        "total_tokens": (obj.get("prompt_eval_count") or 0)
                                         + (obj.get("eval_count") or 0)}}
                return


def mini_guarded_stream(messages, tools, temperature, max_tokens, emit, chunk_factory, model):
    """minicpm5 的**带守卫直通**（流式）。所有 content 由本函数自己 emit。

    返回 (content, tool_calls, usage, 是否拦下了异常输出)。
    """
    content, tcs, usage = "", [], {}
    blocked = False

    for attempt in (1, 2):
        msgs = messages if attempt == 1 else inject_advice(messages, GUARD_HINT)
        parts, tcs, usage = [], {}, {}
        sent, pending = "", ""       # sent=已发出去；pending=还没放行的前缀
        live = False                 # 一旦放行过，就不能再撤回
        last_check = 0               # 上次做周期性检测时的长度
        bad = None

        for obj in mini_call_stream(msgs, tools, temperature, max_tokens, model):
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    emit(chunk_factory({"reasoning_content": d["reasoning_content"]}))
                piece = d.get("content") or ""
                if piece:
                    parts.append(piece)
                    release = None
                    if not live:
                        pending += piece
                        isbad, hits, ratio, names = script_is_bad(pending)
                        if isbad:
                            bad = (hits, ratio, names)
                            break
                        if len(pending) >= GUARD_WINDOW:
                            release, pending = pending, ""
                    else:
                        release = piece

                    # ---- 放行前的统一校验（**先判后发**，大分片也不会整块漏出去）----
                    if release is not None:
                        probe = sent + release
                        isbad, hits, ratio, names = script_is_bad(probe[-GUARD_TAIL:])
                        if isbad:
                            # 只放行这一片里干净的前缀，从第一个怪字处切
                            keep = safe_prefix(release)
                            if keep:
                                sent += release[:keep]
                                emit(chunk_factory({"content": release[:keep]}))
                            bad = (hits, ratio, names)
                            break
                        if len(probe) - last_check >= 200:
                            last_check = len(probe)
                            looping, period = has_short_period(probe)
                            if looping:
                                bad = (0, 1.0, {f"复读机退化(周期{period})"})
                                break
                        live = True
                        sent += release
                        emit(chunk_factory({"content": release}))
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    slot = tcs.setdefault(i, {"id": tc.get("id") or f"call_{i}",
                                              "type": "function",
                                              "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
            if bad:
                break

        content = "".join(parts)
        cat = "/".join(sorted(bad[2])) if bad else ""
        if bad and not live and attempt == 1:
            log(f"⚠ 小模型守卫：输出文字系统异常（{cat}，{bad[1]:.0%}）→ 丢弃并带强约束重试一次")
            continue
        if bad:
            where = "已放行之后" if live else "重试一次后仍然"
            log(f"⚠ 小模型守卫：{where}检测到输出异常（{cat}，{bad[1]:.0%}）→ 掐断，不把这段展示出来")
            # 已经放行过的前缀收不回来（那部分本来就是干净的），但**不返回**含垃圾的全文
            content = (sent if live else "") + GUARD_NOTICE
            emit(chunk_factory({"content": GUARD_NOTICE}))
            blocked = True
            return content, [], usage, blocked
        # 干净：把可能还没放行的尾巴补发出去
        if pending:
            emit(chunk_factory({"content": pending}))
        if attempt == 2:
            log("✅ 小模型守卫：重试后输出正常")
        return content, [tcs[k] for k in sorted(tcs)], usage, blocked

    return content, [], usage, blocked


def mini_guarded_once(messages, tools, temperature, max_tokens, model):
    """非流式版本：返回 (content, tool_calls, usage, 是否拦下)。同样走原生口 + think:false。"""
    for attempt in (1, 2):
        msgs = messages if attempt == 1 else inject_advice(messages, GUARD_HINT)
        payload = {"model": model, "messages": trim_to_fit(msgs, CONFIG["prompt_budget"]),
                   "stream": False, "think": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        if tools:
            payload["tools"] = tools
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(f"{da.MINI_API}/api/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        with _opener.open(req, timeout=900) as r:
            obj = json.loads(r.read().decode("utf-8"))
        msg = obj.get("message") or {}
        content = msg.get("content") or ""
        tcs = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {}, ensure_ascii=False)
            tcs.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                        "function": {"name": fn.get("name") or "", "arguments": args}})
        usage = {"prompt_tokens": obj.get("prompt_eval_count") or 0,
                 "completion_tokens": obj.get("eval_count") or 0,
                 "total_tokens": (obj.get("prompt_eval_count") or 0) + (obj.get("eval_count") or 0)}
        isbad, hits, ratio, names = script_is_bad(content)
        cat = "/".join(sorted(names))
        if not isbad and content:
            loop, period = has_short_period(content)
            if loop:
                isbad, cat, ratio = True, f"复读机退化(周期{period})", 1.0
            else:
                cleaned, hit, dratio = clean_degeneration(content)
                if hit:
                    log(f"⚠ 小模型守卫：段落重复率 {dratio:.0%}，"
                        f"{len(content)} → {len(cleaned)} 字符，已去重")
                    content = cleaned
        if isbad and attempt == 1:
            log(f"⚠ 小模型守卫：输出异常（{cat}，{ratio:.0%}）→ 丢弃并带强约束重试一次")
            continue
        if isbad:
            log(f"⚠ 小模型守卫：重试仍然异常（{cat}，{ratio:.0%}）→ 拦下")
            return GUARD_NOTICE.strip(), [], usage, True
        if attempt == 2:
            log("✅ 小模型守卫：重试后输出正常")
        return content, tcs, usage, False
    return "", [], {}, True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "local-heavy/1.0"

    def log_message(self, *a):  # 静音默认访问日志
        pass

    # ---------- 通用 ----------
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            # ⚠️ 这里**故意只列本服务真正提供的模型**。
            # 之前 local-mini 直连 Ollama，DSH 的"获取可用模型"会把 Ollama 里
            # 全部模型都列出来（含 minicpm5-2b-128k）—— 那个一被选中就会把
            # Bonsai 挤出显存撞死。改成走本服务后，这条路就堵上了。
            now = int(time.time())
            data = [{"id": MODEL_ID, "object": "model", "created": now, "owned_by": "local-heavy",
                     "name": "本地 Heavy（Bonsai×3视角 + minicpm5 裁决）"}]
            data += [{"id": m, "object": "model", "created": now, "owned_by": "local-mini",
                      "name": "MiniCPM5-2B（经怪语言守卫 · 32768 ctx）"} for m in MINI_SERVED]
            self._json(200, {"object": "list", "data": data})
        elif self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(200, {"status": "ok", "model": MODEL_ID, "served": list(SERVED_MODELS)})
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": "only /v1/chat/completions is served",
                                       "type": "invalid_request_error"}})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})
            return

        messages = body.get("messages") or []
        tools = body.get("tools") or []
        stream = bool(body.get("stream"))
        temperature = float(body.get("temperature", 0.3) or 0.3)
        max_tokens = int(body.get("max_tokens") or 1500)
        req_model = (body.get("model") or MODEL_ID).strip() or MODEL_ID
        if req_model not in SERVED_MODELS:
            # 本服务是本地模型的**唯一前门**：只提供上面这几个 id。
            # 想直连 Ollama 请用 11434；这里拦住是为了不让"顺手选个别的模型"
            # 把 Bonsai 挤出显存（128k 那个就是这么出事的）。
            self._json(404, {"error": {
                "message": f"model '{req_model}' is not served here; served: {list(SERVED_MODELS)}",
                "type": "invalid_request_error"}})
            return
        t0 = time.time()

        # ---------- SSE 基础设施 ----------
        # ⚠️ 踩过的坑（2026-10-07，用户那边"卡在准备搜索网页"5 分钟）：
        # 原来流式响应发完 [DONE] 后**不关连接、也没有 Content-Length**，
        # 客户端看不到响应体结束（llama-server 原生是会关连接的）。
        # 现在：立刻回 headers + 心跳帧 + 结束时 Connection: close。
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        resp_model = req_model
        wlock = threading.Lock()
        hb_stop = threading.Event()
        sse_started = False

        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": resp_model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        def emit(obj):
            with wlock:
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True          # 响应体必须能结束
            sse_started = True
            emit(chunk({"role": "assistant", "content": ""}))

            def heartbeat():
                # 流水线可能要几十秒；每 10 秒发一个空帧，避免客户端"流空闲"超时
                while not hb_stop.wait(10):
                    try:
                        emit(chunk({"content": ""}))
                    except Exception:  # noqa: BLE001
                        break
            threading.Thread(target=heartbeat, daemon=True).start()

        def fail(code, message):
            """统一错误出口：流式已经开始就发错误帧（HTTP 状态改不了了），否则发 JSON。"""
            hb_stop.set()
            log(f"返回错误 {code}: {str(message)[:160]}")
            if not sse_started:
                self._json(code, {"error": {"message": message, "type": "upstream_error"}})
                return
            try:
                emit({"error": {"message": message, "code": code, "type": "upstream_error"}})
                emit(chunk({"content": f"\n[本地 Heavy 错误] {str(message)[:300]}"}))
                emit(chunk({}, "stop"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:  # noqa: BLE001
                log(f"写错误帧失败：{e}")

        # ---------- 小模型直通 + 怪语言守卫 ----------
        if req_model in MINI_SERVED:
            if max_tokens < MINI_MIN_MAX_TOKENS:
                log(f"小模型路由：max_tokens {max_tokens} 太小（thinking 会吃光预算导致空回答）"
                    f"→ 抬到 {MINI_MIN_MAX_TOKENS}")
                max_tokens = MINI_MIN_MAX_TOKENS
            try:
                if stream:
                    content, tool_calls, usage, blocked = mini_guarded_stream(
                        messages, tools, temperature, max_tokens, emit, chunk, MINI_UPSTREAM)
                else:
                    content, tool_calls, usage, blocked = mini_guarded_once(
                        messages, tools, temperature, max_tokens, MINI_UPSTREAM)
            except Exception as e:  # noqa: BLE001
                return fail(502, f"minicpm5 调用失败: {e}")
            hb_stop.set()
            # 工具参数照样走同一套校验/修复链（小模型最容易漏必填参数）
            if tool_calls:
                repair_tool_calls(tool_calls, tools)
                repair_tool_calls_with_model(tool_calls, tools)
                clean_tool_call_args(tool_calls)
            log(f"完成 {time.time() - t0:.1f}s  模式=mini直通  输出{len(content)}字  "
                f"工具调用{len(tool_calls)}个" + ("  ⚠守卫拦下" if blocked else ""))
            if not stream:
                self._json(200, {
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": resp_model,
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": content,
                        **({"tool_calls": tool_calls} if tool_calls else {})},
                        "finish_reason": "tool_calls" if tool_calls else "stop"}],
                    "usage": usage or {},
                })
                return
            # 流式：content 已由 mini_guarded_stream 逐段发完，这里只补工具调用帧/结束帧
            if tool_calls:
                emit(chunk({"tool_calls": [
                    {"index": i, "id": tc.get("id") or f"call_{i}", "type": "function",
                     "function": {"name": (tc.get("function") or {}).get("name", ""),
                                  "arguments": (tc.get("function") or {}).get("arguments", "")}}
                    for i, tc in enumerate(tool_calls)]}))
                emit(chunk({}, "tool_calls"))
            else:
                emit(chunk({}, "stop"))
            if usage:
                emit({"id": cid, "object": "chat.completion.chunk", "created": created,
                      "model": resp_model, "choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        ask = last_user_text(messages)
        question = ask or flatten(messages)
        # 只在「用户新开一轮」时做预分析；tool 结果续跑不重复付这份时间。
        # 另外：上下文已经很大时**跳过预分析** —— 那一轮本来就很慢（本地 27B 要先吃几万 token），
        # 再叠 7 次调用纯属雪上加霜；而且大上下文时多视角的边际收益很低。
        ctx_tokens = rough_tokens(messages)
        last_role = (messages[-1].get("role") if messages else "")
        big_ctx = ctx_tokens > CONFIG["preamble_max_tokens"]
        use_preamble = (CONFIG["preamble"] and last_role == "user" and not big_ctx
                        and 0 < len(question) < CONFIG["max_ctx"])
        if CONFIG["preamble"] and last_role == "user" and big_ctx:
            log(f"上下文约 {ctx_tokens} token > {CONFIG['preamble_max_tokens']}，跳过预分析（只做直通调用）")

        content, tool_calls, usage = "", [], {}
        stream_state = {"live": False}       # 是否已经把 content 实时转发出去了
        if tools:
            # ---- 模式 A：agent 在用（带 tools）：先多视角定案，再带 tools 行动 ----
            final_messages, advice_on = list(messages), False
            if use_preamble and ask:
                try:
                    prog = ((lambda m: emit(chunk({"reasoning_content": m + "\n"})))
                            if stream else None)
                    final_messages = inject_advice(messages, pre_analysis(ask, progress=prog))
                    advice_on = True
                except Exception as e:  # noqa: BLE001
                    log(f"pre-analysis 失败（降级直通）：{e}")
                    final_messages = list(messages)

            def call_tools(msgs):
                if stream:
                    # 真流式：Bonsai 一吐 token 就转给 DSH（工具参数先攒着，校验后再发）
                    c, t, u, _cut = relay_stream(msgs, tools, temperature, max_tokens, emit, chunk)
                    stream_state["live"] = True
                    return c, t, u
                msg, usg = boss_with_tools(msgs, tools, temperature, max_tokens)
                return (msg.get("content") or ""), (msg.get("tool_calls") or []), usg

            try:
                content, tool_calls, usage = call_tools(final_messages)
            except UpstreamError as e:
                detail = str(e.body)[:300]
                # 兜底：注入过建议却把上游搞 5xx 了 → 用原始消息重试一次，
                # 保证「加了预分析反而把能跑的请求搞坏」这种事故永不再发生
                if advice_on and e.code >= 500:
                    log(f"注入建议后上游 {e.code}，降级为原始消息重试：{detail[:120]}")
                    try:
                        content, tool_calls, usage = call_tools(list(messages))
                    except Exception as e2:  # noqa: BLE001
                        log(f"降级重试仍失败：{e2}")
                        return fail(502, f"upstream {e.code}: {detail}")
                else:
                    return fail(502, f"upstream {e.code}: {detail}")
            except Exception as e:  # noqa: BLE001
                return fail(502, f"Bonsai 调用失败: {e}")

            # ---- 工具参数校验 + 自修复 ----
            # DSH 的 schema 校验很严：少一个必需参数，整次调用就废（用户截图里的
            # `missing required property "new_string"`）。这里在**交给 DSH 之前**先修：
            #   ① 别名纠正（new_str → new_string 之类）
            #   ② 空值补齐（模型想删文本时 new_string 就是空串，而空值参数会被上游解析丢掉）
            #   ③ 还不行就带着明确的错误提示，让模型重调一次
            if tool_calls:
                # ① 规则修（免费、确定性）：解包 / 类型校准 / 别名 / 空值补齐
                n_fix = repair_tool_calls(tool_calls, tools)
                probs = validate_tool_calls(tool_calls, tools)
                # ①' 语义核对：edit 的 old_string 必须真在文件里（治 `old_string was not found`）
                for _tc in tool_calls:
                    _fn = _tc.get("function") or {}
                    _args = _parse_args(_tc)
                    if isinstance(_args, dict):
                        _ok, _why = verify_edit_against_file(_fn.get("name"), _args)
                        if not _ok:
                            probs.append(f"{_fn.get('name')}: {_why}")
                # ② 小模型修（实测：机械变换很拿手，0.2s；但**内容不许它编**，上面会再核一遍）
                n_llm = 0
                if probs:
                    n_llm = repair_tool_calls_with_model(tool_calls, tools)
                    if n_llm:
                        probs = validate_tool_calls(tool_calls, tools)
                        for _tc in tool_calls:
                            _fn = _tc.get("function") or {}
                            _args = _parse_args(_tc)
                            if isinstance(_args, dict):
                                _ok, _why = verify_edit_against_file(_fn.get("name"), _args)
                                if not _ok:
                                    probs.append(f"{_fn.get('name')}: {_why}")
                # ③ 还不行 → 带提示让大模型重调一次
                if probs:
                    log(f"参数仍不合格：{probs} → 让大模型带提示重调一次")
                    hint = ("你上一次的工具调用参数不合格：" + "；".join(probs) +
                            "。请重新调用同一个工具，并遵守：\n"
                            "1) 参数名必须与 schema 完全一致；\n"
                            "2) 如果是**删除**文字，`new_string` 传空字符串（不能省略该参数）；\n"
                            "3) `old_string` 和 `new_string` **不能相同**，且 `old_string` 不能为空；\n"
                            "4) 要改的内容请给**新的、不同的**文本；不确定原文就先调 `read` 读文件。")
                    try:
                        content, tool_calls, usage = call_tools(
                            inject_advice(final_messages, hint))
                        repair_tool_calls(tool_calls, tools)
                        repair_tool_calls_with_model(tool_calls, tools)
                        probs = validate_tool_calls(tool_calls, tools)
                        log(f"大模型重调后：{probs or '校验通过 ✅'}")
                    except Exception as e:  # noqa: BLE001
                        log(f"重调请求失败（保留原调用交给 DSH 报错）：{e}")

                if probs:
                    # ④ 熔断：同一非法调用反复出现就中断循环，改为返回可读说明
                    if note_invalid_call(tool_calls, probs):
                        log("⚠ 同一非法工具调用反复失败 → 熔断，改为返回说明（避免无限重试）")
                        content = (content or "") + LOOP_BREAK_MESSAGE
                        tool_calls = []
                    else:
                        log("保留这次非法调用交给 DSH 报错（下一次再失败就熔断）")
                else:
                    if n_fix or n_llm:
                        log(f"工具参数已修复并通过校验（规则 {n_fix} 处 / 小模型 {n_llm} 处）✅")
                    _LOOP_STATE.clear()
        else:
            # ---- 模式 B：普通聊天（无 tools）：跑完整 Heavy 流水线 ----
            if use_preamble:
                try:
                    if stream:
                        # 前几步多视角照旧，最后一锤子边生成边发
                        content, usage = heavy_stream(question, emit, chunk)
                        stream_state["live"] = True
                    else:
                        content, _, _ = da.heavy(question, verbose=False)
                except Exception as e:  # noqa: BLE001
                    return fail(502, f"流水线失败: {e}")
            else:
                try:
                    msg, usage = boss_with_tools(messages, [], temperature, max_tokens)
                    content = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls") or []
                except Exception as e:  # noqa: BLE001
                    return fail(502, f"Bonsai 直通失败: {e}")

        hb_stop.set()

        # 复读机清理：本地低比特模型写长文时会"同一段循环 N 次"，这里去重只留真内容。
        # ⚠️ 实时转发过的 content **不能再改**（客户端已经看见了），只能在流里掐断。
        if CONFIG["degeneration_guard"] and content and not stream_state["live"]:
            cleaned, hit, ratio = clean_degeneration(content)
            if hit:
                log(f"⚠ 检测到复读机退化：段落重复率 {ratio:.1%}，"
                    f"{len(content)} → {len(cleaned)} 字符，已按首次出现去重")
                content = cleaned

        # ⚠️ 同样的退化会出现在**工具调用的参数**里！实测：《尾巴与耳朵的敏感篇》45 KB
        # 不是 content 输出的，而是模型用 write 工具一次性写进 arguments 的，
        # 日志里表现为"输出0字 工具调用1个"，所以只清理 content 根本抓不到。
        if CONFIG["degeneration_guard"] and tool_calls:
            clean_tool_call_args(tool_calls)

        dt = time.time() - t0
        log(f"完成 {dt:.1f}s  模式={'tools' if tools else 'chat'}  "
            f"预分析={'有' if use_preamble else '无'}  输出{len(content)}字  工具调用{len(tool_calls)}个")

        if not stream:
            self._json(200, {
                "id": cid, "object": "chat.completion",
                "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": content,
                    **({"tool_calls": tool_calls} if tool_calls else {})},
                    "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": usage or {},
            })
            return

        # ---------- 发出结果帧（headers 已在开头发过）----------
        if stream_state["live"]:
            # content 已经实时转发过了；这里只补**结束帧**（和校验/修复后的工具调用帧）
            if tool_calls:
                emit(chunk({"tool_calls": [
                    {"index": i, "id": tc.get("id") or f"call_{i}", "type": "function",
                     "function": {"name": (tc.get("function") or {}).get("name", ""),
                                  "arguments": (tc.get("function") or {}).get("arguments", "")}}
                    for i, tc in enumerate(tool_calls)]}))
                emit(chunk({}, "tool_calls"))
            else:
                emit(chunk({}, "stop"))
        elif tool_calls:
            emit(chunk({"tool_calls": [
                {"index": i, "id": tc.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": (tc.get("function") or {}).get("name", ""),
                              "arguments": (tc.get("function") or {}).get("arguments", "")}}
                for i, tc in enumerate(tool_calls)]}))
            emit(chunk({}, "tool_calls"))
        elif content:
            step = 32
            for i in range(0, len(content), step):
                emit(chunk({"content": content[i:i + step]}))
            emit(chunk({}, "stop"))
        else:
            emit(chunk({"content": "(本地 Heavy 未产出内容)"}))
            emit(chunk({}, "stop"))
        if usage:
            emit({"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ID, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-preamble", action="store_true",
                    help="关掉多视角预分析（退化成单发 Bonsai 直通，仅用于对照）")
    a = ap.parse_args()
    CONFIG["preamble"] = not a.no_preamble

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log(f"本地 Heavy 服务已启动 http://{a.host}:{a.port}/v1  "
        f"模型 id={MODEL_ID}  预分析={'开' if CONFIG['preamble'] else '关'}")
    log(f"上游：Bonsai {da.BOSS}（{da.BOSS_MODEL}）+ minicpm5 {da.MINI_API}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("收到 Ctrl+C，退出")
        srv.shutdown()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()




