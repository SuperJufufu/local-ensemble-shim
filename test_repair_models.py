#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""test_repair_models.py —— 比一比「谁来修工具参数」更准：

  A) 规则修复      ：我写的别名 + 空值补齐（免费、瞬时）
  B) minicpm5-2b   ：小模型，把 schema + 坏参数 + 报错 丢给它修（快、格式稳）
  C) Bonsai-27B    ：大模型修（慢，但更聪明）

评分标准：修完之后能否通过 **DSH 那套 schema 校验**（这就是用户看到的报错来源）。
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402

OLLAMA = "http://127.0.0.1:11434/api/chat"
MINI = "minicpm5-2b-32k"  # 与 DSH 的 local-mini 路由同一实例，避免重复加载
BOSS = "http://127.0.0.1:8010/v1/chat/completions"
BOSS_MODEL = "bonsai-2-27b-abliterated-mtp"
TARGET = r"C:\Users\Administrator\Desktop\abligrated\尾巴与耳朵的敏感篇.txt"

TOOLS = [
    {"type": "function", "function": {
        "name": "edit", "description": "把文件里的 old_string 替换成 new_string",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"}},
            "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "写入（覆盖）文件",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "读取文件",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "offset": {"type": "integer"}},
            "required": ["file_path"]}}},
]
SCHEMAS = h._schemas(TOOLS)

# (说明, 工具名, 原始 arguments 字符串)
CASES = [
    ("缺 new_string（想删文本）", "edit",
     json.dumps({"file_path": TARGET, "old_string": "喵喵整个人都轻颤了一下"}, ensure_ascii=False)),
    ("键名写成 new_str", "edit",
     json.dumps({"file_path": TARGET, "old_string": "猫娘", "new_str": "小喵"}, ensure_ascii=False)),
    ("新旧文本相同", "edit",
     json.dumps({"file_path": TARGET, "old_string": "猫娘", "new_string": "猫娘"}, ensure_ascii=False)),
    ("路径键名写成 path", "edit",
     json.dumps({"path": TARGET, "old_string": "A", "new_string": "B"}, ensure_ascii=False)),
    ("write 缺 content", "write", json.dumps({"file_path": TARGET}, ensure_ascii=False)),
    ("参数被多包了一层", "edit",
     json.dumps({"arguments": {"file_path": TARGET, "old_string": "A", "new_string": "B"}}, ensure_ascii=False)),
    ("arguments 是二次编码的字符串", "edit",
     json.dumps(json.dumps({"file_path": TARGET, "old_string": "A", "new_string": "B"}, ensure_ascii=False),
                ensure_ascii=False)),
    ("offset 传成字符串", "read",
     json.dumps({"file_path": TARGET, "offset": "10"}, ensure_ascii=False)),
    ("new_string 传成了数组", "edit",
     json.dumps({"file_path": TARGET, "old_string": "A", "new_string": ["B", "C"]}, ensure_ascii=False)),
    ("缺 old_string", "edit",
     json.dumps({"file_path": TARGET, "new_string": "B"}, ensure_ascii=False)),
]


FILE_TEXT = ""
try:
    FILE_TEXT = Path(TARGET).read_text(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def check(tool_name, args, semantic=True):
    """按 DSH 的 schema 校验（返回问题列表，空=合法）。

    semantic=True 时额外查**语义**：edit 的 old_string 必须真的出现在文件里。
    ⚠️ 必须查：小模型会把 old_string 修成"删除需要替换的内容"这种**编造文本**，
    结构校验能过，但到 DSH 那里必然 `old_string was not found`。
    """
    if not isinstance(args, dict):
        return [f"顶层不是 JSON 对象（是 {type(args).__name__}）"]
    tc = [{"id": "x", "type": "function",
           "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False)}}]
    # 加严：类型也要对（validate_tool_calls 只看必需键，这里额外查类型）
    probs = h.validate_tool_calls(tc, TOOLS)
    props = (SCHEMAS.get(tool_name) or {}).get("properties") or {}
    for k, v in args.items():
        spec = props.get(k) or {}
        t = spec.get("type")
        if t == "string" and not isinstance(v, str):
            probs.append(f"{k} 类型应为 string")
        if t == "integer" and not isinstance(v, int):
            probs.append(f"{k} 类型应为 integer")
        if t == "boolean" and not isinstance(v, bool):
            probs.append(f"{k} 类型应为 boolean")
    if semantic and tool_name == "edit" and isinstance(args.get("old_string"), str) \
            and args["old_string"] and FILE_TEXT and args["old_string"] not in FILE_TEXT:
        probs.append("old_string 不在文件里（编造）")
    return probs


def repair_llm(model, tool_name, raw_args, problem):
    """让模型给出修复后的参数 JSON。"""
    prompt = (
        "你是工具调用参数修复器。\n"
        f"工具名：{tool_name}\n"
        f"工具 schema：{json.dumps(SCHEMAS[tool_name], ensure_ascii=False)}\n"
        f"当前参数（不合法）：{raw_args}\n"
        f"校验错误：{problem}\n\n"
        "请输出修复后的参数，**只输出一个 JSON 对象**，不要解释、不要 markdown 代码块。\n"
        "规则：参数名必须与 schema 完全一致；类型必须匹配；\n"
        "如果业务是删除文字，new_string 传空字符串 \"\"（但要保留这个参数）；\n"
        "old_string 和 new_string 不能是同一段文字。"
    )
    t0 = time.time()
    if model == "mini":
        payload = {"model": MINI, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "think": False,
                   "options": {"num_ctx": 8192, "temperature": 0.0, "num_predict": 500}}
        with h._opener.open(urllib.request.Request(  # noqa: SLF001
                OLLAMA, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}), timeout=300) as r:
            txt = (json.loads(r.read().decode("utf-8")).get("message") or {}).get("content", "")
    else:
        payload = {"model": BOSS_MODEL, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": 500, "stream": False}
        with h._opener.open(urllib.request.Request(  # noqa: SLF001
                BOSS, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}), timeout=600) as r:
            txt = ((json.loads(r.read().decode("utf-8")).get("choices") or [{}])[0]
                   .get("message") or {}).get("content", "")
    dt = time.time() - t0
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None, dt
    try:
        return json.loads(m.group(0)), dt
    except Exception:  # noqa: BLE001
        return None, dt


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ROUNDS = 3
    score = {"规则修复": 0, "minicpm5": 0, "Bonsai": 0}
    tsum = {"minicpm5": 0.0, "Bonsai": 0.0}
    print(f"{'用例':<24}{'规则':>6}{'minicpm5':>10}{'Bonsai':>8}")
    print("-" * 52)
    for label, tool, raw in CASES:
        # A) 规则修复
        tc = [{"id": "x", "type": "function", "function": {"name": tool, "arguments": raw}}]
        h.repair_tool_calls(tc, TOOLS)
        try:
            a_args = json.loads(tc[0]["function"]["arguments"])
        except Exception:  # noqa: BLE001
            a_args = {}
        a_ok = isinstance(a_args, dict) and not check(tool, a_args)
        # 原始问题描述（给 LLM 用）
        probs = check(tool, a_args if a_args else {})
        problem = "; ".join(h.validate_tool_calls(
            [{"id": "x", "type": "function", "function": {"name": tool, "arguments": raw}}], TOOLS)) or "参数不合法"

        b_ok = c_ok = False
        for tag in ("mini", "Bonsai"):
            fixed, dt = repair_llm(tag, tool, raw, problem)
            ok = bool(fixed) and not check(tool, fixed)
            if tag == "mini":
                b_ok = ok
                tsum["minicpm5"] += dt
            else:
                c_ok = ok
                tsum["Bonsai"] += dt
        score["规则修复"] += a_ok
        score["minicpm5"] += b_ok
        score["Bonsai"] += c_ok
        print(f"{label:<24}{'✅' if a_ok else '❌':>6}{'✅' if b_ok else '❌':>10}{'✅' if c_ok else '❌':>8}")
    n = len(CASES)
    print("-" * 52)
    print(f"{'合计':<24}{score['规则修复']:>4}/{n}{score['minicpm5']:>8}/{n}{score['Bonsai']:>6}/{n}")
    print(f"\n平均耗时：minicpm5 {tsum['minicpm5']/n:.1f}s/次   Bonsai {tsum['Bonsai']/n:.1f}s/次")


