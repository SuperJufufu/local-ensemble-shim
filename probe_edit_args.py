#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe_edit_args.py —— 复现「edit 工具缺 new_string」这个报错。

给模型和 DSH 一模一样的工具 schema（read / write / edit），
让它做一次修改，然后检查每次返回的参数**键集合**是否齐全。
跑多轮统计失败率，并记录它用什么错键名替代。
"""
import json
import sys
import urllib.request
from collections import Counter

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BOSS = "http://127.0.0.1:8010/v1/chat/completions"
MODEL = "bonsai-2-27b-abliterated-mtp"

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "读取 UTF-8 文本文件",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "offset": {"type": "integer"}, "limit": {"type": "integer"}},
            "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "写入（覆盖）UTF-8 文本文件",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "把文件里的 old_string 字面量替换成 new_string",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string", "description": "要被替换的原文"},
            "new_string": {"type": "string", "description": "替换后的新文本"},
            "replace_all": {"type": "boolean"}},
            "required": ["file_path", "old_string", "new_string"]}}},
]

SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

TARGET = r"C:\Users\Administrator\Desktop\abligrated\尾巴与耳朵的敏感篇.txt"

PROMPTS = [
    (f"请用 edit 工具，把 {TARGET} 里的「猫娘」这个称呼改成「小喵」。",
     "短替换"),
    (f"请用 edit 工具，把 {TARGET} 开头第一句做一次润色改写，让它更生动。",
     "改写一句"),
    (f"请用 edit 工具，在 {TARGET} 的结尾追加一段 100 字左右的收尾描写。",
     "追加一段"),
]


def call(prompt):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "tools": TOOLS, "tool_choice": "auto",
               "temperature": 0.3, "max_tokens": 600, "stream": False,
               "frequency_penalty": 0.2, "presence_penalty": 0.1}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BOSS, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=600) as r:
        resp = json.loads(r.read().decode("utf-8"))
    msg = ((resp.get("choices") or [{}])[0].get("message") or {})
    return msg.get("tool_calls") or []


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    rounds = 3
    stats = Counter()
    bad_keys = Counter()
    for prompt, label in PROMPTS:
        for i in range(rounds):
            try:
                calls = call(prompt)
            except Exception as e:  # noqa: BLE001
                print(f"[{label} #{i+1}] 请求失败 {e}")
                continue
            if not calls:
                print(f"[{label} #{i+1}] ❌ 没调工具")
                stats["没调工具"] += 1
                continue
            tc = calls[0]
            name = (tc.get("function") or {}).get("name")
            raw = (tc.get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except Exception:  # noqa: BLE001
                args = {}
            keys = set(args.keys())
            req = set(SCHEMAS.get(name, {}).get("required", []))
            missing = req - keys
            ok = not missing
            stats["合法" if ok else "缺参数"] += 1
            for m in missing:
                bad_keys[m] += 1
            extra = keys - req
            print(f"[{label} #{i+1}] 工具={name} 键={sorted(keys)} "
                  f"{'✅' if ok else '❌ 缺 ' + str(sorted(missing))}"
                  f"{'  (多余键: ' + str(sorted(extra)) + ')' if extra else ''}")
    print("\n统计：", dict(stats))
    if bad_keys:
        print("缺失的必需参数：", dict(bad_keys))
