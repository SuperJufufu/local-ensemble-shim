#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe_empty_arg.py —— 验证假设：模型**删除文本**时会把 new_string 传成空串，
而 llama.cpp 的工具解析会把空参数丢掉 → 上游报 missing required property "new_string"。

对照组：非空替换（应当 100% 合法）。
"""
import json
import sys
import urllib.request
from collections import Counter

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BOSS = "http://127.0.0.1:8010/v1/chat/completions"
MODEL = "bonsai-2-27b-abliterated-mtp"
TARGET = r"C:\Users\Administrator\Desktop\abligrated\尾巴与耳朵的敏感篇.txt"

TOOLS = [{"type": "function", "function": {
    "name": "edit", "description": "把文件里的 old_string 字面量替换成 new_string（替换为空字符串即删除）",
    "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"}},
        "required": ["file_path", "old_string", "new_string"]}}}]

CASES = [
    ("删除", f"请用 edit 工具，把 {TARGET} 里那句「喵喵整个人都轻颤了一下」整句删掉。"),
    ("替换", f"请用 edit 工具，把 {TARGET} 里的「猫娘」改成「小喵」。"),
    ("删除", f"请用 edit 工具，删除 {TARGET} 中重复出现的多余段落。"),
]


def call(prompt):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "tools": TOOLS, "tool_choice": "auto", "temperature": 0.3,
               "max_tokens": 500, "stream": False,
               "frequency_penalty": 0.2, "presence_penalty": 0.1}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BOSS, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=600) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return (((resp.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    stat = Counter()
    for label, prompt in CASES:
        for i in range(2):
            calls = call(prompt)
            if not calls:
                print(f"[{label} #{i+1}] 没调工具"); stat["没调工具"] += 1; continue
            raw = (calls[0].get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except Exception:  # noqa: BLE001
                args = {}
            has = "new_string" in args
            val = args.get("new_string")
            empty = has and (val == "" or val is None)
            print(f"[{label} #{i+1}] 键={sorted(args.keys())}  "
                  f"new_string={'缺失 ❌' if not has else ('空串' if empty else f'{len(val)} 字符')}"
                  f"{'   ← 正是报错原因' if not has else ''}")
            stat["缺键" if not has else ("空串" if empty else "非空")] += 1
    print("\n统计：", dict(stat))
