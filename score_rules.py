#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""score_rules.py —— 规则修复的得分（用**文件里真实存在**的 old_string，避免判据误伤）。

同时验证「语义核对」：edit 的 old_string 必须真在文件里，否则视为编造。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402

P = r"C:\Users\Administrator\Desktop\abligrated\尾巴与耳朵的敏感篇.txt"
lines = [x.strip() for x in Path(P).read_text(encoding="utf-8", errors="replace").splitlines()
         if len(x.strip()) > 12]
OLD = lines[0][:18]          # 文件里真实存在的一句
print(f"用作 old_string 的真实片段：{OLD!r}\n")

TOOLS = [
    {"type": "function", "function": {"name": "edit", "parameters": {
        "type": "object", "properties": {
            "file_path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
        "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"}},
        "required": ["file_path"]}}},
]

CASES = [
    ("缺 new_string（删文本）", "edit", {"file_path": P, "old_string": OLD}, True),
    ("键名 new_str", "edit", {"file_path": P, "old_string": OLD, "new_str": "新内容"}, True),
    ("路径键 path", "edit", {"path": P, "old_string": OLD, "new_string": "新内容"}, True),
    ("write 缺 content", "write", {"file_path": P}, True),
    ("多包一层 arguments", "edit",
     {"arguments": {"file_path": P, "old_string": OLD, "new_string": "新"}}, True),
    ("二次编码字符串", "edit",
     json.dumps({"file_path": P, "old_string": OLD, "new_string": "新"}, ensure_ascii=False), True),
    ("offset 传成字符串", "read", {"file_path": P, "offset": "10"}, True),
    ("new_string 传成数组", "edit",
     {"file_path": P, "old_string": OLD, "new_string": ["A", "B"]}, True),
    # 下面三例**本来就修不了 / 必须拦住**，考的是「会不会编造」
    ("缺 old_string(应拒绝)", "edit", {"file_path": P, "new_string": "B"}, False),
    ("新旧相同(应拒绝)", "edit", {"file_path": P, "old_string": OLD, "new_string": OLD}, False),
    ("编造原文(必须拦住)", "edit",
     {"file_path": P, "old_string": "这段文字文件里根本不存在喵", "new_string": "X"}, False),
]

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    good = 0
    for label, tool, argv, should_fix in CASES:
        raw = argv if isinstance(argv, str) else json.dumps(argv, ensure_ascii=False)
        tc = [{"id": "x", "type": "function", "function": {"name": tool, "arguments": raw}}]
        h.repair_tool_calls(tc, TOOLS)
        try:
            a = json.loads(tc[0]["function"]["arguments"])
        except Exception:  # noqa: BLE001
            a = {}
        probs = h.validate_tool_calls(tc, TOOLS)
        sem_ok, sem_msg = (h.verify_edit_against_file(tool, a) if isinstance(a, dict)
                           else (False, "解析失败"))
        fixed_ok = isinstance(a, dict) and not probs and sem_ok
        marked = "✅" if (fixed_ok == should_fix) else "❌"
        good += (fixed_ok == should_fix)
        print(f"  {marked} {label:<22} 期望{'能修' if should_fix else '拒绝'}"
              f"  实际{'已修好' if fixed_ok else '未修/拒绝'}"
              f"  结构={probs or 'OK'} 语义={sem_msg or 'OK'}")
    print(f"\n规则修复符合预期：{good}/{len(CASES)}")
    print("（前 8 例应当被修好；后 2 例应当拒绝，因为原文缺失/新旧相同 —— 拒绝后走熔断提示让模型先 read）")
