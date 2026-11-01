#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""test_guard.py —— 单元测试：复读机清理（content + 工具参数）与裁剪保护。

用法： python test_guard.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAIL.append(msg)


print("[1] 复读机清理：正常文章不该被误伤")
normal = "第一段正常内容。\n\n第二段不同的内容。\n\n第三段又是新内容。\n\n第四段继续。\n\n第五段。\n\n第六段结尾。"
out, hit, ratio = h.clean_degeneration(normal)
check(not hit and out == normal, f"6 段各不相同 → 不触发（重复率 {ratio:.0%}）")

print("\n[2] 复读机清理：退化文本应当被折叠成唯一内容")
degen = "\n\n".join(["同一段话反复出现喵。" * 2, "第二句也不停重复喵。" * 2] * 40)
out2, hit2, ratio2 = h.clean_degeneration(degen)
check(hit2, f"触发清理（重复率 {ratio2:.0%}）")
check(len(out2) < len(degen) * 0.1, f"折叠后 {len(out2)} 字符 < 原始 10%（原始 {len(degen)}）")
n_p, n_u, r2 = h.degeneration_report(out2)
check(r2 == 0, f"清理后重复率归零（{n_p} 段 / 唯一 {n_u} 段）")

print("\n[3] 工具参数清理：write 工具的 content 参数退化")
big_degen = "\n\n".join(["循环段落内容示例。" * 3] * 30)
calls = [{"id": "c1", "type": "function",
          "function": {"name": "write",
                       "arguments": json.dumps({"file_path": r"C:\x.txt", "content": big_degen},
                                               ensure_ascii=False)}}]
n = h.clean_tool_call_args(calls)
args = json.loads(calls[0]["function"]["arguments"])
check(n == 1, f"报告清理了 {n} 处参数")
check(len(args["content"]) < len(big_degen) * 0.2,
      f"参数被折叠：{len(big_degen)} → {len(args['content'])} 字符")
check(args["file_path"] == r"C:\x.txt", "其它参数（路径）原样保留")

print("\n[4] 工具参数清理：正常参数不动")
ok_args = json.dumps({"path": "a.txt", "max": 10}, ensure_ascii=False)
calls2 = [{"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": ok_args}}]
check(h.clean_tool_call_args(calls2) == 0, "短参数不触发")
check(calls2[0]["function"]["arguments"] == ok_args, "参数内容未被改写")

print("\n[5] 裁剪：必须保住「用户提问」（否则模板 500 No user query found）")
m = [{"role": "system", "content": "SYS"},
     {"role": "user", "content": "请继续写我们的文章（唯一真实提问）"}]
for i in range(40):
    m.append({"role": "assistant", "content": "",
              "tool_calls": [{"id": str(i), "type": "function",
                              "function": {"name": "read_file", "arguments": "{}"}}]})
    m.append({"role": "tool", "content": "x" * 3000, "tool_call_id": str(i)})
m.append({"role": "user", "content": "<tool_response>" + "y" * 2000 + "</tool_response>"})
t = h.trim_to_fit(m, 6000)
check(h.rough_tokens(t) <= 6000, f"裁到预算内（{h.rough_tokens(t)} ≤ 6000）")
check(any(x.get("role") == "user" and "请继续写" in str(x.get("content")) for x in t),
      "真实提问被保住")
check(t[0]["role"] == "system", "system 仍在首位")

print("\n[6] 工具参数校验 + 自修复（用户截图里的三类报错）")
TOOLS = [
    {"type": "function", "function": {"name": "edit", "parameters": {
        "type": "object",
        "properties": {"file_path": {}, "old_string": {}, "new_string": {}, "replace_all": {}},
        "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"file_path": {}, "content": {}},
        "required": ["file_path", "content"]}}},
]


def mk(name, obj):
    return [{"id": "c", "type": "function",
             "function": {"name": name, "arguments": json.dumps(obj, ensure_ascii=False)}}]


c = mk("edit", {"file_path": "a.txt", "old_string": "旧句"})   # new_string 整个缺失
_before = h.validate_tool_calls(c, TOOLS)
h.repair_tool_calls(c, TOOLS)
check(_before and not h.validate_tool_calls(c, TOOLS),
      f'缺 new_string → 补空串修好（原来报 {_before}）')
check(json.loads(c[0]["function"]["arguments"])["new_string"] == "", "补的是空串（=删除语义）")

c = mk("edit", {"file_path": "a.txt", "old_string": "旧", "new_str": "新"})  # 键名写错
h.repair_tool_calls(c, TOOLS)
check(json.loads(c[0]["function"]["arguments"]).get("new_string") == "新", "new_str → new_string 别名纠正")

c = mk("edit", {"file_path": "a.txt", "old_string": "一样", "new_string": "一样"})
check("old_string 与 new_string 相同" in " ".join(h.validate_tool_calls(c, TOOLS)),
      "old_string == new_string 会被识别出来（交给模型重调）")

c = mk("write", {"file_path": "a.txt"})
h.repair_tool_calls(c, TOOLS)
check(json.loads(c[0]["function"]["arguments"]).get("content") == "", "write 缺 content 补空串")

c = mk("edit", {"file_path": "a.txt", "old_string": "旧", "new_string": "新"})
_before_args = c[0]["function"]["arguments"]
check(h.repair_tool_calls(c, TOOLS) == 0 and c[0]["function"]["arguments"] == _before_args,
      "合法调用不动它（无误伤）")

c = mk("edit", {"old_string": "旧", "new_string": "新"})       # 缺 file_path
h.repair_tool_calls(c, TOOLS)
check("file_path" not in json.loads(c[0]["function"]["arguments"]) or
      json.loads(c[0]["function"]["arguments"])["file_path"] != "",
      "缺 file_path **不会**被补成空串（那会变成另一个错误）")

print("\n[7] 循环熔断：同一非法调用反复出现必须中断")
h._LOOP_STATE.clear()
bad = mk("edit", {"file_path": "a.txt", "old_string": "一样", "new_string": "一样"})
probs = h.validate_tool_calls(bad, TOOLS)
check(bool(probs), f"识别出非法调用：{probs}")
r1 = h.note_invalid_call(bad, probs)
r2 = h.note_invalid_call(bad, probs)
check(not r1 and r2, "第 1 次不熔断（让 DSH 报一次错），第 2 次熔断")
other = mk("edit", {"file_path": "b.txt", "old_string": "x", "new_string": "x"})
check(not h.note_invalid_call(other, probs), "不同的非法调用各自计数，不被误熔断")
check("edit" in h.LOOP_BREAK_MESSAGE and "write" in h.LOOP_BREAK_MESSAGE,
      "熔断消息给出可执行的替代做法（read / write / 切云端）")

print("\n" + "=" * 52)
print(f"  {'✅ 全部通过' if not FAIL else '❌ 失败 ' + str(len(FAIL)) + ' 项'}")
sys.exit(1 if FAIL else 0)
