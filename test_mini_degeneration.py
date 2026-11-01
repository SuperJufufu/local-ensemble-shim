"""复现「小模型输出诡异语言（卡纳达/马拉雅拉姆等）」。

假设：MiniCPM5-2B 只有 2.5B，当 DSH 把「长 system + 一堆工具 schema + 多轮历史」
一起塞进去时，它会失去连贯性，从高熵区采样 → 掉进训练里罕见的印度语系 token。

做法：同一句话，在 5 种「输入形态」下各问一遍，看什么时候开始吐怪字符。
  A 短问句、无工具                  ← 基线
  B 短问句 + 21 个工具 schema       ← DSH agent 模式的真实形状
  C 长 system(~6k token) + 工具
  D 长 system + 工具 + 8 轮历史
  E 长 system + 工具 + 短问句 + 思考开启

判据：回复里出现「中文/英文之外」的罕见文字（卡纳达 U+0C80-0CFF、马拉雅拉姆
U+0D00-0D7F、泰卢固、泰米尔、天城文、阿拉伯、希伯来、泰文）占比过高即判定异常。

用法：python test_mini_degeneration.py [--model minicpm5-2b-32k] [--ctx 32768]
"""
import argparse
import json
import time
import unicodedata
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

WEIRD_BLOCKS = [
    (0x0900, 0x097F, "天城文"), (0x0C00, 0x0C7F, "泰卢固"), (0x0C80, 0x0CFF, "卡纳达"),
    (0x0D00, 0x0D7F, "马拉雅拉姆"), (0x0B80, 0x0BFF, "泰米尔"), (0x0E00, 0x0E7F, "泰文"),
    (0x0600, 0x06FF, "阿拉伯"), (0x0590, 0x05FF, "希伯来"), (0x0980, 0x09FF, "孟加拉"),
    (0x0A00, 0x0A7F, "古木基"), (0x0A80, 0x0AFF, "古吉拉特"), (0x0E80, 0x0EFF, "老挝"),
    (0x1000, 0x109F, "缅甸"), (0x1780, 0x17FF, "高棉"),
]


def weird_ratio(text):
    """返回 (罕见字符数, 占比, 命中的文字名集合)，只统计字母类字符。"""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0, 0.0, set()
    hits, names = 0, set()
    for c in letters:
        cp = ord(c)
        for lo, hi, name in WEIRD_BLOCKS:
            if lo <= cp <= hi:
                hits += 1
                names.add(name)
                break
    return hits, hits / len(letters), names


TOOLS = [
    ("read", "读取文件内容", {"file_path": {"type": "string"}, "offset": {"type": "integer"},
                              "limit": {"type": "integer"}}, ["file_path"]),
    ("write", "写入（覆盖）文件", {"file_path": {"type": "string"}, "content": {"type": "string"}},
     ["file_path", "content"]),
    ("edit", "把文件里的 old_string 替换成 new_string",
     {"file_path": {"type": "string"}, "old_string": {"type": "string"},
      "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
     ["file_path", "old_string", "new_string"]),
    ("bash", "执行 PowerShell 命令", {"command": {"type": "string"},
                                      "timeout_ms": {"type": "integer"}}, ["command"]),
    ("glob", "按路径模式找文件", {"pattern": {"type": "string"}, "path": {"type": "string"}},
     ["pattern"]),
    ("grep", "按正则搜索文件内容", {"pattern": {"type": "string"}, "path": {"type": "string"},
                                    "include": {"type": "string"}}, ["pattern"]),
    ("web_search", "搜索网络", {"queries": {"type": "array", "items": {"type": "string"}}},
     ["queries"]),
    ("web_fetch", "抓取网页内容", {"url": {"type": "string"}}, ["url"]),
    ("todo_write", "记录任务列表", {"todos": {"type": "array", "items": {"type": "object"}}},
     ["todos"]),
    ("team_task_create", "创建共享任务", {"subject": {"type": "string"},
                                          "description": {"type": "string"}}, ["subject", "description"]),
    ("team_task_list", "列出共享任务", {"status": {"type": "string"}}, []),
    ("team_task_get", "读取一个共享任务", {"task_id": {"type": "string"}}, ["task_id"]),
    ("team_task_update", "更新共享任务", {"task_id": {"type": "string"}, "action": {"type": "string"},
                                          "expected_revision": {"type": "integer"}},
     ["task_id", "action", "expected_revision"]),
    ("spawn_teammate", "创建一个队友", {"name": {"type": "string"}, "description": {"type": "string"},
                                        "prompt": {"type": "string"}}, ["name", "description", "prompt"]),
    ("send_message", "给队友发消息", {"target": {"type": "string"}, "message": {"type": "string"}},
     ["target", "message"]),
    ("list_agents", "列出所有成员", {}, []),
    ("wait_agent", "等待队友", {"timeout_ms": {"type": "integer"}}, []),
    ("create_goal", "创建一个持久目标", {"objective": {"type": "string"}}, ["objective"]),
    ("get_goal", "读取当前目标", {}, []),
    ("update_goal", "更新目标", {"goal_id": {"type": "string"}, "revision": {"type": "number"},
                                  "action": {"type": "string"}}, ["goal_id", "revision", "action"]),
    ("ask_user_question", "向用户提问", {"questions": {"type": "array", "items": {"type": "object"}}},
     ["questions"]),
]


def dsh_tools():
    out = []
    for name, desc, props, req in TOOLS:
        out.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    return out


SYSTEM_HEAD = """你是 DeepSeek Harness（DSH）里的编码助手，运行在 Windows 上，工作目录是 C:\\Users\\Administrator\\Desktop。

# 工作方式
- 需要看文件内容时必须调用 read；不要凭空猜文件内容。
- 改动文件用 edit（定点替换）或 write（整文件覆盖）。
- 不确定路径时先用 glob，找内容用 grep。
- 长任务先 todo_write 记计划。

# 语气
- 用中文回答，简洁直接，不要客套。

# 工具使用规则
- 调用工具前先想清楚参数；参数必须符合 schema。
- 一次只做一件明确的事。
- 工具返回后再决定下一步。
"""

FILLER = ("- 历史约定 %02d：本地模型优先；涉及精确编辑时先 read 再 edit；"
          "批处理脚本保持 ASCII；不要擅自重启用户的服务；改动前先备份。\n")


def long_system(n=140):
    return SYSTEM_HEAD + "\n# 项目长期约定\n" + "".join(FILLER % i for i in range(n))


QUESTION = "你好，先简单介绍一下你能做什么。"


def ask(model, ctx, messages, tools=None, think=False, max_tokens=300, timeout=600, temp=0.7):
    payload = {"model": model, "messages": messages, "stream": False,
               "max_tokens": max_tokens, "temperature": temp}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if think:
        payload["think"] = True
    req = urllib.request.Request(OLLAMA + "/v1/chat/completions",
                                 data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with _opener.open(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode("utf-8"))
    msg = ((obj.get("choices") or [{}])[0].get("message") or {})
    return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
            "reasoning": msg.get("reasoning") or "", "wall_s": round(time.time() - t0, 1),
            "finish": (obj.get("choices") or [{}])[0].get("finish_reason"),
            "usage": obj.get("usage") or {}}


def turns(n):
    hist = []
    for i in range(n):
        hist.append({"role": "user", "content": "第 %d 步：看一下 README 的前几行。" % (i + 1)})
        hist.append({"role": "assistant", "content": "好，我先读 README。"})
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minicpm5-2b-32k")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    cases = [
        ("A 短问句·无工具", [{"role": "user", "content": QUESTION}], None, False),
        ("B 短问句+21工具", [{"role": "user", "content": QUESTION}], dsh_tools(), False),
        ("C 长system+工具", [{"role": "system", "content": long_system()},
                             {"role": "user", "content": QUESTION}], dsh_tools(), False),
        ("D 长system+工具+8轮", [{"role": "system", "content": long_system()}] + turns(8) +
                              [{"role": "user", "content": QUESTION}], dsh_tools(), False),
        ("E 长system+工具+思考", [{"role": "system", "content": long_system()},
                                 {"role": "user", "content": QUESTION}], dsh_tools(), True),
    ]

    print("模型 %s ｜ num_ctx %d ｜ temp %s ｜ 每种形态 %d 次 ｜ 长 system 约 %d 字\n"
          % (args.model, args.ctx, args.temp, args.samples, len(long_system())))
    bad = 0
    total = 0
    for label, msgs, tools, think in cases:
        for si in range(args.samples):
            total += 1
            try:
                r = ask(args.model, args.ctx, msgs, tools, think, temp=args.temp)
            except Exception as e:  # noqa: BLE001
                print("  %-20s ❌ 请求失败：%s" % (label, e))
                bad += 1
                continue
            hits, ratio, names = weird_ratio(r["content"])
            flag = (ratio > 0.05) or (hits >= 8)
            if flag:
                bad += 1
            print("  %-20s #%d prompt=%5d tok %4.1fs %s  怪字 %d (%.0f%%) %s"
                  % (label, si + 1, (r["usage"].get("prompt_tokens") or 0), r["wall_s"],
                     "🚨 诡异语言" if flag else "✅ 正常", hits, ratio * 100,
                     "/".join(sorted(names)) if names else ""))
            if flag or si == 0:
                print("      finish=%s tool_calls=%d 正文=%r"
                      % (r["finish"], len(r["tool_calls"]), r["content"][:110]))

    print("\n" + "=" * 62)
    print("出现诡异语言：%d/%d 次采样" % (bad, total))
    return 0 if bad == 0 else 1

    print("\n" + "=" * 62)
    print("出现诡异语言：%d/%d 种输入形态" % (bad, len(cases)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
