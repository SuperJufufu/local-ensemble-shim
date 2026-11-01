"""验证 DSH 的 local-mini 路由（Ollama OpenAI 兼容口）能否扛住真实 DSH 请求。

检查四件事：
  1. SSE 流式：真的一帧一帧回来，且 finish_reason 正常收尾；
  2. 工具调用：tools 声明被接受，tool_calls 以 delta 形式分片到达并能拼回完整参数；
  3. 32k 上下文真的生效：塞进 ~9k token 的长 system 后 prompt_tokens 远大于 4096；
  4. 长 system + 工具同时存在时不会 500。

用法：python test_local_mini_route.py
"""
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:11434/v1"
MODEL = "minicpm5-2b-32k"
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取一个文件的文本内容",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"}},
            "required": ["path"],
        },
    },
}]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  — " + detail) if detail else ""))


def stream(messages, tools=None, max_tokens=400, timeout=180, include_usage=False):
    payload = {"model": MODEL, "messages": messages, "stream": True,
               "max_tokens": max_tokens, "temperature": 0.2}
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(BASE + "/chat/completions",
                                data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer ollama-local"})
    frames, content, tool_calls, finish, usage = [], [], {}, None, None
    t0 = time.time()
    ttfb = None
    with _opener.open(req, timeout=timeout) as r:
        status = r.status
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            if ttfb is None:
                ttfb = time.time() - t0
            frames.append(body)
            try:
                obj = json.loads(body)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0),
                                                 {"name": None, "args": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
    return {"status": status, "frames": frames, "content": "".join(content),
            "tool_calls": tool_calls, "finish": finish, "usage": usage,
            "ttfb": ttfb, "total": time.time() - t0}


print("== 1. SSE 流式 ==")
r = stream([{"role": "user", "content": "用中文从 1 数到 30，每个数字后面加一个逗号，只输出数字。"}],
           max_tokens=600)   # ⚠️ 走 OpenAI 兼容口时 thinking 不会被关掉，给少了会被思考吃光、content 为空
check("HTTP 200", r["status"] == 200, "status=%s" % r["status"])
check("SSE 帧数 > 20（真流式而非一次性）", len(r["frames"]) > 20, "frames=%d" % len(r["frames"]))
check("首帧延迟 < 3s", (r["ttfb"] or 99) < 3, "ttfb=%.2fs" % (r["ttfb"] or -1))
check("有正文", bool(r["content"].strip()), repr(r["content"][:60]))
check("finish_reason 正常收尾", r["finish"] in ("stop", "length"), "finish=%s" % r["finish"])

print("\n== 1b. stream_options.include_usage 是否被支持 ==")
ru = stream([{"role": "user", "content": "只回复 OK"}], max_tokens=10, include_usage=True)
check("include_usage 时末帧带 usage", bool(ru["usage"]),
      json.dumps(ru["usage"] or {}, ensure_ascii=False) + "（空=不支持，DSH 不依赖它，仅记录）")
print("  （此项仅记录，不计入成败）")
results.pop()

print("\n== 2. 工具调用（流式 delta 拼装）==")
r2 = stream([{"role": "system", "content": "你是编码助手，需要读文件时必须调用 read_file 工具。"},
             {"role": "user", "content": "请读一下 C:\\Windows\\win.ini 的内容。"}],
            tools=TOOLS, max_tokens=300)
check("HTTP 200", r2["status"] == 200, "status=%s" % r2["status"])
got_tc = bool(r2["tool_calls"])
check("返回了 tool_calls", got_tc,
      json.dumps(r2["tool_calls"], ensure_ascii=False)[:160] if got_tc
      else ("退化成正文: " + repr(r2["content"][:80])))
if got_tc:
    slot = r2["tool_calls"][0]
    check("工具名正确", slot["name"] == "read_file", str(slot["name"]))
    try:
        args = json.loads(slot["args"] or "{}")
        ok = isinstance(args.get("path"), str) and args["path"].strip() != ""
        check("参数是合法 JSON 且 path 非空", ok, json.dumps(args, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        check("参数是合法 JSON", False, "%s :: %r" % (e, slot["args"][:120]))

print("\n== 3. 32k 上下文生效（长 system 不被静默截断）==")
filler = ("这是一个用于填充上下文长度的事实条目，编号 %04d，"
          "内容包括若干无关紧要的描述文字，用来把 prompt 撑到几千 token 以上。\n")
big = "".join(filler % i for i in range(260))
# 用非流式调用：OpenAI 兼容口只在非流式（或 stream_options）下回 usage，
# 而 prompt_tokens 正是判断「有没有被截到 4096」的证据。
req3 = urllib.request.Request(
    BASE + "/chat/completions",
    data=json.dumps({"model": MODEL, "stream": False, "max_tokens": 400,
                     "messages": [{"role": "system", "content": "以下是一批参考资料：\n" + big +
                                   "\n回答时只说 OK。"},
                                  {"role": "user", "content": "只回复两个字母：OK"}]}).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with _opener.open(req3, timeout=180) as resp:
    obj3 = json.loads(resp.read().decode("utf-8"))
pt = ((obj3.get("usage") or {}).get("prompt_tokens") or 0)
body3 = ((obj3.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
check("HTTP 200", resp.status == 200, "status=%s" % resp.status)
check("prompt_tokens > 4096（证明 num_ctx=32768 而非默认 4096）", pt > 4096,
      "prompt_tokens=%d" % pt)
check("长上下文下仍有正文", bool(body3.strip()), repr(body3[:40]))
print("  注：mini 是 thinking 模型，reasoning 先占 token；max_tokens 给小了"
      "（如 20）会整段预算被思考吃掉、content 返回空串。")
print("  本次 usage=" + json.dumps(obj3.get("usage") or {}, ensure_ascii=False))

print("\n== 4. 长 system + 工具并存 ==")
r4 = stream([{"role": "system", "content": "参考资料：\n" + big + "\n你是编码助手，需要读文件时调用 read_file。"},
             {"role": "user", "content": "读一下 D:\\a.txt"}], tools=TOOLS, max_tokens=200)
check("HTTP 200", r4["status"] == 200, "status=%s" % r4["status"])
check("未 500 / 未报上下文超限", r4["status"] == 200 and r4["finish"] is not None,
      "finish=%s" % r4["finish"])

bad = [n for n, ok, _ in results if not ok]
print("\n" + "=" * 60)
print("通过 %d/%d" % (len(results) - len(bad), len(results)))
if bad:
    print("失败项：" + "；".join(bad))
sys.exit(1 if bad else 0)
