"""走 shim 的 minicpm5 路由（带守卫）端到端测试。

重点验证主人明确提的那条：**不影响工具调用和干活**。
顺便量一下守卫的「先扣 48 字符」到底加了多长首字延迟。

用法：python test_mini_via_shim.py
"""
import json
import re
import sys
import time
import unicodedata
import urllib.request

SHIM = "http://127.0.0.1:8020/v1/chat/completions"
OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "minicpm5-2b-32k"
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

RANGES = [(0x0900, 0x097F), (0x0980, 0x09FF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF),
          (0x0D00, 0x0D7F), (0x0E00, 0x0E7F), (0x0600, 0x06FF), (0x0590, 0x05FF)]

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "读取文件内容",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  — " + detail) if detail else ""))


def weird(text):
    return sum(1 for c in text if unicodedata.category(c).startswith("L")
               and any(lo <= ord(c) <= hi for lo, hi in RANGES))


def stream(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    frames, content, tcs, finish, ttfb = 0, [], {}, None, None
    t0 = time.time()
    with _opener.open(req, timeout=timeout) as r:
        status = r.status
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            frames += 1
            if ttfb is None:
                ttfb = time.time() - t0
            try:
                obj = json.loads(body)
            except Exception:  # noqa: BLE001
                continue
            for ch in obj.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = tcs.setdefault(tc.get("index", 0), {"name": None, "args": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
    return {"status": status, "frames": frames, "content": "".join(content), "tcs": tcs,
            "finish": finish, "ttfb": ttfb, "total": time.time() - t0}


print("== 1. 流式聊天（走 shim，带守卫）==")
# 刻意问一个**较长**的回答：短回答本来就没什么可"逐字"的，
# 用短回答测流式会得出误导性的结论（之前就是这么误判过一次）。
LONG_Q = "用中文介绍一下 S3 存储，分 4 点说，每点一句话，总共 150 字左右。"
r = stream(SHIM, {"model": MODEL, "stream": True, "max_tokens": 600, "temperature": 0.3,
                  "messages": [{"role": "user", "content": LONG_Q}]})
check("HTTP 200", r["status"] == 200, f"status={r['status']}")
check("是真流式（多帧）", r["frames"] > 10, f"frames={r['frames']}")
check("有正文", len(r["content"]) > 40, f"{len(r['content'])} 字：{r['content'][:50]!r}")
check("没有怪语言", weird(r["content"]) == 0, f"怪字 {weird(r['content'])}")
check("正常收尾", r["finish"] == "stop", f"finish={r['finish']}")
shim_ttfb = r["ttfb"]

print("\n== 1b. 极短回答（< 观察窗）也不会变成空 ==")
rs = stream(SHIM, {"model": MODEL, "stream": True, "max_tokens": 200, "temperature": 0.2,
                   "messages": [{"role": "user", "content": "只回复两个字：正常"}]})
check("有正文且收尾正常", bool(rs["content"].strip()) and rs["finish"] == "stop",
      f"{rs['content']!r} finish={rs['finish']}")

print("\n== 2. 工具调用（主人明确要求：不能受影响）==")
r2 = stream(SHIM, {"model": MODEL, "stream": True, "max_tokens": 300, "temperature": 0.2,
                   "messages": [{"role": "system", "content": "需要读文件时必须调用 read_file 工具。"},
                                {"role": "user", "content": "读一下 C:\\Windows\\win.ini"}],
                   "tools": TOOLS, "tool_choice": "auto"})
got = bool(r2["tcs"])
check("HTTP 200", r2["status"] == 200, f"status={r2['status']}")
check("返回 tool_calls", got,
      json.dumps(r2["tcs"], ensure_ascii=False)[:120] if got else ("退化成正文: " + repr(r2["content"][:60])))
if got:
    slot = r2["tcs"][0]
    check("工具名正确", slot["name"] == "read_file", str(slot["name"]))
    try:
        args = json.loads(slot["args"] or "{}")
        check("参数是合法 JSON 且 path 非空", isinstance(args.get("path"), str) and args["path"].strip(),
              json.dumps(args, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        check("参数是合法 JSON", False, f"{e} :: {slot['args'][:100]!r}")
check("工具调用帧正常收尾", r2["finish"] == "tool_calls", f"finish={r2['finish']}")

print("\n== 3. 非流式 ==")
req = urllib.request.Request(SHIM, data=json.dumps(
    {"model": MODEL, "stream": False, "max_tokens": 80,
     "messages": [{"role": "user", "content": "只回复两个字：正常"}]}).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with _opener.open(req, timeout=300) as resp:
    obj = json.loads(resp.read().decode("utf-8"))
body = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content", "")
check("HTTP 200 + 有正文", resp.status == 200 and bool(body.strip()), repr(body[:50]))
check("没有怪语言", weird(body) == 0)

print("\n== 4. 守卫的首字延迟代价（shim vs 直连 Ollama）==")
rd = stream(OLLAMA, {"model": MODEL, "stream": True, "max_tokens": 600, "temperature": 0.3,
                     "messages": [{"role": "user", "content": LONG_Q}]})
delta = (shim_ttfb or 0) - (rd["ttfb"] or 0)
print(f"     直连首帧 {rd['ttfb']:.2f}s ｜ 走 shim（含观察窗）{shim_ttfb:.2f}s ｜ 差 {delta:+.2f}s")
check("守卫增加的首字延迟 < 1s", delta < 1.0, f"{delta:+.2f}s")
check("直连也没有怪语言作对照", weird(rd["content"]) == 0)

bad = [n for n, ok, _ in results if not ok]
print("\n" + "=" * 60)
print("通过 %d/%d" % (len(results) - len(bad), len(results)))
if bad:
    print("失败项：" + "；".join(bad))
sys.exit(1 if bad else 0)
