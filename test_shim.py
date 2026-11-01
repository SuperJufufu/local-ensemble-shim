#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""test_shim.py —— 验证本地 Heavy 服务（heavy_server.py）三种模式：
   1) 无 tools 非流式（完整 Heavy 流水线）
   2) 带 tools（agent 模式，必须保住工具调用）
   3) stream=true（SSE 流式，DSH 实际用的就是这个）
"""
import json
import sys
import time
import urllib.request

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE = "http://127.0.0.1:8020/v1"


def post(payload, stream=False):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/chat/completions", data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer placeholder"})
    return _opener.open(req, timeout=900)


def t1_chat():
    print("\n=== 1) 无 tools · 非流式（完整 Heavy 流水线）===")
    t0 = time.time()
    with post({"model": "dual-heavy", "stream": False,
               "messages": [{"role": "user", "content": "3.11 和 3.9 哪个大？只回答那个数。"}]}) as r:
        body = json.loads(r.read().decode("utf-8"))
    msg = body["choices"][0]["message"]
    print(f"  耗时 {time.time()-t0:.1f}s")
    print(f"  回答: {msg.get('content')!r}")
    print(f"  finish: {body['choices'][0]['finish_reason']}")
    print(f"  usage: {body.get('usage')}")
    return "3.9" in (msg.get("content") or "")


def t4_tools_with_system():
    """回归：DSH 的真实请求形状 —— 第一条就是 system，后面才是 user。
    这正是 502 "System message must be at the beginning" 的触发条件。"""
    print("\n=== 4) 带 tools + 首条 system（DSH 真实形状）===")
    t0 = time.time()
    try:
        with post({"model": "dual-heavy", "stream": False, "tools": TOOLS,
                   "messages": [
                       {"role": "system", "content": "You are an AI agent powered by DeepSeek Harness."},
                       {"role": "user", "content": "帮我查一下上海现在的天气。"}]}) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
        return False
    msg = body["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print(f"  耗时 {time.time()-t0:.1f}s")
    print(f"  tool_calls: {[(c['function']['name'], c['function']['arguments']) for c in calls] or '（无）'}")
    return bool(calls)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询指定城市的当前天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def t2_tools():
    print("\n=== 2) 带 tools · 非流式（agent 模式）===")
    t0 = time.time()
    with post({"model": "dual-heavy", "stream": False, "tools": TOOLS,
               "messages": [{"role": "user", "content": "帮我查一下上海现在的天气。"}]}) as r:
        body = json.loads(r.read().decode("utf-8"))
    msg = body["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print(f"  耗时 {time.time()-t0:.1f}s")
    print(f"  tool_calls: {[(c['function']['name'], c['function']['arguments']) for c in calls] or '（无）'}")
    print(f"  finish: {body['choices'][0]['finish_reason']}")
    return bool(calls) and calls[0]["function"]["name"] == "get_weather"


def t3_stream():
    print("\n=== 3) stream=true（SSE，DSH 用的这条）===")
    t0 = time.time()
    frames, text, finish = 0, [], None
    with post({"model": "dual-heavy", "stream": True,
               "messages": [{"role": "user", "content": "一个数加 17，再乘 3，再减 5，结果是 100。这个数是多少？只回答数字。"}]}) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            frames += 1
            obj = json.loads(data)
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    out = "".join(text)
    print(f"  耗时 {time.time()-t0:.1f}s  收到 {frames} 个 SSE 帧  finish={finish}")
    print(f"  拼出的回答: {out!r}")
    return frames > 1 and finish == "stop" and "18" in out


def t5_stream_with_tools():
    """回归：**流式 + 工具** —— 之前从没测过这个组合，
    结果 shim 发完 [DONE] 不关连接，客户端看不到响应体结束 → 一直等（用户那边卡 5 分钟）。"""
    print("\n=== 5) stream=true + tools + 首条 system（真实 agent 场景）===")
    t0 = time.time()
    frames, finish, calls = 0, None, []
    try:
        r = post({"model": "dual-heavy", "stream": True, "tools": TOOLS,
                  "messages": [
                      {"role": "system", "content": "You are an AI agent powered by DeepSeek Harness."},
                      {"role": "user", "content": "帮我查一下上海现在的天气。"}]})
        with r:
            for raw in r:                      # 若响应体不结束，这里会一直挂着（就是那个 bug）
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                frames += 1
                obj = json.loads(data)
                for ch in obj.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                        calls.append(tc.get("function", {}).get("name") or f"<arg片#{tc.get('index')}>")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 读流失败/超时：{e}")
        return False
    print(f"  耗时 {time.time()-t0:.1f}s  帧数 {frames}  finish={finish}")
    print(f"  工具调用帧：{calls}")
    ok = finish == "tool_calls" and any(c == "get_weather" for c in calls)
    print(f"  {'✅' if ok else '❌'} 流式帧能正常结束且带工具调用")
    return ok


def t6_oversized_context():
    """回归：**请求超过 Bonsai 的 32768** —— 用户实际撞到的 502
    （上游 400 request (40698 tokens) exceeds the available context size (32768 tokens)）。
    期望：shim 自动裁剪最老的历史，**不报错**，任务继续。
    """
    print("\n=== 6) 超大上下文（约 5 万 token）自动裁剪 ===")
    msgs = [{"role": "system", "content": "You are an AI agent powered by DeepSeek Harness."}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"第{i}轮问题：" + "背景资料" * 300})
        msgs.append({"role": "assistant", "content": f"第{i}轮回答：" + "分析内容" * 300})
    msgs.append({"role": "user", "content": "最终问题：帮我查一下上海现在的天气。"})
    chars = sum(len(str(m.get("content"))) for m in msgs)
    print(f"  构造消息 {len(msgs)} 条 / {chars:,} 字符（约 {int(chars*0.7):,} token > 32768）")
    t0 = time.time()
    try:
        with post({"model": "dual-heavy", "stream": False, "tools": TOOLS, "messages": msgs}) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
        return False
    msg = body["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print(f"  耗时 {time.time()-t0:.1f}s  tool_calls={[(c['function']['name'], c['function']['arguments']) for c in calls] or '（无）'}")
    print(f"  content: {(msg.get('content') or '')[:80]!r}")
    ok = bool(calls) or bool(msg.get("content"))
    print(f"  {'✅' if ok else '❌'} 没有 502，正常返回")
    return ok


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    r1, r2, r3, r4, r5 = t1_chat(), t2_tools(), t3_stream(), t4_tools_with_system(), t5_stream_with_tools()
    r6 = t6_oversized_context()
    print("\n" + "=" * 52)
    print(f"  1) 流水线回答正确(3.9)      : {'✅' if r1 else '❌'}")
    print(f"  2) agent 模式保住工具调用    : {'✅' if r2 else '❌'}")
    print(f"  3) SSE 流式可用             : {'✅' if r3 else '❌'}")
    print(f"  4) 首条 system 不炸(502)    : {'✅' if r4 else '❌'}")
    print(f"  5) 流式+工具能正常结束       : {'✅' if r5 else '❌'}")
    print(f"  6) 超大上下文自动裁剪        : {'✅' if r6 else '❌'}")
    sys.exit(0 if (r1 and r2 and r3 and r4 and r5 and r6) else 1)
