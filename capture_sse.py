#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""capture_sse.py —— 抓「流式 + 工具调用」的原始 SSE 帧，
把 llama-server 原生格式 与 我的 shim 的格式并排对比。
"""
import json
import sys
import urllib.request

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询指定城市的当前天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def cap(url, payload, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Authorization": "Bearer x"})
    frames = []
    with _opener.open(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8").rstrip("\n")
            if not line:
                continue
            frames.append(line)
    for f in frames[:40]:
        print("  " + (f[:220] + ("…" if len(f) > 220 else "")))
    if len(frames) > 40:
        print(f"  …（共 {len(frames)} 行）")
    return frames


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    msgs = [{"role": "system", "content": "You are an AI agent."},
            {"role": "user", "content": "帮我查一下上海现在的天气。"}]

    cap("http://127.0.0.1:8010/v1/chat/completions",
        {"model": "bonsai-2-27b-abliterated-mtp", "messages": msgs, "tools": TOOLS,
         "tool_choice": "auto", "temperature": 0.0, "max_tokens": 300, "stream": True},
        "① llama-server（Bonsai 原生）· 流式 + 工具")

    cap("http://127.0.0.1:8020/v1/chat/completions",
        {"model": "dual-heavy", "messages": msgs, "tools": TOOLS,
         "temperature": 0.0, "max_tokens": 300, "stream": True},
        "② 我的 shim（dual-heavy）· 流式 + 工具")
