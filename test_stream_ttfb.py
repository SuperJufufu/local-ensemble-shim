#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""test_stream_ttfb.py —— 验证「真流式」：首字节延迟（TTFB）必须远小于总耗时。

同时确认：流式 + 工具调用仍然能正常收尾（finish_reason=tool_calls）。
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE = "http://127.0.0.1:8020/v1"

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "读取本地文件",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]

LONG_PROMPT = ("请写一段 400 字左右的散文，描写猫娘午后在窗台晒太阳的场景，"
               "要有细节和心情描写。")


def stream_case(messages, tools=None, label=""):
    payload = {"model": "dual-heavy", "stream": True, "messages": messages}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(f"{BASE}/chat/completions",
                                 data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer x"})
    t0 = time.time()
    ttfb = ttr = ttfc = None
    content, finish, frames, reasoning = [], None, 0, 0
    with _opener.open(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if ttfb is None:
                ttfb = time.time() - t0
            if body == "[DONE]":
                break
            frames += 1
            try:
                obj = json.loads(body)
            except Exception:  # noqa: BLE001
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    if ttr is None:
                        ttr = time.time() - t0
                    reasoning += len(d["reasoning_content"])
                if d.get("content"):
                    if ttfc is None:
                        ttfc = time.time() - t0
                    content.append(d["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.time() - t0
    text = "".join(content)
    print(f"  {label}")
    print(f"    首帧 {ttfb:.2f}s ｜ 首条思考 {('%.2f' % ttr) if ttr is not None else '—'}s "
          f"｜ 首个正文 {('%.2f' % ttfc) if ttfc is not None else '—'}s ｜ 总耗时 {total:.2f}s")
    print(f"    SSE 帧 {frames} ｜ 正文 {len(text)} 字 ｜ 思考流 {reasoning} 字 ｜ finish={finish}")
    return {"ttfb": ttfb, "ttr": ttr, "ttfc": ttfc, "total": total,
            "n": len(text), "finish": finish}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print("=== 1) 普通聊天（无 tools）：长文写作 ===")
    r1 = stream_case([{"role": "user", "content": LONG_PROMPT}], label="长文写作")
    print("\n=== 2) agent 模式（带 tools）：流式 + 工具调用 ===")
    r2 = stream_case(
        [{"role": "system", "content": "You are a DSH agent."},
         {"role": "user", "content": "读一下 C:\\Users\\Administrator\\Desktop\\README-桌面规划.md"}],
        tools=TOOLS, label="工具调用")
    print("\n" + "=" * 54)
    # 真流式的判据：正文不是一次性吐出，而是**持续**到达
    span = (r1["total"] - r1["ttfc"]) if r1.get("ttfc") is not None else 0
    live = r1.get("ttfc") is not None and span >= 0.5 and r1["n"] > 0
    prog = (r2.get("ttr") is not None) and (r2["ttr"] < r2["total"] * 0.9)
    ok2 = r2["finish"] == "tool_calls"
    print(f"  1) 正文真流式（持续到达 {span:.2f}s）: {'✅' if live else '❌'}  "
          f"首个正文 {r1['ttfc']:.2f}s / 总 {r1['total']:.2f}s")
    print(f"  2) agent 模式有实时进度（思考流）    : {'✅' if prog else '❌'}  "
          f"首条进度 {r2['ttr'] and round(r2['ttr'], 2)}s / 总 {r2['total']:.2f}s")
    print(f"  3) 流式+工具仍能正常收尾           : {'✅' if ok2 else '❌'}  finish={r2['finish']}")
    print("\n  提示：想最快 → 用 `--no-preamble` 起的实例（退化成单发流式，省掉多视角那 ~15 秒）")
    sys.exit(0 if (live and ok2) else 1)
