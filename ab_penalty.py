#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""ab_penalty.py —— A/B 验证「重复惩罚」对本地模型长文退化的作用。

同一提示词、同一温度，只切换 frequency/presence penalty，比较：
  * 段落重复率
  * 唯一段落数
  * 是否触发复读机清理
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402

BOSS = "http://127.0.0.1:8010/v1/chat/completions"
PROMPT = ("写一段猫娘午后在窗台晒太阳的日常场景，要求细节丰富、有对话和心理描写，"
          "篇幅 1200 字以上。不要只写开头，要把整个下午写完。")
MAX_TOKENS = 3000
TEMP = 0.7


def gen(freq, pres):
    payload = {"model": "bonsai-2-27b-abliterated-mtp",
               "messages": [{"role": "user", "content": PROMPT}],
               "temperature": TEMP, "max_tokens": MAX_TOKENS, "stream": False,
               "frequency_penalty": freq, "presence_penalty": pres}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BOSS, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with h._opener.open(req, timeout=900) as r:  # noqa: SLF001
        resp = json.loads(r.read().decode("utf-8"))
    txt = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return txt, time.time() - t0, resp.get("usage") or {}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(f"提示词：{PROMPT[:40]}…  max_tokens={MAX_TOKENS}  temp={TEMP}\n")
    for label, f, p in (("A 关惩罚 (0 / 0)", 0.0, 0.0), ("B 开惩罚 (0.4 / 0.3)", 0.4, 0.3)):
        txt, dt, usage = gen(f, p)
        n_p, n_u, ratio = h.degeneration_report(txt)
        clean, hit, _ = h.clean_degeneration(txt)
        print(f"{label}")
        print(f"   输出 {len(txt):,} 字符  {n_p} 段 / 唯一 {n_u} 段  重复率 {ratio:>5.1%}  "
              f"触发清理={hit}  用时 {dt:.0f}s  完成token={usage.get('completion_tokens')}")
        print()
