#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""score_repair.py —— 用「结构 + 语义」双重标准，多轮评估谁来修工具参数最靠谱。

语义标准很关键：小模型会把 old_string 修成"删除需要替换的内容"这种**编造文本**，
结构合法但到 DSH 必然 `old_string was not found`。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402
from test_repair_models import TOOLS, CASES, check, repair_llm  # noqa: E402

ROUNDS = 3

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    tot = {"规则": 0, "mini": 0, "Bonsai": 0, "编造": 0, "解析失败": 0}
    n_cases = len(CASES)
    total_mini = n_cases * ROUNDS
    print(f"\n=== 结构 + 语义 双重评分：{n_cases} 例 × {ROUNDS} 轮 = {total_mini} 次（每个模型）===\n")
    print(f"{'用例':<26}{'规则':>6}{'minicpm5':>10}{'Bonsai':>8}   备注")
    print("-" * 72)
    for label, tool, raw in CASES:
        # 规则
        tc = [{"id": "x", "type": "function", "function": {"name": tool, "arguments": raw}}]
        h.repair_tool_calls(tc, TOOLS)
        try:
            a = json.loads(tc[0]["function"]["arguments"])
        except Exception:  # noqa: BLE001
            a = {}
        rule_ok = isinstance(a, dict) and not check(tool, a)
        tot["规则"] += bool(rule_ok)

        mini_ok = boss_ok = 0
        note = []
        for _ in range(ROUNDS):
            for tag, key in (("mini", "mini"), ("Bonsai", "Bonsai")):
                fixed, _ = repair_llm(tag, tool, raw, "参数不合法")
                if fixed is None:
                    if tag == "mini":
                        tot["解析失败"] += 1
                    continue
                probs = check(tool, fixed)
                if probs and any("编造" in p for p in probs):
                    if tag == "mini":
                        tot["编造"] += 1
                    note.append("编造原文")
                if not probs:
                    if tag == "mini":
                        mini_ok += 1
                    else:
                        boss_ok += 1
        tot["mini"] += mini_ok
        tot["Bonsai"] += boss_ok
        print(f"{label:<26}{'✅' if rule_ok else '❌':>6}{mini_ok:>7}/{ROUNDS}{boss_ok:>6}/{ROUNDS}   "
              f"{','.join(sorted(set(note))) if note else ''}")
    print("-" * 72)
    print(f"规则修复（确定性，单次） : {tot['规则']}/{n_cases}")
    print(f"minicpm5-2b ({ROUNDS} 轮合计)   : {tot['mini']}/{total_mini}")
    print(f"Bonsai-27B  ({ROUNDS} 轮合计)   : {tot['Bonsai']}/{total_mini}")
    print(f"  └ minicpm5 解析失败 {tot['解析失败']} 次，编造 old_string {tot['编造']} 次")
