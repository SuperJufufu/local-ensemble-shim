#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dual_agent.py —— 双模型协作原型（Grok-Heavy 风格）

角色分工（按实测能力不对称设计）：
  minicpm5-2b  → 分派者(plan) / 压缩者(compress)：快、会格式、但不擅长推理
  Bonsai-27B   → 多视角工作者(workers) / 裁决者(synthesize) / 挑刺者(critique)

流水线：
  plan(Bonsai) → K 个不同视角的 Bonsai 作答 → compress(mini 去重抽取)
             → synthesize(Bonsai 裁决，指出分歧) → critique(Bonsai 挑刺) → 最终答案

用法：
  python dual_agent.py --ask "问题"
  python dual_agent.py --eval          # 与「单发 Bonsai」对照跑 5 道题
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

_ARGS = None
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BOSS = "http://127.0.0.1:8010"
MINI_API = "http://127.0.0.1:11434"
BOSS_MODEL = "bonsai-2-27b-abliterated-mtp"
# 用 DSH 里注册的同一个派生模型（num_ctx=32768），这样 DSH 与 heavy shim
# 命中 Ollama 里同一个常驻实例，不会为了不同 ctx 反复卸载/重载。
MINI_MODEL = "minicpm5-2b-32k"


def _post(url, payload, timeout=600):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def boss(system, user, temperature=0.6, max_tokens=900):
    r = _post(f"{BOSS}/v1/chat/completions", {
        "model": BOSS_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens, "stream": False})
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""


def mini(system, user, temperature=0.2, max_tokens=500):
    r = _post(f"{MINI_API}/api/chat", {
        "model": MINI_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "think": False,
        # 不显式传 num_ctx：继承模型的 32768，避免与 DSH 抢同一个 runner 时抖 ctx
        "options": {"temperature": temperature, "num_predict": max_tokens}})
    return (r.get("message") or {}).get("content", "") or ""


def final_of(text: str) -> str:
    """从回复里抽 FINAL: 那一行；抽不到就把最后一行当答案。"""
    m = re.findall(r"FINAL\s*[:：]\s*(.+)", text or "")
    if m:
        return m[-1].strip()
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1][:80] if lines else ""


# ---------- 视角（这就是「heavy」多样性的来源）----------
LENSES = [
    ("严谨推导",
     "你负责严格推导。凡是数值比较，必须把小数位对齐再比（例：3.11 与 3.9 → 3.11 与 3.90）；"
     "凡有单位，必须换算；每步都写出来。最后一行必须是 `FINAL: <答案>`。"),
    ("常识直觉",
     "你负责快速给出符合常识的答案，但必须写出你依赖的常识假设。最后一行必须是 `FINAL: <答案>`。"),
    ("挑刺反例",
     "你负责挑刺：主动找反例、找常见陷阱（小数位数、计数时的重复、否定与语序、前提是否被偷换）。"
     "如果题目本身有歧义，直接指出来。最后一行必须是 `FINAL: <最可能的答案>`。"),
]


def plan(question, trace, planner="boss"):
    """拆解子问题。**实测默认用 Bonsai**：minicpm5 会把「拆解」当「作答」，
    直接抢答且答错（见 --control 对照），所以它只适合干压缩/整理，不适合干判断。"""
    sys_p = ("你是任务拆解器。你【绝对不要回答问题】，只把问题拆成 1-3 个必须回答的子问题。"
             "只输出 JSON 数组，例如 [\"子问题1\",\"子问题2\"]，不要任何解释、不要给答案。")
    ask = f"要拆解的问题：{question}"
    raw = boss(sys_p, ask, temperature=0.0, max_tokens=200) if planner == "boss" \
        else mini(sys_p, ask, max_tokens=200)
    subs = []
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            subs = [str(x) for x in json.loads(m.group(0))][:3]
        except Exception:  # noqa: BLE001
            subs = []
    if not subs:
        subs = [question]
    trace.append((f"plan({planner})", raw.strip()[:200]))
    return subs


def work(question, subs, trace):
    """K 个视角各自作答（Bonsai）。"""
    sub_text = "\n".join(f"- {s}" for s in subs)
    out = []
    for name, lens in LENSES:
        t0 = time.time()
        txt = boss(lens, f"问题：{question}\n\n需要覆盖的子问题：\n{sub_text}\n\n请作答。",
                   temperature=0.7, max_tokens=700)
        ans = final_of(txt)
        out.append({"lens": name, "answer": ans, "raw": txt})
        trace.append((f"worker[{name}]", f"{ans!r}  ({time.time()-t0:.1f}s)"))
    return out


def compress(question, cands, trace):
    """minicpm5 去重 + 抽取，不推理，只整理。"""
    block = "\n\n".join(f"【{c['lens']}】\n{c['raw'][:600]}" for c in cands)
    sys_c = ("你是信息整理器，不要推理、不要给新答案。"
             "把下面几份候选答案里的【最终结论】和【关键理由】抽出来，去重，"
             "只输出 JSON 数组：[{\"lens\":\"...\",\"answer\":\"...\",\"reason\":\"...\"}]")
    raw = mini(sys_c, f"问题：{question}\n\n{block}", max_tokens=600)
    trace.append(("compress(mini)", raw.strip()[:300]))
    return raw


def synthesize(question, compressed, trace):
    sys_s = ("你是最终裁决者。下面几个视角的结论可能一致也可能冲突。"
             "要求：1) 指出分歧点；2) 用最可靠的那条理由定案；3) 明确给出唯一答案。"
             "最后一行必须是 `FINAL: <答案>`，答案要短。")
    txt = boss(sys_s, f"问题：{question}\n\n各视角整理结果：\n{compressed}", temperature=0.2, max_tokens=700)
    trace.append(("synthesize(boss)", txt.strip()[:400]))
    return final_of(txt), txt


def critique(question, draft, trace):
    sys_k = ("你是挑刺者。针对下面的答案，找【具体可验证】的错误：数值算错、计数漏重、"
             "前提被偷换、单位没换算。如果确实没问题，只回 `OK`。"
             "如果发现问题，回 `FIX: <正确做法>` 并给出正确最终答案行 `FINAL: <答案>`。")
    txt = boss(sys_k, f"问题：{question}\n\n待检查答案：{draft}", temperature=0.0, max_tokens=500)
    trace.append(("critique(boss)", txt.strip()[:300]))
    m = re.search(r"FINAL\s*[:：]\s*(.+)", txt)
    if txt.strip().upper().startswith("OK") or not m:
        return draft, False
    new = m.group(1).strip()
    # 挑刺者常常写一堆 "FIX:" 但其实结论没变 —— 只有答案真的不同才算改过
    if new == draft or new.startswith(draft) or draft.startswith(new):
        return draft, False
    return new, True


def heavy(question, verbose=True):
    """完整 heavy 流水线，返回 (最终答案, trace, 耗时)。"""
    trace = []
    t0 = time.time()
    subs = plan(question, trace, getattr(_ARGS, "planner", "boss"))
    cands = work(question, subs, trace)
    comp = compress(question, cands, trace)
    draft, _ = synthesize(question, comp, trace)
    final, fixed = critique(question, draft, trace)
    dt = time.time() - t0
    if verbose:
        print(f"\n【Heavy 流水线】问题：{question}")
        for k, v in trace:
            print(f"   {k:<18} {v}")
        print(f"   → 最终答案：{final!r}   （挑刺{'改过答案' if fixed else '未发现问题'}，{dt:.1f}s）")
    return final, trace, dt


def single(question):
    """对照：单发 Bonsai，直接回答。"""
    t0 = time.time()
    txt = boss("只回答答案本身，尽量短。", question, temperature=0.0, max_tokens=400)
    return final_of(txt), time.time() - t0


# ---------- 评测 ----------
EVAL = [
    ("小数比较", "3.11 和 3.9 哪个大？只回答那个数。", lambda t: "3.9" in t),
    ("两步算术", "一个数加 17，再乘 3，再减 5，结果是 100。这个数是多少？只回答数字。", lambda t: "18" in t),
    ("计数", "从 1 数到 100，数字 7 一共出现多少次？只回答数字。", lambda t: "20" in t),
    ("排队逻辑", "A、B、C 三人排队。A 不站在最前面，C 站在最后面。谁站在第一个？只回答字母。",
     lambda t: t.strip().startswith("B") or "B" in t.split("\n")[-1][:12]),
    ("比较链", "小明比小红高，小红比小刚高。谁最矮？只回答名字。", lambda t: "小刚" in t),
]


def run_eval():
    print("=" * 66)
    print("对照实验：单发 Bonsai  vs  双模型 Heavy 流水线")
    print("=" * 66)
    rows = []
    for name, q, ok in EVAL:
        s_ans, s_t = single(q)
        s_ok = ok(s_ans)
        h_ans, _, h_t = heavy(q, verbose=False)
        h_ok = ok(h_ans)
        rows.append((name, s_ok, s_ans[:26], h_ok, h_ans[:26], s_t, h_t))
        print(f"  {name:<8} 单发 {'✅' if s_ok else '❌'} {s_ans[:20]:<22} "
              f"Heavy {'✅' if h_ok else '❌'} {h_ans[:20]:<22} ({h_t:.0f}s)")
    s_score = sum(r[1] for r in rows)
    h_score = sum(r[3] for r in rows)
    print("-" * 66)
    print(f"  单发 Bonsai {s_score}/{len(rows)}    Heavy 流水线 {h_score}/{len(rows)}")


def self_consistency(question, n=3):
    """对照实验：同样单发，但温度 0.7 采 n 次多数表决（无视角提示）。"""
    from collections import Counter
    answers = []
    t0 = time.time()
    for _ in range(n):
        txt = boss("只回答答案本身，尽量短。", question, temperature=0.7, max_tokens=300)
        answers.append(final_of(txt))
    c = Counter(answers)
    return answers, c.most_common(1)[0][0], time.time() - t0


# 两个「单发会错」的陷阱题，用来对照：视角 vs 纯多采样
TRAPS = [
    ("小数比较", "3.11 和 3.9 哪个大？只回答那个数。", lambda t: "3.9" in t),
    ("两步算术", "一个数加 17，再乘 3，再减 5，结果是 100。这个数是多少？只回答数字。", lambda t: "18" in t),
]


def run_control():
    print("=" * 70)
    print("对照：单发(T=0)  vs  纯多采样多数表决(T=0.7 ×3)  vs  视角 Heavy")
    print("=" * 70)
    for name, q, ok in TRAPS:
        s, _ = single(q)
        samples, maj, dt = self_consistency(q, 3)
        h, _, ht = heavy(q, verbose=False)
        print(f"  {name}")
        print(f"    单发 T=0        {'✅' if ok(s) else '❌'}  {s!r}")
        print(f"    多采样×3        {'✅' if ok(maj) else '❌'}  样本={samples} → 多数={maj!r}  ({dt:.1f}s)")
        print(f"    视角 Heavy      {'✅' if ok(h) else '❌'}  {h!r}  ({ht:.1f}s)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--planner", choices=["boss", "mini"], default="boss")
    a = ap.parse_args()
    globals()["_ARGS"] = a
    if a.eval:
        run_eval()
    elif a.control:
        run_control()
    elif a.ask:
        heavy(a.ask)
    else:
        heavy("一个数加 17，再乘 3，再减 5，结果是 100。这个数是多少？")




