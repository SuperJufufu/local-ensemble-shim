"""长上下文「大海捞针」实测：小模型把 ctx 拉大之后，到底能不能用？

背景：minicpm5-2b 原生 context length = 131072，Ollama 里能拉到 128k（实测显存 7.25 GB）。
但「能加载」不等于「能检索」。这个脚本量的是后者。

做法：把一条唯一事实按不同深度埋进一坨文本里，问模型要它。
  * 3 个深度：开头 10% / 正中间 50% / 结尾 90%（经典 lost-in-the-middle）
  * 2 个长度：~2k token 对照 / ~18k token
  * 对照温度 0（要的是检索能力，不是文采）

用法：
  python test_needle_ctx.py                     # 默认打本机 minicpm5-2b-32k
  python test_needle_ctx.py --model minicpm5-2b-128k --ctx 131072
"""
import argparse
import json
import random
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

NEEDLE = "【重要档案】地下三层的保险柜密码是 47-ALPHA-19，仅此一处记录。"
QUESTION = "地下三层的保险柜密码是什么？只输出密码本身，不要解释。"

# 模板句故意做得「像资料但各不相同」，避免用纯重复文本把针衬得太显眼。
SUBJECTS = ["第七车间的质检记录", "三号仓库的盘点差异", "西区供电的巡检日志", "备用泵房的检修单",
            "旧档案室的移交清单", "化验室的本月耗材", "南门岗的访客登记", "冷库二期的温控曲线",
            "试验田的土壤采样", "二号线的班次调整", "锅炉房的压力记录", "天台水箱的清洗单"]
VERBS = ["显示", "记录", "表明", "标注", "提到", "反映出", "备注", "印证"]
DETAILS = ["与上月基本持平", "比去年同期略高", "存在两处待复核的偏差", "已在当日完成闭环",
           "需要下周例会再确认", "未发现异常波动", "已移交下一班次跟进", "数据待补录完整",
           "按规程完成了抽检", "负责人为李工", "负责人为王工", "已归档至本年度卷宗"]


def wang_text(rng, n_para):
    out = []
    for i in range(n_para):
        lines = []
        for _ in range(rng.randint(3, 6)):
            lines.append("%s%s%s（编号 %s-%04d）。" % (
                rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(DETAILS),
                rng.choice("ABCDEFG"), rng.randint(1, 9999)))
        out.append("".join(lines))
    return "\n".join(out)


def build(rng, needle_at, target_tokens, tpc):
    """needle_at ∈ [0,1]：针放在总长度的大致位置。问题永远放在最后。"""
    est_chars = max(200, int(target_tokens / tpc))
    pre_chars = int(est_chars * needle_at)
    post_chars = est_chars - pre_chars
    pre = wang_text(rng, max(1, pre_chars // 165))
    post = wang_text(rng, max(1, post_chars // 165))
    return ("以下是一批内部资料，请通读：\n" + pre + "\n" + NEEDLE + "\n" + post
            + "\n\n" + QUESTION)


def ask(model, ctx, prompt, timeout=900):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "think": False,
               "options": {"num_ctx": ctx, "temperature": 0, "num_predict": 40}}
    req = urllib.request.Request(OLLAMA + "/api/chat",
                                 data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with _opener.open(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode("utf-8"))
    return {
        "text": (obj.get("message") or {}).get("content", "") or "",
        "prompt_tokens": obj.get("prompt_eval_count", 0),
        "eval_s": round((obj.get("prompt_eval_duration") or 0) / 1e9, 2),
        "out_tokens": obj.get("eval_count", 0),
        "wall_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minicpm5-2b-32k")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tpc", type=float, default=0.67, help="token/字符（中文散文实测约 0.67）")
    ap.add_argument("--targets", default="2000,18000",
                    help="要测的目标 token 数，逗号分隔（实际值会因估算偏差而略小）")
    args = ap.parse_args()

    print("模型 %s ｜ num_ctx %d ｜ 针 = %s" % (args.model, args.ctx, NEEDLE[:24] + "…"))
    print("问题：%s\n" % QUESTION)

    rows = []
    for target in [int(x) for x in args.targets.split(",")]:
        label = "~%dk token" % round(target / 1000)
        for depth, at in (("开头 10%", 0.10), ("正中间 50%", 0.50), ("结尾 90%", 0.90)):
            rng = random.Random(hash((label, depth)) & 0xFFFF)
            prompt = build(rng, at, target, args.tpc)
            r = ask(args.model, args.ctx, prompt)
            ok = "47-ALPHA-19" in r["text"]
            rows.append((label, depth, r, ok))
            print("  %-16s %-10s prompt=%6d tok  预填 %5.1fs  墙钟 %5.1fs  %s  →  %r"
                  % (label, depth, r["prompt_tokens"], r["eval_s"], r["wall_s"],
                     "✅ 找到" if ok else "❌ 没找到", r["text"][:60]))

    hit = sum(1 for _, _, _, ok in rows if ok)
    print("\n" + "=" * 66)
    print("检索命中 %d/%d" % (hit, len(rows)))
    for label in dict.fromkeys(r[0] for r in rows):
        sub = [r for r in rows if r[0] == label]
        print("  %-18s %d/%d" % (label, sum(1 for r in sub if r[3]), len(sub)))
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
