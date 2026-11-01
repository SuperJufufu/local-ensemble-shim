"""小模型「怪语言守卫」的单元测试 + 假上游端到端测试。

背景：用户截图里 minicpm5 整段输出卡纳达文（"你怎么又切回这个诡异语言了"）。
守卫要保证的是**用户永远看不到那屏垃圾**，而不是"把模型修好"。

本测试用一个假上游（monkeypatch `mini_call_stream`）注入各种输出，
检查守卫是否：正常放行 / 异常丢弃并重试 / 重试仍异常就拦下 / 放行之后异常就掐断。

用法：python test_guard_scripts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402

# 用户截图里那段真东西（卡纳达文）
KANNADA = ("(๑>؂<๑) ಈ ಬದಲಿಗೆ ನೀಳು ಬಿದ್ದಿದೆ! 👆\n\n"
           "ನಾನು ನಿಜವಾಗಿಯೂ \"ಬದಲಿಗೆ\" ಎಂದರೆ ಮನಗೆ ಹೋಗಿದೆ, ಕೆಲ ಪ್ರಾತರ್ಥಗಳು ಅಲ್ಲ! 😊\n\n"
           "ನಿಮಗೆ ಸಾಲು, ನಾನು ನಿಜವಾಗಿಯೂ ಸೇರಿಕೊ ಸೇರಲು ಬರು ತಿಳಿದಿದ್ದೇನೆ? ✨")
CLEAN = "好的喵~ 我来帮你看看这个文件，先读一下前几行再说。"
MIXED_TAIL = ("先给你一个正常的开头，这段是完全正常的中文回答，用来占满放行窗口 —— "
              "因为守卫要先扣住开头几十个字符判定干净才放行，所以这里必须写够长，"
              "才能真正走到「放行之后才变怪」那条路径上。\n\n" + KANNADA)

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  — " + detail) if detail else ""))


def fake_upstream(pieces_by_call):
    """pieces_by_call: list[list[str]]，第 n 次调用用第 n 个分片序列（越界则用最后一个）。"""
    state = {"n": 0}

    def gen(messages, tools, temperature, max_tokens, model):
        i = min(state["n"], len(pieces_by_call) - 1)
        state["n"] += 1
        for p in pieces_by_call[i]:
            yield {"choices": [{"index": 0, "delta": {"content": p}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    gen.state = state
    return gen


def run_stream(pieces_by_call, tools=None):
    emit_log = []

    def emit(obj):
        emit_log.append(obj)

    def chunk(delta, finish=None):
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    h.mini_call_stream = fake_upstream(pieces_by_call)
    content, tcs, usage, blocked = h.mini_guarded_stream(
        [{"role": "user", "content": "帮我看看文件"}], tools, 0.7, 300, emit, chunk, "minicpm5-2b-32k")
    streamed = "".join(f["choices"][0]["delta"].get("content") or ""
                       for f in emit_log if f.get("choices"))
    return content, streamed, tcs, usage, blocked, emit_log


print("== 1. 检测器本身 ==")
bad, hits, ratio, names = h.script_is_bad(KANNADA)
check("用户截图里那段卡纳达文被判为异常", bad, f"命中 {hits} 字 / {ratio:.0%} / {'/'.join(sorted(names))}")
bad2, hits2, ratio2, _ = h.script_is_bad(CLEAN)
check("正常中文回答不误伤", not bad2, f"命中 {hits2} 字 / {ratio2:.0%}")
bad3, _, _, _ = h.script_is_bad("OK")
check("极短回答不误伤（字母不足 8 个直接放行）", not bad3)
bad4, hits4, ratio4, _ = h.script_is_bad(
    "这句话里引用了一个词 ಠ_ಠ 还有 ಇದು，其余都是中文，不应该被判为乱码。" )
check("少量外来字符混在中文里不误伤", not bad4, f"命中 {hits4} 字 / {ratio4:.0%}")
bad5, _, _, _ = h.script_is_bad("文件写好了，路径是 C:\\Users\\Administrator\\Desktop\\报告.md ✅")
check("中英混合的正常长回复不误伤", not bad5)

print("\n== 2. 正常输出：原样放行 ==")
content, streamed, tcs, usage, blocked, _ = run_stream([[CLEAN]])
check("没有拦下", not blocked)
check("客户端拿到的内容和模型输出一致", streamed == CLEAN and content == CLEAN,
      f"streamed={streamed[:40]!r}")

print("\n== 3. 一直吐卡纳达文：重试一次后拦下 ==")
content, streamed, tcs, usage, blocked, _ = run_stream([[KANNADA], [KANNADA]])
check("标记为已拦下", blocked)
check("客户端**没有**收到任何卡纳达文", "ಈ" not in streamed and "ನಾನು" not in streamed)
check("客户端收到可读说明", "本地小模型守卫" in streamed)
check("上游被调用了 2 次（含带强约束的重试）", h.mini_call_stream.state["n"] == 2,
      f"实际 {h.mini_call_stream.state['n']} 次")

print("\n== 4. 第一次坏、重试正常：用户只看到正常回答 ==")
content, streamed, tcs, usage, blocked, _ = run_stream([[KANNADA], [CLEAN]])
check("不算拦下（重试救回来了）", not blocked)
check("客户端拿到的是重试后的正常文本", streamed == CLEAN, repr(streamed[:50]))
check("上游被调用了 2 次", h.mini_call_stream.state["n"] == 2)
check("重试那次确实带了强约束提示（中文/禁其他文字系统）",
      "简体中文" in h.GUARD_HINT and "严禁" in h.GUARD_HINT)

print("\n== 5. 放行之后才变怪：掐断 + 说明（已发出的字收不回） ==")
# 关键：干净前缀必须**单独一个分片**送达（≥ 放行窗口），才走得到"放行之后"那条路
_prefix, _kan = MIXED_TAIL.split("\n\n", 1)
content, streamed, tcs, usage, blocked, _ = run_stream([[_prefix + "\n\n", _kan]])
check("标记为已拦下", blocked)
check("已放行的正常前缀仍在（收不回，这是诚实的代价）", "先给你一个正常的开头" in streamed)
check("卡纳达文没有漏出去", "ಈ" not in streamed and "ನೀಳು" not in streamed)
check("尾部有可读说明", "本地小模型守卫" in streamed)

print("\n== 6. 纯英文长回答不误伤 ==")
en = ("Here is the plan: read the file first, then edit it with a precise replacement. "
      "Keep the batch files ASCII-only and back up before writing.") * 2
content, streamed, _, _, blocked, _ = run_stream([[en]])
check("英文正常放行", not blocked and streamed == en)

print("\n== 7. 退化（复读机）也会被掐断 ==")
loop = "这是一段会重复的话。" * 80
content, streamed, _, _, blocked, _ = run_stream([[loop]])
check("重复内容被识别并掐断", blocked, f"streamed={len(streamed)} 字")

print("\n== 8. 工具调用输出**不受影响**（主人明确要求） ==")


def fake_toolcall():
    calls = []

    def gen(messages, tools, temperature, max_tokens, model):
        calls.append(messages)
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]}}]}
        for piece in ['{"path":', ' "C:\\\\a.txt"}']:
            yield {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": piece}}]}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    gen.calls = calls
    return gen


emit_log = []
h.mini_call_stream = fake_toolcall()
content, tcs, usage, blocked = h.mini_guarded_stream(
    [{"role": "user", "content": "读一下 a.txt"}], [{"type": "function", "function": {"name": "read_file"}}],
    0.7, 300, lambda o: emit_log.append(o),
    lambda delta, finish=None: {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
    "minicpm5-2b-32k")
check("工具调用原样透出", len(tcs) == 1 and tcs[0]["function"]["name"] == "read_file",
      str(tcs)[:100])
check("参数分片被拼回完整 JSON", json.loads(tcs[0]["function"]["arguments"])["path"] == "C:\\a.txt")
check("没有触发守卫、没有多余重试", not blocked and len(h.mini_call_stream.calls) == 1)
check("守卫没有往内容里加任何东西",
      "".join(f["choices"][0]["delta"].get("content") or "" for f in emit_log if f.get("choices")) == "")

bad_list = [n for n, ok, _ in results if not ok]
print("\n" + "=" * 62)
print("通过 %d/%d" % (len(results) - len(bad_list), len(results)))
if bad_list:
    print("失败项：" + "；".join(bad_list))
sys.exit(1 if bad_list else 0)
