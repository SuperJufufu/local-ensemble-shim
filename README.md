# local-ensemble-shim

Turn two weak local models into **one usable "heavy" model behind a single OpenAI-compatible API**: the bigger one drafts from multiple viewpoints, the smaller one judges, and a guard layer fixes the failure modes local models actually have.

**Built for the failure modes, not the happy path:**

- degeneration loops — a 2-bit-quantized model repeating one paragraph N times
- missing / malformed tool-call arguments, and a model that *fabricates* `old_string` it never saw (detected by reading the real file)
- thinking models burning the whole `max_tokens` budget on reasoning → empty `content`
- client vs `llama-server` context-limit mismatch → auto-trim + retry at half size
- "streaming" that isn't streaming — SSE buffered upstream, UI hanging for minutes

Works with Ollama + `llama-server` and any GGUF; no model weights in this repo.

`local-llm` `ollama` `llama-cpp` `openai-api` `ensemble` `tool-calling` `quantization` `context-window` `python`

---

## 中文说明

## 为什么需要它

本机跑得动的模型（27B 三元量化 / 2B 小模型）单独干活时有两类硬伤：

| 问题 | 表现 |
|---|---|
| 强模型慢但会写 | 27B 写长文质量够，但慢，且**复读机退化**（同一段循环 N 次） |
| 弱模型快但不能当 agent | 2B 工具调用结构能出，但**漏必填参数**、会**编造 `old_string`**、thinking 模型还会把 `max_tokens` 全烧在 reasoning 上导致 `content` 返回空串 |

所以这里做的不是"再训一个模型"，而是**用编排 + 校验把它们拼成一个能用的模型**：
多视角并行出稿 → 裁决者选/合并 → 输出守卫 → 工具参数自修复。

## 结构

```
heavy_server.py      OpenAPI 兼容服务（8020）：/v1/models、/v1/chat/completions
                     - 路由：dual-heavy / minicpm5-2b-32k / mini-guarded
                     - SSE 真流式（首个进度帧 0.0s 可达）
                     - 上下文超限自动裁剪 + 减半重试
dual_agent.py        多视角 → 裁决 的执行体（含思考流播报）
*.Modelfile          Ollama 上下文派生（32k / 128k），防止 4096 默认值静默截断
start-heavy-server.bat / stop-heavy-server.bat
test_*.py            自测：路由、真流式、流式工具调用、上下文未截断、大海捞针、
                     退化检测、编辑参数修复、回归
ab_penalty.py        A/B 对比不同惩罚参数下的退化率
capture_sse.py       SSE 帧抓取，用于核对"是不是真的流式"
check_degeneration.py / score_repair.py / score_rules.py   退化与修复打分
read-dsh-session.js  读 DSH 会话日志（zstd 多帧），复现具体上下文
```

## 踩过的坑（都写进代码了）

1. **不能在开头之外再放 system 消息** —— DSH 的首条就是 system，再放一条会被上游 502。
2. **流式响应必须关连接**，否则客户端等不到"响应体结束"，界面卡几分钟。
3. **上下文计数不一致**（DSH 与 llama-server 各自算），超限时表现为 400 —— 改成自动裁剪最老对话 + 减半重试。
4. **Ollama 不会双载同模型的不同 ctx**，而是驱逐重载 → 三处配置里的模型名和 ctx 必须统一，否则每轮抖动 2.4s。
5. **`think:false` 只在 Ollama 原生 `/api/chat` 生效**，OpenAI 兼容口不认。
6. **2.13bpw 量化模型写长文会退化成同段循环** → 后置清理（content + 工具参数两处）+ 温和重复惩罚；清理时必须保住"用户提问"，否则模板报 `No user query found in messages.`
7. **小模型会编造 `old_string`**（实测 30 次编了 21 次）→ shim 在本机直接读文件核对，编造的原文永远进不了下游。

## 快速开始

```bat
:: 1) 先起 Ollama，并拉好 minicpm5（或任意 chat 模型）
:: 2) 起 27B（llama-server，按你自己的路径改 start-heavy-server.bat）
:: 3) 起 shim
start-heavy-server.bat
:: 4) 在任意 OpenAI 兼容客户端里填 http://127.0.0.1:8020/v1，模型选 dual-heavy
```

依赖：Python 3.11+、`requests`；llama.cpp 的 `llama-server`、Ollama。

## 已知限制

- 编排层每轮多 4~7 次本地调用，**慢 4~12 秒**；流式只让"已生成的字"立刻可见，不会让流水线本身变快。
- 弱模型**不能碰内容**（原文一缺就编造），只适合机械变换（解包、类型转换、改名）。
- 32k 与 128k 显存不可兼得：128k 会直接把 27B 挤死（Ollama 不看剩余显存，照加载不误）。

## 依赖的本地组件

llama.cpp（`llama-server`）、Ollama、一个 27B 级 GGUF + 一个 2B 级 GGUF。仓库里**不含任何模型权重**。
