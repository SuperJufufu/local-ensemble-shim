# local-ensemble-shim

**An OpenAI-compatible server that turns two small local models into one usable "heavy" model — plus the guard rails local models actually need in production.**

Local models small enough to run on a consumer GPU have two hard failure modes, and prompt engineering
does not fix either of them:

| Problem | What actually happens |
|---|---|
| The good writer is slow | A 27B quantized model writes well but runs slowly, and **degenerates into loop-repetition** (same paragraph N times) |
| The fast model cannot act as an agent | A 2B model emits a syntactically valid tool call, but **omits required arguments**, **fabricates `old_string`** for edits, and thinking models burn the whole `max_tokens` budget on reasoning, returning an **empty `content`** |

So this project does not train another model. It composes the two you already have into one that works:
**multi-view generation → judge/merge → output guard → tool-argument self-repair**. All of it sits behind
a standard `/v1/chat/completions` endpoint, so any OpenAI-compatible client works unchanged.

---

## Quick start

Requirements: Python 3.11+, `requests`, [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`),
[Ollama](https://ollama.com), one 27B-class GGUF and one 2B-class GGUF. **No model weights are included
in this repo.**

```bash
git clone https://github.com/SuperJufufu/local-ensemble-shim
cd local-ensemble-shim
pip install requests

# 1. Start Ollama and pull the small model
ollama pull openbmb/minicpm5-2b

# 2. Start the 27B backend (edit paths in start-heavy-server.bat first)
./start-heavy-server.bat

# 3. Start the shim
python heavy_server.py
```

Then point any OpenAI-compatible client at:

```
base_url = http://127.0.0.1:8020/v1
model    = dual-heavy        # or mini-guarded, minicpm5-2b-32k
```

Verify it is alive:

```bash
curl -s http://127.0.0.1:8020/v1/models
curl -s http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dual-heavy","messages":[{"role":"user","content":"Say hi in five words."}]}'
```

---

## Routes

| Route | What it does |
|---|---|
| `dual-heavy` | Both models answer in parallel, a judge picks or merges the result |
| `mini-guarded` | 2B model with output guards + tool-argument repair (fast, cheap) |
| `minicpm5-2b-32k` | Plain pass-through with the correct context size |

The server speaks real SSE streaming (first progress frame arrives at 0.0 s), auto-trims when the
context is exceeded and retries at half size, and repairs tool-call arguments before they reach the
client.

---

## Repository layout

```
heavy_server.py         OpenAI-compatible server on :8020 — /v1/models, /v1/chat/completions
                        routing (dual-heavy / minicpm5-2b-32k / mini-guarded),
                        real SSE streaming, context-overflow trim + retry-at-half
dual_agent.py           multi-view → judge execution body (with thinking-stream narration)
*.Modelfile             Ollama context derivations (32k / 128k) so the 4096 default cannot silently truncate
start-heavy-server.bat  / stop-heavy-server.bat
test_*.py               self-tests: routing, real streaming, streaming tool calls,
                        context-not-truncated, needle-in-haystack, degeneration detection,
                        edit-argument repair, regression
ab_penalty.py           A/B different repetition penalties, report degeneration rate
capture_sse.py          capture raw SSE frames — to check whether streaming is *actually* streaming
check_degeneration.py   degeneration detection
score_repair.py / score_rules.py   scoring for repair and rule checks
probe_edit_args.py / probe_empty_arg.py   probe specific failure modes
read-dsh-session.js     read DSH session logs (multi-frame zstd) to reproduce a real context
```

---

## Failure modes this exists to handle

Every item below was hit in practice and is now handled in code, not in a prompt.

1. **No system message except the first one.** An upstream proxy returns 502 if a second `system`
   message appears mid-conversation. The shim folds extra instructions into the first message.
2. **Streaming responses must close the connection**, otherwise the client never sees "end of body"
   and the UI hangs for minutes.
3. **Context counters disagree** between DSH and `llama-server`; going over shows up as a 400. Fixed by
   trimming the oldest turns and retrying at half size instead of failing.
4. **Ollama will not keep two contexts of the same model loaded** — it evicts and reloads, costing
   ~2.4 s per turn of thrash. Model name and context size must match across all three config points.
5. **`think: false` only works on Ollama's native `/api/chat`**, not on the OpenAI-compatible endpoint.
6. **A 2.13 bpw quantized model degenerates into repeating the same paragraph** on long output. Fixed
   with post-processing in both `content` and tool arguments, plus a mild repetition penalty.
   The cleanup must preserve the user turn, or the template rejects it with
   `No user query found in messages.`
7. **Small models fabricate `old_string`** — measured: 21 fabrications out of 30 attempts. The shim
   reads the file locally and verifies the quoted text, so a fabricated original never reaches the
   downstream editor.

---

## Known limitations

- The orchestration layer costs **4–7 extra local calls per turn, i.e. 4–12 s slower**. Streaming only
  makes already-generated text visible sooner; it does not make the pipeline faster.
- The weak model **must not touch content** (it fabricates whenever something is missing). It is safe
  only for mechanical transforms: unpacking, type conversion, renaming.
- 32k and 128k contexts cannot coexist in VRAM — 128k will OOM the 27B outright (Ollama does not check
  remaining VRAM before loading).

---

## License

MIT — see [LICENSE](LICENSE).
