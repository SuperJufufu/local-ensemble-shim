"""列出 Hermes（千夏）state.db 里最近的对话消息，看有没有卡纳达/怪语言那一段。"""
import json
import sqlite3

DB = r"C:\Users\Administrator\AppData\Local\Hermes\state.db"
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()

cols = [c[1] for c in cur.execute("PRAGMA table_info('messages')")]
print("messages 列:", cols)
n = cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
print("总消息数:", n)

order = "id" if "id" in cols else "rowid"
print("\n=== 最近 15 条 ===")
sel = ", ".join(cols)
for row in cur.execute("SELECT %s FROM messages ORDER BY %s DESC LIMIT 15" % (sel, order)):
    d = dict(zip(cols, row))
    txt = (d.get("content") or "")
    if isinstance(txt, str) and len(txt) > 220:
        txt = txt[:220] + "…"
    print("-", json.dumps(d, ensure_ascii=False, default=str)[:700])

print("\n=== sessions 表最近 10 条 ===")
scols = [c[1] for c in cur.execute("PRAGMA table_info('sessions')")]
print("sessions 列:", scols)
for row in cur.execute("SELECT %s FROM sessions ORDER BY rowid DESC LIMIT 10" % ", ".join(scols)):
    d = dict(zip(scols, row))
    print("-", json.dumps(d, ensure_ascii=False, default=str)[:500])
