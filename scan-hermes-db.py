"""在 Hermes（千夏）的 state.db 里找「诡异文字」输出，定位是哪次会话产生的。

只读打开（mode=ro），绝不写用户的数据库。
用法：python scan-hermes-db.py
"""
import json
import re
import sqlite3
import unicodedata

DB = r"C:\Users\Administrator\AppData\Local\Hermes\state.db"

RANGES = [(0x0900, 0x097F, "天城文"), (0x0980, 0x09FF, "孟加拉"), (0x0B80, 0x0BFF, "泰米尔"),
          (0x0C00, 0x0C7F, "泰卢固"), (0x0C80, 0x0CFF, "卡纳达"), (0x0D00, 0x0D7F, "马拉雅拉姆"),
          (0x0E00, 0x0E7F, "泰文"), (0x0600, 0x06FF, "阿拉伯"), (0x0590, 0x05FF, "希伯来")]

TEXTY = ("content", "text", "message", "body", "data", "value", "payload",
         "result", "output", "response", "transcript", "raw", "reply", "answer")


def weird_ratio(text):
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0, 0.0, set()
    hits, names = 0, set()
    for c in letters:
        cp = ord(c)
        for lo, hi, name in RANGES:
            if lo <= cp <= hi:
                hits += 1
                names.add(name)
                break
    return hits, hits / len(letters), names


con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()
tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("表: %s\n" % ", ".join(tabs))

total_hits = 0
for t in tabs:
    try:
        cols = [c[1] for c in cur.execute("PRAGMA table_info('%s')" % t)]
    except Exception:
        continue
    tcols = [c for c in cols if c.lower() in TEXTY]
    if not tcols:
        continue
    what = ", ".join(tcols) + (", rowid" if "rowid" in cols else "")
    try:
        n = cur.execute("SELECT COUNT(*) FROM '%s'" % t).fetchone()[0]
    except Exception:
        continue
    print("--- 表 %s（%d 行）文本列: %s" % (t, n, tcols))
    try:
        rows = cur.execute("SELECT rowid, %s FROM '%s'" % (", ".join(tcols), t)).fetchall()
    except Exception as e:
        print("    读取失败: %s" % e)
        continue
    for row in rows:
        rid, vals = row[0], row[1:]
        for col, v in zip(tcols, vals):
            if not isinstance(v, str) or len(v) < 20:
                continue
            hits, ratio, names = weird_ratio(v)
            if hits >= 8 and ratio > 0.03:
                total_hits += 1
                idx = re.search(r"[\u0900-\u0d7f]", v)
                start = max(0, (idx.start() if idx else 0) - 200)
                print("\n  🚨 %s.rowid=%s.%s  怪字 %d (%.0f%%) %s  长度 %d"
                      % (t, rid, col, hits, ratio * 100, "/".join(sorted(names)), len(v)))
                print("     " + json.dumps(v[start:start + 500], ensure_ascii=False))

print("\n共 %d 处命中" % total_hits)
