#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""check_degeneration.py —— 用真实产出验证「复读机清理」是否有效。

用法： python check_degeneration.py [文件]
默认检查 桌面\abligrated\尾巴与耳朵的敏感篇.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_server as h  # noqa: E402

DEFAULT = r"C:\Users\Administrator\Desktop\abligrated\尾巴与耳朵的敏感篇.txt"

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    p = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    raw = p.read_text(encoding="utf-8", errors="replace")
    n_p, n_u, ratio = h.degeneration_report(raw)
    print(f"文件: {p.name}")
    print(f"原始  : {len(raw):,} 字符 / {n_p} 段 / 唯一 {n_u} 段 / 重复率 {ratio:.1%}")

    clean, hit, _ = h.clean_degeneration(raw)
    print(f"清理  : {len(clean):,} 字符  触发={hit}  剩下 {len(clean)/max(1,len(raw)):.1%}")
    cp, cu, cr = h.degeneration_report(clean)
    print(f"清理后: {cp} 段 / 唯一 {cu} 段 / 重复率 {cr:.1%}")
    print()
    print(f"惩罚参数: frequency_penalty={h.CONFIG['frequency_penalty']}  "
          f"presence_penalty={h.CONFIG['presence_penalty']}")
    print()
    # 对照：正常文章应当判定为「未触发」
    for f in sorted(Path(p.parent).glob("*.txt")):
        if f.name == p.name:
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        _, _, r = h.degeneration_report(t)
        _, hitt, _ = h.clean_degeneration(t)
        print(f"  对照组 {f.name:<24} {len(t):>6,} 字符  重复率 {r:>5.1%}  触发清理={hitt}")
