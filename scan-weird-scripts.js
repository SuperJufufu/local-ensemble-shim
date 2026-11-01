#!/usr/bin/env node
/* 扫全部 DSH 会话，找「印度语系等罕见文字」出现在哪些会话里（按字符区间判断，不靠精确短语）。 */
const fs = require("fs"), path = require("path"), zlib = require("zlib");
const MAGIC = 4247762216;
function scanFrames(buffer) {
  const frames = []; let offset = 0;
  while (offset < buffer.length) {
    const start = offset;
    if (buffer.length - offset < 4) return { frames };
    if (buffer.readUInt32LE(offset) !== MAGIC) return { frames };
    offset += 4;
    if (offset === buffer.length) return { frames };
    const d = buffer.readUInt8(offset); offset += 1;
    const csf = d >>> 6, ss = (d & 32) !== 0, ck = (d & 4) !== 0, df = d & 3;
    const dictB = df === 3 ? 4 : df;
    const csB = csf === 0 ? (ss ? 1 : 0) : 1 << csf;
    const rest = (ss ? 0 : 1) + dictB + csB;
    if (buffer.length - offset < rest) return { frames };
    offset += rest;
    for (;;) {
      if (buffer.length - offset < 3) return { frames };
      const bh = buffer.readUIntLE(offset, 3); offset += 3;
      const last = (bh & 1) !== 0, bt = (bh >>> 1) & 3, bs = bh >>> 3;
      const pb = bt === 1 ? 1 : bs;
      if (buffer.length - offset < pb) return { frames };
      offset += pb;
      if (last) break;
    }
    if (ck) { if (buffer.length - offset < 4) return { frames }; offset += 4; }
    frames.push([start, offset]);
  }
  return { frames };
}
const RANGES = [[0x0900,0x097f,"天城文"],[0x0980,0x09ff,"孟加拉"],[0x0a00,0x0a7f,"古木基"],
  [0x0a80,0x0aff,"古吉拉特"],[0x0b00,0x0b7f,"奥里亚"],[0x0b80,0x0bff,"泰米尔"],
  [0x0c00,0x0c7f,"泰卢固"],[0x0c80,0x0cff,"卡纳达"],[0x0d00,0x0d7f,"马拉雅拉姆"],
  [0x0e00,0x0e7f,"泰文"],[0x0e80,0x0eff,"老挝"],[0x1000,0x109f,"缅甸"],[0x1780,0x17ff,"高棉"],
  [0x0590,0x05ff,"希伯来"],[0x0600,0x06ff,"阿拉伯"]];
function weird(text) {
  const counts = {};
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    for (const [lo, hi, name] of RANGES) if (cp >= lo && cp <= hi) { counts[name] = (counts[name] || 0) + 1; break; }
  }
  return counts;
}
function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out); else if (e.name.endsWith(".jsonl.zstd")) out.push(p);
  }
  return out;
}
const root = process.argv[2] || "C:/Users/Administrator/.dsh/sessions";
let scanned = 0, hits = 0;
for (const f of walk(root)) {
  let text;
  try {
    const buf = fs.readFileSync(f);
    const { frames } = scanFrames(buf);
    text = Buffer.concat(frames.map(([a, b]) => zlib.zstdDecompressSync(buf.subarray(a, b)))).toString("utf8");
  } catch (e) { console.log(`  [skip] ${f}: ${e.message}`); continue; }
  scanned++;
  const c = weird(text);
  const total = Object.values(c).reduce((a, b) => a + b, 0);
  if (total >= 5) {
    hits++;
    const mt = fs.statSync(f).mtime.toISOString().slice(0, 19);
    console.log(`\n🚨 ${path.basename(path.dirname(f))}  (${mt}, ${text.length} 字节)`);
    console.log(`   ${path.dirname(f)}`);
    console.log(`   怪字合计 ${total}: ` + Object.entries(c).map(([k, v]) => `${k} ${v}`).join(", "));
    const idx = text.search(/[\u0900-\u0dff\u0e00-\u0eff]/u);
    console.log(`   上下文: ${JSON.stringify(text.slice(Math.max(0, idx - 120), idx + 120))}`);
  }
}
console.log(`\n扫了 ${scanned} 个会话，命中 ${hits} 个`);
