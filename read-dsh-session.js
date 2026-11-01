#!/usr/bin/env node
/*
 * 读 DSH 的会话日志：<session dir>/session.v4.jsonl.zstd -> 纯文本 JSONL
 *
 * DSH 的会话文件是「多个独立的、带校验和的 zstd 帧」首尾相接，不是单个流。
 * Node 的 createZstdDecompress 会在这里报 "Unknown frame descriptor"，
 * 所以这里按 DSH 自己的 scanZstdFrames 逻辑先切帧，再逐帧解压。
 *
 * 用法：
 *   node read-dsh-session.js <文件或目录> [--out 输出文件] [--grep 关键词]
 *   目录会给每个 *.jsonl.zstd 输出一份 .jsonl，并汇总「哪个会话命中了关键词」
 */
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

const ZSTD_MAGIC = 4247762216;

function scanZstdFrames(buffer, maxFrames = Number.POSITIVE_INFINITY) {
  const frames = [];
  let offset = 0;
  while (offset < buffer.length) {
    const start = offset;
    if (buffer.length - offset < 4) return { frames, tornStart: start };
    if (buffer.readUInt32LE(offset) !== ZSTD_MAGIC) {
      throw new Error(`invalid frame magic at byte ${offset}`);
    }
    offset += 4;
    if (offset === buffer.length) return { frames, tornStart: start };
    const descriptor = buffer.readUInt8(offset);
    offset += 1;
    const contentSizeFlag = descriptor >>> 6;
    const singleSegment = (descriptor & 32) !== 0;
    const checksum = (descriptor & 4) !== 0;
    const dictionaryFlag = descriptor & 3;
    const dictionaryBytes = dictionaryFlag === 3 ? 4 : dictionaryFlag;
    const contentSizeBytes = contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : 1 << contentSizeFlag;
    const remainingHeaderBytes = (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes;
    if (buffer.length - offset < remainingHeaderBytes) return { frames, tornStart: start };
    offset += remainingHeaderBytes;
    for (;;) {
      if (buffer.length - offset < 3) return { frames, tornStart: start };
      const blockHeader = buffer.readUIntLE(offset, 3);
      offset += 3;
      const lastBlock = (blockHeader & 1) !== 0;
      const blockType = (blockHeader >>> 1) & 3;
      const blockSize = blockHeader >>> 3;
      const payloadBytes = blockType === 1 ? 1 : blockSize;
      if (buffer.length - offset < payloadBytes) return { frames, tornStart: start };
      offset += payloadBytes;
      if (lastBlock) break;
    }
    if (checksum) {
      if (buffer.length - offset < 4) return { frames, tornStart: start };
      offset += 4;
    }
    frames.push({ start, end: offset });
    if (frames.length === maxFrames) return { frames };
  }
  return { frames };
}

function decodeSession(file) {
  const buf = fs.readFileSync(file);
  const { frames, tornStart } = scanZstdFrames(buf);
  const parts = [];
  for (const f of frames) {
    parts.push(zlib.zstdDecompressSync(buf.subarray(f.start, f.end)));
  }
  return { text: Buffer.concat(parts).toString("utf8"), frames: frames.length, torn: tornStart !== undefined };
}

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (e.name.endsWith(".jsonl.zstd")) out.push(p);
  }
  return out;
}

function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0) {
    console.error("用法: node read-dsh-session.js <文件或目录> [--out 文件] [--grep 关键词]");
    process.exit(2);
  }
  const target = argv[0];
  const outIdx = argv.indexOf("--out");
  const grepIdx = argv.indexOf("--grep");
  const outArg = outIdx >= 0 ? argv[outIdx + 1] : null;
  const grepArg = grepIdx >= 0 ? argv[grepIdx + 1] : null;

  const files = fs.statSync(target).isDirectory() ? walk(target) : [target];
  const hits = [];
  for (const f of files) {
    let r;
    try {
      r = decodeSession(f);
    } catch (e) {
      console.log(`  [skip] ${f}  ← ${e.message}`);
      continue;
    }
    let dest = null;
    if (outArg && files.length === 1) dest = outArg;
    else if (!grepArg) dest = f.replace(/\.zstd$/, "");
    if (dest) fs.writeFileSync(dest, r.text, "utf8");
    const n = r.text.split("\n").filter(Boolean).length;
    if (grepArg && r.text.includes(grepArg)) {
      const line = r.text.split("\n").find((l) => l.includes(grepArg)) || "";
      hits.push({ file: f, line: line.slice(0, 200) });
    }
    console.log(`  ${f}\n    frames=${r.frames}${r.torn ? " (尾部有不完整帧)" : ""}  行=${n}  字节=${r.text.length}${dest ? `  -> ${dest}` : ""}`);
  }
  if (grepArg) {
    console.log(`\n命中 "${grepArg}" 的会话：${hits.length}`);
    for (const h of hits) console.log(`  ${path.basename(path.dirname(h.file))}\n    ${h.line}`);
  }
}

main();
