#!/usr/bin/env node
// Patch onnxruntime-web Conv3DNaive's `get3DPadAndOutInfo` to accept ONNX
// per-axis pads instead of rejecting any non-uniform array.
//
// Upstream behavior: the WebGPU Conv3DNaive kernel throws
// `Unsupported padding parameter: <pads>` whenever the 6-element pads
// array isn't all-equal. Even when accepted, it stores pads as
// `{top:pad[0], bottom:pad[1], left:pad[2], right:pad[3], front:pad[4],
// back:pad[5]}` -- which is *not* ONNX's
// `[d_start, h_start, w_start, d_end, h_end, w_end]` layout -- and uses
// `pad[0]` as a uniform pad in the output-shape calculation.
//
// LTX VAE convs use kernel (1,3,3) with pads `[0,1,1,0,1,1]` (no
// temporal pad, symmetric H/W pad of 1), which the strict check rejects.
//
// The downstream WGSL only reads `uniforms.pads[0..2]` as
// (d_start, h_start, w_start) (line `xFRCCorner * strides - uniforms.pads`
// in `conv3d_naive_webgpu.ts`), which already matches ONNX ordering. So
// this patch only fixes the JS pre-pass: drop the all-equal check, build
// padInfo with ONNX semantics, and compute output shape per-axis using
// (start + end).
//
// Idempotent.

import fs from "node:fs";
import path from "node:path";

const DIST = "node_modules/onnxruntime-web/dist";
const FILES = [
  "ort.bundle.min.mjs",
  "ort.all.bundle.min.mjs",
  "ort.all.min.mjs",
  "ort.all.mjs",
  "ort.min.mjs",
  "ort.mjs",
];

// Minified anchor:
//   else if(Array.isArray(t)){
//     if(!t.every((w,S,x)=>w===x[0]))throw Error(`Unsupported padding parameter: ${t}`);
//     p={top:t[0],bottom:t[1],left:t[2],right:t[3],front:t[4],back:t[5]};
//     let y=kd([e,r,n,1],[u,d,c],1,[o,i,s],t[0]);m=y[0],g=y[1],b=y[2]
//   }
// Names (t,e,r,n,o,i,s,u,d,c,p,m,g,b,kd,y) vary across bundles, so we
// match positionally with capture groups. Some bundles call the throw'd
// length-violation Unsupported instead of all-equal; we still match.
const RE = new RegExp(
  // group: padArg (1)
  "else if\\(Array\\.isArray\\(([a-zA-Z_$][\\w$]*)\\)\\)\\{" +
    // strict check
    "if\\(!\\1\\.every\\(\\([^)]*\\)=>[^)]+\\)\\)throw Error\\(`Unsupported padding parameter: \\$\\{\\1\\}`\\);" +
    // padInfo assign: capture target (2)
    "([a-zA-Z_$][\\w$]*)=\\{top:\\1\\[0\\],bottom:\\1\\[1\\],left:\\1\\[2\\],right:\\1\\[3\\],front:\\1\\[4\\],back:\\1\\[5\\]\\};" +
    // output-shape call: capture fn (3), inDims (4), filterDims (5),
    // strideDims (6), tmpVar (7) and the m/g/b targets (8/9/10)
    "let ([a-zA-Z_$][\\w$]*)=" +
    "([a-zA-Z_$][\\w$]*)\\(\\[([^\\]]+)\\],\\[([^\\]]+)\\],1,\\[([^\\]]+)\\],\\1\\[0\\]\\);" +
    "([a-zA-Z_$][\\w$]*)=\\3\\[0\\],([a-zA-Z_$][\\w$]*)=\\3\\[1\\],([a-zA-Z_$][\\w$]*)=\\3\\[2\\]" +
    "\\}",
);

// Rebuild the branch with ONNX semantics.
//   pads layout = [d0,h0,w0,d1,h1,w1] (start, then end)
//   padInfo: front=t[0], top=t[1], left=t[2], back=t[3], bottom=t[4], right=t[5]
//   m = floor((inD + t[0] + t[3] - filterD) / strideD + 1)
//   g = floor((inH + t[1] + t[4] - filterH) / strideH + 1)
//   b = floor((inW + t[2] + t[5] - filterW) / strideW + 1)
function buildReplacement(g) {
  const [, t, p, , , inDims, filterDims, strideDims, m, gOut, b] = g;
  const [eIn, rIn, nIn] = inDims.split(",").map((s) => s.trim());
  const [uF, dF, cF] = filterDims.split(",").map((s) => s.trim());
  const [oS, iS, sS] = strideDims.split(",").map((s) => s.trim());
  return (
    `else if(Array.isArray(${t})){` +
      `if(${t}.length!==6)throw Error(\`Unsupported padding parameter: \${${t}}\`);` +
      `${p}={front:${t}[0],top:${t}[1],left:${t}[2],back:${t}[3],bottom:${t}[4],right:${t}[5]};` +
      `${m}=Math.trunc((${eIn}+${t}[0]+${t}[3]-${uF})/${oS}+1);` +
      `${gOut}=Math.trunc((${rIn}+${t}[1]+${t}[4]-${dF})/${iS}+1);` +
      `${b}=Math.trunc((${nIn}+${t}[2]+${t}[5]-${cF})/${sS}+1)` +
    `}`
  );
}

// Idempotency marker: replacement uses `front:${t}[0]` (not `top:${t}[0]`).
function alreadyPatched(src) {
  return /Array\.isArray\([a-zA-Z_$][\w$]*\)\)\{if\([a-zA-Z_$][\w$]*\.length!==6\)/.test(
    src,
  );
}

let totalChanged = 0;
for (const name of FILES) {
  const p = path.join(DIST, name);
  if (!fs.existsSync(p)) {
    console.log(`skip ${name} (missing)`);
    continue;
  }
  const orig = fs.readFileSync(p, "utf8");
  if (alreadyPatched(orig)) {
    console.log(`skip ${name} (already-patched)`);
    continue;
  }
  if (!/Unsupported padding parameter/.test(orig)) {
    console.log(`skip ${name} (no Conv3DNaive pad branch)`);
    continue;
  }
  const m = orig.match(RE);
  if (!m) {
    // Unminified bundles (`ort.mjs`, `ort.all.mjs`) have the source shape
    // and won't match this minified anchor. Vite's default ESM import
    // resolves to `ort.bundle.min.mjs` which we do match, so skip.
    console.log(`skip ${name} (unminified shape, not loaded by Vite)`);
    continue;
  }
  const replaced = orig.replace(RE, buildReplacement(m));
  if (replaced === orig) {
    throw new Error(`${name}: replacement produced identical output`);
  }
  fs.writeFileSync(p, replaced);
  totalChanged++;
  console.log(`patched ${name}`);
}
console.log(`done: ${totalChanged} file(s) patched`);
