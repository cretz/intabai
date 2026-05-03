#!/usr/bin/env node
// Patches onnxruntime-web Pow op (binary-op pow_custom WGSL helper) to
// short-circuit `b == 2` to a fp32-intermediate square.
//
// Upstream issue: pow_custom routes ALL exponents (including integer 2)
// through the WGSL builtin `pow(f32(abs(a)), f32(b))`, which lowers to
// `exp(log(x) * y)`. For fp16 inputs in PixelNorm-style chains
// (`Pow(x, 2) -> ReduceMean -> Sqrt -> Div`), the exp/log path is
// numerically lossy compared to a direct multiply, and `pow(0, 2)` is
// implementation-defined (some drivers return NaN from `log(0)*2`).
//
// Earlier revision of this patch used `return a * a;` directly. For LTX
// VAE the second PixelNorm chain consumes conv3d output where ~30% of
// positions saturate to fp16 ±Inf, and fp16 `a*a` on Inf collapses to a
// different bit pattern than wasm's exp/log fallback path produces from
// the same input. That made the EPs disagree at pow_2 even though every
// upstream op matched within fp16 ULP (see notes/console.log bisect on
// dec_block_00_res_0_tapped, 2026-05-06). Fix is to compute the square
// in fp32 and only narrow at the final output cast — fp16 +/-Inf squared
// becomes fp32 +Inf which casts back to fp16 +Inf consistently across
// EPs.
//
// Fix: insert
//     if (b == ${type}(2.0)) { let af = f32(a); return ${type}(af * af); }
// at the top of pow_custom. The vector path calls pow_custom per-lane so
// it picks up the optimization automatically. For f32-typed pow_custom
// the f32 cast collapses to identity which the shader compiler removes.
//
// Idempotent — re-running on an already-patched file is a no-op. Also
// rewrites the older `return a * a;` form in-place if it is the active
// patch state (so `node patch-ort-pow-square.mjs` upgrades any existing
// install without needing a reinstall).

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

const ID = "[a-zA-Z_$][a-zA-Z_$0-9]*";

function patch(src) {
    if (!/fn pow_custom\(a :/.test(src)) {
        return { src, changed: 0, note: "no pow_custom shader" };
    }
    // Already-patched (new fp32-intermediate form): no-op.
    if (/if \(b == \$\{[^}]+\}\(2\.0\)\) \{\s*let af = f32\(a\);\s*return \$\{[^}]+\}\(af \* af\);/.test(src)) {
        return { src, changed: 0, note: "already-patched (fp32)" };
    }

    // Upgrade path: previous revision injected `return a * a;`. Rewrite it
    // to the fp32-intermediate form in-place. We need the type ident so
    // we can interpolate it into the new return. The original injection
    // template was:
    //   if (b == ${X}(2.0)) {
    //     return a * a;
    //   }
    const upgradeRe = new RegExp(
        "if \\(b == (\\$\\{" + ID + "\\})\\(2\\.0\\)\\) \\{\\n(\\s+)return a \\* a;\\n(\\s+)\\}",
    );
    const upgradeMatch = src.match(upgradeRe);
    if (upgradeMatch) {
        const [, typeExpr, indentInner, indentOuter] = upgradeMatch;
        const replacement =
            `if (b == ${typeExpr}(2.0)) {\n` +
            `${indentInner}let af = f32(a);\n` +
            `${indentInner}return ${typeExpr}(af * af);\n` +
            `${indentOuter}}`;
        const before = src;
        src = src.replace(upgradeRe, () => replacement);
        if (before === src) {
            throw new Error("pow_custom: upgrade substitution did not modify source");
        }
        return { src, changed: 1, typeExpr, note: "upgraded a*a -> fp32(af*af)" };
    }

    // First-time install. Match the function header + the existing
    // `if (b == ${X}(0.0))` line so we can inject the b==2 short-circuit
    // between them. Capture the type ident.
    const re = new RegExp(
        "(fn pow_custom\\(a : (\\$\\{(" + ID + ")\\}), b : \\2\\) -> \\2 \\{\\n)(\\s+)(if \\(b == \\2\\(0\\.0\\)\\))",
    );
    const m = src.match(re);
    if (!m) {
        throw new Error("pow_custom: header + b==0 anchor not found");
    }
    const [, header, typeExpr, , indent, b0check] = m;
    const inject =
        `${indent}if (b == ${typeExpr}(2.0)) {\n` +
        `${indent}  let af = f32(a);\n` +
        `${indent}  return ${typeExpr}(af * af);\n` +
        `${indent}}\n${indent}`;
    const before = src;
    // Use a function replacement to avoid `$` chars in captured WGSL
    // template-literal text being interpreted as backreference tokens.
    src = src.replace(re, () => `${header}${inject}${b0check}`);
    if (before === src) {
        throw new Error("pow_custom: substitution did not modify source");
    }
    return { src, changed: 1, typeExpr, note: "first-time install" };
}

for (const name of FILES) {
    const p = path.join(DIST, name);
    if (!fs.existsSync(p)) {
        console.log(`skip ${name} (missing)`);
        continue;
    }
    const orig = fs.readFileSync(p, "utf8");
    const { src, changed, note, typeExpr } = patch(orig);
    if (src === orig) {
        console.log(`skip ${name} (${note || "no change"})`);
        continue;
    }
    fs.writeFileSync(p, src);
    console.log(`patched ${name} [typeExpr=${typeExpr}, edits=${changed}]`);
}
