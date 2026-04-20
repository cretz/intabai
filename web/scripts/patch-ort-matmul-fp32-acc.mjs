#!/usr/bin/env node
// Patches onnxruntime-web bundled MatMul WebGPU shader to accumulate in fp32.
// Upstream bug: `var acc: array<vec4<${type}>` uses fp16 for accumulator when
// inputs are fp16, giving ~sqrt(N)*eps_fp16 error on long dot products
// (confetti output on Fastwan transformer blocks).
//
// Fix pattern matches ORT PR #20486 (Attention fp32 compute) and applies it
// to MatMul: accumulate in fp32, keep fp16 multiplies (trade: full f32 compute
// would be closer to PR #20486 but requires more template surgery).
//
// Idempotent — re-running on an already-patched file is a no-op.

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

function detectTypeParam(src) {
    const m = src.match(/var acc: array<vec4<\$\{([a-zA-Z_]+)\}>, rowPerThread>;/);
    return m ? m[1] : null;
}

function patch(src) {
    const matmulPatched = src.includes("var acc: array<vec4<f32>, rowPerThread>;");
    const attnPatched = /var value: f32 = 0\.0;\n\s{2,}for \(var w: u32 = 0u; w < uniforms\.K/.test(src);
    const softmaxPatched = src.includes("var<workgroup> rowSumShared : f32;");
    if (matmulPatched && attnPatched && softmaxPatched) {
        return { src, changed: 0, note: "already-patched" };
    }
    const typeParam = matmulPatched ? null : detectTypeParam(src);
    if (!matmulPatched && !typeParam) {
        return { src, changed: 0, note: "no vec4 matmul shader found" };
    }
    const tVar = typeParam ? "${" + typeParam + "}" : null;
    let changed = 0;
    const sub = (re, repl) => {
        const before = src;
        src = src.replace(re, repl);
        if (before !== src) changed++;
        else throw new Error(`pattern did not match: ${re}`);
    };

    if (matmulPatched) {
        // skip all MatMul edits; fall through to AttentionScore
    } else {
    // 1. vec4 acc declaration -> f32
    sub(
        /var acc: array<vec4<\$\{[a-zA-Z_]+\}>, rowPerThread>;/,
        "var acc: array<vec4<f32>, rowPerThread>;",
    );

    // 2. non-vec4 acc declaration -> f32
    sub(
        /var acc ?: array<array<\$\{[a-zA-Z_]+\}, colPerThread>, rowPerThread>;/,
        (m) => m.replace(/\$\{[a-zA-Z_]+\}/, "f32"),
    );

    // 3. vec4 multiply-add: BCachedN * ACachedN[i] or ACached.{x,y,z,w}
    // wrap rhs in vec4<f32>(...) so the fp16 product is promoted before the
    // fp32 accumulate. 8 occurrences total (4 transposeA branches x 2).
    const mulPat = /acc\[i\] = (BCached[0-3]) \* (ACached(?:[0-3]\[i\]|\.[xyzw])) \+ acc\[i\];/g;
    const mulMatches = src.match(mulPat) || [];
    if (mulMatches.length !== 8) {
        throw new Error(`expected 8 vec4 mul-adds, found ${mulMatches.length}`);
    }
    src = src.replace(mulPat, "acc[i] = vec4<f32>($1 * $2) + acc[i];");
    changed++;

    // 4. non-vec4 multiply-add (single-line variant)
    sub(
        /acc\[innerRow\]\[innerCol\] = acc\[innerRow\]\[innerCol\] \+ ACached \* BCached\[innerCol\];/,
        "acc[innerRow][innerCol] = acc[innerRow][innerCol] + f32(ACached * BCached[innerCol]);",
    );

    // 5. non-vec4 multiply-add (multiline variant)
    sub(
        /acc\[innerRow\]\[innerCol\] = acc\[innerRow\]\[innerCol\] \+\n(\s*)ACached \* BCached\[innerCol\];/,
        "acc[innerRow][innerCol] = acc[innerRow][innerCol] +\n$1f32(ACached * BCached[innerCol]);",
    );

    // 6. vec4 mm_write -> cast acc back to fp16 before store
    sub(
        /mm_write\(batch, globalRow \+ innerRow, globalCol, acc\[innerRow\]\);/,
        `mm_write(batch, globalRow + innerRow, globalCol, vec4<${tVar}>(acc[innerRow]));`,
    );

    // 7. non-vec4 mm_write variant 1 (seq-access, single line)
    sub(
        /mm_write\(batch, gRow, gCol, acc\[innerRow\]\[innerCol\]\);/,
        `mm_write(batch, gRow, gCol, ${tVar}(acc[innerRow][innerCol]));`,
    );

    // 8. non-vec4 mm_write variant 2 (non-seq, multiline)
    sub(
        /mm_write\(batch, globalRow \+ innerRow, globalCol \+ innerCol,\n(\s*)acc\[innerRow\]\[innerCol\]\);/,
        `mm_write(batch, globalRow + innerRow, globalCol + innerCol,\n$1${tVar}(acc[innerRow][innerCol]));`,
    );
    }

    if (!attnPatched) {
    // 9. AttentionScore (softmax(QK^T)·V) fp16 accumulator — same bug class,
    // missed by upstream PR #20486 which only fixed the Q·Kᵀ kernel.
    // Target the decl at `createVxAttentionScoreProgramInfo`. Anchor on
    // the TILE_SIZE loop that follows so we don't touch reduce ops that
    // share the `var value = ${X.type.storage}(0);` prefix.
    const attnDeclRe = /var value = \$\{([a-zA-Z_$][a-zA-Z_$0-9]*)\.type\.storage\}\(0\);\n(\s{2,})for \(var w: u32 = 0u; w < uniforms\.K; w \+= TILE_SIZE\) \{/;
    const attnMatch = src.match(attnDeclRe);
    if (!attnMatch) throw new Error("AttentionScore decl anchor not found");
    const attnHelper = attnMatch[1];
    const tStore = "${" + attnHelper + ".type.storage}";
    src = src.replace(
        attnDeclRe,
        `var value: f32 = 0.0;\n$2for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {`,
    );
    changed++;
    sub(
        /value \+= tileQ\[TILE_SIZE \* local_id\.y \+ k\] \* tileV\[TILE_SIZE \* k \+ local_id\.x\];/,
        `value += f32(tileQ[TILE_SIZE * local_id.y + k] * tileV[TILE_SIZE * k + local_id.x]);`,
    );
    sub(
        /output\[outputIdx\] = value;/,
        `output[outputIdx] = ${tStore}(value);`,
    );
    }

    if (softmaxPatched) {
        return { src, changed, typeParam, note: "softmax already patched" };
    }
    // 10-16. Softmax fp16 accumulator — same bug class as MatMul. Reductions
    // over long axes (attention scores, seq_len up to 8190) accumulate in
    // ${valueType} (f16 for fp16 inputs), giving sqrt(N)*eps_fp16 drift plus
    // systematic under-summation (small values rounded to 0 against large
    // running sum). Fix: store threadShared as f32, accumulate sums in f32,
    // cast back to valueType only at final write.
    //
    // Anchor on `rowMaxShared : ${VN};` to detect the valueType variable name
    // (minified → `$`, unminified → `valueType`). Then use `${VN.replace("f16",
    // "f32")}` inline in the template so the per-invocation shader text is
    // correct for whatever components the dispatcher chose.
    const smAnchor = src.match(/var<workgroup> rowMaxShared : \$\{([a-zA-Z_$][a-zA-Z_$0-9]*)\};/);
    if (!smAnchor) throw new Error("softmax rowMaxShared anchor not found");
    const VN = smAnchor[1];
    const f32Expr = "${" + VN + '.replace("f16","f32")}';
    const VNre = VN.replace(/\$/g, "\\$");

    // 10. rowSumShared -> f32 scalar
    sub(
        new RegExp(`var<workgroup> rowSumShared : \\$\\{${VNre}\\};`),
        "var<workgroup> rowSumShared : f32;",
    );

    // 11. threadShared array element -> valueType-as-f32
    sub(
        new RegExp(`var<workgroup> threadShared : array<\\$\\{${VNre}\\}, (\\$\\{[a-zA-Z_$][a-zA-Z_$0-9]*\\})>;`),
        `var<workgroup> threadShared : array<${f32Expr}, $1>;`,
    );

    // 12. max-phase store: cast f16 threadMax to f32 before storing
    sub(
        /threadShared\[lindex\] = threadMax;/,
        `threadShared[lindex] = ${f32Expr}(threadMax);`,
    );

    // 13. sum accumulator init -> f32 zero
    sub(
        new RegExp(`var threadSum = \\$\\{${VNre}\\}\\(0\\.0\\);`),
        `var threadSum = ${f32Expr}(0.0);`,
    );

    // 14. sum accumulate: cast exp() result (f16) to f32 before add
    sub(
        /threadSum \+= subExp;/,
        `threadSum += ${f32Expr}(subExp);`,
    );

    // 15. rowSumShared store: drop valueType() cast — sumVector now returns
    // an f32 scalar (threadShared is f32), and rowSumShared is f32.
    sub(
        new RegExp(
            `rowSumShared = \\$\\{${VNre}\\}\\((\\$\\{[a-zA-Z_$][a-zA-Z_$0-9]*\\("threadShared\\[0\\]",\\s*[a-zA-Z_$][a-zA-Z_$0-9]*\\)\\})\\);`,
        ),
        "rowSumShared = $1;",
    );

    // 15b. rowMaxShared write: threadShared is now vec<f32>, so maxVector(...)
    // returns an f32 scalar. WGSL vec<f16>(f32) isn't allowed (needs matching
    // scalar type), so insert an explicit scalar cast to valueType's elem type.
    const scalarElemExpr = "${" + VN + '.includes("f16") ? "f16" : "f32"}';
    sub(
        new RegExp(
            `rowMaxShared = \\$\\{${VNre}\\}\\((\\$\\{[a-zA-Z_$][a-zA-Z_$0-9]*\\("threadShared\\[0\\]",\\s*[a-zA-Z_$][a-zA-Z_$0-9]*\\)\\})\\);`,
        ),
        `rowMaxShared = \${${VN}}(${scalarElemExpr}($1));`,
    );

    // 16. final divide: promote numerator to f32, divide by f32 rowSumShared,
    // cast result back to valueType. rowMaxShared stays valueType (max is
    // exact for f16, no precision loss going through the f32 tree reduction
    // since all stored values are already representable in f16).
    sub(
        /var value = exp\(getValue\(row, col, row_stride\) - rowMaxShared\) \/ rowSumShared;/,
        `var value = \${${VN}}(${f32Expr}(exp(getValue(row, col, row_stride) - rowMaxShared)) / rowSumShared);`,
    );

    return { src, changed, typeParam, valueTypeVar: VN };
}

const files = FILES.length ? FILES : fs.readdirSync(DIST);
for (const name of files) {
    const p = path.join(DIST, name);
    if (!fs.existsSync(p)) {
        console.log(`skip ${name} (missing)`);
        continue;
    }
    const orig = fs.readFileSync(p, "utf8");
    const { src, changed, note, typeParam } = patch(orig);
    if (src === orig) {
        console.log(`skip ${name} (${note || "no change"})`);
        continue;
    }
    fs.writeFileSync(p, src);
    console.log(`patched ${name} [typeParam=${typeParam}, edits=${changed}]`);
}
