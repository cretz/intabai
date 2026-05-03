#!/usr/bin/env node
// Patches onnxruntime-web bundled MatMul WebGPU shader to accumulate in fp32.
// Upstream bug: `var acc: array<vec4<${type}>` uses fp16 for accumulator when
// inputs are fp16, giving ~sqrt(N)*eps_fp16 error on long dot products
// (confetti output on Fastwan transformer blocks).
//
// Fix pattern matches ORT PR #20486 (Attention fp32 compute) and applies it
// to MatMul: cast operands to fp32 BEFORE the multiply, accumulate in fp32.
// (Earlier revisions kept the multiply in fp16 and only widened the
// accumulator; that was a half-fix — late-block weights with magnitude 4-5
// lose bits in the fp16 product before the fp32 add ever sees them.)
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
    // Idempotency keys gate on the NEW pre-multiply form so re-running the
    // script over an old half-fix file detects work to do.
    const matmulPatched = /acc\[i\] = vec4<f32>\(BCached[0-3]\) \* vec4<f32>\(ACached/.test(src);
    const attnPatched = /value \+= f32\(tileQ\[[^\]]+\]\) \* f32\(tileV\[/.test(src);
    const softmaxPatched = src.includes("var<workgroup> rowSumShared : f32;");
    const matNaivePatched = /var values_f32: array<f32,/.test(src);
    const softmaxExpInF32 = /let subExp = exp\(\$\{[^}]+\.replace\("f16","f32"\)\}\(getValue/.test(src);
    if (matmulPatched && attnPatched && softmaxPatched && matNaivePatched && softmaxExpInF32) {
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

    if (!matmulPatched) {
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

    // 3. vec4 multiply-add: cast BOTH operands to vec4<f32> BEFORE the
    // multiply so the product is computed in fp32, then add to the fp32
    // accumulator. 8 occurrences total (4 transposeA branches x 2).
    const mulPat = /acc\[i\] = (BCached[0-3]) \* (ACached(?:[0-3]\[i\]|\.[xyzw])) \+ acc\[i\];/g;
    const mulMatches = src.match(mulPat) || [];
    if (mulMatches.length !== 8) {
        throw new Error(`expected 8 vec4 mul-adds, found ${mulMatches.length}`);
    }
    // ACached side is a scalar f16 (ACached0[i] or ACached.x); WGSL doesn't
    // allow vec4<f32>(f16_scalar), so cast it through f32 and let
    // vec<f32> * f32 broadcast.
    src = src.replace(mulPat, "acc[i] = vec4<f32>($1) * f32($2) + acc[i];");
    changed++;

    // 4. non-vec4 multiply-add (single-line variant)
    sub(
        /acc\[innerRow\]\[innerCol\] = acc\[innerRow\]\[innerCol\] \+ ACached \* BCached\[innerCol\];/,
        "acc[innerRow][innerCol] = acc[innerRow][innerCol] + f32(ACached) * f32(BCached[innerCol]);",
    );

    // 5. non-vec4 multiply-add (multiline variant)
    sub(
        /acc\[innerRow\]\[innerCol\] = acc\[innerRow\]\[innerCol\] \+\n(\s*)ACached \* BCached\[innerCol\];/,
        "acc[innerRow][innerCol] = acc[innerRow][innerCol] +\n$1f32(ACached) * f32(BCached[innerCol]);",
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
        `value += f32(tileQ[TILE_SIZE * local_id.y + k]) * f32(tileV[TILE_SIZE * k + local_id.x]);`,
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
        `var value = \${${VN}}(exp(${f32Expr}(getValue(row, col, row_stride)) - ${f32Expr}(rowMaxShared)) / rowSumShared);`,
    );

    // 17/18. exp() in fp16 underflows for scores < ~-12 (fp16 min normal
    // 6e-5 ≈ exp(-9.7); subnormals down to exp(-16)). For long-axis
    // attention (e.g. cross-attn over 256 text tokens) this zeros out
    // many small probabilities BEFORE the fp32 sum reduction sees them,
    // producing a different distribution than wasm's f32-exp softmax.
    // Push the f32 cast to the *input* of exp() so the underflow happens
    // at f32 range (~exp(-87)) instead of f16 range. Two lines: subExp
    // (sum-pass) and final value (write-pass). Both shapes share the
    // same text after step 16 above.
    sub(
        /let subExp = exp\(getValue\(row, col, row_stride\) - rowMaxShared\);/,
        `let subExp = exp(${f32Expr}(getValue(row, col, row_stride)) - ${f32Expr}(rowMaxShared));`,
    );

    // 19. (TODO next session) Vec4 softmax generator (separate from the
    // scalar one above) emits `vec4<f16>(vec4<f32>(exp(f16)) / sumF32)` at
    // runtime — the verbose log confirms it dispatches for late-block LTX
    // attention. Source location not yet found in the bundle (this scalar
    // generator's text doesn't match). Hunt next session: grep for another
    // shader producing the `enable f16; ... rowMaxShared : vec4<f16>` form.

    // 17. MatMulNaive fp16 accumulator — same bug class as the vec4 MatMul
    // path. Used for MatMul shapes that don't fit the packed kernel; the
    // LTX transformer dispatches to this for at least some block matmuls,
    // so the vec4 patch alone does not fix late-block drift. Body shape:
    //   var values: array<${out.type.value}, ${N}>;
    //   for (k loop) { values[Y] = fma(${b.type.value}(a_data[...]), b_dataZ, values[Y]); }
    //   for (i loop) { var value = values[i]; ... bias ... activation ... write }
    // Fix: parallel f32 accumulator (values_f32), pre-multiply f32 cast,
    // then materialize values[i] = fp16(values_f32[i]) just before the
    // existing read so bias/activation/write still operate on the output
    // type. KNOWN UNFIXED SIBLING: GroupedConv-Vectorize uses an identical
    // fp16 fma accumulator pattern. Probably matters for VAE conv ops; not
    // patched here because the current focus is transformer drift.
    if (!matNaivePatched) {
        // The fma line lives inside a separate JS helper (`W()` minified,
        // `calcResult()` unminified) that builds the k-loop body string;
        // the WGSL template just interpolates its return value. So we apply
        // three independent substitutions, anchored on stable text.

        // 17a. Capture out-helper + count var from the values declaration,
        // and inject `var values_f32: array<f32, ${count}>;` between it
        // and the `for (var k: u32 = 0u; k < uniforms.K` k-loop opener
        // (this opener uniquely identifies the MatMulNaive WGSL template).
        const declRe =
            /(var values: array<\$\{(\w+)\.type\.value\}, \$\{(\w+)\}>;)\n(\s*)(for \(var k: u32 = 0u; k < uniforms\.K)/;
        const declMatch = src.match(declRe);
        if (!declMatch) {
            throw new Error("MatMulNaive: values decl + k-loop anchor not found");
        }
        const outH = declMatch[2];
        const cntV = declMatch[3];
        sub(
            declRe,
            `$1\n$4var values_f32: array<f32, \${${cntV}}>;\n$4$5`,
        );

        // 17b. The fma line. Match operand shape, ignore minified vs.
        // unminified spacing in the `=== 1 ? "" : ...` ternary.
        const fmaRe =
            /values\[(\$\{\w+\})\] = fma\((\$\{\w+\.type\.value\})\(a_data(\$\{\w+ ?=== ?1 ?\? ?"" ?: ?`\[\$\{\w+\}\]`\})\), b_data(\$\{\w+\}), values\[\$\{\w+\}\]\);/g;
        const fmaMatches = (src.match(fmaRe) || []).length;
        if (fmaMatches === 0) {
            throw new Error("MatMulNaive: no fma line matched");
        }
        src = src.replace(
            fmaRe,
            "values_f32[$1] = f32($2(a_data$3)) * f32(b_data$4) + values_f32[$1];",
        );
        changed++;

        // 17c. Cast values_f32 -> values inside the i-loop, just before the
        // existing `var value = values[i];` read so bias/activation/write
        // continue operating on the output type unchanged.
        const readRe = new RegExp(
            `(for \\(var i = 0u; i < \\$\\{${cntV}\\}u; i\\+\\+\\) \\{\\n)(\\s*)(var value = values\\[i\\];)`,
        );
        sub(
            readRe,
            `$1$2values[i] = \${${outH}.type.value}(values_f32[i]);\n$2$3`,
        );
    }

    return { src, changed, typeParam };
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
