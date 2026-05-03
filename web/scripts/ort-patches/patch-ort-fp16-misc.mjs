#!/usr/bin/env node
// Patches remaining onnxruntime-web fp16 accumulator gaps that the existing
// MatMul / AttentionScore / Softmax / MatMulNaive / Conv3DNaive patches don't
// cover. Audit-flagged kernels:
//   - GroupedConv naive (`var value: T = T(0); value += xVal * wVal;`)
//   - GroupedConv-Vectorize (`var values: array<T,N>; values[i] = fma(...)`)
//   - 7 naive reduce ops: ReduceLogSum, ReduceL1, ReduceL2, ReduceLogSumExp,
//     ReduceProd, ReduceSum, ReduceSumSquare
//     (ReduceMean naive already correct; ReduceMax/ReduceMin are exact for fp16)
//
// Pattern: cast loads/products to f32, accumulate in f32, cast back to the
// graph storage/value type only at the final write so downstream ops still
// receive correctly-typed buffers.
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

// ID = backtick-quoted JS identifier (works for minified n/o and unminified input/output)
const ID = "[a-zA-Z_$][a-zA-Z_$0-9]*";

// Builders return the inner-array text (without the surrounding `[]`).
// `form` is "min" (compact) or "un" (newline-separated, indented).
function joinArr(form, parts) {
    if (form === "min") return parts.join(",");
    const indent = "        ";
    return "\n" + parts.map((p) => indent + p).join(",\n") + "\n      ";
}

// Detect already-patched arrays by their f32 accumulator marker.
function alreadyPatched(arrBody) {
    return /value_f32/.test(arrBody);
}

function buildLogSum(I, O, form) {
    return joinArr(form, [
        "`var value_f32: f32 = 0.0; var value = ${" + O + ".type.storage}(0);`",
        '""',
        "`value_f32 += f32(${" + I + '.getByIndices("input_indices")});`',
        "`value = ${" + O + ".type.value}(log(value_f32));`",
    ]);
}
function buildL1(I, O, form) {
    return joinArr(form, [
        "`var value_f32: f32 = 0.0; var value = ${" + O + ".type.storage}(0);`",
        '""',
        "`value_f32 += abs(f32(${" + I + '.getByIndices("input_indices")}));`',
        "`value = ${" + O + ".type.value}(value_f32);`",
    ]);
}
function buildL2(I, O, form) {
    return joinArr(form, [
        "`var t_f32: f32 = 0.0; var value_f32: f32 = 0.0; var value = ${" + O + ".type.value}(0);`",
        '""',
        "`t_f32 = f32(${" + I + '.getByIndices("input_indices")}); value_f32 += (t_f32 * t_f32);`',
        "`value = ${" + O + ".type.value}(sqrt(value_f32));`",
    ]);
}
function buildLogSumExp(I, O, form) {
    return joinArr(form, [
        "`var value_f32: f32 = 0.0; var value = ${" + O + ".type.storage}(0);`",
        '""',
        "`value_f32 += exp(f32(${" + I + '.getByIndices("input_indices")}));`',
        "`value = ${" + O + ".type.value}(log(value_f32));`",
    ]);
}
function buildProd(I, O, form) {
    return joinArr(form, [
        "`var value_f32: f32 = 1.0; var value = ${" + O + ".type.storage}(1);`",
        '""',
        "`value_f32 *= f32(${" + I + '.getByIndices("input_indices")});`',
        "`value = ${" + O + ".type.value}(value_f32);`",
    ]);
}
function buildSum(I, O, form) {
    return joinArr(form, [
        "`var value_f32: f32 = 0.0; var value = ${" + O + ".type.storage}(0);`",
        '""',
        "`value_f32 += f32(${" + I + '.getByIndices("input_indices")});`',
        "`value = ${" + O + ".type.value}(value_f32);`",
    ]);
}
function buildSumSquare(I, O, form) {
    return joinArr(form, [
        "`var t_f32: f32 = 0.0; var value_f32: f32 = 0.0; var value = ${" + O + ".type.value}(0);`",
        '""',
        "`t_f32 = f32(${" + I + '.getByIndices("input_indices")}); value_f32 += t_f32 * t_f32;`',
        "`value = ${" + O + ".type.value}(value_f32);`",
    ]);
}

const REDUCERS = [
    ["ReduceLogSum", buildLogSum],
    ["ReduceL1", buildL1],
    ["ReduceL2", buildL2],
    ["ReduceLogSumExp", buildLogSumExp],
    ["ReduceProd", buildProd],
    ["ReduceSum", buildSum],
    ["ReduceSumSquare", buildSumSquare],
];

function patchReducerArray(src, name, build) {
    // Minified: `,(n,o)=>[...]\)` immediately followed by the call site that
    // contains `"NAME"`.  We capture (n,o)=>[...] and check via the gap to
    // confirm "NAME" appears soon after — the `st(t,"NAME",e,(n,o)=>[...])`
    // call has the name BEFORE the arrow, so look for `"NAME"` BEFORE.
    let changed = 0;
    // Form: st(t,"NAME",e,(IN,OUT)=>[ARR])    (minified)
    const reMin = new RegExp(
        `("${name}"[^()]*?,\\s*\\((${ID}),\\s*(${ID})\\)\\s*=>\\s*\\[)([\\s\\S]*?)(\\]\\))`,
    );
    const m1 = src.match(reMin);
    if (m1) {
        const [whole, prefix, inI, outI, arrBody, suffix] = m1;
        if (!alreadyPatched(arrBody)) {
            const newBody = build(inI, outI, "min");
            src = src.replace(whole, prefix + newBody + suffix);
            changed++;
        }
    } else {
        // Form: const reduceOp = (input, output) => [\n ... \n];\n runReduceProgram(context, "NAME"
        // Tempered lazy match so we don't span across the previous reducer's
        // `const reduceOp` block (multiple of those exist in the unminified
        // bundle).
        const reUn = new RegExp(
            `(const reduceOp = \\((${ID}),\\s*(${ID})\\)\\s*=>\\s*\\[)((?:(?!const reduceOp)[\\s\\S])*?)(\\];\\s*runReduceProgram\\(context,\\s*"${name}")`,
        );
        const m2 = src.match(reUn);
        if (m2) {
            const [whole, prefix, inI, outI, arrBody, suffix] = m2;
            if (!alreadyPatched(arrBody)) {
                const newBody = build(inI, outI, "un");
                src = src.replace(whole, prefix + newBody + suffix);
                changed++;
            }
        } else {
            throw new Error(`reducer ${name}: neither minified nor unminified pattern matched`);
        }
    }
    return { src, changed };
}

function patchGroupedConvNaive(src) {
    // Decl + bias-line interpolation pattern lives uniquely in the GroupedConv
    // shader template. Anchor on the unique `name:"GroupedConv"` later in the
    // file by checking it exists; the actual edit anchors on the WGSL decl.
    let changed = 0;

    if (!/name:\s*"GroupedConv"/.test(src)) {
        return { src, changed, note: "no GroupedConv found" };
    }

    const declAlready = /var value_f32_acc: f32 = 0\.0;\n\s+\$\{[a-zA-Z_$][a-zA-Z_$0-9]*\}\n\s+var value: \$\{[a-zA-Z_$][a-zA-Z_$0-9]*\.type\.value\} = \$\{[a-zA-Z_$][a-zA-Z_$0-9]*\.type\.value\}\(value_f32_acc\);/.test(src);
    const bodyAlready = /value_f32_acc \+= f32\(xVal\) \* f32\(wVal\);/.test(src);

    if (declAlready && bodyAlready) {
        return { src, changed, note: "GroupedConv naive already-patched" };
    }

    const sub = (re, repl) => {
        const before = src;
        src = src.replace(re, repl);
        if (before === src) throw new Error(`GroupedConv naive: pattern did not match: ${re}`);
        changed++;
    };

    // 1. Replace `var value: T = T(0);` decl + the `${v}` body interpolation
    //    plus a new cast line before `${i}` (bias). Anchor: the unique 4-line
    //    sequence inside the GroupedConv shader template.
    if (!declAlready) {
        const declReDyn = new RegExp(
            "var value: (\\$\\{(" + ID + ")\\.type\\.value\\}) = \\$\\{\\2\\.type\\.value\\}\\(0\\);\\n(\\s+)(\\$\\{(" + ID + ")\\})\\n(\\s+)(\\$\\{(" + ID + ")\\})",
        );
        const m = src.match(declReDyn);
        if (!m) {
            throw new Error("GroupedConv naive: decl+v+i anchor not found");
        }
        // m[1] = `${x.type.value}` literal, m[2] = x ident, m[4] = `${v}`, m[6] = `${i}`
        sub(
            declReDyn,
            `var value_f32_acc: f32 = 0.0;\n$3$4\n$3var value: $1 = $1(value_f32_acc);\n$6$7`,
        );
    }

    // 2. Body lines inside `${v}`: change both NHWC and NCHW occurrences.
    if (!bodyAlready) {
        const before = src;
        src = src.replace(
            /value \+= xVal \* wVal;/g,
            "value_f32_acc += f32(xVal) * f32(wVal);",
        );
        if (before === src) throw new Error("GroupedConv naive: no `value += xVal * wVal;` lines found");
        changed++;
    }

    return { src, changed };
}

function patchGroupedConvVectorize(src) {
    // Markers unique to the vectorized variant.
    if (!/var x_vals: array<\$\{[a-zA-Z_$][a-zA-Z_$0-9]*\.type\.value\},/.test(src)) {
        return { src, changed: 0, note: "no GroupedConv-Vectorize found" };
    }
    let changed = 0;
    const declAlready = /var values_f32: array<f32,/.test(src);
    const fmaAlready = /values_f32\[i\] = f32\(x_vals\[/.test(src);
    const readAlready = /var value = \$\{[a-zA-Z_$][a-zA-Z_$0-9]*\.type\.value\}\(values_f32\[i\]\);/.test(src);

    if (declAlready && fmaAlready && readAlready) {
        return { src, changed, note: "GroupedConv-Vectorize already-patched" };
    }

    const sub = (re, repl) => {
        const before = src;
        src = src.replace(re, repl);
        if (before === src) throw new Error(`GroupedConv-Vectorize: pattern did not match: ${re}`);
        changed++;
    };

    if (!declAlready) {
        // Add `var values_f32: array<f32, ${s}>;` next to the existing values decl.
        const declRe = new RegExp(
            "(var values: array<\\$\\{(" + ID + ")\\.type\\.value\\}, (\\$\\{" + ID + "\\})>;)",
        );
        const m = src.match(declRe);
        if (!m) throw new Error("GroupedConv-Vectorize: values decl not found");
        sub(declRe, `$1 var values_f32: array<f32, $3>;`);
    }

    if (!fmaAlready) {
        sub(
            /values\[i\] = fma\(x_vals\[i \* u32\(uniforms\.strides\[1\]\) \+ w_width\], w_val, values\[i\]\);/,
            "values_f32[i] = f32(x_vals[i * u32(uniforms.strides[1]) + w_width]) * f32(w_val) + values_f32[i];",
        );
    }

    if (!readAlready) {
        const readRe = new RegExp(
            "var value = values\\[i\\];",
        );
        // Need the output type ident for the cast back. The vec template uses
        // `${w.type.value}` for the values array element type — capture it via
        // a nearby anchor.
        const wIdMatch = src.match(
            new RegExp(
                "var values: array<\\$\\{(" + ID + ")\\.type\\.value\\},",
            ),
        );
        if (!wIdMatch) throw new Error("GroupedConv-Vectorize: cannot resolve w-helper ident");
        const wId = wIdMatch[1];
        sub(readRe, `var value = \${${wId}.type.value}(values_f32[i]);`);
    }

    return { src, changed };
}

function patch(src) {
    let totalChanged = 0;
    const log = [];

    for (const [name, build] of REDUCERS) {
        const { src: ns, changed } = patchReducerArray(src, name, build);
        src = ns;
        totalChanged += changed;
        if (changed) log.push(name);
    }

    {
        const r = patchGroupedConvNaive(src);
        src = r.src;
        totalChanged += r.changed;
        if (r.changed) log.push("GroupedConv-naive");
    }
    {
        const r = patchGroupedConvVectorize(src);
        src = r.src;
        totalChanged += r.changed;
        if (r.changed) log.push("GroupedConv-Vectorize");
    }

    return { src, changed: totalChanged, log };
}

for (const name of FILES) {
    const p = path.join(DIST, name);
    if (!fs.existsSync(p)) {
        console.log(`skip ${name} (missing)`);
        continue;
    }
    const orig = fs.readFileSync(p, "utf8");
    const { src, changed, log } = patch(orig);
    if (src === orig) {
        console.log(`skip ${name} (already-patched)`);
        continue;
    }
    fs.writeFileSync(p, src);
    console.log(`patched ${name} [edits=${changed}, ops=${log.join(",")}]`);
}
