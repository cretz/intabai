#!/usr/bin/env node
// Patches onnxruntime-web Conv3DNaive WebGPU shader to handle fp16 inputs.
//
// Upstream bug: helper signatures (`fn getX(...) -> f32`), the final store
// (`result[global_idx] = f32(value)`), and the bias add all hardcode f32
// while input/output storage is f16 for fp16 models. WGSL rejects the
// pipeline at compile time, Chrome logs `[Invalid ComputePipeline]`, and
// session.run() silently emits bias-only output.
//
// Fix: promote f16 values to f32 on load (inside the getX/getW helpers and
// around the bias add). Keep the accumulator as f32 (free precision win,
// same pattern as the MatMul fp32-acc patch). Cast back to the output
// storage type at the final write.
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

function patch(src) {
    if (!/name:\s*"Conv3DNaive"/.test(src)) {
        return { src, changed: 0, note: "no Conv3DNaive shader" };
    }
    const biasPatched = src.includes(
        "value = value + f32(getBiasByOutputCoords(coords))",
    );
    // Conv3DNaive-specific accumulator pattern (the AttentionScore kernel
    // also uses `var value: f32 = 0.0;`, so anchor on the next line that's
    // unique to Conv3DNaive's outer filter loop).
    const accumTyped = /var value: f32 = 0\.0;\n\s+for \(var wF = 0u;/.test(src);
    if (biasPatched && accumTyped) {
        return { src, changed: 0, note: "already-patched" };
    }

    // Capture the storage-type JS variable used in Conv3DNaive's template.
    // The `getBiasByOutputCoords` return annotation inlines it as
    // `-> ${T?`vec4<${V}>`:V}` where V is our target (scalar f16/f32 storage).
    // Allow optional whitespace (unminified uses `T ? ... : V`).
    const typeMatch = src.match(
        /fn getBiasByOutputCoords\(coords : array<u32, 5>\) -> \$\{[a-zA-Z_$][a-zA-Z_$0-9]*\s*\?\s*`vec4<\$\{([a-zA-Z_$][a-zA-Z_$0-9]*)\}>`\s*:\s*\1\s*\} \{/,
    );
    if (!typeMatch) {
        throw new Error("could not capture Conv3DNaive storage-type variable");
    }
    const tVar = typeMatch[1];
    const tExpr = "${" + tVar + "}";

    let changed = 0;
    const sub = (re, repl) => {
        const before = src;
        src = src.replace(re, repl);
        if (before === src) throw new Error(`pattern did not match: ${re}`);
        changed++;
    };

    if (!biasPatched) {
        // 1. Helper returns (getX + getW): promote f16 load to f32.
        const returnPat =
            /return (\$\{[a-zA-Z_$][a-zA-Z_$0-9]*\.getByIndices\("aIndices"\)\});/g;
        const returnMatches = src.match(returnPat) || [];
        if (returnMatches.length !== 2) {
            throw new Error(
                `expected 2 getByIndices returns for Conv3DNaive, found ${returnMatches.length}`,
            );
        }
        src = src.replace(returnPat, "return f32($1);");
        changed++;

        // 2. Bias add: cast `${t}` bias to f32 before f32 accumulator add.
        sub(
            /"value = value \+ getBiasByOutputCoords\(coords\)"/,
            '"value = value + f32(getBiasByOutputCoords(coords))"',
        );

        // 3. Final store: cast f32 accumulator to output storage type.
        sub(
            /result\[global_idx\] = f32\(value\);/,
            `result[global_idx] = ${tExpr}(value);`,
        );
    }

    if (!accumTyped) {
        // 4. Explicit f32 typing on the accumulator. Removes WGSL inference
        //    ambiguity now that loads are f32-promoted and final store casts
        //    to `${t}`.
        sub(
            /var value = 0\.0;\n/,
            "var value: f32 = 0.0;\n",
        );
    }

    return { src, changed, typeVar: tVar };
}

for (const name of FILES) {
    const p = path.join(DIST, name);
    if (!fs.existsSync(p)) {
        console.log(`skip ${name} (missing)`);
        continue;
    }
    const orig = fs.readFileSync(p, "utf8");
    const { src, changed, note, typeVar } = patch(orig);
    if (src === orig) {
        console.log(`skip ${name} (${note || "no change"})`);
        continue;
    }
    fs.writeFileSync(p, src);
    console.log(`patched ${name} [typeVar=${typeVar}, edits=${changed}]`);
}
