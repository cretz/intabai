// LTX-Video 2B 0.9.8 distilled generate entry point.
//
// T2V multi-scale flow (matches configs/ltxv-2b-0.9.8-distilled.yaml):
//   1. T5-XXL text encode (24 q4f16 blocks layer-streamed).
//   2. First pass at downscaled resolution (target * 0.6667, rounded down to
//      a 32-multiple): 7 from_checkpoint sigmas down to 0.7250.
//   3. Spatial upscale: latent -> un_normalize -> upscaler doubles spatial
//      -> normalize -> AdaIN match channel mean/std to the first-pass output.
//   4. Re-noise to sigma=0.9094 (`x = (1-s)*upsampled + s*noise`), then
//      second pass at 2x downscaled resolution: 3 sigmas down to 0.4219,
//      with a final dt down to 0.
//   5. Unpatchify, un-normalize, VAE decode, render to ImageBitmap[].
//
// Output resolution is the second-pass resolution (== 2 * downscaled),
// which is generally larger than the original target. The reference space
// bilinear-resizes back; we skip that and just deliver the higher-res frames.
//
// I2V conditioning (input image as first latent frame) and per-block
// transformer caching across passes are deferred.

import type { ModelCache } from "../shared/model-cache";
import { f16BitsToF32, f32ToF16Bits } from "../sd15/fp16";
import { LtxT5Embedding } from "./embedding";
import {
  loadLtxLatentStats,
  normalizeLatentF16,
  unNormalizeLatentF16,
} from "./latent-stats";
import {
  LTX_T5_EMBEDDING_Q8_FILE,
  LTX_T5_EMBEDDING_SCALES_FILE,
  ltxT5BlockFiles,
  ltxT5ShellPostFile,
  ltxTransformerFiles,
  ltxUpscalerFile,
  ltxVaeFiles,
} from "./models";
import {
  buildIndicesGrid,
  latentNumTokens,
  patchifyF16,
  pixelToLatentShape,
  unpatchifyF16,
  LTX_LATENT_CHANNELS,
  LTX_VAE_SPATIAL_DOWNSCALE,
  type LtxLatentShape,
} from "./patchifier";
import { LtxT5Encoder } from "./text-encoder";
import { loadTokenizer, tokenize } from "./tokenizer";
import { LtxTransformer, LTX_TX_PATCH_CHANNELS } from "./transformer";
import { LtxUpscaler } from "./upscaler";
import { LtxVae, type VaeShardTiming } from "./vae";

/** LTX configs use 256-token T5 context. */
const LTX_T5_MAX_LEN = 256;

/** Target output resolution before the multi-scale upsample. The actual
 *  delivered resolution is `2 * floor(TARGET * DOWNSCALE / 32) * 32`,
 *  so 256 -> first-pass 160 -> second-pass / output 320. Frame count
 *  must be 8k+1. */
const TARGET_FRAMES = 25;
const TARGET_HW = 384;
const DOWNSCALE = 0.6666666;
/** HF Space app.py uses FPS = 30.0; matches the frame_rate the model
 *  was trained with for indices_grid time scaling. */
const DEFAULT_FPS = 30;

/** Diagnostic bypass: skip T5 + both denoise passes + upscaler. Decode
 *  fresh N(0,1) noise at highShape directly. If decoded frames look like
 *  random colored noise -> VAE is fine, bug is upstream (sampler/transformer).
 *  If still tiled 10x10 -> bug is in the VAE export. */
const SKIP_TRANSFORMER = false;

/** Diagnostic transformer fidelity test: run ONE forward step on
 *  canonical deterministic input (all-ones hidden, zeros enc_hidden,
 *  timestep=1.0, latent shape (2, 4, 4) = 32 tokens) and log sample
 *  values, then abort. Cross-check vs `ltx/reference_transformer.py`
 *  output. Overrides SKIP_TRANSFORMER. */
const CANONICAL_TX_TEST = false;

/** Diagnostic T5 fidelity test: run text encoder, dump first 16 channels of
 *  several token positions, then abort. Two runs with different prompts
 *  should produce visibly different numbers; identical numbers mean T5 is
 *  broken or being bypassed. */
const T5_DEBUG_DUMP = false;

/** Verbose end-to-end pipeline instrumentation. Logs (mean, std, min, max,
 *  first-3-samples) at every checkpoint between T5 and final pixels. Run
 *  twice with different prompts and diff the logs to find where prompt
 *  dependence vanishes. */
const VERBOSE_DEBUG = false;

/** Diagnostic per-component tap inside the transformer step: log fp16
 *  stats after shell_pre, every block, and shell_post. Combined with
 *  LTX_ABORT_AFTER_PASS1_STEP1 this localizes the wasm/webgpu divergence
 *  to a specific component in ~1 wasm step (~50s) instead of a full run.
 *  Diff the [v] block_NN lines between EPs; the first one to diverge
 *  meaningfully is the buggy component. */
const LTX_PER_BLOCK_TAP = false;

/** Diagnostic abort: throw after pass1 step 1 completes, before step 2.
 *  Pairs with LTX_PER_BLOCK_TAP. Wasm cost ~50s, webgpu ~10s. */
const LTX_ABORT_AFTER_PASS1_STEP1 = false;

class LtxAbortAfterStep1 extends Error {
  constructor() {
    super("LTX_ABORT_AFTER_PASS1_STEP1: diagnostic abort, expected");
    this.name = "LtxAbortAfterStep1";
  }
}

/** Diagnostic dump: during pass1 step 1, capture the 6 fp16 inputs that
 *  block_26 will consume (hidden = block_25 output, plus the 5 aux tensors
 *  from shell_pre) and download as a single .bin bundle. Use the dump as
 *  the input for an in-browser block_26 wasm-vs-webgpu diff with REAL
 *  production-shape activations instead of synthetic sinFill. After dump
 *  triggers, throws LtxAbortAfterDumpInputs to skip the rest of the run.
 *
 *  Workflow:
 *   1. Set this flag to true, run generate once on either EP.
 *   2. Browser downloads `block_26_pass1_step1_inputs.bin`.
 *   3. Move file to notes/models/ltx/hf-repo/captures/.
 *   4. Set this flag back to false. Re-run model-smoke ltx-block-26-real
 *      candidate which fetches the bundle via the local-models proxy. */
const LTX_DUMP_BLOCK26_INPUTS = false;

/** Diagnostic dump: during pass1 step 1, capture shell_pre.hidden plus
 *  every transformer block's output (block_00..block_27) as one bundle.
 *  Filename `ltx_all_blocks_pass1_step1_${ep}.bin`. Pair with two runs
 *  (one webgpu, one wasm via ?wasmltx) to compare element-wise per
 *  block and find the first block where wgpu/wasm diverge beyond fp16
 *  ULP. Supersedes LTX_DUMP_BLOCK26_INPUTS for upstream-drift hunting. */
const LTX_DUMP_ALL_BLOCKS = false;
const LTX_NUM_BLOCKS = 28;

/** Diagnostic dump: capture every VAE decoder shard's fp16 output (latent_in,
 *  dec_shell_pre.hidden, dec_block_*_*.hidden_out, dec_shell_post.pixels) as
 *  one bundle. Filename `ltx_vae_decode_${ep}.bin`. Pair with two runs
 *  (one webgpu = no flag, one wasm via ?wasmvae) to find the first decoder
 *  shard where wgpu/wasm diverge beyond fp16 ULP. Does NOT abort -- run
 *  completes so you can visually confirm output. */
const LTX_DUMP_VAE_SHARDS = true;

class LtxAbortAfterDumpInputs extends Error {
  constructor() {
    super("LTX_DUMP_BLOCK26_INPUTS: dump complete, expected abort");
    this.name = "LtxAbortAfterDumpInputs";
  }
}

/** Pack {name -> {shape, fp16 bytes}} into a single binary blob:
 *    8 bytes  ASCII magic "LTXBLK26"
 *    4 bytes  uint32 LE header JSON length
 *    H bytes  UTF-8 JSON: {"tensors":{name:{shape,byteOffset,byteLength}}}
 *    N bytes  concatenated fp16 raw bytes (little-endian)
 *  and trigger a browser download. */
function downloadBlock26InputBundle(
  filename: string,
  tensors: Record<string, { shape: number[]; data: Uint16Array }>,
  magic: string = "LTXBLK26",
): void {
  const order = Object.keys(tensors);
  const manifest: Record<string, { shape: number[]; byteOffset: number; byteLength: number }> = {};
  let cursor = 0;
  for (const name of order) {
    const t = tensors[name];
    const byteLength = t.data.byteLength;
    manifest[name] = { shape: t.shape, byteOffset: cursor, byteLength };
    cursor += byteLength;
  }
  const headerJson = JSON.stringify({ tensors: manifest });
  const headerBytes = new TextEncoder().encode(headerJson);
  const total = 8 + 4 + headerBytes.byteLength + cursor;
  const out = new Uint8Array(total);
  let p = 0;
  if (magic.length !== 8) throw new Error(`magic must be 8 ASCII chars, got "${magic}"`);
  out.set(new TextEncoder().encode(magic), p); p += 8;
  new DataView(out.buffer).setUint32(p, headerBytes.byteLength, true); p += 4;
  out.set(headerBytes, p); p += headerBytes.byteLength;
  for (const name of order) {
    const bytes = new Uint8Array(tensors[name].data.buffer, tensors[name].data.byteOffset, tensors[name].data.byteLength);
    out.set(bytes, p); p += bytes.byteLength;
  }
  const blob = new Blob([out], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Distilled schedule from configs/ltxv-2b-0.9.8-distilled.yaml. */
const FIRST_PASS_SIGMAS = [
  1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725,
];
const SECOND_PASS_SIGMAS = [0.9094, 0.725, 0.4219];

/** Decode-time noise mix (matches yaml decode_timestep / decode_noise_scale). */
const DECODE_TIMESTEP = 0.05;
const DECODE_NOISE_SCALE = 0.025;

export interface LtxProgressInfo {
  stage: "load" | "denoise" | "vae";
  pct: number;
  message: string;
}

export interface LtxGenerateArgs {
  cache: ModelCache;
  prompt: string;
  seed?: number;
  inputImage?: ImageBitmap;
  signal?: AbortSignal;
  onProgress?: (info: LtxProgressInfo) => void;
  onPreview?: (frames: ImageBitmap[]) => void;
  onDebug?: (msg: string) => void;
}

export interface LtxGenerateResult {
  frames: ImageBitmap[];
  fps: number;
  seed: number;
}

function statsF16(label: string, arr: Uint16Array, debug: (m: string) => void): void {
  if (!VERBOSE_DEBUG) return;
  let mn = Infinity, mx = -Infinity, sum = 0;
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    const v = f16BitsToF32(arr[i]);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  const mean = sum / n;
  let varAcc = 0;
  for (let i = 0; i < n; i++) {
    const d = f16BitsToF32(arr[i]) - mean;
    varAcc += d * d;
  }
  const std = Math.sqrt(varAcc / n);
  const f = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(4);
  const s0 = f(f16BitsToF32(arr[0]));
  const s1 = f(f16BitsToF32(arr[Math.min(1, n - 1)]));
  const sm = f(f16BitsToF32(arr[Math.floor(n / 2)]));
  const sl = f(f16BitsToF32(arr[n - 1]));
  debug(
    `[v] ${label} n=${n} mean=${f(mean)} std=${f(std)} min=${f(mn)} max=${f(mx)} ` +
      `[0]=${s0} [1]=${s1} [n/2]=${sm} [last]=${sl}`,
  );
}

function statsF32(label: string, arr: Float32Array, debug: (m: string) => void): void {
  if (!VERBOSE_DEBUG) return;
  let mn = Infinity, mx = -Infinity, sum = 0;
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  const mean = sum / n;
  let varAcc = 0;
  for (let i = 0; i < n; i++) {
    const d = arr[i] - mean;
    varAcc += d * d;
  }
  const std = Math.sqrt(varAcc / n);
  const f = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(4);
  const s0 = f(arr[0]);
  const s1 = f(arr[Math.min(1, n - 1)]);
  const sm = f(arr[Math.floor(n / 2)]);
  const sl = f(arr[n - 1]);
  debug(
    `[v] ${label} n=${n} mean=${f(mean)} std=${f(std)} min=${f(mn)} max=${f(mx)} ` +
      `[0]=${s0} [1]=${s1} [n/2]=${sm} [last]=${sl}`,
  );
}

function providers(role?: "t5" | "shellPre" | "vae"): ("webgpu" | "wasm")[] {
  const params = new URLSearchParams(location.search);
  if (params.has("wasmltx")) return ["wasm"];
  if (role === "t5" && params.has("wasmt5")) return ["wasm"];
  if (role === "shellPre" && params.has("wasmshellpre")) return ["wasm"];
  if (role === "vae" && params.has("wasmvae")) return ["wasm"];
  const out: ("webgpu" | "wasm")[] = [];
  if ("gpu" in navigator) out.push("webgpu");
  out.push("wasm");
  return out;
}

export async function generateLtx(args: LtxGenerateArgs): Promise<LtxGenerateResult> {
  const { cache, prompt, onDebug, onProgress, signal } = args;
  const debug = onDebug ?? (() => {});

  const SEED = (args.seed ?? Math.floor(Math.random() * 0x7fffffff)) >>> 0;
  let s = SEED || 1;
  const rand01 = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return ((s >>> 8) & 0xffffff) / 0x1000000;
  };

  if (LTX_LATENT_CHANNELS !== LTX_TX_PATCH_CHANNELS) {
    throw new Error(
      `LTX channel mismatch: latent=${LTX_LATENT_CHANNELS} tx=${LTX_TX_PATCH_CHANNELS}`,
    );
  }

  // ---- Diagnostic: transformer fidelity test ---------------------------
  if (CANONICAL_TX_TEST) {
    debug(`[ltx] CANONICAL_TX_TEST: running 1 transformer step on canonical input`);
    onProgress?.({ stage: "load", pct: 0, message: "tx canonical test" });

    const canonShape: LtxLatentShape = { t: 2, h: 4, w: 4 };
    const nTokens = canonShape.t * canonShape.h * canonShape.w; // 32
    const nText = 256;

    // hidden = all ones, fp16
    const hiddenF16 = new Uint16Array(nTokens * LTX_TX_PATCH_CHANNELS);
    const oneF16 = f32ToF16Bits(1.0);
    for (let i = 0; i < hiddenF16.length; i++) hiddenF16[i] = oneF16;

    // encoder_hidden_states = zeros, fp16
    const encHidden = new Uint16Array(1 * nText * 4096);

    // encoder_attention_mask = all ones, int64
    const encMask = new BigInt64Array(nText);
    for (let i = 0; i < nText; i++) encMask[i] = 1n;

    // indices_grid: same as patchifier.buildIndicesGrid with FPS=30.
    const indicesGrid = buildIndicesGrid(canonShape, DEFAULT_FPS);

    // timestep = 1.0
    const timestep = new Uint16Array([f32ToF16Bits(1.0)]);

    const tx = new LtxTransformer(cache, ltxTransformerFiles(), providers(), providers("shellPre"));
    signal?.throwIfAborted();
    const txResult = await tx.runStep(
      {
        hiddenStates: hiddenF16,
        indicesGrid,
        encoderHiddenStates: encHidden,
        encoderAttentionMask: encMask,
        timestep,
        nTokens,
        nText,
      },
      (info) => {
        onProgress?.({
          stage: "denoise",
          pct: (info.blockIndex + 1) / info.totalBlocks,
          message: `tx-test block ${info.blockIndex + 1}/${info.totalBlocks}`,
        });
      },
    );

    const np = txResult.noisePred;
    const C = LTX_TX_PATCH_CHANNELS;
    debug(`[ltx-tx-test] noise_pred length=${np.length} (expect ${nTokens * C})`);
    // Overall stats
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (let i = 0; i < np.length; i++) {
      const v = f16BitsToF32(np[i]);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
    }
    const mean = sum / np.length;
    let varAcc = 0;
    for (let i = 0; i < np.length; i++) {
      const d = f16BitsToF32(np[i]) - mean;
      varAcc += d * d;
    }
    const std = Math.sqrt(varAcc / np.length);
    debug(
      `[ltx-tx-test] np overall: mean=${mean.toFixed(5)} std=${std.toFixed(5)} ` +
        `min=${mn.toFixed(5)} max=${mx.toFixed(5)}`,
    );

    const dumpTok = (tok: number) => {
      if (tok >= nTokens) return;
      const vals: string[] = [];
      let s8 = 0;
      for (let c = 0; c < 8; c++) {
        const v = f16BitsToF32(np[tok * C + c]);
        vals.push((v >= 0 ? "+" : "") + v.toFixed(5));
        s8 += v;
      }
      debug(`[ltx-tx-test] token${tok.toString().padStart(4)} ch0..7=[${vals.join(", ")}] sum8=${s8 >= 0 ? "+" : ""}${s8.toFixed(5)}`);
    };
    dumpTok(0);
    dumpTok(1);
    if (nTokens > 64) dumpTok(64);
    dumpTok(Math.floor(nTokens / 2));
    dumpTok(nTokens - 1);

    // Per-channel sums over all tokens (first 8 channels).
    const chSums: number[] = [];
    for (let c = 0; c < 8; c++) {
      let s = 0;
      for (let t = 0; t < nTokens; t++) s += f16BitsToF32(np[t * C + c]);
      chSums.push(s);
    }
    debug(
      `[ltx-tx-test] per-channel sums (over ${nTokens} tokens) ch0..7=[` +
        chSums.map((v) => (v >= 0 ? "+" : "") + v.toFixed(4)).join(", ") +
        "]",
    );
    onProgress?.({ stage: "vae", pct: 1, message: "tx test done" });
    throw new Error("CANONICAL_TX_TEST done; aborting");
  }

  // Resolution arithmetic. downscaled is the first-pass target; upscaled
  // (=2x) is the second-pass and final output target.
  const downscaledHw = Math.max(
    LTX_VAE_SPATIAL_DOWNSCALE,
    Math.floor((TARGET_HW * DOWNSCALE) / LTX_VAE_SPATIAL_DOWNSCALE) *
      LTX_VAE_SPATIAL_DOWNSCALE,
  );
  const upscaledHw = downscaledHw * 2;
  const lowShape = pixelToLatentShape(TARGET_FRAMES, downscaledHw, downscaledHw);
  const highShape: LtxLatentShape = { t: lowShape.t, h: lowShape.h * 2, w: lowShape.w * 2 };

  // ---- Diagnostic bypass: skip transformer/sampler, decode noise directly --
  if (SKIP_TRANSFORMER) {
    debug(`[ltx] SKIP_TRANSFORMER: decoding fresh noise at highShape`);
    const stats = await loadLtxLatentStats(cache, debug);
    const highSpatial = highShape.t * highShape.h * highShape.w;
    const noiseLatent = new Uint16Array(LTX_LATENT_CHANNELS * highSpatial);
    for (let i = 0; i < noiseLatent.length; i += 2) {
      const u1 = Math.max(rand01(), 1e-7);
      const u2 = rand01();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;
      noiseLatent[i] = f32ToF16Bits(r * Math.cos(theta));
      if (i + 1 < noiseLatent.length) {
        noiseLatent[i + 1] = f32ToF16Bits(r * Math.sin(theta));
      }
    }
    // Reference order: mix decode noise in NORMALIZED space, THEN un_normalize.
    // (pipeline_ltx_video.py:1321-1322 mixes pre-`vae_decode`, which calls
    // `un_normalize_latents` internally before hitting the VAE.)
    noiseMixLatentF16(noiseLatent, DECODE_NOISE_SCALE, rand01);
    unNormalizeLatentF16(noiseLatent, stats, highSpatial);

    signal?.throwIfAborted();
    onProgress?.({ stage: "vae", pct: 0, message: "bypass decode" });
    const vae = new LtxVae(cache, ltxVaeFiles(), providers("vae"));
    const tDec = performance.now();
    const decResult = await vae.decode({
      latent: noiseLatent,
      shape: highShape,
      decodeTimestep: DECODE_TIMESTEP,
      decodeNoiseScale: 0,
      rand01,
      signal,
      onDebug: debug,
      onShard: (info: VaeShardTiming) => {
        onProgress?.({
          stage: "vae",
          pct: (info.shardIndex + 1) / info.totalShards,
          message: `vae ${info.label}`,
        });
        debug(
          `[ltx]   ${info.label} (${info.shardIndex + 1}/${info.totalShards}): ` +
            `load=${info.loadMs.toFixed(0)}ms run=${info.runMs.toFixed(0)}ms`,
        );
      },
    });
    debug(`[ltx] VAE decode OK in ${(performance.now() - tDec).toFixed(0)} ms`);

    let pmin = Infinity;
    let pmax = -Infinity;
    let pmean = 0;
    for (let i = 0; i < decResult.pixels.length; i++) {
      const v = f16BitsToF32(decResult.pixels[i]);
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
      pmean += v;
    }
    pmean /= decResult.pixels.length;
    debug(
      `[ltx] pixels: min=${pmin.toFixed(3)} max=${pmax.toFixed(3)} ` +
        `mean=${pmean.toFixed(3)}`,
    );
    const frames = await framesToBitmaps(
      decResult.pixels,
      decResult.pixelFrames,
      decResult.pixelHeight,
      decResult.pixelWidth,
    );
    onProgress?.({ stage: "vae", pct: 1, message: "done (bypass)" });
    return { frames, fps: DEFAULT_FPS, seed: SEED };
  }

  // Coarse work-fraction split for progress. First pass dominates step
  // count, second pass dominates per-step cost (4x tokens), VAE decode
  // is at the bigger res so non-trivial.
  const W_T5 = 0.10;
  const W_PASS1 = 0.20;
  const W_UPSCALE = 0.05;
  const W_PASS2 = 0.30;
  const W_DECODE = 0.35;

  // ---- T5 text encode ----------------------------------------------------
  debug(`[ltx] T5 encode (seed=${SEED})`);
  onProgress?.({ stage: "load", pct: 0, message: "loading T5 embedding" });
  signal?.throwIfAborted();

  const [q8Buf, scalesBuf] = await Promise.all([
    cache.loadFile(LTX_T5_EMBEDDING_Q8_FILE),
    cache.loadFile(LTX_T5_EMBEDDING_SCALES_FILE),
  ]);
  const embedding = new LtxT5Embedding(q8Buf, scalesBuf);

  const tokenizer = await loadTokenizer(cache);
  const tokenized = tokenize(tokenizer, prompt, LTX_T5_MAX_LEN);
  debug(`[ltx] tokenized: ${tokenized.validLength}/${LTX_T5_MAX_LEN} real tokens`);

  const encoder = new LtxT5Encoder(
    cache,
    { blocks: ltxT5BlockFiles(), shellPost: ltxT5ShellPostFile() },
    embedding,
    providers("t5"),
  );

  signal?.throwIfAborted();
  const tT5 = performance.now();
  const t5Result = await encoder.encode(tokenized.ids, tokenized.validLength, (info) => {
    onProgress?.({
      stage: "load",
      pct: W_T5 * ((info.blockIndex + 1) / info.totalBlocks),
      message: `T5 ${info.blockIndex + 1}/${info.totalBlocks}`,
    });
  });
  debug(`[ltx] T5 OK in ${(performance.now() - tT5).toFixed(0)} ms`);

  if (T5_DEBUG_DUMP) {
    debug(`[ltx-t5-dump] validLength=${t5Result.validLength} seqLen=${t5Result.seqLen}`);
    debug(`[ltx-t5-dump] tokenIds[0..7]=[${Array.from(tokenized.ids.slice(0, 8), (b) => Number(b)).join(", ")}]`);
    const fmt = (x: number): string => (x >= 0 ? "+" : "") + x.toFixed(4);
    const sampleRow = (label: string, h: Uint16Array, tok: number, D: number) => {
      const off = tok * D;
      const vals: number[] = [];
      for (let j = 0; j < 16; j++) vals.push(f16BitsToF32(h[off + j]));
      let sumAbs = 0;
      for (let j = 0; j < D; j++) sumAbs += Math.abs(f16BitsToF32(h[off + j]));
      debug(`[ltx-${label}] tok${tok} ch0..15=[${vals.map(fmt).join(", ")}] |row|/D=${(sumAbs / D).toFixed(4)}`);
    };
    sampleRow("t5", t5Result.hiddenStates, 0, 4096);
    sampleRow("t5", t5Result.hiddenStates, Math.max(0, t5Result.validLength - 1), 4096);

    // One full transformer step on canonical low-res shape with the real
    // T5 output. If two prompts give different noise_pred -> cross-attn
    // works -> bug is in sampler/upscaler/VAE. If identical -> cross-attn
    // is structurally dead (dig into block export / caption_projection).
    const dbgEncMask = new BigInt64Array(t5Result.seqLen);
    for (let i = 0; i < t5Result.validLength; i++) dbgEncMask[i] = 1n;
    const dbgShape: LtxLatentShape = pixelToLatentShape(TARGET_FRAMES, downscaledHw, downscaledHw);
    const dbgNTokens = latentNumTokens(dbgShape);
    const dbgIndices = buildIndicesGrid(dbgShape, DEFAULT_FPS);
    const dbgHidden = new Uint16Array(dbgNTokens * LTX_TX_PATCH_CHANNELS);
    for (let i = 0; i < dbgHidden.length; i++) dbgHidden[i] = f32ToF16Bits(0.5);
    const dbgTimestep = new Uint16Array([f32ToF16Bits(1.0)]);
    const dbgTx = new LtxTransformer(cache, ltxTransformerFiles(), providers(), providers("shellPre"));
    const tDbg = performance.now();
    const dbgOut = await dbgTx.runStep({
      hiddenStates: dbgHidden,
      indicesGrid: dbgIndices,
      encoderHiddenStates: t5Result.hiddenStates,
      encoderAttentionMask: dbgEncMask,
      timestep: dbgTimestep,
      nTokens: dbgNTokens,
      nText: t5Result.seqLen,
    });
    debug(`[ltx-tx-dump] step done in ${((performance.now() - tDbg) / 1000).toFixed(1)}s`);
    debug(`[ltx-tx-dump] shape t=${dbgShape.t} h=${dbgShape.h} w=${dbgShape.w} nTokens=${dbgNTokens}`);
    sampleRow("tx", dbgOut.noisePred, 0, LTX_TX_PATCH_CHANNELS);
    sampleRow("tx", dbgOut.noisePred, 1, LTX_TX_PATCH_CHANNELS);
    sampleRow("tx", dbgOut.noisePred, Math.floor(dbgNTokens / 2), LTX_TX_PATCH_CHANNELS);
    sampleRow("tx", dbgOut.noisePred, dbgNTokens - 1, LTX_TX_PATCH_CHANNELS);
    let txMin = Infinity, txMax = -Infinity, txSum = 0;
    for (let i = 0; i < dbgOut.noisePred.length; i++) {
      const v = f16BitsToF32(dbgOut.noisePred[i]);
      if (v < txMin) txMin = v;
      if (v > txMax) txMax = v;
      txSum += v;
    }
    debug(`[ltx-tx-dump] overall: mean=${fmt(txSum / dbgOut.noisePred.length)} min=${fmt(txMin)} max=${fmt(txMax)}`);
    throw new Error("T5_DEBUG_DUMP done; aborting");
  }

  const encMask = new BigInt64Array(t5Result.seqLen);
  for (let i = 0; i < t5Result.validLength; i++) encMask[i] = 1n;

  // ---- Stats + transformer (shared across passes) ------------------------
  const stats = await loadLtxLatentStats(cache, debug);
  const tx = new LtxTransformer(cache, ltxTransformerFiles(), providers(), providers("shellPre"));

  // ---- Pass 1: low-res denoise -------------------------------------------
  debug(
    `[ltx] pass1: ${TARGET_FRAMES}x${downscaledHw}x${downscaledHw} -> latent ` +
      `t=${lowShape.t} h=${lowShape.h} w=${lowShape.w}`,
  );
  let hiddenF32 = initNoiseHidden(lowShape, rand01);
  statsF32("pass1 init hidden", hiddenF32, debug);
  await runDenoisePass({
    hiddenF32,
    shape: lowShape,
    sigmas: [...FIRST_PASS_SIGMAS, 0],
    tx,
    t5Result,
    encMask,
    signal,
    debug,
    label: "pass1",
    onStep: (frac) =>
      onProgress?.({
        stage: "denoise",
        pct: W_T5 + W_PASS1 * frac,
        message: `pass1 ${(frac * 100).toFixed(0)}%`,
      }),
  });

  // ---- Upscale latents ---------------------------------------------------
  signal?.throwIfAborted();
  onProgress?.({ stage: "denoise", pct: W_T5 + W_PASS1, message: "upscale" });
  statsF32("pass1 final hidden", hiddenF32, debug);
  // Latent at end of pass 1, in normalized space, NCDHW.
  const lowLatentNorm = patchHiddenToLatent(hiddenF32, lowShape);
  statsF16("lowLatent (normalized)", lowLatentNorm, debug);
  // Reference for AdaIN (deep copy, normalized).
  const refLatentNorm = lowLatentNorm.slice();

  // Decoder/upscaler want raw (un-normalized) latents.
  const lowSpatial = lowShape.t * lowShape.h * lowShape.w;
  unNormalizeLatentF16(lowLatentNorm, stats, lowSpatial);
  statsF16("lowLatent (unnormalized, fed to upscaler)", lowLatentNorm, debug);

  const upscaler = new LtxUpscaler(cache, ltxUpscalerFile(), providers());
  const upscaleResult = await upscaler.run({ latent: lowLatentNorm, shape: lowShape, signal });
  statsF16("upscaler output (raw)", upscaleResult.latent, debug);
  debug(
    `[ltx] upscale: ${lowShape.t}x${lowShape.h}x${lowShape.w} -> ` +
      `${upscaleResult.shape.t}x${upscaleResult.shape.h}x${upscaleResult.shape.w} ` +
      `(load=${upscaleResult.loadMs.toFixed(0)}ms run=${upscaleResult.runMs.toFixed(0)}ms)`,
  );
  if (
    upscaleResult.shape.t !== highShape.t ||
    upscaleResult.shape.h !== highShape.h ||
    upscaleResult.shape.w !== highShape.w
  ) {
    throw new Error(
      `upscale shape mismatch: got ${upscaleResult.shape.t}x${upscaleResult.shape.h}x${upscaleResult.shape.w} ` +
        `expected ${highShape.t}x${highShape.h}x${highShape.w}`,
    );
  }

  // Back to normalized space, then AdaIN to the low-res reference.
  const highLatentNorm = upscaleResult.latent;
  const highSpatial = highShape.t * highShape.h * highShape.w;
  normalizeLatentF16(highLatentNorm, stats, highSpatial);
  statsF16("highLatent (re-normalized)", highLatentNorm, debug);
  adainPerChannel(highLatentNorm, highSpatial, refLatentNorm, lowSpatial);
  statsF16("highLatent (after AdaIN)", highLatentNorm, debug);

  // ---- Re-noise to sigma=0.9094 + Pass 2 ---------------------------------
  hiddenF32 = patchifyHiddenFromLatentF16(highLatentNorm, highShape);
  statsF32("hiddenF32 (after patchify, pre-renoise)", hiddenF32, debug);
  const reNoiseSigma = SECOND_PASS_SIGMAS[0];
  for (let i = 0; i < hiddenF32.length; i += 2) {
    const u1 = Math.max(rand01(), 1e-7);
    const u2 = rand01();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    const n0 = r * Math.cos(theta);
    const n1 = r * Math.sin(theta);
    hiddenF32[i] = (1 - reNoiseSigma) * hiddenF32[i] + reNoiseSigma * n0;
    if (i + 1 < hiddenF32.length) {
      hiddenF32[i + 1] = (1 - reNoiseSigma) * hiddenF32[i + 1] + reNoiseSigma * n1;
    }
  }

  debug(
    `[ltx] pass2: ${TARGET_FRAMES}x${upscaledHw}x${upscaledHw} -> latent ` +
      `t=${highShape.t} h=${highShape.h} w=${highShape.w}`,
  );
  statsF32("pass2 init hidden (after re-noise)", hiddenF32, debug);
  await runDenoisePass({
    hiddenF32,
    shape: highShape,
    sigmas: [...SECOND_PASS_SIGMAS, 0],
    tx,
    t5Result,
    encMask,
    signal,
    debug,
    label: "pass2",
    onStep: (frac) =>
      onProgress?.({
        stage: "denoise",
        pct: W_T5 + W_PASS1 + W_UPSCALE + W_PASS2 * frac,
        message: `pass2 ${(frac * 100).toFixed(0)}%`,
      }),
  });

  // ---- Decode + render ---------------------------------------------------
  statsF32("pass2 final hidden", hiddenF32, debug);
  const finalLatent = patchHiddenToLatent(hiddenF32, highShape);
  statsF16("finalLatent (normalized)", finalLatent, debug);
  // Reference order: mix decode noise in NORMALIZED space, THEN un_normalize.
  // (pipeline_ltx_video.py:1321-1322 mixes pre-`vae_decode`, which calls
  // `un_normalize_latents` internally before hitting the VAE.)
  noiseMixLatentF16(finalLatent, DECODE_NOISE_SCALE, rand01);
  statsF16("finalLatent (post noise-mix, normalized)", finalLatent, debug);
  unNormalizeLatentF16(finalLatent, stats, highSpatial);
  statsF16("finalLatent (unnormalized, fed to VAE)", finalLatent, debug);

  signal?.throwIfAborted();
  debug(`[ltx] VAE decode at ${upscaledHw}x${upscaledHw}`);
  const vae = new LtxVae(cache, ltxVaeFiles(), providers("vae"));
  const tDec = performance.now();
  const vaeDumps = LTX_DUMP_VAE_SHARDS
    ? new Map<string, { data: Uint16Array; dims: number[] }>()
    : undefined;
  const decResult = await vae.decode({
    latent: finalLatent,
    shape: highShape,
    decodeTimestep: DECODE_TIMESTEP,
    decodeNoiseScale: 0,
    rand01,
    signal,
    onDebug: debug,
    tap: vaeDumps
      ? (label, data, dims) => {
          vaeDumps.set(label, { data: new Uint16Array(data), dims: dims.slice() });
        }
      : undefined,
    onShard: (info: VaeShardTiming) => {
      onProgress?.({
        stage: "vae",
        pct: W_T5 + W_PASS1 + W_UPSCALE + W_PASS2 + W_DECODE * ((info.shardIndex + 1) / info.totalShards),
        message: `vae ${info.label}`,
      });
      debug(
        `[ltx]   ${info.label} (${info.shardIndex + 1}/${info.totalShards}): ` +
          `load=${info.loadMs.toFixed(0)}ms run=${info.runMs.toFixed(0)}ms`,
      );
    },
  });
  debug(`[ltx] VAE decode OK in ${(performance.now() - tDec).toFixed(0)} ms`);
  statsF16("vae pixels (raw fp16)", decResult.pixels, debug);

  if (vaeDumps) {
    const epTag = providers("vae")[0];
    const tensors: Record<string, { shape: number[]; data: Uint16Array }> = {};
    for (const [label, { data, dims }] of vaeDumps) {
      tensors[label] = { shape: dims, data };
    }
    downloadBlock26InputBundle(
      `ltx_vae_decode_${epTag}.bin`,
      tensors,
      "LTXVAE13",
    );
    debug(`[ltx] LTX_DUMP_VAE_SHARDS: bundle downloaded (${vaeDumps.size} shards)`);
  }

  signal?.throwIfAborted();
  {
    let pmin = Infinity;
    let pmax = -Infinity;
    let pmean = 0;
    for (let i = 0; i < decResult.pixels.length; i++) {
      const v = f16BitsToF32(decResult.pixels[i]);
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
      pmean += v;
    }
    pmean /= decResult.pixels.length;
    debug(
      `[ltx] pixels: min=${pmin.toFixed(3)} max=${pmax.toFixed(3)} ` +
        `mean=${pmean.toFixed(3)}`,
    );
  }
  const frames = await framesToBitmaps(
    decResult.pixels,
    decResult.pixelFrames,
    decResult.pixelHeight,
    decResult.pixelWidth,
    debug,
  );
  onProgress?.({ stage: "vae", pct: 1, message: "done" });

  return { frames, fps: DEFAULT_FPS, seed: SEED };
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** In-place: latent[i] = (1-s)*latent[i] + s*N(0,1). NCDHW or any layout
 *  (operates element-wise). No-op when scale<=0. */
function noiseMixLatentF16(
  latent: Uint16Array,
  scale: number,
  rand01: () => number,
): void {
  if (scale <= 0) return;
  const oneMinus = 1 - scale;
  for (let i = 0; i < latent.length; i += 2) {
    const u1 = Math.max(rand01(), 1e-7);
    const u2 = rand01();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    const n0 = r * Math.cos(theta);
    const n1 = r * Math.sin(theta);
    latent[i] = f32ToF16Bits(f16BitsToF32(latent[i]) * oneMinus + n0 * scale);
    if (i + 1 < latent.length) {
      latent[i + 1] = f32ToF16Bits(f16BitsToF32(latent[i + 1]) * oneMinus + n1 * scale);
    }
  }
}

function initNoiseHidden(shape: LtxLatentShape, rand01: () => number): Float32Array {
  const n = shape.t * shape.h * shape.w * LTX_TX_PATCH_CHANNELS;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rand01(), 1e-7);
    const u2 = rand01();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    out[i] = r * Math.cos(theta);
    if (i + 1 < n) out[i + 1] = r * Math.sin(theta);
  }
  return out;
}

interface DenoisePassArgs {
  hiddenF32: Float32Array;
  shape: LtxLatentShape;
  /** Strict-decreasing sigma list, padded with terminal 0. dt at step i
   *  is `sigmas[i] - sigmas[i+1]`. */
  sigmas: number[];
  tx: LtxTransformer;
  t5Result: { hiddenStates: Uint16Array; seqLen: number };
  encMask: BigInt64Array;
  signal?: AbortSignal;
  debug: (msg: string) => void;
  label: string;
  onStep?: (frac: number) => void;
}

/** RF sampler loop. Mutates `hiddenF32` in place. */
async function runDenoisePass(args: DenoisePassArgs): Promise<void> {
  const { hiddenF32, shape, sigmas, tx, t5Result, encMask, signal, debug, label, onStep } = args;
  const nTokens = latentNumTokens(shape);
  if (hiddenF32.length !== nTokens * LTX_TX_PATCH_CHANNELS) {
    throw new Error(
      `${label}: hidden length ${hiddenF32.length} != ${nTokens * LTX_TX_PATCH_CHANNELS}`,
    );
  }
  const indicesGrid = buildIndicesGrid(shape, DEFAULT_FPS);
  const numSteps = sigmas.length - 1;
  const hiddenF16 = new Uint16Array(hiddenF32.length);

  for (let step = 0; step < numSteps; step++) {
    signal?.throwIfAborted();
    const sigma = sigmas[step];
    const sigmaNext = sigmas[step + 1];
    const dt = sigma - sigmaNext;

    statsF32(`${label} step ${step + 1}/${numSteps} pre-tx hidden`, hiddenF32, debug);
    for (let i = 0; i < hiddenF32.length; i++) hiddenF16[i] = f32ToF16Bits(hiddenF32[i]);

    const tStep = performance.now();
    const dumping = LTX_DUMP_BLOCK26_INPUTS && label === "pass1" && step === 0;
    const dumpingAll = LTX_DUMP_ALL_BLOCKS && label === "pass1" && step === 0;
    const dumpCaptures = dumping ? new Map<string, Uint16Array>() : undefined;
    const dumpAllCaptures = dumpingAll ? new Map<string, Uint16Array>() : undefined;
    const tap = (LTX_PER_BLOCK_TAP || dumping || dumpingAll)
      ? (lbl: string, data: Uint16Array) => {
          if (LTX_PER_BLOCK_TAP) {
            statsF16(`${label} step ${step + 1}/${numSteps} ${lbl}`, data, debug);
          }
          if (dumpCaptures) {
            const aux = lbl.startsWith("shell_pre.") && lbl !== "shell_pre.hidden";
            if (aux) {
              dumpCaptures.set(lbl.slice("shell_pre.".length), new Uint16Array(data));
            } else if (lbl === "block_25") {
              dumpCaptures.set("hidden_states", new Uint16Array(data));
              const NT = nTokens;
              const NX = t5Result.seqLen;
              const tsModLen = dumpCaptures.get("timestep_mod")!.length;
              const ebLen = dumpCaptures.get("encoder_attn_bias")!.length;
              const epTag = providers()[0];
              downloadBlock26InputBundle(`block_26_pass1_step1_inputs_${epTag}.bin`, {
                hidden_states: { shape: [1, NT, 2048], data: dumpCaptures.get("hidden_states")! },
                freqs_cos: { shape: [1, NT, 2048], data: dumpCaptures.get("freqs_cos")! },
                freqs_sin: { shape: [1, NT, 2048], data: dumpCaptures.get("freqs_sin")! },
                timestep_mod: { shape: [1, 1, tsModLen], data: dumpCaptures.get("timestep_mod")! },
                encoder_proj: { shape: [1, NX, 2048], data: dumpCaptures.get("encoder_proj")! },
                encoder_attn_bias: { shape: [1, 1, ebLen], data: dumpCaptures.get("encoder_attn_bias")! },
              });
              debug(`[ltx] LTX_DUMP_BLOCK26_INPUTS: bundle downloaded, aborting`);
              throw new LtxAbortAfterDumpInputs();
            }
          }
          if (dumpAllCaptures) {
            if (lbl === "shell_pre.hidden") {
              dumpAllCaptures.set("shell_pre_hidden", new Uint16Array(data));
            } else if (/^block_\d+$/.test(lbl)) {
              dumpAllCaptures.set(lbl, new Uint16Array(data));
              const idx = parseInt(lbl.slice("block_".length), 10);
              if (idx === LTX_NUM_BLOCKS - 1) {
                const NT = nTokens;
                const epTag = providers()[0];
                const tensors: Record<string, { shape: number[]; data: Uint16Array }> = {
                  shell_pre_hidden: { shape: [1, NT, 2048], data: dumpAllCaptures.get("shell_pre_hidden")! },
                };
                for (let i = 0; i < LTX_NUM_BLOCKS; i++) {
                  const k = `block_${String(i).padStart(2, "0")}`;
                  tensors[k] = { shape: [1, NT, 2048], data: dumpAllCaptures.get(k)! };
                }
                downloadBlock26InputBundle(
                  `ltx_all_blocks_pass1_step1_${epTag}.bin`,
                  tensors,
                  "LTXALL28",
                );
                debug(`[ltx] LTX_DUMP_ALL_BLOCKS: bundle downloaded, aborting`);
                throw new LtxAbortAfterDumpInputs();
              }
            }
          }
        }
      : undefined;
    const txResult = await tx.runStep(
      {
        hiddenStates: hiddenF16,
        indicesGrid,
        encoderHiddenStates: t5Result.hiddenStates,
        encoderAttentionMask: encMask,
        timestep: new Uint16Array([f32ToF16Bits(sigma)]),
        nTokens,
        nText: t5Result.seqLen,
      },
      (info) => {
        const frac = (step + (info.blockIndex + 1) / info.totalBlocks) / numSteps;
        onStep?.(frac);
      },
      tap,
    );
    const stepMs = performance.now() - tStep;

    // Per-token noisePred sample. If t0/t1/t_hw/tlast collapse to the
    // same value, RoPE positions aren't differentiating tokens.
    const np = txResult.noisePred;
    const C = LTX_TX_PATCH_CHANNELS;
    const sampleTok = (tok: number): string => {
      let s = 0;
      for (let c = 0; c < 8; c++) s += f16BitsToF32(np[tok * C + c]);
      return s.toFixed(4);
    };
    const tHw = Math.min(shape.h * shape.w, nTokens - 1);
    debug(
      `[ltx]   np samples: t0=${sampleTok(0)} t1=${sampleTok(1)} ` +
        `t_hw=${sampleTok(tHw)} tlast=${sampleTok(nTokens - 1)}`,
    );

    statsF16(`${label} step ${step + 1}/${numSteps} noisePred`, txResult.noisePred, debug);

    for (let i = 0; i < hiddenF32.length; i++) {
      hiddenF32[i] = hiddenF32[i] - dt * f16BitsToF32(txResult.noisePred[i]);
    }
    debug(
      `[ltx] ${label} step ${step + 1}/${numSteps} sigma=${sigma.toFixed(4)} ` +
        `dt=${dt.toFixed(4)} in ${stepMs.toFixed(0)} ms`,
    );
    statsF32(`${label} step ${step + 1}/${numSteps} post-step hidden`, hiddenF32, debug);

    if (LTX_ABORT_AFTER_PASS1_STEP1 && label === "pass1" && step === 0) {
      debug(`[ltx] LTX_ABORT_AFTER_PASS1_STEP1: aborting after pass1 step 1`);
      throw new LtxAbortAfterStep1();
    }
  }
}

/** Patch-order fp32 hidden -> NCDHW fp16 latent (un-patchified). */
function patchHiddenToLatent(hidden: Float32Array, shape: LtxLatentShape): Uint16Array {
  const f16 = new Uint16Array(hidden.length);
  for (let i = 0; i < hidden.length; i++) f16[i] = f32ToF16Bits(hidden[i]);
  return unpatchifyF16(f16, shape);
}

/** NCDHW fp16 latent -> patch-order fp32 hidden. */
function patchifyHiddenFromLatentF16(latent: Uint16Array, shape: LtxLatentShape): Float32Array {
  const patched = patchifyF16(latent, shape);
  const out = new Float32Array(patched.length);
  for (let i = 0; i < patched.length; i++) out[i] = f16BitsToF32(patched[i]);
  return out;
}

/** Adaptive instance norm per channel: rescale `dst` so each channel's
 *  mean/std (over T*H*W) matches `ref`'s. NCDHW layout. Mutates dst. */
function adainPerChannel(
  dst: Uint16Array,
  dstSpatial: number,
  ref: Uint16Array,
  refSpatial: number,
): void {
  for (let c = 0; c < LTX_LATENT_CHANNELS; c++) {
    let rMean = 0;
    const rBase = c * refSpatial;
    for (let i = 0; i < refSpatial; i++) rMean += f16BitsToF32(ref[rBase + i]);
    rMean /= refSpatial;
    let rVar = 0;
    for (let i = 0; i < refSpatial; i++) {
      const d = f16BitsToF32(ref[rBase + i]) - rMean;
      rVar += d * d;
    }
    const rStd = Math.sqrt(rVar / refSpatial);

    let dMean = 0;
    const dBase = c * dstSpatial;
    for (let i = 0; i < dstSpatial; i++) dMean += f16BitsToF32(dst[dBase + i]);
    dMean /= dstSpatial;
    let dVar = 0;
    for (let i = 0; i < dstSpatial; i++) {
      const d = f16BitsToF32(dst[dBase + i]) - dMean;
      dVar += d * d;
    }
    const dStd = Math.sqrt(dVar / dstSpatial) || 1e-6;

    const scale = rStd / dStd;
    for (let i = 0; i < dstSpatial; i++) {
      const x = f16BitsToF32(dst[dBase + i]);
      dst[dBase + i] = f32ToF16Bits((x - dMean) * scale + rMean);
    }
  }
}

/** NCDHW [1, 3, T, H, W] fp16 in [-1, 1] -> ImageBitmap[]. */
async function framesToBitmaps(
  pixels: Uint16Array,
  T: number,
  H: number,
  W: number,
  debug?: (m: string) => void,
): Promise<ImageBitmap[]> {
  const plane = H * W;
  const out: ImageBitmap[] = [];
  let totalHash = 0 >>> 0;
  for (let f = 0; f < T; f++) {
    const rgba = new Uint8ClampedArray(plane * 4);
    const rBase = 0 * T * plane + f * plane;
    const gBase = 1 * T * plane + f * plane;
    const bBase = 2 * T * plane + f * plane;
    let rSum = 0, gSum = 0, bSum = 0, frameHash = 0 >>> 0;
    for (let i = 0; i < plane; i++) {
      const r = (f16BitsToF32(pixels[rBase + i]) + 1) * 0.5;
      const g = (f16BitsToF32(pixels[gBase + i]) + 1) * 0.5;
      const b = (f16BitsToF32(pixels[bBase + i]) + 1) * 0.5;
      const ri = Math.max(0, Math.min(255, Math.round(r * 255)));
      const gi = Math.max(0, Math.min(255, Math.round(g * 255)));
      const bi = Math.max(0, Math.min(255, Math.round(b * 255)));
      rgba[i * 4 + 0] = ri;
      rgba[i * 4 + 1] = gi;
      rgba[i * 4 + 2] = bi;
      rgba[i * 4 + 3] = 255;
      rSum += ri; gSum += gi; bSum += bi;
      // FNV-1a-ish 32-bit
      frameHash = (frameHash ^ ri) >>> 0; frameHash = Math.imul(frameHash, 16777619) >>> 0;
      frameHash = (frameHash ^ gi) >>> 0; frameHash = Math.imul(frameHash, 16777619) >>> 0;
      frameHash = (frameHash ^ bi) >>> 0; frameHash = Math.imul(frameHash, 16777619) >>> 0;
    }
    if (debug && VERBOSE_DEBUG) {
      const rMean = rSum / plane, gMean = gSum / plane, bMean = bSum / plane;
      debug(
        `[v] frame ${f}/${T} rgbMean=(${rMean.toFixed(1)}, ${gMean.toFixed(1)}, ${bMean.toFixed(1)}) ` +
          `hash=0x${frameHash.toString(16).padStart(8, "0")}`,
      );
    }
    totalHash = (totalHash ^ frameHash) >>> 0;
    totalHash = Math.imul(totalHash, 16777619) >>> 0;
    out.push(await createImageBitmap(new ImageData(rgba, W, H)));
  }
  if (debug && VERBOSE_DEBUG) {
    debug(`[v] all-frames hash=0x${totalHash.toString(16).padStart(8, "0")} (T=${T} H=${H} W=${W})`);
  }
  return out;
}
