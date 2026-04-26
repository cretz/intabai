// FastWan 2.2 TI2V-5B end-to-end generation orchestrator.
//
// Runs the full pipeline for one 5s 480×480 clip:
//   1. Tokenize prompt (UMT5, pad/truncate to 512).
//   2. JS-side int8 embedding lookup -> fp16 [1, 512, 4096].
//   3. UMT5 text encoder: 24 q4f16 layers + shell_post, one session at a
//      time, output [1, 512, 4096] fp16.
//   4. Init latent noise [1, 48, 21, 30, 30] fp32 (seeded Gaussian).
//   5. 3 DMD denoising steps with direct x0 prediction + re-noise:
//      shell_pre -> 30 block sessions -> shell_post -> x0 + add_noise.
//      Timesteps [1000, 757, 522] are hardcoded training hyperparameters
//      of the DMD distillation; see FastVideo/fastvideo/configs/pipelines
//      /wan.py:130-132 (FastWan2_2_TI2V_5B_Config.dmd_denoising_steps).
//      Transformer works in fp16 on the wire; sampling math in fp32.
//   6. Per-channel denormalize the latent: raw[c] = norm[c]*std[c]+mean[c].
//      Required for lighttae weights (lightx2v pipeline.py:266-267).
//   7. Transpose latent axis order for VAE: [1, 48, 21, 30, 30] (NCTHW)
//      -> [1, 21, 48, 30, 30] (NTCHW).
//   8. LightTAE VAE decode -> [1, 81, 3, 480, 480] fp16, range [0, 1].
//   9. Convert each frame to an ImageBitmap for display.
//
// Progress reporting is bucketed by stage with approximate per-step pct:
// the block loop dominates wall time (~95%), so we weight it accordingly.

import type { ModelCache } from "../shared/model-cache";
import { gaussianNoise, mulberry32 } from "../image-gen/generate-utils";
import { f16ToF32Array, f32ToF16Array, f16BitsToF32 } from "../sd15/fp16";

import { TokenEmbedding } from "./embedding";
import { loadTokenizer, tokenize } from "./tokenizer";
import { TextEncoder } from "./text-encoder";
import {
  Transformer,
  FASTWAN_LATENT_CHANNELS,
  FASTWAN_LATENT_FRAMES,
  fastwanShape,
  type FastwanResolution,
  type FastwanShape,
} from "./transformer";
import { VaeDecoder, LIGHTTAE_OUT_FRAMES } from "./vae";
import {
  FASTWAN_EMBEDDING_Q8_FILE,
  FASTWAN_EMBEDDING_SCALES_FILE,
  FASTWAN_FLOW_SHIFT,
  FASTWAN_NUM_STEPS,
  fastwanTextEncoderLayers,
  fastwanTextEncoderShellPost,
  fastwanTransformerFiles,
  fastwanVaeFile,
  type FastwanTransformerPrecision,
} from "./models";
import { UniPCFlowScheduler } from "./unipc-scheduler";

export const FASTWAN_OUTPUT_FPS = 16;

/** Per-channel latent mean/std for Wan 2.2 (48 channels). Identical to
 *  `notes/models/fastwan/source/vae/config.json` and to the hardcoded
 *  arrays in lightx2v `vae_tiny.py:97-191` inside `Wan2_2_VAE_tiny`.
 *  The transformer operates in normalized latent space; the LightTAE
 *  decoder expects raw (unnormalized) latents. lightx2v's pipeline sets
 *  `need_scaled=True` whenever the weights file name contains "lighttae"
 *  (see `pipeline.py:266-267`), which is our case (`lighttaew2_2.safetensors`).
 *  Formula (from vae_tiny.py:199-202, after collapsing a double-inverse):
 *      raw[c, t, h, w] = normalized[c, t, h, w] * std[c] + mean[c]. */
const VAE_LATENTS_MEAN: readonly number[] = [
  -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
  -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
  -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
  -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
  -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
  0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
];
const VAE_LATENTS_STD: readonly number[] = [
  0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
  0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
  0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
  0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
  0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
  0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

/** Apply per-channel denormalization on a [C, T, H, W] fp32 latent.
 *  Input layout is C-outermost (transformer output, pre-transpose).
 *  Returns a new Float32Array so the caller's latent state isn't mutated
 *  (important when the same latent is used for both preview and the next
 *  denoising step in future code paths). */
function denormalizeLatent(
  src: Float32Array,
  C: number,
  T: number,
  H: number,
  W: number,
): Float32Array {
  const plane = T * H * W;
  if (src.length !== C * plane) {
    throw new Error(`denormalizeLatent: expected ${C * plane}, got ${src.length}`);
  }
  if (VAE_LATENTS_MEAN.length !== C || VAE_LATENTS_STD.length !== C) {
    throw new Error(`denormalizeLatent: constants must have ${C} entries`);
  }
  const out = new Float32Array(src.length);
  for (let c = 0; c < C; c++) {
    const mean = VAE_LATENTS_MEAN[c];
    const std = VAE_LATENTS_STD[c];
    const base = c * plane;
    for (let i = 0; i < plane; i++) {
      out[base + i] = src[base + i] * std + mean;
    }
  }
  return out;
}

export interface ProgressInfo {
  /** 0..1 overall progress. */
  pct: number;
  stage: "tokenize" | "embed" | "text-encoder" | "denoise" | "vae" | "done";
  message: string;
}

export interface GenerateOptions {
  cache: ModelCache;
  prompt: string;
  /** 32-bit unsigned seed. Random if undefined. */
  seed?: number;
  /** Transformer precision. `q4f16` (default) runs on mobile and desktop,
   *  ~2.8 GB download. `fp16` is desktop-only, ~9.4 GB, ~25% faster per
   *  forward pass because ORT-web's MatMulNBits kernel dequants on every
   *  call. Text encoder + VAE precision is fixed. */
  transformerPrecision?: FastwanTransformerPrecision;
  /** Text encoder precision. Default `q4f16` (shipping path, ~47% numeric
   *  drift vs PyTorch but kept for compute parity with the transformer
   *  blocks). `fp16` is opt-in via URL param for A/B testing - matches
   *  PyTorch to cosine=1.0 but costs 9.2 GB download and empirically
   *  slows subsequent transformer block compute ~3x on some GPUs. */
  textEncoderPrecision?: FastwanTransformerPrecision;
  /** Output resolution. The transformer + VAE assets are exported per
   *  resolution; the selected video-gen model entry pins this. */
  resolution: FastwanResolution;
  onProgress?: (info: ProgressInfo) => void;
  /** Verbose per-stage trace (timings, shapes, step-level events). Opt-in
   *  because the scheduler / block loop produce a lot of lines. */
  onDebug?: (msg: string) => void;
  signal?: AbortSignal;
  /** Optional preview hook. When set, the in-flight latent is VAE-decoded
   *  after each non-final denoising step and the resulting frames are
   *  delivered here. Adds ~2-3 s per step to the run, huge UX win on
   *  13-minute generations. `stepIndex` is 0-based. */
  onPreview?: (frames: ImageBitmap[], stepIndex: number) => void;
}

export interface GenerateResult {
  frames: ImageBitmap[];
  fps: number;
  seed: number;
}

/** Stage progress weights chosen so wall time roughly matches: block loop
 *  is ~95% of runtime, text encoder ~4%, everything else negligible. */
const W_TOKENIZE = 0.002;
const W_EMBED = 0.003;
const W_TEXT_ENC = 0.04;
const W_DENOISE = 0.94;
const W_VAE = 0.01;
const W_FRAMES = 0.005;

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException("aborted", "AbortError");
}

export async function generateFastwan(opts: GenerateOptions): Promise<GenerateResult> {
  const { cache, prompt, signal, onProgress, onDebug } = opts;
  const seed = opts.seed ?? Math.floor(Math.random() * 0xffffffff);
  const transformerPrecision: FastwanTransformerPrecision =
    opts.transformerPrecision ?? "q4f16";
  const textEncoderPrecision: FastwanTransformerPrecision =
    opts.textEncoderPrecision ?? "q4f16";
  const shape: FastwanShape = fastwanShape(opts.resolution);
  const txFiles = fastwanTransformerFiles(transformerPrecision, opts.resolution);
  const t0 = performance.now();
  const log = (msg: string) => {
    if (!onDebug) return;
    const ms = (performance.now() - t0).toFixed(0).padStart(6, " ");
    onDebug(`[+${ms}ms] ${msg}`);
  };
  log(`seed=${seed} prompt=${JSON.stringify(prompt.slice(0, 80))}${prompt.length > 80 ? "..." : ""}`);

  // Sanity stats for intermediate tensors - surfaces NaN/range issues
  // without needing to run the full 10-minute pipeline blind. Only
  // computed when the debug log is enabled.
  const statsFp32 = (name: string, arr: Float32Array): void => {
    if (!onDebug) return;
    let mn = Infinity, mx = -Infinity, sum = 0, nan = 0, zero = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isNaN(v)) { nan++; continue; }
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
      if (v === 0) zero++;
    }
    const mean = sum / Math.max(1, arr.length - nan);
    log(
      `stats[${name}] n=${arr.length} min=${mn.toFixed(4)} max=${mx.toFixed(4)} ` +
        `mean=${mean.toFixed(4)} nan=${nan} zeros=${zero}`,
    );
  };
  const statsFp16Bits = (name: string, bits: Uint16Array): void => {
    if (!onDebug) return;
    statsFp32(name, f16ToF32Array(bits));
  };

  const progress = (pct: number, stage: ProgressInfo["stage"], msg: string) =>
    onProgress?.({ pct: Math.min(1, Math.max(0, pct)), stage, message: msg });

  // ---- 1. Tokenize ---------------------------------------------------------
  progress(0, "tokenize", "tokenizing prompt");
  const tokenizer = await loadTokenizer(cache);
  throwIfAborted(signal);
  const { ids, validLength } = tokenize(tokenizer, prompt);
  log(`tokenized: ${validLength} real tokens (padded to ${ids.length})`);
  {
    const validIds = Array.from(ids.subarray(0, validLength));
    log(`input_ids[valid]=${JSON.stringify(validIds)}`);
    const tailStart = Math.max(validLength, ids.length - 4);
    const tailIds = Array.from(ids.subarray(tailStart, ids.length));
    log(`input_ids[pad_tail ${tailStart}..${ids.length}]=${JSON.stringify(tailIds)}`);
  }

  // ---- 2. Embedding lookup -------------------------------------------------
  progress(W_TOKENIZE, "embed", "loading embedding table");
  const [q8Buf, scalesBuf] = await Promise.all([
    cache.loadFile(FASTWAN_EMBEDDING_Q8_FILE),
    cache.loadFile(FASTWAN_EMBEDDING_SCALES_FILE),
  ]);
  throwIfAborted(signal);
  const embedding = new TokenEmbedding(q8Buf, scalesBuf);
  log(`embedding table loaded: ${(q8Buf.byteLength / 1e9).toFixed(2)} GB int8 + scales`);

  // ---- 3. Text encoder -----------------------------------------------------
  progress(W_TOKENIZE + W_EMBED, "text-encoder", "encoding text (0/24)");
  const textEncoder = new TextEncoder(
    cache,
    {
      layers: fastwanTextEncoderLayers(textEncoderPrecision),
      shellPost: fastwanTextEncoderShellPost(textEncoderPrecision),
    },
    embedding,
  );
  const encResult = await textEncoder.encode(
    ids,
    validLength,
    (done, total) => {
      const frac = done / total;
      progress(
        W_TOKENIZE + W_EMBED + W_TEXT_ENC * frac,
        "text-encoder",
        `encoding text (${done}/${total})`,
      );
      throwIfAborted(signal);
    },
    statsFp16Bits,
  );
  const textEmbeds = encResult.hiddenStates; // fp16 bits, [1, 512, 4096]
  log(`text encoder done in ${(performance.now() - t0).toFixed(0)} ms`);
  statsFp16Bits("text_embeds[all]", textEmbeds);
  // Also stat just the real-token slice; the padded region should differ
  // substantially (masked attention means those positions carry less
  // information from the prompt).
  statsFp16Bits(
    `text_embeds[valid:0..${validLength}]`,
    textEmbeds.subarray(0, validLength * 4096),
  );

  // ---- 4. Noise init -------------------------------------------------------
  // Latent [1, 48, 21, latentH, latentW].
  const latentLen =
    FASTWAN_LATENT_CHANNELS *
    FASTWAN_LATENT_FRAMES *
    shape.latentH *
    shape.latentW;
  const rand = mulberry32(seed);
  let latentFp32 = gaussianNoise(latentLen, rand);
  statsFp32("noise_init", latentFp32);

  // ---- 5. Denoising loop ---------------------------------------------------
  const txBase = W_TOKENIZE + W_EMBED + W_TEXT_ENC;
  const transformer = new Transformer(
    cache,
    {
      shellPre: txFiles.shellPre,
      blocks: txFiles.blocks,
      shellPost: txFiles.shellPost,
    },
    shape,
  );
  log(`transformer precision: ${transformerPrecision}`);
  progress(txBase, "denoise", "loading transformer shells");
  await transformer.load();
  throwIfAborted(signal);

  // Pure Gaussian noise is already at sigma=1 (flow_sigma(1000)=1.0), so no
  // rescaling of latentFp32 is needed for DMD's t=1000 first step.

  // VAE is used for optional per-step previews and the final decode. Load
  // once and reuse across all decodes to avoid the ~250 ms load cost
  // repeated 3x.
  const vae = new VaeDecoder(cache, fastwanVaeFile(opts.resolution), shape);
  await vae.load();

  const decodeLatentToBitmaps = async (src: Float32Array): Promise<ImageBitmap[]> => {
    const denormalized = denormalizeLatent(
      src,
      FASTWAN_LATENT_CHANNELS,
      FASTWAN_LATENT_FRAMES,
      shape.latentH,
      shape.latentW,
    );
    const transposedPre = transposeCT(
      denormalized,
      FASTWAN_LATENT_CHANNELS,
      FASTWAN_LATENT_FRAMES,
      shape.latentH,
      shape.latentW,
    );
    const inputFp16 = f32ToF16Array(transposedPre);
    const framesBits = await vae.decode(inputFp16);
    return framesToBitmaps(
      framesBits,
      LIGHTTAE_OUT_FRAMES,
      shape.pixelH,
      shape.pixelW,
      () => {},
    );
  };

  // UniPCMultistepScheduler with flow sigmas + predict_x0 + flow_prediction
  // + bh2. Matches the HF Space (KingNish/wan2-2-fast) exactly: 4 steps,
  // flow_shift=8, solver_order=2. See unipc-scheduler.ts.
  const scheduler = new UniPCFlowScheduler({
    numInferenceSteps: FASTWAN_NUM_STEPS,
    flowShift: FASTWAN_FLOW_SHIFT,
    solverOrder: 2,
    lowerOrderFinal: true,
  });
  log(
    `scheduler=unipc_flow_bh2 num_steps=${FASTWAN_NUM_STEPS} ` +
      `flow_shift=${FASTWAN_FLOW_SHIFT} ` +
      `sigmas=[${Array.from(scheduler.sigmas).map((s) => s.toFixed(4)).join(",")}]`,
  );

  try {
    for (let step = 0; step < FASTWAN_NUM_STEPS; step++) {
      throwIfAborted(signal);
      const sigmaCur = scheduler.sigmas[step];
      // Transformer's shell_pre takes integer timestep in [0, 1000]. With
      // flow sigmas sigma == t/1000, so timestep = round(sigma*1000).
      const timestep = Math.round(sigmaCur * 1000);
      const isFinal = step + 1 === FASTWAN_NUM_STEPS;
      const stepStart = performance.now();
      log(
        `step ${step + 1}/${FASTWAN_NUM_STEPS} starting, timestep=${timestep} ` +
          `sigma=${sigmaCur.toFixed(4)}`,
      );
      const latentFp16 = f32ToF16Array(latentFp32);

      const noisePredFp16 = await transformer.forward(
        {
          latent: latentFp16,
          timestep,
          textEmbeds,
          stepIndex: step,
          onDebug,
          onStatsFp16: statsFp16Bits,
        },
        (sIdx, bDone, bTotal) => {
          const withinStep = bDone / bTotal;
          const frac = (sIdx + withinStep) / FASTWAN_NUM_STEPS;
          progress(
            txBase + W_DENOISE * frac,
            "denoise",
            `step ${sIdx + 1}/${FASTWAN_NUM_STEPS}, block ${bDone}/${bTotal}`,
          );
          throwIfAborted(signal);
        },
      );

      const noisePredFp32 = f16ToF32Array(noisePredFp16);
      statsFp32(`noise_pred[step${step + 1}]`, noisePredFp32);

      // Direct x0 estimate for the preview path (UniPC's corrector mutates
      // sample internally; we compute x0 explicitly here so previews decode
      // the current clean-image estimate rather than the partially-denoised
      // latent). flow_prediction: x0 = x_t - sigma_t * v.
      const x0 = new Float32Array(latentFp32.length);
      for (let i = 0; i < x0.length; i++) {
        x0[i] = latentFp32[i] - sigmaCur * noisePredFp32[i];
      }
      statsFp32(`x0[step${step + 1}]`, x0);

      latentFp32 = scheduler.step(noisePredFp32, latentFp32);
      statsFp32(`latent[after step${step + 1}]`, latentFp32);
      log(`step ${step + 1} done in ${(performance.now() - stepStart).toFixed(0)} ms`);

      // Preview between steps. Skip on the final step — the end-of-run path
      // produces the full result and we don't want to double-decode.
      // Decode x0 (the clean-image estimate), not the re-noised latent. In
      // DMD the between-step latent has been intentionally re-noised to the
      // next timestep's sigma (~0.83 after step 2, etc.), so decoding it
      // shows mostly Gaussian speckle. x0 is the current best clean-latent
      // estimate and is what the user actually wants to preview.
      if (opts.onPreview && !isFinal) {
        const previewStart = performance.now();
        try {
          const previewFrames = await decodeLatentToBitmaps(x0);
          log(
            `preview decode after step ${step + 1} in ` +
              `${(performance.now() - previewStart).toFixed(0)} ms`,
          );
          opts.onPreview(previewFrames, step);
        } catch (err) {
          // Preview is best-effort; never fail the run because of it.
          log(`preview after step ${step + 1} failed: ${(err as Error).message}`);
        }
        throwIfAborted(signal);
      }
    }
  } finally {
    await transformer.release();
  }

  // ---- 6. Final VAE decode -----------------------------------------------
  progress(txBase + W_DENOISE, "vae", "decoding frames");
  let framesFp16Bits: Uint16Array;
  const vaeStart = performance.now();
  try {
    statsFp32("vae_in_latent (normalized, pre-denorm)", latentFp32);
    const denormalized = denormalizeLatent(
      latentFp32,
      FASTWAN_LATENT_CHANNELS,
      FASTWAN_LATENT_FRAMES,
      shape.latentH,
      shape.latentW,
    );
    statsFp32("vae_in_latent (denormalized, post x*std+mean)", denormalized);
    // LightTAE: takes [1, T, C, H, W] fp16, returns all 81 frames at once.
    const transposed = transposeCT(
      denormalized,
      FASTWAN_LATENT_CHANNELS,
      FASTWAN_LATENT_FRAMES,
      shape.latentH,
      shape.latentW,
    );
    framesFp16Bits = await vae.decode(f32ToF16Array(transposed));
  } finally {
    await vae.release();
  }
  log(`VAE decode in ${(performance.now() - vaeStart).toFixed(0)} ms`);
  statsFp16Bits("vae_out_frames", framesFp16Bits);
  throwIfAborted(signal);

  // ---- 8. Frames -> ImageBitmap[] -----------------------------------------
  progress(txBase + W_DENOISE + W_VAE, "vae", "rendering frames");
  const frames = await framesToBitmaps(
    framesFp16Bits,
    LIGHTTAE_OUT_FRAMES,
    shape.pixelH,
    shape.pixelW,
    (done) => {
      const frac = done / LIGHTTAE_OUT_FRAMES;
      progress(
        txBase + W_DENOISE + W_VAE + W_FRAMES * frac,
        "vae",
        `framing ${done}/${LIGHTTAE_OUT_FRAMES}`,
      );
    },
  );

  progress(1, "done", "done");
  log(`total ${(performance.now() - t0).toFixed(0)} ms, ${frames.length} frames`);
  return { frames, fps: FASTWAN_OUTPUT_FPS, seed };
}

/** Permute axes 1<->2 of a [1, C, T, H, W] fp32 tensor to [1, T, C, H, W].
 *  Inner H*W plane is copied contiguously per (c,t) pair. */
function transposeCT(
  src: Float32Array,
  C: number,
  T: number,
  H: number,
  W: number,
): Float32Array {
  const plane = H * W;
  const out = new Float32Array(src.length);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c++) {
      const srcOff = (c * T + t) * plane;
      const dstOff = (t * C + c) * plane;
      out.set(src.subarray(srcOff, srcOff + plane), dstOff);
    }
  }
  return out;
}

/** Convert [1, F, 3, H, W] fp16 bits in [0, 1] to a list of ImageBitmaps.
 *  Our ONNX export is of the raw TAEHV decoder, which outputs [0, 1];
 *  the Wan22TinyVAE wrapper in lightx2v (`vae_tiny.py:72`) adds a
 *  `.mul_(2).sub_(1)` to convert to [-1, 1] for callers, but our export
 *  does not include that step. Hence direct multiply by 255 here. */
async function framesToBitmaps(
  bits: Uint16Array,
  F: number,
  H: number,
  W: number,
  onFrame: (done: number) => void,
): Promise<ImageBitmap[]> {
  const plane = H * W;
  const frameStride = 3 * plane;
  const out: ImageBitmap[] = [];
  for (let f = 0; f < F; f++) {
    const base = f * frameStride;
    const rgba = new Uint8ClampedArray(plane * 4);
    for (let i = 0; i < plane; i++) {
      const r = f16BitsToF32(bits[base + i]);
      const g = f16BitsToF32(bits[base + plane + i]);
      const b = f16BitsToF32(bits[base + 2 * plane + i]);
      rgba[i * 4 + 0] = Math.max(0, Math.min(255, Math.round(r * 255)));
      rgba[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      rgba[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
      rgba[i * 4 + 3] = 255;
    }
    const bitmap = await createImageBitmap(new ImageData(rgba, W, H));
    out.push(bitmap);
    onFrame(f + 1);
  }
  return out;
}
