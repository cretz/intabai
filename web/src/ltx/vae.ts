// LTX 2B distilled VAE, run as one shard at a time so peak GPU footprint
// is one block (~450 MB worst-case for enc_block_08_res_*) instead of the
// full 2.4 GB graph.
//
// Layout matches the per-shard split produced by convert_vae_layers.py:
//   encoder = enc_shell_pre + 9 down_blocks (block_08 split per-resnet)
//             + enc_shell_post
//   decoder = dec_shell_pre + 7 up_blocks (block_00 split as pre + 5
//             per-resnet) + dec_shell_post
//
// Decoder is timestep-conditioned (yaml default decode_timestep = 0.05).
// dec_shell_pre returns scaled_timestep (= timestep * timestep_scale_multiplier,
// fp32) which is threaded through every UNetMidBlock3D up_block and into
// dec_shell_post. dec_block_00_pre runs that block's time_embedder and
// emits ts_embed (fp16, shape [1, in_channels*4, 1, 1, 1]) consumed by
// the per-resnet sub-files.
//
// Plain (non-mid) decoder up_blocks don't take scaled_timestep; we detect
// that by inspecting session.inputNames so the caller doesn't need to know
// which block index is which kind.
//
// io-binding intentionally skipped while the pipeline is being validated.
// Per-shard load/run/release timings are reported via the BlockProgress
// callback so the debug log shows where time is going across the 26 shards.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits, f16BitsToF32, f32ToF16Bits } from "../sd15/fp16";
import {
  LTX_LATENT_CHANNELS,
  LTX_VAE_SPATIAL_DOWNSCALE,
  LTX_VAE_TEMPORAL_DOWNSCALE,
  type LtxLatentShape,
} from "./patchifier";
import { assertF16View, assertF32View, assertOrtOutput } from "./validate";

/** moments output channel count from enc_shell_post (mean 128 + logvar 128). */
const ENC_MOMENTS_CHANNELS = 256;

/** Logical stem of a VAE shard, with the `_pixnorm_fp32` suffix stripped so
 *  dispatch logic (e.g. `stem === "dec_block_00_pre"`) and tap labels keep
 *  matching whether the patched or unpatched graph is loaded. */
function vaeStem(file: OrtModelFile): string {
  const raw = ("graph" in file ? file.graph.name : file.name)
    .replace(/^vae\//, "")
    .replace(/\.onnx$/, "");
  return raw.endsWith("_pixnorm_fp32") ? raw.slice(0, -"_pixnorm_fp32".length) : raw;
}

export interface LtxVaeFiles {
  encShellPre: OrtModelFile;
  encBlocks: OrtModelFile[];
  encShellPost: OrtModelFile;
  decShellPre: OrtModelFile;
  decBlocks: OrtModelFile[];
  decShellPost: OrtModelFile;
}

export interface VaeShardTiming {
  /** "enc:shell_pre", "enc:block_03", "dec:block_00_res_2", ... */
  label: string;
  shardIndex: number;
  totalShards: number;
  loadMs: number;
  runMs: number;
  releaseMs: number;
}

export type VaeShardProgress = (info: VaeShardTiming) => void;

export interface LtxVaeEncodeArgs {
  /** [1, 3, T, H, W] fp16 pixels in [-1, 1]. */
  pixels: Uint16Array;
  pixelFrames: number;
  pixelHeight: number;
  pixelWidth: number;
  /** Sample latent? If false, returns mean (deterministic). */
  sample?: boolean;
  rand01?: () => number;
  signal?: AbortSignal;
  onShard?: VaeShardProgress;
}

export interface LtxVaeEncodeResult {
  /** [1, 128, T, H, W] fp16. */
  latent: Uint16Array;
  shape: LtxLatentShape;
}

export interface LtxVaeDecodeArgs {
  /** [1, 128, T, H, W] fp16 latent (un-normalized; pipeline-level
   *  un_normalize_latents must already have been applied). */
  latent: Uint16Array;
  shape: LtxLatentShape;
  /** scalar in [0, 1]; pipeline default 0.05. Fed into the decoder
   *  shell_pre as the conditioning `timestep`. */
  decodeTimestep?: number;
  /** Mix in noise before decoding: `latent = latent*(1-s) + noise*s`.
   *  Defaults to `decodeTimestep` (matches pipeline_ltx_video default
   *  where decode_noise_scale falls back to decode_timestep). Set to 0
   *  to disable. */
  decodeNoiseScale?: number;
  rand01?: () => number;
  signal?: AbortSignal;
  onShard?: VaeShardProgress;
  /** Per-block diagnostic logger. Receives `[label, dims, stats]` strings. */
  onDebug?: (msg: string) => void;
  /** Per-shard tap for EP-divergence bisection. Called once per decoder
   *  shard with the fp16 output buffer + dims. Labels:
   *    "latent_in"
   *    "dec_shell_pre.hidden"
   *    "dec_block_00_pre.ts_embed"
   *    "dec_block_00_res_{0..4}.hidden_out"
   *    "dec_block_{01..06}.hidden_out"
   *    "dec_shell_post.pixels"
   *  Caller owns the buffer; do not retain past the callback. */
  tap?: (label: string, data: Uint16Array, dims: number[]) => void;
}

export interface LtxVaeDecodeResult {
  /** [1, 3, T, H, W] fp16 pixels. */
  pixels: Uint16Array;
  pixelFrames: number;
  pixelHeight: number;
  pixelWidth: number;
}

export class LtxVae {
  private readonly totalShards: number;

  constructor(
    private readonly cache: ModelCache,
    private readonly files: LtxVaeFiles,
    private readonly providers: ("webgpu" | "wasm")[] = ["webgpu", "wasm"],
  ) {
    this.totalShards =
      2 + files.encBlocks.length + 2 + files.decBlocks.length;
    console.log("[ltx-diag] VAE providers:", providers);
  }

  // -------------------------------------------------------------------------
  // Encoder
  // -------------------------------------------------------------------------

  async encode(args: LtxVaeEncodeArgs): Promise<LtxVaeEncodeResult> {
    const { pixels, pixelFrames, pixelHeight, pixelWidth, signal, onShard } = args;
    const sample = args.sample ?? true;
    const rand01 = args.rand01 ?? Math.random;

    if (
      (pixelFrames - 1) % LTX_VAE_TEMPORAL_DOWNSCALE !== 0 ||
      pixelHeight % LTX_VAE_SPATIAL_DOWNSCALE !== 0 ||
      pixelWidth % LTX_VAE_SPATIAL_DOWNSCALE !== 0
    ) {
      throw new Error(
        `vae.encode: bad pixel shape ${pixelFrames}x${pixelHeight}x${pixelWidth}`,
      );
    }
    const expected = 3 * pixelFrames * pixelHeight * pixelWidth;
    if (pixels.length !== expected) {
      throw new Error(`vae.encode: expected ${expected} fp16 vals, got ${pixels.length}`);
    }

    let shardIdx = 0;
    let hidden: Uint16Array;
    let dims: number[];

    // shell_pre: pixels -> hidden after patchify+conv_in. Output is
    // (1, 128, T, H/4, W/4); the conv_in stride is set to give us the
    // post-patchify shape directly.
    {
      const t = await this.runShard(
        "enc:shell_pre",
        shardIdx++,
        this.files.encShellPre,
        signal,
        async (sess) => {
          const feed = new ort.Tensor(
            "float16",
            pixels,
            [1, 3, pixelFrames, pixelHeight, pixelWidth],
          );
          const r = await sess.run({ pixels: feed });
          const out = pickOutput(r, ["hidden"]);
          assertOrtOutput("vae.enc_shell_pre.hidden", out, { type: "float16" });
          const outDims = (out.dims as number[]).slice();
          let count = 1;
          for (const d of outDims) count *= d;
          const view = assertF16View("vae.enc_shell_pre.hidden.data", out.data, count);
          const data = copyF16Bits(view);
          feed.dispose?.();
          for (const k in r) (r[k] as ort.Tensor).dispose?.();
          return { data, dims: outDims };
        },
        onShard,
      );
      hidden = t.data;
      dims = t.dims;
    }

    // 9 encoder down-block shards (block_08 is split into 2 res sub-files
    // but the I/O signature is the same: hidden_in -> hidden_out).
    for (let i = 0; i < this.files.encBlocks.length; i++) {
      const file = this.files.encBlocks[i];
      const label = `enc:${vaeStem(file)}`;
      const t = await this.runShard(
        label,
        shardIdx++,
        file,
        signal,
        async (sess) => {
          const feed = new ort.Tensor("float16", hidden, dims);
          const r = await sess.run({ hidden_in: feed });
          const out = pickOutput(r, ["hidden_out"]);
          assertOrtOutput(`${label}.hidden_out`, out, { type: "float16" });
          const outDims = (out.dims as number[]).slice();
          let count = 1;
          for (const d of outDims) count *= d;
          const view = assertF16View(`${label}.hidden_out.data`, out.data, count);
          const data = copyF16Bits(view);
          feed.dispose?.();
          for (const k in r) (r[k] as ort.Tensor).dispose?.();
          return { data, dims: outDims };
        },
        onShard,
      );
      hidden = t.data;
      dims = t.dims;
    }

    // shell_post: hidden -> moments [1, 256, T, H, W].
    let moments: Uint16Array;
    let momentsDims: number[];
    {
      const t = await this.runShard(
        "enc:shell_post",
        shardIdx++,
        this.files.encShellPost,
        signal,
        async (sess) => {
          const feed = new ort.Tensor("float16", hidden, dims);
          const r = await sess.run({ hidden: feed });
          const out = pickOutput(r, ["moments"]);
          assertOrtOutput("vae.enc_shell_post.moments", out, { type: "float16" });
          const outDims = (out.dims as number[]).slice();
          let count = 1;
          for (const d of outDims) count *= d;
          const view = assertF16View("vae.enc_shell_post.moments.data", out.data, count);
          const data = copyF16Bits(view);
          feed.dispose?.();
          for (const k in r) (r[k] as ort.Tensor).dispose?.();
          return { data, dims: outDims };
        },
        onShard,
      );
      moments = t.data;
      momentsDims = t.dims;
    }

    if (momentsDims[1] !== ENC_MOMENTS_CHANNELS) {
      throw new Error(
        `vae.encode: moments has ${momentsDims[1]} channels, expected ${ENC_MOMENTS_CHANNELS}`,
      );
    }

    // JS-side: split into mean (first 128) + logvar (last 128, uniform
    // repeat of one channel) and sample. Memory layout: NCDHW row-major.
    const T = momentsDims[2];
    const H = momentsDims[3];
    const W = momentsDims[4];
    const spatial = T * H * W;
    const latent = new Uint16Array(LTX_LATENT_CHANNELS * spatial);

    if (sample) {
      const noise = boxMullerF16(latent.length, rand01);
      for (let c = 0; c < LTX_LATENT_CHANNELS; c++) {
        const meanBase = c * spatial;
        const logvarBase = (LTX_LATENT_CHANNELS + c) * spatial;
        const dstBase = c * spatial;
        for (let i = 0; i < spatial; i++) {
          const mean = f16BitsToF32(moments[meanBase + i]);
          const logvar = f16BitsToF32(moments[logvarBase + i]);
          const std = Math.exp(0.5 * logvar);
          const eps = f16BitsToF32(noise[dstBase + i]);
          latent[dstBase + i] = f32ToF16Bits(mean + std * eps);
        }
      }
    } else {
      // Deterministic: just the mean half.
      latent.set(moments.subarray(0, LTX_LATENT_CHANNELS * spatial));
    }

    return {
      latent,
      shape: { t: T, h: H, w: W },
    };
  }

  // -------------------------------------------------------------------------
  // Decoder
  // -------------------------------------------------------------------------

  async decode(args: LtxVaeDecodeArgs): Promise<LtxVaeDecodeResult> {
    const { shape, signal, onShard } = args;
    const onDebug = args.onDebug ?? (() => {});

    /** For (1, C, T, H, W) fp16: dump overall mean/std and ch0/frame0
     *  spatial std. If spatialStd << overallStd, the layer collapsed
     *  spatial info (= tiled output). */
    const dumpStats = (label: string, data: Uint16Array, dims: number[]) => {
      const [, C, T, H, W] = dims;
      const N = data.length;
      let mean = 0;
      for (let i = 0; i < N; i++) mean += f16BitsToF32(data[i]);
      mean /= N;
      let v = 0;
      for (let i = 0; i < N; i++) {
        const d = f16BitsToF32(data[i]) - mean;
        v += d * d;
      }
      const std = Math.sqrt(v / N);
      // Channel 0, frame 0 spatial slice.
      const plane = H * W;
      const sliceStart = 0 * T * plane + 0 * plane;
      let sMean = 0;
      for (let i = 0; i < plane; i++) sMean += f16BitsToF32(data[sliceStart + i]);
      sMean /= plane;
      let sV = 0;
      for (let i = 0; i < plane; i++) {
        const d = f16BitsToF32(data[sliceStart + i]) - sMean;
        sV += d * d;
      }
      const sStd = Math.sqrt(sV / plane);
      onDebug(
        `[vae-stats] ${label} dims=[${C},${T},${H},${W}] ` +
          `mean=${mean.toFixed(3)} std=${std.toFixed(3)} ` +
          `ch0f0_spatial_std=${sStd.toFixed(4)}`,
      );
    };

    const decodeTimestep = args.decodeTimestep ?? 0.05;
    const decodeNoiseScale = args.decodeNoiseScale ?? decodeTimestep;
    const rand01 = args.rand01 ?? Math.random;
    const expected = LTX_LATENT_CHANNELS * shape.t * shape.h * shape.w;
    if (args.latent.length !== expected) {
      throw new Error(`vae.decode: expected ${expected} fp16 vals, got ${args.latent.length}`);
    }

    // Mix noise into the latent before the decoder, matching
    // pipeline_ltx_video.py: `latents = latents*(1-s) + noise*s`.
    // s defaults to decode_timestep (0.05). Done in JS so the exported
    // dec_shell_pre stays a pure conv_in; numerics in fp32 then re-quant
    // to fp16 to limit the (1-s)*x rounding error at small s.
    let latent: Uint16Array;
    if (decodeNoiseScale > 0) {
      const noise = boxMullerF16(expected, rand01);
      const oneMinusS = 1 - decodeNoiseScale;
      latent = new Uint16Array(expected);
      for (let i = 0; i < expected; i++) {
        const x = f16BitsToF32(args.latent[i]);
        const n = f16BitsToF32(noise[i]);
        latent[i] = f32ToF16Bits(x * oneMinusS + n * decodeNoiseScale);
      }
    } else {
      latent = args.latent;
    }

    let shardIdx = 0;
    let hidden: Uint16Array;
    let dims: number[];
    let scaledTimestep: Float32Array;
    let scaledTimestepDims: number[];

    // shell_pre: latent + raw timestep -> hidden + scaled_timestep (fp32).
    {
      const tsRaw = new Uint16Array([f32ToF16Bits(decodeTimestep)]);
      const t = await this.runShard(
        "dec:shell_pre",
        shardIdx++,
        this.files.decShellPre,
        signal,
        async (sess) => {
          const latentFeed = new ort.Tensor(
            "float16",
            latent,
            [1, LTX_LATENT_CHANNELS, shape.t, shape.h, shape.w],
          );
          const tsFeed = new ort.Tensor("float16", tsRaw, [1]);
          const r = await sess.run({ latent: latentFeed, timestep: tsFeed });
          const hOut = pickOutput(r, ["hidden"]);
          const tsOut = pickOutput(r, ["scaled_timestep"]);
          assertOrtOutput("vae.dec_shell_pre.hidden", hOut, { type: "float16" });
          assertOrtOutput("vae.dec_shell_pre.scaled_timestep", tsOut, { type: "float32" });
          const hDims = (hOut.dims as number[]).slice();
          const tsDims = (tsOut.dims as number[]).slice();
          let hCount = 1;
          for (const d of hDims) hCount *= d;
          let tsCount = 1;
          for (const d of tsDims) tsCount *= d;
          const hView = assertF16View("vae.dec_shell_pre.hidden.data", hOut.data, hCount);
          const tsSrc = assertF32View("vae.dec_shell_pre.scaled_timestep.data", tsOut.data, tsCount);
          const hData = copyF16Bits(hView);
          const tsData = new Float32Array(tsSrc);
          latentFeed.dispose?.();
          tsFeed.dispose?.();
          for (const k in r) (r[k] as ort.Tensor).dispose?.();
          return { data: hData, dims: hDims, ts: tsData, tsDims };
        },
        onShard,
      );
      hidden = t.data;
      dims = t.dims;
      scaledTimestep = (t as { ts: Float32Array }).ts;
      scaledTimestepDims = (t as { tsDims: number[] }).tsDims;
      dumpStats("dec_shell_pre", hidden, dims);
      args.tap?.("dec_shell_pre.hidden", hidden, dims);
    }
    // Stats on the input latent itself (post-noise-mix) for baseline.
    const latentInDims = [1, LTX_LATENT_CHANNELS, shape.t, shape.h, shape.w];
    dumpStats("latent_in", latent, latentInDims);
    args.tap?.("latent_in", latent, latentInDims);

    // ts_embed produced by dec_block_00_pre and consumed by the 5
    // per-resnet sub-files of dec_block_00. Lifetime is just those 6
    // shards; null otherwise.
    let tsEmbed: Uint16Array | null = null;
    let tsEmbedDims: number[] | null = null;

    for (let i = 0; i < this.files.decBlocks.length; i++) {
      const file = this.files.decBlocks[i];
      const stem = vaeStem(file);
      const label = `dec:${stem}`;
      const isPre = stem === "dec_block_00_pre";
      const isMidRes = stem.startsWith("dec_block_00_res_");

      const t = await this.runShard(
        label,
        shardIdx++,
        file,
        signal,
        async (sess) => {
          const feeds: Record<string, ort.Tensor> = {
            hidden_in: new ort.Tensor("float16", hidden, dims),
          };
          const inputs = sess.inputNames;
          if (inputs.includes("scaled_timestep")) {
            feeds.scaled_timestep = new ort.Tensor(
              "float32",
              scaledTimestep,
              scaledTimestepDims,
            );
          }
          if (inputs.includes("ts_embed")) {
            if (!tsEmbed || !tsEmbedDims) {
              throw new Error(`${label}: ts_embed required but not produced yet`);
            }
            feeds.ts_embed = new ort.Tensor("float16", tsEmbed, tsEmbedDims);
          }

          const r = await sess.run(feeds);
          let outData: Uint16Array;
          let outDims: number[];
          if (isPre) {
            const out = pickOutput(r, ["ts_embed"]);
            assertOrtOutput(`${label}.ts_embed`, out, { type: "float16" });
            outDims = (out.dims as number[]).slice();
            let count = 1;
            for (const d of outDims) count *= d;
            const view = assertF16View(`${label}.ts_embed.data`, out.data, count);
            outData = copyF16Bits(view);
          } else {
            const out = pickOutput(r, ["hidden_out"]);
            assertOrtOutput(`${label}.hidden_out`, out, { type: "float16" });
            outDims = (out.dims as number[]).slice();
            let count = 1;
            for (const d of outDims) count *= d;
            const view = assertF16View(`${label}.hidden_out.data`, out.data, count);
            outData = copyF16Bits(view);
          }
          for (const k in feeds) feeds[k].dispose?.();
          for (const k in r) (r[k] as ort.Tensor).dispose?.();
          return { data: outData, dims: outDims };
        },
        onShard,
      );

      if (isPre) {
        tsEmbed = t.data;
        tsEmbedDims = t.dims;
        args.tap?.(`${stem}.ts_embed`, tsEmbed, tsEmbedDims);
        // hidden / dims unchanged: pre only computes ts_embed.
      } else {
        hidden = t.data;
        dims = t.dims;
        dumpStats(stem, hidden, dims);
        args.tap?.(`${stem}.hidden_out`, hidden, dims);
        // After the last block_00 res, drop ts_embed -- subsequent blocks
        // either take scaled_timestep (mid) or nothing (plain).
        if (isMidRes && stem === "dec_block_00_res_4") {
          tsEmbed = null;
          tsEmbedDims = null;
        }
      }
    }

    // shell_post: hidden + scaled_timestep -> pixels (1, 3, T_px, H_px, W_px).
    let pixels: Uint16Array;
    let pixelDims: number[];
    {
      const t = await this.runShard(
        "dec:shell_post",
        shardIdx++,
        this.files.decShellPost,
        signal,
        async (sess) => {
          const hFeed = new ort.Tensor("float16", hidden, dims);
          const tsFeed = new ort.Tensor("float32", scaledTimestep, scaledTimestepDims);
          const r = await sess.run({ hidden: hFeed, scaled_timestep: tsFeed });
          const out = pickOutput(r, ["pixels"]);
          assertOrtOutput("vae.dec_shell_post.pixels", out, { type: "float16" });
          const outDims = (out.dims as number[]).slice();
          let count = 1;
          for (const d of outDims) count *= d;
          const view = assertF16View("vae.dec_shell_post.pixels.data", out.data, count);
          const data = copyF16Bits(view);
          dumpStats("dec_shell_post (pixels)", data, outDims);
          hFeed.dispose?.();
          tsFeed.dispose?.();
          for (const k in r) (r[k] as ort.Tensor).dispose?.();
          return { data, dims: outDims };
        },
        onShard,
      );
      pixels = t.data;
      pixelDims = t.dims;
      args.tap?.("dec_shell_post.pixels", pixels, pixelDims);
    }

    return {
      pixels,
      pixelFrames: pixelDims[2],
      pixelHeight: pixelDims[3],
      pixelWidth: pixelDims[4],
    };
  }

  // -------------------------------------------------------------------------

  private async runShard<T extends { data: Uint16Array; dims: number[] }>(
    label: string,
    shardIndex: number,
    file: OrtModelFile,
    signal: AbortSignal | undefined,
    body: (sess: ort.InferenceSession) => Promise<T>,
    onShard?: VaeShardProgress,
  ): Promise<T> {
    signal?.throwIfAborted();
    const tLoad = performance.now();
    const session = await createSession(this.cache, file, this.providers);
    const loadMs = performance.now() - tLoad;

    let runMs = 0;
    let releaseMs = 0;
    let result: T;
    try {
      const tRun = performance.now();
      result = await body(session);
      runMs = performance.now() - tRun;
    } finally {
      const tRel = performance.now();
      await session.release();
      releaseMs = performance.now() - tRel;
    }
    onShard?.({
      label,
      shardIndex,
      totalShards: this.totalShards,
      loadMs,
      runMs,
      releaseMs,
    });
    return result;
  }
}

function pickOutput(
  results: ort.InferenceSession.OnnxValueMapType,
  preferred: string[],
): ort.Tensor {
  for (const name of preferred) {
    if (results[name]) return results[name] as ort.Tensor;
  }
  const keys = Object.keys(results);
  const t = keys[0] ? results[keys[0]] : undefined;
  if (!t) throw new Error("session produced no output");
  return t as ort.Tensor;
}

function boxMullerF16(n: number, rand01: () => number): Uint16Array {
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rand01(), 1e-7);
    const u2 = rand01();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    out[i] = f32ToF16Bits(r * Math.cos(theta));
    if (i + 1 < n) out[i + 1] = f32ToF16Bits(r * Math.sin(theta));
  }
  return out;
}
