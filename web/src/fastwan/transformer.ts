// FastWan 2.2 TI2V transformer, run as 1 shell_pre + 30 per-block sessions
// + 1 shell_post per denoising step.
//
// The model was exported per-block at q4f16 because the monolithic 9.4 GB
// fp16 transformer won't fit in one WebGPU buffer on most devices. Each
// block is ~92 MB on disk. We load one block at a time, run it, and dispose
// the session before loading the next, so peak GPU memory is one block
// plus the tokens/conditioning tensors rather than 30 * 92 MB.
//
// shell_pre and shell_post are kept loaded across the whole generation
// (180 MB fp16 and 1.2 MB respectively; q4 shell_pre had a NaN bug under
// MatMulNBits acc=4, so we ship fp16). shell_pre is re-run per denoising step
// because it derives timestep_proj/temb from the current timestep.
//
// Input shapes vary by selected resolution (see FastwanShape). For 480x480
// the latent is [1, 48, 21, 30, 30] and patch grid is 21*15*15=4725 tokens;
// for 576x576 the latent is [1, 48, 21, 36, 36] and patch grid is
// 21*18*18=6804 tokens. Patch size is always [1,2,2].
//
// io-binding: all intra-step tensor handoffs stay GPU-resident. shell_pre
// outputs land directly in our GPU tensors, the 30 blocks ping-pong the
// `tokens` hidden state between two GPU buffers while reusing the 5
// invariant conditioning tensors, and shell_post writes noise_pred into
// a GPU buffer that we download to CPU exactly once at the end. A naive
// CPU round-trip between blocks ships ~50 MB per handoff * 30 blocks *
// 4 steps = ~6 GB of PCIe traffic per generation; with io-binding we move
// ~14 MB (latent in, noise_pred out).

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import {
  createGpuTensor,
  destroyGpuTensor,
  getOrtGpuDevice,
  readGpuFp16,
  writeGpuBytes,
} from "./gpu-io";

export const FASTWAN_NUM_BLOCKS = 30;

/** Resolution-independent constants from the pipeline config. */
export const FASTWAN_LATENT_CHANNELS = 48;
export const FASTWAN_LATENT_FRAMES = 21;
export const FASTWAN_PATCH_F = 21;
/** Hidden size inside the transformer. */
export const FASTWAN_HIDDEN = 3072;
/** Text encoder output sequence length (see text-encoder.ts). */
export const FASTWAN_TEXT_SEQ_LEN = 512;
/** Per-token conditioning arity that timestep_proj uses (AdaLayerNorm). */
export const FASTWAN_TIMESTEP_PROJ_ARITY = 6;
/** RoPE head dim. Attention head dim is 128 (hidden=3072 / 24 heads). */
export const FASTWAN_HEAD_DIM = 128;

export type FastwanResolution = 480 | 576;

/** Resolution-bound geometry. Computed once per generation from the user's
 *  selected model and threaded through the pipeline. Latent dim = pixel/16,
 *  patch dim = latent/2 (patch_size=[1,2,2]); seq_len = patch_f*patch_h*patch_w. */
export interface FastwanShape {
  resolution: FastwanResolution;
  /** Output pixel height (= width; we only ship square models). */
  pixelH: number;
  pixelW: number;
  /** Latent height/width (= pixel/16). */
  latentH: number;
  latentW: number;
  /** Patch grid height/width (= latent/2). */
  patchH: number;
  patchW: number;
  /** Packed token count = patchF * patchH * patchW. */
  seqLen: number;
}

export function fastwanShape(resolution: FastwanResolution): FastwanShape {
  const latent = resolution / 16;
  const patch = latent / 2;
  return {
    resolution,
    pixelH: resolution,
    pixelW: resolution,
    latentH: latent,
    latentW: latent,
    patchH: patch,
    patchW: patch,
    seqLen: FASTWAN_PATCH_F * patch * patch,
  };
}

export interface TransformerFiles {
  shellPre: OrtModelFile;
  /** 30 block files, index 0..29. */
  blocks: OrtModelFile[];
  shellPost: OrtModelFile;
}

export type BlockProgress = (
  stepIndex: number,
  blockIndex: number,
  totalBlocks: number,
) => void;

export interface ForwardInput {
  /** Current noisy latent. [1, 48, 21, 30, 30] fp16 bits. */
  latent: Uint16Array;
  /** Current scheduler timestep. Broadcast to [1, 4725] int64 on the wire. */
  timestep: number;
  /** Text embeddings from the text encoder. [1, 512, 4096] fp16 bits. */
  textEmbeds: Uint16Array;
  /** Which denoising step this is (0-indexed). Passed to onBlock so the
   *  caller can drive a progress bar that spans all steps * all blocks. */
  stepIndex: number;
  /** Optional verbose trace of load / run / release timings for each
   *  block plus shell_pre / shell_post. */
  onDebug?: (msg: string) => void;
  /** Optional per-tensor stats hook for diagnosing gray-output bugs.
   *  Called with intermediate fp16-bit tensors at key stages. */
  onStatsFp16?: (name: string, bits: Uint16Array) => void;
}

/** Per-shape element counts for the well-known tensors. Used by readback
 *  for stats/debug output. */
function shapeSizes(shape: FastwanShape) {
  const seq = shape.seqLen;
  const txt = FASTWAN_TEXT_SEQ_LEN;
  return {
    tokens: 1 * seq * FASTWAN_HIDDEN,
    encProj: 1 * txt * FASTWAN_HIDDEN,
    timestepProj: 1 * seq * FASTWAN_TIMESTEP_PROJ_ARITY * FASTWAN_HIDDEN,
    temb: 1 * seq * FASTWAN_HIDDEN,
    freqs: 1 * seq * 1 * FASTWAN_HEAD_DIM,
    noisePred:
      1 *
      FASTWAN_LATENT_CHANNELS *
      FASTWAN_LATENT_FRAMES *
      shape.latentH *
      shape.latentW,
  };
}

export class Transformer {
  private shellPre: ort.InferenceSession | null = null;
  private shellPost: ort.InferenceSession | null = null;

  constructor(
    private readonly cache: ModelCache,
    private readonly files: TransformerFiles,
    private readonly shape: FastwanShape,
    private readonly providers: ("webgpu" | "wasm")[] = ["webgpu", "wasm"],
  ) {
    if (files.blocks.length !== FASTWAN_NUM_BLOCKS) {
      throw new Error(
        `expected ${FASTWAN_NUM_BLOCKS} transformer blocks, got ${files.blocks.length}`,
      );
    }
  }

  async load(): Promise<void> {
    if (!this.shellPre) {
      this.shellPre = await createSession(
        this.cache,
        this.files.shellPre,
        this.providers,
      );
    }
    if (!this.shellPost) {
      this.shellPost = await createSession(
        this.cache,
        this.files.shellPost,
        this.providers,
      );
    }
  }

  async release(): Promise<void> {
    if (this.shellPre) {
      await this.shellPre.release();
      this.shellPre = null;
    }
    if (this.shellPost) {
      await this.shellPost.release();
      this.shellPost = null;
    }
  }

  /** One denoising step: shell_pre -> 30 blocks -> shell_post. Returns
   *  noise_pred as fp16 bits, shape [1, 48, 21, 30, 30]. */
  async forward(input: ForwardInput, onBlock?: BlockProgress): Promise<Uint16Array> {
    if (!this.shellPre || !this.shellPost) {
      throw new Error("Transformer.load() must be called before forward()");
    }

    const device = getOrtGpuDevice();
    if (!device) {
      throw new Error(
        "Transformer: ORT-web did not materialize a WebGPU device; " +
          "io-binding requires the webgpu EP",
      );
    }
    const debug = input.onDebug;
    const stats = input.onStatsFp16;
    const wantsStats = stats && input.stepIndex === 0;

    const shape = this.shape;
    const sizes = shapeSizes(shape);

    // Expand scalar timestep to [1, seqLen] int64. Wan 2.2 TI2V uses
    // expand_timesteps=True: every packed token gets the same timestep.
    // int64 stays on CPU; ORT routes shape/index ops to the wasm EP anyway.
    const timestepArr = new BigInt64Array(shape.seqLen);
    const tBig = BigInt(input.timestep);
    for (let i = 0; i < shape.seqLen; i++) timestepArr[i] = tBig;

    // ---- allocate GPU tensors ------------------------------------------------
    // All created fresh per forward() so the buffers are cleanly destroyed
    // before we return; cost is ~2 ms of WebGPU buffer-object churn per step,
    // negligible versus the ~3-minute step wall time.
    const gpuLatent = createGpuTensor(device, "float16", [
      1,
      FASTWAN_LATENT_CHANNELS,
      FASTWAN_LATENT_FRAMES,
      shape.latentH,
      shape.latentW,
    ]);
    const gpuEncoderHS = createGpuTensor(device, "float16", [
      1,
      FASTWAN_TEXT_SEQ_LEN,
      4096,
    ]);
    const gpuTokensA = createGpuTensor(device, "float16", [
      1,
      shape.seqLen,
      FASTWAN_HIDDEN,
    ]);
    const gpuTokensB = createGpuTensor(device, "float16", [
      1,
      shape.seqLen,
      FASTWAN_HIDDEN,
    ]);
    const gpuEncProj = createGpuTensor(device, "float16", [
      1,
      FASTWAN_TEXT_SEQ_LEN,
      FASTWAN_HIDDEN,
    ]);
    const gpuTimestepProj = createGpuTensor(device, "float16", [
      1,
      shape.seqLen,
      FASTWAN_TIMESTEP_PROJ_ARITY,
      FASTWAN_HIDDEN,
    ]);
    const gpuTemb = createGpuTensor(device, "float16", [
      1,
      shape.seqLen,
      FASTWAN_HIDDEN,
    ]);
    const gpuFreqsCos = createGpuTensor(device, "float32", [
      1,
      shape.seqLen,
      1,
      FASTWAN_HEAD_DIM,
    ]);
    const gpuFreqsSin = createGpuTensor(device, "float32", [
      1,
      shape.seqLen,
      1,
      FASTWAN_HEAD_DIM,
    ]);
    const gpuNoisePred = createGpuTensor(device, "float16", [
      1,
      FASTWAN_LATENT_CHANNELS,
      FASTWAN_LATENT_FRAMES,
      shape.latentH,
      shape.latentW,
    ]);

    const destroyAll = () => {
      for (const t of [
        gpuLatent,
        gpuEncoderHS,
        gpuTokensA,
        gpuTokensB,
        gpuEncProj,
        gpuTimestepProj,
        gpuTemb,
        gpuFreqsCos,
        gpuFreqsSin,
        gpuNoisePred,
      ]) destroyGpuTensor(t);
    };

    try {
      // ---- shell_pre --------------------------------------------------------
      writeGpuBytes(device, gpuLatent, input.latent);
      writeGpuBytes(device, gpuEncoderHS, input.textEmbeds);
      const preFeeds: Record<string, ort.Tensor> = {
        hidden_states: gpuLatent,
        timestep: new ort.Tensor("int64", timestepArr, [1, shape.seqLen]),
        encoder_hidden_states: gpuEncoderHS,
      };
      const preFetches: Record<string, ort.Tensor> = {
        tokens: gpuTokensA,
        enc_proj: gpuEncProj,
        timestep_proj: gpuTimestepProj,
        temb: gpuTemb,
        freqs_cos: gpuFreqsCos,
        freqs_sin: gpuFreqsSin,
      };
      const tPre = performance.now();
      await this.shellPre.run(preFeeds, preFetches);
      debug?.(`shell_pre: ${(performance.now() - tPre).toFixed(0)} ms`);

      if (wantsStats) {
        const [tokensBits, encProjBits, timestepProjBits, tembBits] =
          await Promise.all([
            readGpuFp16(device, gpuTokensA, sizes.tokens),
            readGpuFp16(device, gpuEncProj, sizes.encProj),
            readGpuFp16(device, gpuTimestepProj, sizes.timestepProj),
            readGpuFp16(device, gpuTemb, sizes.temb),
          ]);
        stats!("shell_pre.tokens", tokensBits);
        stats!("shell_pre.enc_proj", encProjBits);
        stats!("shell_pre.timestep_proj", timestepProjBits);
        stats!("shell_pre.temb", tembBits);
      }

      // ---- block loop with double-buffered session loading + ping-pong ------
      const loadBlock = (i: number) =>
        createSession(this.cache, this.files.blocks[i], this.providers);

      // After shell_pre, tokens are in gpuTokensA. Each block reads from
      // `currTokens` and writes to `nextTokens`; then we swap. Conditioning
      // tensors (encProj/timestepProj/freqsCos/freqsSin) are the same across
      // all 30 blocks, so we just hand them back in as feeds each iteration.
      let currTokens = gpuTokensA;
      let nextTokens = gpuTokensB;

      let currentLoad = loadBlock(0);
      for (let i = 0; i < FASTWAN_NUM_BLOCKS; i++) {
        const blockSession = await currentLoad;
        const tRun = performance.now();
        const nextLoad: Promise<ort.InferenceSession> | null =
          i + 1 < FASTWAN_NUM_BLOCKS ? loadBlock(i + 1) : null;
        try {
          const feeds: Record<string, ort.Tensor> = {
            hidden_states: currTokens,
            encoder_hidden_states: gpuEncProj,
            timestep_proj: gpuTimestepProj,
            freqs_cos: gpuFreqsCos,
            freqs_sin: gpuFreqsSin,
          };
          const fetches: Record<string, ort.Tensor> = {
            hidden_states_out: nextTokens,
          };
          await blockSession.run(feeds, fetches);
          debug?.(`block ${i}: ${(performance.now() - tRun).toFixed(0)} ms`);

          // Stats readback for first and last block on step 0 only - cheap
          // sanity check that block output isn't all NaN / all zero.
          const isFirstOrLast = i === 0 || i === FASTWAN_NUM_BLOCKS - 1;
          if (wantsStats && isFirstOrLast) {
            const outBits = await readGpuFp16(device, nextTokens, sizes.tokens);
            stats!(
              `block_${i.toString().padStart(2, "0")}.out`,
              outBits,
            );
          }
        } catch (err) {
          if (nextLoad) {
            nextLoad.then((s) => s.release()).catch(() => {});
          }
          throw err;
        } finally {
          await blockSession.release();
        }
        onBlock?.(input.stepIndex, i + 1, FASTWAN_NUM_BLOCKS);

        // Ping-pong: the block just wrote its output into nextTokens, which
        // becomes the input for block i+1.
        const swap = currTokens;
        currTokens = nextTokens;
        nextTokens = swap;

        currentLoad = nextLoad as Promise<ort.InferenceSession>;
      }

      // ---- shell_post -------------------------------------------------------
      const ppfArr = new BigInt64Array([BigInt(FASTWAN_PATCH_F)]);
      const pphArr = new BigInt64Array([BigInt(shape.patchH)]);
      const ppwArr = new BigInt64Array([BigInt(shape.patchW)]);
      const postFeeds: Record<string, ort.Tensor> = {
        hidden_states: currTokens,
        temb: gpuTemb,
        ppf: new ort.Tensor("int64", ppfArr, []),
        pph: new ort.Tensor("int64", pphArr, []),
        ppw: new ort.Tensor("int64", ppwArr, []),
      };
      const postFetches: Record<string, ort.Tensor> = {
        noise_pred: gpuNoisePred,
      };
      const tPost = performance.now();
      await this.shellPost.run(postFeeds, postFetches);
      debug?.(`shell_post: ${(performance.now() - tPost).toFixed(0)} ms`);

      // Final readback to CPU (scheduler math runs in fp32 in JS).
      return await readGpuFp16(device, gpuNoisePred, sizes.noisePred);
    } finally {
      destroyAll();
    }
  }
}

