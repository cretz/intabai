// FastWan 2.2 TI2V transformer, run as 1 shell_pre + 30 per-block sessions
// + 1 shell_post per denoising step.
//
// The model was exported per-block at q4f16 because the monolithic 9.4 GB
// fp16 transformer won't fit in one WebGPU buffer on most devices (see
// worklog entry 2026-04-18). Each block is ~92 MB on disk. We load one
// block at a time, run it, and dispose the session before loading the
// next, so peak GPU memory is one block plus the tokens/conditioning
// tensors rather than 30 × 92 MB.
//
// shell_pre and shell_post are kept loaded across the whole generation
// (52 MB and 1.2 MB respectively). shell_pre is re-run per denoising step
// because it derives timestep_proj/temb from the current timestep; the
// encoder projection and RoPE frequencies it emits are invariant across
// steps, so we cache those locally and hand them back into the block loop.
//
// Input shapes are fixed to the 480 × 832 × 81f layout the model was
// trained on: latent [1, 48, 21, 30, 52], patch [1,2,2], so the packed
// token sequence is 21 × 15 × 26 = 8190. If we ever support other
// resolutions we'll need to recompute ppf/pph/ppw and the noise shape.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits, f16ToF32Array } from "../sd15/fp16";

export const FASTWAN_NUM_BLOCKS = 30;

/** Fixed latent geometry at the default 480×832 resolution, 81 output
 *  frames. Latent frame count, latent spatial dims, and patch grid are
 *  derived from the pipeline config. */
export const FASTWAN_LATENT_CHANNELS = 48;
export const FASTWAN_LATENT_FRAMES = 21;
export const FASTWAN_LATENT_H = 30;
export const FASTWAN_LATENT_W = 52;
export const FASTWAN_PATCH_F = 21;
export const FASTWAN_PATCH_H = 15;
export const FASTWAN_PATCH_W = 26;
/** seq_len after patching = ppf * pph * ppw. */
export const FASTWAN_SEQ_LEN =
  FASTWAN_PATCH_F * FASTWAN_PATCH_H * FASTWAN_PATCH_W;
/** Hidden size inside the transformer. Fixed by the model config. */
export const FASTWAN_HIDDEN = 3072;
/** Text encoder output sequence length (see text-encoder.ts). */
export const FASTWAN_TEXT_SEQ_LEN = 512;
/** Per-token conditioning arity that timestep_proj uses (AdaLayerNorm). */
export const FASTWAN_TIMESTEP_PROJ_ARITY = 6;
/** RoPE head dim. Attention head dim is 128 (hidden=3072 / 24 heads). */
export const FASTWAN_HEAD_DIM = 128;

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
  /** Current noisy latent. [1, 48, 21, 30, 52] fp16 bits. */
  latent: Uint16Array;
  /** Current scheduler timestep. Broadcast to [1, 8190] int64 on the wire. */
  timestep: number;
  /** Text embeddings from the text encoder. [1, 512, 4096] fp16 bits. */
  textEmbeds: Uint16Array;
  /** Which denoising step this is (0-indexed). Passed to onBlock so the
   *  caller can drive a progress bar that spans all steps * all blocks. */
  stepIndex: number;
  /** Optional verbose trace of load / run / release timings for each
   *  block plus shell_pre / shell_post. Opt-in; noisy but essential when
   *  diagnosing a hang vs a slow block. */
  onDebug?: (msg: string) => void;
  /** Optional per-tensor stats hook for diagnosing gray-output bugs.
   *  Called with intermediate fp16-bit tensors at key stages. */
  onStatsFp16?: (name: string, bits: Uint16Array) => void;
}

export class Transformer {
  private shellPre: ort.InferenceSession | null = null;
  private shellPost: ort.InferenceSession | null = null;

  constructor(
    private readonly cache: ModelCache,
    private readonly files: TransformerFiles,
  ) {
    if (files.blocks.length !== FASTWAN_NUM_BLOCKS) {
      throw new Error(
        `expected ${FASTWAN_NUM_BLOCKS} transformer blocks, got ${files.blocks.length}`,
      );
    }
  }

  async load(): Promise<void> {
    if (!this.shellPre) {
      this.shellPre = await createSession(this.cache, this.files.shellPre, [
        "webgpu",
        "wasm",
      ]);
    }
    if (!this.shellPost) {
      this.shellPost = await createSession(this.cache, this.files.shellPost, [
        "webgpu",
        "wasm",
      ]);
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
   *  noise_pred as fp16 bits, shape [1, 48, 21, 30, 52]. */
  async forward(input: ForwardInput, onBlock?: BlockProgress): Promise<Uint16Array> {
    if (!this.shellPre || !this.shellPost) {
      throw new Error("Transformer.load() must be called before forward()");
    }

    // Expand scalar timestep to [1, 8190] int64. Wan 2.2 TI2V uses
    // expand_timesteps=True: every packed token gets the same timestep.
    const timestepArr = new BigInt64Array(FASTWAN_SEQ_LEN);
    const tBig = BigInt(input.timestep);
    for (let i = 0; i < FASTWAN_SEQ_LEN; i++) timestepArr[i] = tBig;

    // shell_pre: latent + timestep + text embeds -> tokens + enc_proj +
    // timestep_proj + temb + RoPE freqs.
    const preFeeds: Record<string, ort.Tensor> = {
      hidden_states: new ort.Tensor(
        "float16",
        input.latent,
        [
          1,
          FASTWAN_LATENT_CHANNELS,
          FASTWAN_LATENT_FRAMES,
          FASTWAN_LATENT_H,
          FASTWAN_LATENT_W,
        ],
      ),
      timestep: new ort.Tensor("int64", timestepArr, [1, FASTWAN_SEQ_LEN]),
      encoder_hidden_states: new ort.Tensor(
        "float16",
        input.textEmbeds,
        [1, FASTWAN_TEXT_SEQ_LEN, 4096],
      ),
    };
    const debug = input.onDebug;
    const tPre = performance.now();
    const preOut = await this.shellPre.run(preFeeds);
    debug?.(`shell_pre: ${(performance.now() - tPre).toFixed(0)} ms`);
    // Copy out so we can release intermediate session tensors cleanly
    // between blocks. WebGPU-backed ORT tensors can be invalidated by
    // subsequent runs on the same session, so a fresh copy is safest.
    let tokens = copyF16Bits(preOut.tokens.data as ArrayBufferView);
    const encProj = copyF16Bits(preOut.enc_proj.data as ArrayBufferView);
    const timestepProj = copyF16Bits(preOut.timestep_proj.data as ArrayBufferView);
    const temb = copyF16Bits(preOut.temb.data as ArrayBufferView);
    const freqsCos = new Float32Array(preOut.freqs_cos.data as Float32Array);
    const freqsSin = new Float32Array(preOut.freqs_sin.data as Float32Array);
    const stats = input.onStatsFp16;
    if (stats && input.stepIndex === 0) {
      stats("shell_pre.tokens", tokens);
      stats("shell_pre.enc_proj", encProj);
      stats("shell_pre.timestep_proj", timestepProj);
      stats("shell_pre.temb", temb);
    }
    if (debug && input.stepIndex === 0) {
      // Hex dumps of first 32 values of each shell_pre output, for
      // byte-diffing against a Python CPU ONNX reference. Pair with the
      // scripts/dump-reference-shell-pre.py script.
      dumpHex(debug, "shell_pre.tokens[tok=0,0:32]", tokens, 0);
      dumpHex(debug, "shell_pre.enc_proj[tok=0,0:32]", encProj, 0);
      dumpHex(debug, "shell_pre.temb[tok=0,0:32]", temb, 0);
      // freqs are fp32; dump separately.
      dumpHexF32(debug, "shell_pre.freqs_cos[tok=0,0:32]", freqsCos, 0);
      dumpHexF32(debug, "shell_pre.freqs_sin[tok=0,0:32]", freqsSin, 0);
    }
    const tokensDims = [1, FASTWAN_SEQ_LEN, FASTWAN_HIDDEN];
    const encProjDims = [1, FASTWAN_TEXT_SEQ_LEN, FASTWAN_HIDDEN];
    const timestepProjDims = [
      1,
      FASTWAN_SEQ_LEN,
      FASTWAN_TIMESTEP_PROJ_ARITY,
      FASTWAN_HIDDEN,
    ];
    const freqsDims = [1, FASTWAN_SEQ_LEN, 1, FASTWAN_HEAD_DIM];

    // Block loop with double-buffered session loading: while block N runs
    // we start compiling block N+1, so the ~500 ms load cost of the next
    // block overlaps the ~7 s run cost of the current block. Peak GPU
    // residency is 2 block sessions (~2 x 92 MB q4 graphs + activations),
    // still well under mobile maxBufferSize.
    const loadBlock = (i: number) =>
      createSession(this.cache, this.files.blocks[i], ["webgpu", "wasm"]);
    let currentLoad = loadBlock(0);
    for (let i = 0; i < FASTWAN_NUM_BLOCKS; i++) {
      const blockSession = await currentLoad;
      const tRun = performance.now();
      // Kick off next-block load now so its compile overlaps this run.
      let nextLoad: Promise<ort.InferenceSession> | null = null;
      if (i + 1 < FASTWAN_NUM_BLOCKS) {
        nextLoad = loadBlock(i + 1);
      }
      try {
        const feeds: Record<string, ort.Tensor> = {
          hidden_states: new ort.Tensor("float16", tokens, tokensDims),
          encoder_hidden_states: new ort.Tensor("float16", encProj, encProjDims),
          timestep_proj: new ort.Tensor(
            "float16",
            timestepProj,
            timestepProjDims,
          ),
          freqs_cos: new ort.Tensor("float32", freqsCos, freqsDims),
          freqs_sin: new ort.Tensor("float32", freqsSin, freqsDims),
        };
        const result = await blockSession.run(feeds);
        debug?.(`block ${i}: ${(performance.now() - tRun).toFixed(0)} ms`);
        const out =
          result.hidden_states_out ?? result[Object.keys(result)[0]];
        tokens = copyF16Bits(out.data as ArrayBufferView);
        if (stats && input.stepIndex === 0 && (i === 0 || i === FASTWAN_NUM_BLOCKS - 1)) {
          stats(`block_${i.toString().padStart(2, "0")}.out`, tokens);
        }
        if (debug && input.stepIndex === 0 && i === 0) {
          dumpHex(debug, "block_00.out[tok=0,0:32]", tokens, 0);
        }
      } catch (err) {
        // Block run failed — make sure the prefetched next session is
        // cleaned up before we propagate, otherwise its GPU memory leaks.
        if (nextLoad) {
          nextLoad.then((s) => s.release()).catch(() => {});
        }
        throw err;
      } finally {
        await blockSession.release();
      }
      onBlock?.(input.stepIndex, i + 1, FASTWAN_NUM_BLOCKS);
      currentLoad = nextLoad as Promise<ort.InferenceSession>;
    }

    // shell_post: tokens + temb + int64 patch dims -> noise_pred.
    const ppfArr = new BigInt64Array([BigInt(FASTWAN_PATCH_F)]);
    const pphArr = new BigInt64Array([BigInt(FASTWAN_PATCH_H)]);
    const ppwArr = new BigInt64Array([BigInt(FASTWAN_PATCH_W)]);
    const postFeeds: Record<string, ort.Tensor> = {
      hidden_states: new ort.Tensor("float16", tokens, tokensDims),
      temb: new ort.Tensor("float16", temb, tokensDims),
      ppf: new ort.Tensor("int64", ppfArr, []),
      pph: new ort.Tensor("int64", pphArr, []),
      ppw: new ort.Tensor("int64", ppwArr, []),
    };
    const tPost = performance.now();
    const postOut = await this.shellPost.run(postFeeds);
    debug?.(`shell_post: ${(performance.now() - tPost).toFixed(0)} ms`);
    const noise =
      postOut.noise_pred ?? postOut[Object.keys(postOut)[0]];
    return copyF16Bits(noise.data as ArrayBufferView);
  }
}

function dumpHex(
  debug: (msg: string) => void,
  label: string,
  bits: Uint16Array,
  offset: number,
): void {
  const slice = bits.subarray(offset, offset + 32);
  const f32 = f16ToF32Array(slice);
  const hex = Array.from(slice).map((b) => b.toString(16).padStart(4, "0")).join(",");
  debug(`${label} f32=[${Array.from(f32).map((v) => v.toFixed(4)).join(",")}]`);
  debug(`${label} hex=${hex}`);
}

function dumpHexF32(
  debug: (msg: string) => void,
  label: string,
  values: Float32Array,
  offset: number,
): void {
  const slice = values.subarray(offset, offset + 32);
  const hex = Array.from(slice).map((v) => {
    const u32 = new Uint32Array(new Float32Array([v]).buffer)[0];
    return u32.toString(16).padStart(8, "0");
  }).join(",");
  debug(`${label} f32=[${Array.from(slice).map((v) => v.toFixed(6)).join(",")}]`);
  debug(`${label} hex32=${hex}`);
}
