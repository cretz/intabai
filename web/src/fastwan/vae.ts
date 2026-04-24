// LightTAE (Wan 2.2) VAE decoder. Distilled Conv2D decoder from
// `lightx2v/Autoencoders` (lighttaew2_2.safetensors), exported to ONNX fp16
// at hf-repo/onnx/vae_decoder.onnx (36.6 MB). Substitutes for the full
// Wan 2.2 VAE which is 2.63 GB with 3D causal convs that ORT-web has no
// working kernels for; LightTAE is pure Conv2D, same op family as SD1.5
// VAE decode, and produced a clean WebGPU smoke on 2026-04-18.
//
// Input latent shape varies by selected resolution:
//   [1, T=21, C=48, H, W] fp16 where H=W=resolution/16 (30 for 480, 36 for 576).
// Output:
//   [1, F=81, 3, resolution, resolution] fp16, channel-first RGB in [-1, 1].
// The UI-side framing code (rgba bytes, videoframes) lives in the
// pipeline, not here - this module just runs the ONNX session.
//
// WanVaeDecoder below is the full AutoencoderKLWan path. It is DEAD CODE,
// kept for reference only - no caller, not wired into the pipeline. See
// the banner above the class for context.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits } from "../sd15/fp16";
import {
  copyGpuTensorsBatch,
  createGpuTensor,
  destroyGpuTensor,
  getOrtGpuDevice,
  readGpuFp16,
  writeGpuBytes,
} from "./gpu-io";

/** Resolution-independent constants. Spatial dims live on FastwanShape. */
export const LIGHTTAE_LATENT_T = 21;
export const LIGHTTAE_LATENT_C = 48;
export const LIGHTTAE_OUT_FRAMES = 81;

import type { FastwanShape } from "./transformer";

export class VaeDecoder {
  private session: ort.InferenceSession | null = null;

  constructor(
    private readonly cache: ModelCache,
    private readonly file: OrtModelFile,
    private readonly shape: FastwanShape,
  ) {}

  async load(): Promise<void> {
    if (this.session) return;
    this.session = await createSession(this.cache, this.file, ["webgpu", "wasm"]);
  }

  async release(): Promise<void> {
    if (!this.session) return;
    await this.session.release();
    this.session = null;
  }

  /** Decode latents (fp16 bits, shape [1, 21, 48, latentH, latentW]) to
   *  frames (fp16 bits, shape [1, 81, 3, pixelH, pixelW], channel-first
   *  RGB in [-1,1]). */
  async decode(latents: Uint16Array): Promise<Uint16Array> {
    if (!this.session) throw new Error("VaeDecoder.load() must be called first");
    const feeds: Record<string, ort.Tensor> = {
      latents: new ort.Tensor(
        "float16",
        latents,
        [1, LIGHTTAE_LATENT_T, LIGHTTAE_LATENT_C, this.shape.latentH, this.shape.latentW],
      ),
    };
    const results = await this.session.run(feeds);
    const key = "frames" in results ? "frames" : Object.keys(results)[0];
    const out = results[key];
    if (!out) throw new Error("vae_decoder produced no output");
    return copyF16Bits(out.data as ArrayBufferView);
  }
}

// ============================================================================
// DEAD CODE - WanVaeDecoder below is no longer wired into the pipeline.
//
// Originally gated behind `?vaekl=1` for a final-decode quality A/B against
// LightTAE. The A/B confirmed the swap did not improve quality (see
// notes/image.png vs notes/image-previous.png), so generate.ts always uses
// LightTAE now. The class is retained because the ONNX export pipeline
// (web/scripts/export-fastwan-vae-kl-streaming.py with --decompose-conv3d),
// the 32-slot streaming-cache contract, and the GPU-resident io-binding
// loop took real work to land. If we ever revisit Wan VAE on the web (say,
// once ORT-web ships a working Conv3D kernel) this is the entry point.
//
// Constants below are frozen at the original 480x832 / T=21 export and
// will not match a fresh re-export. Re-probe via the export script before
// re-enabling. No model files reference WAN_VAE_LATENT_* anymore;
// fastwan/models.ts no longer ships the decoder_init/step ONNX entries.
// ============================================================================

// ---- Full AutoencoderKLWan streaming decoder ------------------------------
//
// Two ONNX graphs (decoder_init, decoder_step) share a 32-tensor fp16 cache
// I/O contract. Pipeline:
//   - init(latent_frame_0)                 -> frames[T=1]  + 32 cache_out
//   - step(latent_frame_i, 32 cache_in)    -> frames[T=4]  + 32 cache_out   (x20)
// Total: 1 + 20*4 = 81 output frames. Each call emits frames as
// [1, 3, T, 480, 832] fp16 in [-1, 1] (torch.clamp in the export wrapper).
//
// io-binding: the 32-tensor cache (1.4 GB fp16) stays GPU-resident across
// all 20 step iterations. Per-step we keep two pre-allocated cache sets
// and ping-pong between them via GPU-to-GPU copies. Only the per-frame
// latent (~0.3 MB) is uploaded and the per-call frames output
// (~9.6 MB for step, ~2.4 MB for init) is downloaded. Without this the
// pipeline round-trips ~56 GB across PCIe per decode; with it, ~200 MB.

export const WAN_VAE_LATENT_C = 48;
export const WAN_VAE_LATENT_T = 21;
export const WAN_VAE_LATENT_H = 30;
export const WAN_VAE_LATENT_W = 52;
export const WAN_VAE_OUT_FRAMES = 81;
export const WAN_VAE_OUT_H = 480;
export const WAN_VAE_OUT_W = 832;

export const WAN_VAE_CACHE_SHAPES: readonly (readonly number[])[] = [
  [1, 48, 2, 30, 52],
  ...Array.from({ length: 11 }, () => [1, 1024, 2, 30, 52]),
  ...Array.from({ length: 7 }, () => [1, 1024, 2, 60, 104]),
  [1, 1024, 2, 120, 208],
  ...Array.from({ length: 5 }, () => [1, 512, 2, 120, 208]),
  [1, 512, 2, 240, 416],
  ...Array.from({ length: 6 }, () => [1, 256, 2, 240, 416]),
];

function cacheName(kind: "in" | "out", i: number): string {
  return `cache_${kind}_${i.toString().padStart(2, "0")}`;
}

function cacheElementCount(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function cacheByteLength(shape: readonly number[]): number {
  return cacheElementCount(shape) * 2;
}

/** Extract a single time-slice of a [1, C, T, H, W] fp16 latent into a
 *  contiguous [1, C, 1, H, W] buffer. Source is not contiguous along T. */
function sliceLatentFrame(
  src: Uint16Array,
  t: number,
  C: number,
  T: number,
  H: number,
  W: number,
): Uint16Array {
  const plane = H * W;
  const out = new Uint16Array(C * plane);
  for (let c = 0; c < C; c++) {
    const srcOff = (c * T + t) * plane;
    out.set(src.subarray(srcOff, srcOff + plane), c * plane);
  }
  return out;
}

/** Copy frames output [1, 3, frameCount, H, W] into the packed
 *  [F_total, 3, H, W] output buffer at offset `writeFrameOffset`.
 *  Converts the model's NCTHW ordering to the per-frame NCHW layout that
 *  `framesToBitmaps` expects. */
function writeFramesTransposed(
  outBits: Uint16Array,
  framesBits: Uint16Array,
  frameCount: number,
  writeFrameOffset: number,
  H: number,
  W: number,
): void {
  const plane = H * W;
  for (let f = 0; f < frameCount; f++) {
    for (let c = 0; c < 3; c++) {
      const srcOff = (c * frameCount + f) * plane;
      const dstOff = ((writeFrameOffset + f) * 3 + c) * plane;
      outBits.set(framesBits.subarray(srcOff, srcOff + plane), dstOff);
    }
  }
}

export class WanVaeDecoder {
  private initSession: ort.InferenceSession | null = null;
  private stepSession: ort.InferenceSession | null = null;

  constructor(
    private readonly cache: ModelCache,
    private readonly initFile: OrtModelFile,
    private readonly stepFile: OrtModelFile,
  ) {}

  async release(): Promise<void> {
    if (this.initSession) {
      await this.initSession.release();
      this.initSession = null;
    }
    if (this.stepSession) {
      await this.stepSession.release();
      this.stepSession = null;
    }
  }

  /** Decode latent [1, C=48, T=21, H=30, W=52] fp16 bits to frames
   *  [1, F=81, 3, 480, 832] fp16 bits. Native output range is [-1, 1];
   *  caller is responsible for rescaling before display. Progress callback
   *  fires once per frame-group (init: 1 frame, each step: 4 frames).
   *
   *  All intermediate cache tensors stay GPU-resident across the full
   *  decode via io-binding - see module header for rationale. */
  async decode(
    latents: Uint16Array,
    onProgress?: (doneFrames: number, totalFrames: number) => void,
  ): Promise<Uint16Array> {
    const C = WAN_VAE_LATENT_C;
    const T = WAN_VAE_LATENT_T;
    const H = WAN_VAE_LATENT_H;
    const W = WAN_VAE_LATENT_W;
    const F = WAN_VAE_OUT_FRAMES;
    const outH = WAN_VAE_OUT_H;
    const outW = WAN_VAE_OUT_W;
    const expected = C * T * H * W;
    if (latents.length !== expected) {
      throw new Error(
        `WanVaeDecoder.decode: expected latent length ${expected}, got ${latents.length}`,
      );
    }

    const out = new Uint16Array(F * 3 * outH * outW);
    let framesDone = 0;

    // Pre-create the init session so ORT has materialized its WebGPU device
    // before we try to allocate GPU buffers.
    this.initSession = await createSession(this.cache, this.initFile, [
      "webgpu",
      "wasm",
    ]);
    const device = getOrtGpuDevice();
    if (!device) {
      throw new Error(
        "WanVaeDecoder: ORT-web did not materialize a WebGPU device; " +
          "io-binding requires the webgpu EP",
      );
    }

    // Allocate two cache sets (A, B) for ping-pong. Each iteration reads
    // from one set and writes to the other, then we GPU-copy the writes
    // back for the next iteration (simpler than swapping feed bindings
    // across 64 tensors). Both sets persist for the whole decode.
    const cacheA: ort.Tensor[] = WAN_VAE_CACHE_SHAPES.map((shape) =>
      createGpuTensor(device, "float16", shape),
    );
    const cacheB: ort.Tensor[] = WAN_VAE_CACHE_SHAPES.map((shape) =>
      createGpuTensor(device, "float16", shape),
    );

    // Latent-per-frame input: [1, C, 1, H, W] fp16. Reused across init +
    // every step; we re-upload per frame with writeGpuBytes.
    const latentFrameDims = [1, C, 1, H, W] as const;
    const gpuLatent = createGpuTensor(
      device,
      "float16",
      Array.from(latentFrameDims),
    );

    // Frames output buffers. Init emits 1 frame, step emits 4 frames;
    // allocate separate fixed-shape buffers so we don't reallocate.
    const initFramesDims = [1, 3, 1, outH, outW] as const;
    const stepFramesDims = [1, 3, 4, outH, outW] as const;
    const gpuFramesInit = createGpuTensor(
      device,
      "float16",
      Array.from(initFramesDims),
    );
    const gpuFramesStep = createGpuTensor(
      device,
      "float16",
      Array.from(stepFramesDims),
    );

    const initFrameCount = 1 * 3 * 1 * outH * outW;
    const stepFrameCount = 1 * 3 * 4 * outH * outW;

    try {
      // ---- init: frame 0 --------------------------------------------------
      {
        const frame0 = sliceLatentFrame(latents, 0, C, T, H, W);
        writeGpuBytes(device, gpuLatent, frame0);

        // Init writes to cacheA as cache_out_NN; cacheB is unused on this
        // first call.
        const feeds: Record<string, ort.Tensor> = { latent: gpuLatent };
        const fetches: Record<string, ort.Tensor> = { frames: gpuFramesInit };
        for (let i = 0; i < WAN_VAE_CACHE_SHAPES.length; i++) {
          fetches[cacheName("out", i)] = cacheA[i];
        }
        await this.initSession.run(feeds, fetches);
        const framesBits = await readGpuFp16(
          device,
          gpuFramesInit,
          initFrameCount,
        );
        writeFramesTransposed(out, framesBits, 1, 0, outH, outW);
        framesDone = 1;
        onProgress?.(framesDone, F);
      }

      // Release the init session before loading step. Cache A survives
      // (we own its GPUBuffers, not the session).
      await this.initSession.release();
      this.initSession = null;

      // ---- step: frames 1..20 (4 frames each) -----------------------------
      this.stepSession = await createSession(this.cache, this.stepFile, [
        "webgpu",
        "wasm",
      ]);

      // Ping-pong: on iteration t we feed `inSet` as cache_in and receive
      // into `outSet` as cache_out, then copy outSet -> inSet on GPU and
      // repeat. After init, cacheA holds the most recent state, so inSet
      // starts as cacheA.
      let inSet = cacheA;
      let outSet = cacheB;

      const copyPairs: Array<{
        src: ort.Tensor;
        dst: ort.Tensor;
        byteLength: number;
      }> = WAN_VAE_CACHE_SHAPES.map((shape, i) => ({
        src: outSet[i],
        dst: inSet[i],
        byteLength: cacheByteLength(shape),
      }));

      for (let t = 1; t < T; t++) {
        const frameT = sliceLatentFrame(latents, t, C, T, H, W);
        writeGpuBytes(device, gpuLatent, frameT);

        const feeds: Record<string, ort.Tensor> = { latent: gpuLatent };
        const fetches: Record<string, ort.Tensor> = { frames: gpuFramesStep };
        for (let i = 0; i < WAN_VAE_CACHE_SHAPES.length; i++) {
          feeds[cacheName("in", i)] = inSet[i];
          fetches[cacheName("out", i)] = outSet[i];
        }
        await this.stepSession.run(feeds, fetches);

        const framesBits = await readGpuFp16(
          device,
          gpuFramesStep,
          stepFrameCount,
        );
        writeFramesTransposed(out, framesBits, 4, framesDone, outH, outW);
        framesDone += 4;

        // GPU-side copy cache_out -> cache_in for next iteration.
        if (t + 1 < T) {
          for (let i = 0; i < WAN_VAE_CACHE_SHAPES.length; i++) {
            copyPairs[i].src = outSet[i];
            copyPairs[i].dst = inSet[i];
          }
          copyGpuTensorsBatch(device, copyPairs);
        }

        onProgress?.(framesDone, F);
      }
    } finally {
      if (this.initSession) {
        try { await this.initSession.release(); } catch { /* ignore */ }
        this.initSession = null;
      }
      if (this.stepSession) {
        try { await this.stepSession.release(); } catch { /* ignore */ }
        this.stepSession = null;
      }
      for (const t of cacheA) destroyGpuTensor(t);
      for (const t of cacheB) destroyGpuTensor(t);
      destroyGpuTensor(gpuLatent);
      destroyGpuTensor(gpuFramesInit);
      destroyGpuTensor(gpuFramesStep);
    }

    if (framesDone !== F) {
      throw new Error(`WanVaeDecoder: expected ${F} frames, got ${framesDone}`);
    }
    return out;
  }
}
