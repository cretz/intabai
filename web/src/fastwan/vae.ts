// LightTAE (Wan 2.2) VAE decoder. Distilled Conv2D decoder from
// `lightx2v/Autoencoders` (lighttaew2_2.safetensors), exported to ONNX fp16
// at hf-repo/onnx/vae_decoder.onnx (36.6 MB). Substitutes for the full
// Wan 2.2 VAE which is 2.63 GB with 3D causal convs that ORT-web has no
// working kernels for; LightTAE is pure Conv2D, same op family as SD1.5
// VAE decode, and produced a clean WebGPU smoke on 2026-04-18.
//
// Input latent shape is fixed by the transformer geometry:
//   [1, T=21, C=48, H=30, W=52] fp16
// Output:
//   [1, F=81, 3, 480, 832] fp16, channel-first RGB in [-1, 1].
// The UI-side framing code (rgba bytes, videoframes) lives in the
// pipeline, not here - this module just runs the ONNX session.
//
// WanVaeDecoder below is the full AutoencoderKLWan path. Two sequential
// ONNX graphs (init + step) sharing a 32-tensor cache contract; gated in
// generate.ts behind `?vaekl=1`.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits } from "../sd15/fp16";

export const LIGHTTAE_LATENT_T = 21;
export const LIGHTTAE_LATENT_C = 48;
export const LIGHTTAE_LATENT_H = 30;
export const LIGHTTAE_LATENT_W = 52;
export const LIGHTTAE_OUT_FRAMES = 81;
export const LIGHTTAE_OUT_H = 480;
export const LIGHTTAE_OUT_W = 832;

export class VaeDecoder {
  private session: ort.InferenceSession | null = null;

  constructor(
    private readonly cache: ModelCache,
    private readonly file: OrtModelFile,
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

  /** Decode latents (fp16 bits, shape [1, 21, 48, 30, 52]) to frames
   *  (fp16 bits, shape [1, 81, 3, 480, 832], channel-first RGB in [-1,1]). */
  async decode(latents: Uint16Array): Promise<Uint16Array> {
    if (!this.session) throw new Error("VaeDecoder.load() must be called first");
    const feeds: Record<string, ort.Tensor> = {
      latents: new ort.Tensor(
        "float16",
        latents,
        [1, LIGHTTAE_LATENT_T, LIGHTTAE_LATENT_C, LIGHTTAE_LATENT_H, LIGHTTAE_LATENT_W],
      ),
    };
    const results = await this.session.run(feeds);
    const key = "frames" in results ? "frames" : Object.keys(results)[0];
    const out = results[key];
    if (!out) throw new Error("vae_decoder produced no output");
    return copyF16Bits(out.data as ArrayBufferView);
  }
}

// ---- Full AutoencoderKLWan streaming decoder ------------------------------
//
// Two ONNX graphs (decoder_init, decoder_step) share a 32-tensor fp16 cache
// I/O contract. Pipeline:
//   - init(latent_frame_0)                 -> frames[T=1]  + 32 cache_out
//   - step(latent_frame_i, 32 cache_in)    -> frames[T=4]  + 32 cache_out   (x20)
// Total: 1 + 20*4 = 81 output frames. Each call emits frames as
// [1, 3, T, 480, 832] fp16 in [-1, 1] (torch.clamp in the export wrapper).
//
// Cache shapes (captured by the export probe, fixed at default resolution):
// 1.4 GB fp16 total. All 32 entries are carried CPU<->GPU every step today
// because there's no io-binding yet; that's the biggest perf knob left.

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
   *  fires once per frame-group (init: 1 frame, each step: 4 frames). */
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

    // ---- init: frame 0 -----------------------------------------------------
    this.initSession = await createSession(this.cache, this.initFile, [
      "webgpu",
      "wasm",
    ]);
    let caches: Uint16Array[];
    try {
      const frame0 = sliceLatentFrame(latents, 0, C, T, H, W);
      const feeds: Record<string, ort.Tensor> = {
        latent: new ort.Tensor("float16", frame0, [1, C, 1, H, W]),
      };
      const results = await this.initSession.run(feeds);
      const frames = results["frames"];
      if (!frames) throw new Error("decoder_init produced no frames output");
      writeFramesTransposed(out, copyF16Bits(frames.data as ArrayBufferView), 1, 0, outH, outW);
      framesDone = 1;
      onProgress?.(framesDone, F);
      // Harvest cache_out_NN tensors as Uint16Array (fp16 bits).
      caches = WAN_VAE_CACHE_SHAPES.map((_, i) => {
        const t = results[cacheName("out", i)];
        if (!t) throw new Error(`decoder_init missing ${cacheName("out", i)}`);
        return copyF16Bits(t.data as ArrayBufferView);
      });
    } finally {
      await this.initSession.release();
      this.initSession = null;
    }

    // ---- step: frames 1..20 (4 frames each) --------------------------------
    this.stepSession = await createSession(this.cache, this.stepFile, [
      "webgpu",
      "wasm",
    ]);
    try {
      for (let t = 1; t < T; t++) {
        const frameT = sliceLatentFrame(latents, t, C, T, H, W);
        const feeds: Record<string, ort.Tensor> = {
          latent: new ort.Tensor("float16", frameT, [1, C, 1, H, W]),
        };
        for (let i = 0; i < WAN_VAE_CACHE_SHAPES.length; i++) {
          feeds[cacheName("in", i)] = new ort.Tensor(
            "float16",
            caches[i],
            WAN_VAE_CACHE_SHAPES[i].slice(),
          );
        }
        const results = await this.stepSession.run(feeds);
        const frames = results["frames"];
        if (!frames) throw new Error("decoder_step produced no frames output");
        writeFramesTransposed(out, copyF16Bits(frames.data as ArrayBufferView), 4, framesDone, outH, outW);
        framesDone += 4;
        // Update caches for next iteration.
        for (let i = 0; i < WAN_VAE_CACHE_SHAPES.length; i++) {
          const co = results[cacheName("out", i)];
          if (!co) throw new Error(`decoder_step missing ${cacheName("out", i)}`);
          caches[i] = copyF16Bits(co.data as ArrayBufferView);
        }
        onProgress?.(framesDone, F);
      }
    } finally {
      await this.stepSession.release();
      this.stepSession = null;
    }

    if (framesDone !== F) {
      throw new Error(`WanVaeDecoder: expected ${F} frames, got ${framesDone}`);
    }
    return out;
  }
}
