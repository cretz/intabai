// LightTAE (Wan 2.2) VAE encoder. Companion to vae.ts (decoder), used
// only on the I2V path: encodes the user's optional input image to a
// single latent frame which the denoise loop pins as frame 0 of the
// latent buffer.
//
// Input  shape: [1, T=4, 3, pixelH, pixelW] fp16 in [0, 1].
//                T=4 because the encoder's temporal pool stride is 4 and
//                T must be a multiple of 4. For single-image conditioning
//                the caller repeats the same image 4x (matches the
//                reference encode_video last-frame-repeat pad).
// Output shape: [1, T_lat=1, 48, latentH, latentW] fp16 (raw latents).
//                Caller must per-channel normalize via VAE_LATENTS_MEAN/STD
//                in generate.ts before splicing into the transformer's
//                normalized-space latent buffer.
//
// Validated 2026-05-01: ONNX vs PyTorch parity max abs diff 9.77e-4;
// real-image round-trip 26.4 dB PSNR with [0,1] input range.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits } from "../sd15/fp16";

import type { FastwanShape } from "./transformer";

/** Number of input frames the encoder ONNX is exported with. T must be a
 *  multiple of 4. For I2V conditioning we always feed 4 copies of the
 *  same image so the export matches inference. */
export const VAE_ENCODER_INPUT_FRAMES = 4;

/** Single output latent frame. T_in / 4 = 1. */
export const VAE_ENCODER_OUTPUT_LATENT_FRAMES = 1;

export class VaeEncoder {
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

  /** Encode RGB frames (fp16 bits, [1, 4, 3, pixelH, pixelW] NTCHW, [0,1])
   *  to raw latents (fp16 bits, [1, 1, 48, latentH, latentW] NTCHW). */
  async encode(frames: Uint16Array): Promise<Uint16Array> {
    if (!this.session) throw new Error("VaeEncoder.load() must be called first");
    const expected = VAE_ENCODER_INPUT_FRAMES * 3 * this.shape.pixelH * this.shape.pixelW;
    if (frames.length !== expected) {
      throw new Error(
        `VaeEncoder.encode: expected ${expected} fp16 elements, got ${frames.length}`,
      );
    }
    const feeds: Record<string, ort.Tensor> = {
      frames: new ort.Tensor("float16", frames, [
        1,
        VAE_ENCODER_INPUT_FRAMES,
        3,
        this.shape.pixelH,
        this.shape.pixelW,
      ]),
    };
    const results = await this.session.run(feeds);
    const key = "latents" in results ? "latents" : Object.keys(results)[0];
    const out = results[key];
    if (!out) throw new Error("vae_encoder produced no output");
    return copyF16Bits(out.data as ArrayBufferView);
  }
}
