// SD1.5 VAE encoder. Wraps an ORT-web InferenceSession around
// vae_encoder/model.onnx (~68 MB, fp16) and exposes a single encode() call
// that turns a user-supplied RGB image into a clean (noise-free) latent in
// the same model-space representation the rest of the pipeline uses.
//
// Input:
//   sample         float16 [B, 3, H, W]       RGB image, scaled from
//                                              [0, 255] to [-1, 1] (the
//                                              same convention the VAE
//                                              decoder produces on output)
// Output:
//   latent_sample  float16 [B, 4, H/8, W/8]   raw VAE latent (the
//                                              deterministic mean of the
//                                              latent distribution; the
//                                              ONNX export samples-or-mean
//                                              once at export time so we do
//                                              not have to). We then
//                                              multiply by VAE_SCALING_FACTOR
//                                              (0.18215) so the value handed
//                                              back is in the same scaled-
//                                              down "model space" as the
//                                              UNet's input/output latents
//                                              and the random init noise.
//                                              The decoder reverses this by
//                                              dividing by 0.18215 before
//                                              decoding.
//
// Used by:
//   1. image-gen img2img: encode the user's reference image, partial-noise
//      it via DdimScheduler.addNoise, and start the denoise loop from a
//      mid-schedule timestep instead of pure Gaussian noise.
//   2. video-gen (Phase 2, AnimateDiff): encode reference frames into
//      latents for I2V / SparseCtrl-style conditioning. Same call shape;
//      AnimateDiff just batches multiple frames through it.
//
// RAM policy: load -> encode -> release. Same lifecycle as VaeDecoder.
// Activation memory at 512x512 is the inverse of decode (image-space input
// shrinks 8x to latent space) so OOM risk is lower than the decoder, but we
// still release immediately so the encoder is not resident during the
// (much heavier) UNet phase.
//
// fp16 boundary lives at the model edge, identical to unet.ts / vae.ts: the
// caller passes a Float32Array of pixels in [-1, 1], we convert to fp16 for
// the ONNX run, and the returned latent is a Float32Array in pipeline
// (model) space.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { f16ToF32Array, f32ToF16Array } from "./fp16";
import { VAE_SCALING_FACTOR } from "./vae";
import { createSession, type OrtModelFile } from "./ort-helpers";

function defaultProviders(): string[] {
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

/** Convert an RGBA ImageData to a CHW Float32Array in [-1, 1]. The VAE was
 *  trained on (image / 127.5) - 1 normalization (same as the decoder's
 *  output range), so the encoder expects the same. Alpha is dropped. */
export function imageDataToChw(image: ImageData): Float32Array {
  const { width, height, data } = image;
  const planeSize = width * height;
  const chw = new Float32Array(3 * planeSize);
  for (let i = 0; i < planeSize; i++) {
    const o = i * 4;
    chw[i] = data[o] / 127.5 - 1;
    chw[planeSize + i] = data[o + 1] / 127.5 - 1;
    chw[2 * planeSize + i] = data[o + 2] / 127.5 - 1;
  }
  return chw;
}

export interface VaeEncoderOptions {
  /** Latent scaling factor. SD1.5 = 0.18215, SDXL = 0.13025. */
  scalingFactor?: number;
  /** Boundary dtype for the input image (and output latent). nmkd SD1.5
   *  fp16 export uses fp16 throughout; diffusers/optimum SDXL exports
   *  use fp32 even when weights are fp16 internally. Default fp16. */
  boundaryDtype?: "float16" | "float32";
}

export class VaeEncoder {
  private constructor(
    private session: ort.InferenceSession | null,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly scalingFactor: number,
    private readonly boundaryDtype: "float16" | "float32",
  ) {}

  static async load(
    cache: ModelCache,
    file: OrtModelFile,
    options: VaeEncoderOptions = {},
  ): Promise<VaeEncoder> {
    const session = await createSession(cache, file, defaultProviders());

    if (session.inputNames.length === 0) {
      throw new Error("VAE encoder ONNX has no inputs");
    }
    const inputName = session.inputNames[0];
    let outputName = "latent_sample";
    if (!session.outputNames.includes(outputName)) {
      outputName = session.outputNames[0];
    }

    return new VaeEncoder(
      session,
      inputName,
      outputName,
      options.scalingFactor ?? VAE_SCALING_FACTOR,
      options.boundaryDtype ?? "float16",
    );
  }

  /**
   * Encode an RGB image into a clean pipeline-space latent.
   *
   * @param chw       Float32Array length 3*H*W (channels-first, [-1, 1]).
   *                  Use imageDataToChw() to build this from an ImageData.
   * @param imageH    Image height in pixels. Must be a multiple of 8.
   * @param imageW    Image width in pixels. Must be a multiple of 8.
   * @returns         Float32Array length 4 * (imageH/8) * (imageW/8), in
   *                  the scaled-down "model space" the UNet operates in.
   */
  async encode(chw: Float32Array, imageH: number, imageW: number): Promise<Float32Array> {
    if (!this.session) throw new Error("VAE encoder has been released");
    if (imageH % 8 !== 0 || imageW % 8 !== 0) {
      throw new Error(`VAE encoder: image dims must be multiples of 8, got ${imageW}x${imageH}`);
    }
    const expectedIn = 3 * imageH * imageW;
    if (chw.length !== expectedIn) {
      throw new Error(`VAE encoder: chw length ${chw.length} != expected ${expectedIn}`);
    }

    const inputTensor =
      this.boundaryDtype === "float32"
        ? new ort.Tensor("float32", new Float32Array(chw), [1, 3, imageH, imageW])
        : new ort.Tensor("float16", f32ToF16Array(chw), [1, 3, imageH, imageW]);
    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: inputTensor,
    };
    const results = await this.session.run(feeds);
    const out = results[this.outputName];
    if (!out) {
      throw new Error(`VAE encoder did not produce output "${this.outputName}"`);
    }

    const latentH = imageH / 8;
    const latentW = imageW / 8;
    const expectedOutLen = 4 * latentH * latentW;

    // Same Float16Array detection dance as vae.ts: native Float16Array,
    // legacy Uint16Array bit patterns, or fp32. Match by constructor name
    // because the TS lib types lag the runtime.
    let raw: Float32Array;
    const ctorName = (out.data as ArrayLike<number>).constructor.name;
    if (ctorName === "Float16Array") {
      const half = out.data as ArrayLike<number>;
      if (half.length !== expectedOutLen) {
        throw new Error(`VAE encoder output length ${half.length} != expected ${expectedOutLen}`);
      }
      raw = new Float32Array(half);
    } else if (out.type === "float16") {
      const half = out.data as Uint16Array;
      if (half.length !== expectedOutLen) {
        throw new Error(`VAE encoder output length ${half.length} != expected ${expectedOutLen}`);
      }
      raw = f16ToF32Array(half);
    } else {
      const data = out.data as Float32Array;
      if (data.length !== expectedOutLen) {
        throw new Error(`VAE encoder output length ${data.length} != expected ${expectedOutLen}`);
      }
      raw = data;
    }

    // Apply the VAE scaling factor: pipeline-space latents are stored
    // pre-multiplied by this constant (0.18215 for SD1.5, 0.13025 for
    // SDXL family). Without this scaling the addNoise() formula would be
    // off and the denoise loop would diverge from a noise distribution it
    // was never trained on.
    const scaled = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) scaled[i] = raw[i] * this.scalingFactor;
    return scaled;
  }

  async release(): Promise<void> {
    if (!this.session) return;
    await this.session.release();
    this.session = null;
  }
}
