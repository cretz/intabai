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
