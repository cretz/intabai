// SDXL UNet wrapper. Mirrors web/src/sd15/unet.ts in structure but with the
// SDXL-specific input shape:
//
//   sample                float16 [B, 4, H/8, W/8]   noisy latent
//   timestep              int64   [1] (or float16)   current diffusion step
//   encoder_hidden_states float16 [B, 77, 2048]      concat(CLIP-L, bigG)
//                                                     hidden states (penultimate
//                                                     layer of each)
//   text_embeds           float16 [B, 1280]          pooled output of bigG only
//   time_ids              float16 [B, 6]             [orig_h, orig_w,
//                                                     crop_top, crop_left,
//                                                     target_h, target_w]
//
// Output:
//   out_sample            float16 [B, 4, H/8, W/8]   predicted noise
//
// Latent channel count and the basic add-noise / DDIM math are the same as
// SD1.5 - SDXL just feeds the cross-attention layers a wider hidden state
// (2048-dim concat instead of SD1.5's 768) and adds the two extra
// conditioning tensors. The DdimScheduler from sd15/scheduler.ts works
// unchanged because the alpha schedule is identical (scaled_linear, beta
// 0.00085 -> 0.012).
//
// Same fp16 boundary policy as the SD1.5 wrapper: callers pass Float32Array
// inputs and get Float32Array outputs; this class round-trips through fp16
// only at the model edge.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { f16ToF32Array, f32ToF16Array, f32ToF16Bits } from "../sd15/fp16";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";

/** SDXL latent has 4 channels, same as SD1.5. */
export const SDXL_UNET_LATENT_CHANNELS = 4;

/** SDXL CLIP context length, same 77 as SD1.5. */
export const SDXL_CONTEXT_LENGTH = 77;

/** Concatenated CLIP-L (768) + bigG (1280) penultimate hidden state width. */
export const SDXL_HIDDEN_SIZE = 2048;

/** bigG pooled output width. */
export const SDXL_POOLED_SIZE = 1280;

function defaultProviders(): string[] {
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

export interface SdxlUnetOptions {
  /** Boundary dtype for sample / hidden_states / text_embeds / time_ids
   *  inputs and the out_sample output. Diffusers/optimum SDXL exports
   *  default to fp32 at the boundary even when weights are stored fp16
   *  (segmind-vega is one example: the repo name says "fp16" but it
   *  refers to the weight storage; runtime tensors round-trip fp32).
   *  Other exports (webnn/sdxl-turbo q4f16) use fp16 here. Hardcode per
   *  bundle. */
  boundaryDtype?: "float16" | "float32";
}

export class SdxlUnet {
  private constructor(
    private session: ort.InferenceSession | null,
    private readonly sampleInputName: string,
    private readonly timestepInputName: string,
    private readonly hiddenStatesInputName: string,
    private readonly textEmbedsInputName: string,
    private readonly timeIdsInputName: string,
    private readonly outputName: string,
    private readonly timestepIsInt64: boolean,
    private readonly boundaryDtype: "float16" | "float32",
  ) {}

  static async load(
    cache: ModelCache,
    model: OrtModelFile,
    options: SdxlUnetOptions = {},
  ): Promise<SdxlUnet> {
    const session = await createSession(cache, model, defaultProviders());

    const inputNames = session.inputNames;
    const findInput = (...candidates: string[]): string => {
      for (const c of candidates) {
        if (inputNames.includes(c)) return c;
      }
      for (const name of inputNames) {
        for (const c of candidates) {
          if (name.toLowerCase().includes(c.toLowerCase())) return name;
        }
      }
      throw new Error(
        `SDXL UNet ONNX is missing an input matching ${candidates.join(" / ")}; have [${inputNames.join(", ")}]`,
      );
    };
    const sampleInputName = findInput("sample");
    const timestepInputName = findInput("timestep");
    const hiddenStatesInputName = findInput("encoder_hidden_states", "encoder_hidden_state");
    const textEmbedsInputName = findInput("text_embeds");
    const timeIdsInputName = findInput("time_ids");

    let outputName = "out_sample";
    if (!session.outputNames.includes(outputName)) {
      outputName = session.outputNames[0];
    }

    // Diffusers/optimum fp32-boundary exports (segmind-vega) use int64 for
    // timestep. The webnn q4f16 exports use fp16 everywhere including
    // timestep. Derive from the declared boundary dtype.
    const timestepIsInt64 = (options.boundaryDtype ?? "float16") === "float32";

    return new SdxlUnet(
      session,
      sampleInputName,
      timestepInputName,
      hiddenStatesInputName,
      textEmbedsInputName,
      timeIdsInputName,
      outputName,
      timestepIsInt64,
      options.boundaryDtype ?? "float16",
    );
  }

  /**
   * One UNet forward pass for a single conditioning. Caller invokes twice
   * per denoising step (uncond, cond) and combines via the scheduler's
   * CFG helper. Same shape as the SD1.5 Unet.predictNoise but with the
   * extra SDXL conditioning tensors.
   *
   * @param latent          Float32Array length 4*H*W (latent dims).
   * @param timestep        Integer diffusion timestep.
   * @param hiddenStates    Float32Array length 77*2048 (concat penultimate
   *                        from CLIP-L and bigG).
   * @param pooledTextEmbeds Float32Array length 1280 (bigG pooled).
   * @param timeIds         Float32Array length 6.
   * @param spatialH        Latent height (image height / 8).
   * @param spatialW        Latent width  (image width  / 8).
   */
  async predictNoise(
    latent: Float32Array,
    timestep: number,
    hiddenStates: Float32Array,
    pooledTextEmbeds: Float32Array,
    timeIds: Float32Array,
    spatialH: number,
    spatialW: number,
  ): Promise<Float32Array> {
    if (!this.session) throw new Error("SDXL UNet has been released");

    const expectedLatentLen = SDXL_UNET_LATENT_CHANNELS * spatialH * spatialW;
    if (latent.length !== expectedLatentLen) {
      throw new Error(
        `latent length ${latent.length} != expected ${expectedLatentLen} for ${spatialH}x${spatialW}`,
      );
    }
    if (hiddenStates.length !== SDXL_CONTEXT_LENGTH * SDXL_HIDDEN_SIZE) {
      throw new Error(
        `hiddenStates length ${hiddenStates.length} != expected ${SDXL_CONTEXT_LENGTH * SDXL_HIDDEN_SIZE}`,
      );
    }
    if (pooledTextEmbeds.length !== SDXL_POOLED_SIZE) {
      throw new Error(
        `pooledTextEmbeds length ${pooledTextEmbeds.length} != expected ${SDXL_POOLED_SIZE}`,
      );
    }
    if (timeIds.length !== 6) {
      throw new Error(`timeIds length ${timeIds.length} != expected 6`);
    }

    // Build inputs in the boundary dtype declared on the export. fp32 for
    // diffusers/optimum exports (segmind-vega), fp16 for q4f16 exports
    // (webnn/sdxl-turbo). Caller picks via SdxlUnetOptions.boundaryDtype.
    const fp32 = this.boundaryDtype === "float32";
    const sampleTensor = fp32
      ? new ort.Tensor("float32", new Float32Array(latent), [
          1,
          SDXL_UNET_LATENT_CHANNELS,
          spatialH,
          spatialW,
        ])
      : new ort.Tensor("float16", f32ToF16Array(latent), [
          1,
          SDXL_UNET_LATENT_CHANNELS,
          spatialH,
          spatialW,
        ]);
    const hiddenStatesTensor = fp32
      ? new ort.Tensor("float32", new Float32Array(hiddenStates), [
          1,
          SDXL_CONTEXT_LENGTH,
          SDXL_HIDDEN_SIZE,
        ])
      : new ort.Tensor("float16", f32ToF16Array(hiddenStates), [
          1,
          SDXL_CONTEXT_LENGTH,
          SDXL_HIDDEN_SIZE,
        ]);
    const textEmbedsTensor = fp32
      ? new ort.Tensor("float32", new Float32Array(pooledTextEmbeds), [1, SDXL_POOLED_SIZE])
      : new ort.Tensor("float16", f32ToF16Array(pooledTextEmbeds), [1, SDXL_POOLED_SIZE]);
    const timeIdsTensor = fp32
      ? new ort.Tensor("float32", new Float32Array(timeIds), [1, 6])
      : new ort.Tensor("float16", f32ToF16Array(timeIds), [1, 6]);

    let timestepTensor: ort.Tensor;
    if (this.timestepIsInt64) {
      timestepTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(timestep)]), [1]);
    } else if (fp32) {
      timestepTensor = new ort.Tensor("float32", Float32Array.from([timestep]), [1]);
    } else {
      timestepTensor = new ort.Tensor("float16", Uint16Array.from([f32ToF16Bits(timestep)]), [1]);
    }

    const feeds: Record<string, ort.Tensor> = {
      [this.sampleInputName]: sampleTensor,
      [this.timestepInputName]: timestepTensor,
      [this.hiddenStatesInputName]: hiddenStatesTensor,
      [this.textEmbedsInputName]: textEmbedsTensor,
      [this.timeIdsInputName]: timeIdsTensor,
    };

    const results = await this.session.run(feeds);
    const out = results[this.outputName];
    if (!out) {
      throw new Error(`SDXL UNet did not produce output "${this.outputName}"`);
    }
    const ctorName = (out.data as ArrayLike<number>).constructor.name;
    if (ctorName === "Float16Array") {
      const half = out.data as ArrayLike<number>;
      if (half.length !== expectedLatentLen) {
        throw new Error(`SDXL UNet output length ${half.length} != expected ${expectedLatentLen}`);
      }
      return new Float32Array(half);
    } else if (out.type === "float16") {
      const half = out.data as Uint16Array;
      if (half.length !== expectedLatentLen) {
        throw new Error(`SDXL UNet output length ${half.length} != expected ${expectedLatentLen}`);
      }
      return f16ToF32Array(half);
    } else {
      const data = out.data as Float32Array;
      if (data.length !== expectedLatentLen) {
        throw new Error(`SDXL UNet output length ${data.length} != expected ${expectedLatentLen}`);
      }
      return new Float32Array(data);
    }
  }

  async release(): Promise<void> {
    if (!this.session) return;
    await this.session.release();
    this.session = null;
  }
}

/** Build the SDXL `time_ids` tensor for a txt2img / img2img run with no
 *  cropping. Layout (per diffusers SDXL pipeline):
 *  [original_h, original_w, crop_top, crop_left, target_h, target_w] */
export function buildTimeIds(width: number, height: number): Float32Array {
  return new Float32Array([height, width, 0, 0, height, width]);
}
