// SD1.5 UNet wrapper. The big one. Wraps an ORT-web InferenceSession around
// unet/model.onnx (~1.72 GB) from tlwu/stable-diffusion-v1-5-onnxruntime
// and exposes a single predictNoise() call that the denoising loop drives
// once per (step, conditioning) pair.
//
// Inputs (standard diffusers SD1.5 ONNX export):
//   sample               float32 [B, 4, H/8, W/8]   the noisy latent
//   timestep             float32 [1] or int64 [1]   the current diffusion timestep
//   encoder_hidden_states float32 [B, 77, 768]      CLIP text embeddings
//
// Output:
//   out_sample           float32 [B, 4, H/8, W/8]   predicted noise
//
// Naming varies slightly across exports; we discover input/output names from
// the session metadata at load time and key the feed dict by what we find,
// the same way text-encoder.ts does.
//
// CFG mode: there are two valid approaches, with a real tradeoff:
//
//   (a) Batched: batch=2, run UNet ONCE per step with [uncond_emb, cond_emb]
//       stacked. ~1.5x compute, 2x GPU activation memory.
//   (b) Serial:  batch=1, run UNet TWICE per step. ~2x compute, 1x GPU
//       activation memory.
//
// We default to (b) because UNet activations on top of the 1.7 GB weight
// buffer are what makes mobile WebGPU OOM, and the user has been clear that
// in-browser RAM headroom matters more than absolute speed. (a) can be
// added later as a "fast mode" toggle once we know our memory budget.
//
// All compute happens through ORT. The wrapper holds no state between
// predictNoise calls beyond the session itself.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { f16ToF32Array, f32ToF16Array, f32ToF16Bits } from "./fp16";
import { createSession, type OrtModelFile } from "./ort-helpers";

/** SD1.5 latent has 4 channels. Hard-coded by the trained UNet. */
export const UNET_LATENT_CHANNELS = 4;

/** Latent height/width in pixels = image height/width / 8. */
export const VAE_SCALE_FACTOR = 8;

function defaultProviders(): string[] {
  // UNet WANTS WebGPU. The whole point of running this in a browser is the
  // GPU acceleration. If WebGPU is unavailable we fall through to wasm so
  // the user gets *some* result, but that path is unusably slow for SD1.5
  // UNet (multi-minute per step). Capability checks in the tool's main.ts
  // should warn the user before reaching this point.
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

export class Unet {
  private constructor(
    private session: ort.InferenceSession | null,
    private readonly sampleInputName: string,
    private readonly timestepInputName: string,
    private readonly embeddingInputName: string,
    private readonly outputName: string,
    private readonly timestepIsInt64: boolean,
  ) {}

  static async load(cache: ModelCache, model: OrtModelFile): Promise<Unet> {
    // External-data layout is required for the SD1.5 UNet (~1.72 GB):
    // loading a monolithic ONNX of that size overflows the wasm 32-bit
    // linear memory cap during session create with std::bad_alloc, because
    // the parser materializes initializers in wasm space before handing
    // them to the WebGPU EP. External data lets the WebGPU EP claim weight
    // tensors directly into GPU buffers. createSession() handles the blob
    // URL plumbing for both layouts; the SD1.5 nmkd UNet always passes
    // {graph, data, dataPath: "weights.pb"} here.
    const session = await createSession(cache, model, defaultProviders());

    // Discover input names. SD1.5 exports universally use "sample",
    // "timestep", and "encoder_hidden_states", but we look them up by
    // pattern so a slightly differently-named finetune still works.
    const inputNames = session.inputNames;
    const findInput = (...candidates: string[]): string => {
      for (const c of candidates) {
        if (inputNames.includes(c)) return c;
      }
      // Fallback: substring match (case-insensitive).
      for (const name of inputNames) {
        for (const c of candidates) {
          if (name.toLowerCase().includes(c.toLowerCase())) return name;
        }
      }
      throw new Error(
        `UNet ONNX is missing an input matching ${candidates.join(" / ")}; have [${inputNames.join(", ")}]`,
      );
    };
    const sampleInputName = findInput("sample");
    const timestepInputName = findInput("timestep");
    const embeddingInputName = findInput("encoder_hidden_states", "encoder_hidden_state");

    let outputName = "out_sample";
    if (!session.outputNames.includes(outputName)) {
      outputName = session.outputNames[0];
    }

    const meta = (
      session as unknown as {
        inputMetadata?: Record<string, { type?: string }>;
      }
    ).inputMetadata;
    const timestepType = meta?.[timestepInputName]?.type;
    const timestepIsInt64 = timestepType === "int64";

    return new Unet(
      session,
      sampleInputName,
      timestepInputName,
      embeddingInputName,
      outputName,
      timestepIsInt64,
    );
  }

  /**
   * Run one UNet forward pass for a single conditioning. Caller invokes
   * twice per denoising step (uncond, cond) and combines via the
   * scheduler's CFG helper.
   *
   * @param latent     Float32Array length C*H*W (C=4) for batch=1.
   * @param timestep   Integer diffusion timestep.
   * @param embedding  Float32Array length 77*768 from the text encoder.
   * @param spatialH   Latent height (image height / 8). Typical: 64.
   * @param spatialW   Latent width  (image width  / 8). Typical: 64.
   */
  async predictNoise(
    latent: Float32Array,
    timestep: number,
    embedding: Float32Array,
    spatialH: number,
    spatialW: number,
  ): Promise<Float32Array> {
    if (!this.session) throw new Error("UNet has been released");

    const expectedLatentLen = UNET_LATENT_CHANNELS * spatialH * spatialW;
    if (latent.length !== expectedLatentLen) {
      throw new Error(
        `latent length ${latent.length} != expected ${expectedLatentLen} for ${spatialH}x${spatialW}`,
      );
    }
    if (embedding.length !== 77 * 768) {
      throw new Error(`embedding length ${embedding.length} != expected ${77 * 768}`);
    }

    // nmkd's SD1.5 export is fp16 throughout: the UNet declares fp16
    // sample, fp16 encoder_hidden_states, and fp16 timestep inputs (and
    // fp16 output). Our scheduler / CFG / VAE-decode pipeline operates in
    // fp32 because the rounding error of doing those at half precision
    // accumulates visibly over many steps. So we round-trip through fp16
    // only at the model boundary, on each predictNoise call.
    const sampleTensor = new ort.Tensor("float16", f32ToF16Array(latent), [
      1,
      UNET_LATENT_CHANNELS,
      spatialH,
      spatialW,
    ]);

    const embeddingTensor = new ort.Tensor("float16", f32ToF16Array(embedding), [1, 77, 768]);

    // Timestep tensor: dtype probed at load time. nmkd's UNet uses fp16
    // here too; older exports used int64 or fp32, so we keep the dtype
    // probe and add the fp16 path.
    let timestepTensor: ort.Tensor;
    if (this.timestepIsInt64) {
      timestepTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(timestep)]), [1]);
    } else {
      timestepTensor = new ort.Tensor("float16", Uint16Array.from([f32ToF16Bits(timestep)]), [1]);
    }

    const feeds: Record<string, ort.Tensor> = {
      [this.sampleInputName]: sampleTensor,
      [this.timestepInputName]: timestepTensor,
      [this.embeddingInputName]: embeddingTensor,
    };

    const results = await this.session.run(feeds);
    const out = results[this.outputName];
    if (!out) {
      throw new Error(`UNet did not produce output "${this.outputName}"`);
    }
    // Convert the output to a fresh Float32Array. Three possible shapes
    // depending on browser + ORT-web version:
    //   1. Native Float16Array (modern Chrome/Edge with TC39 Float16Array
    //      shipped). out.data[i] already returns a JS number; copying via
    //      new Float32Array(half) iterates and converts cleanly.
    //   2. Uint16Array of raw fp16 bit patterns (older browsers, ORT-web
    //      polyfill). Needs our hand-rolled f16BitsToF32 over each entry.
    //   3. Float32Array (would only happen if the UNet were exported fp32).
    //
    // Detect by constructor name because TypeScript's lib.dom.d.ts in the
    // versions we're on doesn't yet include Float16Array in its type
    // hierarchy.
    const ctorName = (out.data as ArrayLike<number>).constructor.name;
    let result: Float32Array;
    if (ctorName === "Float16Array") {
      const half = out.data as ArrayLike<number>;
      if (half.length !== expectedLatentLen) {
        throw new Error(`UNet output length ${half.length} != expected ${expectedLatentLen}`);
      }
      result = new Float32Array(half);
    } else if (out.type === "float16") {
      const half = out.data as Uint16Array;
      if (half.length !== expectedLatentLen) {
        throw new Error(`UNet output length ${half.length} != expected ${expectedLatentLen}`);
      }
      result = f16ToF32Array(half);
    } else {
      const data = out.data as Float32Array;
      if (data.length !== expectedLatentLen) {
        throw new Error(`UNet output length ${data.length} != expected ${expectedLatentLen}`);
      }
      result = new Float32Array(data);
    }
    return result;
  }

  async release(): Promise<void> {
    if (!this.session) return;
    await this.session.release();
    this.session = null;
  }
}
