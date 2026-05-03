// LTX spatial latent upscaler. Single 252 MB fp16 ONNX session that
// doubles spatial dims and keeps temporal dims unchanged.
//
//   in:  [1, 128, T, H, W] fp16 (un-normalized latent space)
//   out: [1, 128, T, 2H, 2W] fp16
//
// Sized small enough to load + run + release as one session per call;
// no per-block streaming.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { copyF16Bits } from "../sd15/fp16";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { LTX_LATENT_CHANNELS, type LtxLatentShape } from "./patchifier";
import { assertArrayType, assertF16View, assertOrtOutput } from "./validate";

export interface LtxUpscalerArgs {
  /** [1, 128, T, H, W] fp16 un-normalized latent. */
  latent: Uint16Array;
  shape: LtxLatentShape;
  signal?: AbortSignal;
}

export interface LtxUpscalerResult {
  /** [1, 128, T, 2H, 2W] fp16 un-normalized latent. */
  latent: Uint16Array;
  shape: LtxLatentShape;
  loadMs: number;
  runMs: number;
}

export class LtxUpscaler {
  constructor(
    private readonly cache: ModelCache,
    private readonly file: OrtModelFile,
    private readonly providers: ("webgpu" | "wasm")[] = ["webgpu", "wasm"],
  ) {}

  async run(args: LtxUpscalerArgs): Promise<LtxUpscalerResult> {
    const { latent, shape, signal } = args;
    const expected = LTX_LATENT_CHANNELS * shape.t * shape.h * shape.w;
    assertArrayType("upscaler.latent", latent, Uint16Array);
    if (latent.length !== expected) {
      throw new Error(
        `upscaler: expected ${expected} fp16 vals, got ${latent.length}`,
      );
    }

    signal?.throwIfAborted();
    const tLoad = performance.now();
    const session = await createSession(this.cache, this.file, this.providers);
    const loadMs = performance.now() - tLoad;

    let out: Uint16Array;
    let outShape: LtxLatentShape;
    let runMs = 0;
    try {
      const feed = new ort.Tensor("float16", latent, [
        1,
        LTX_LATENT_CHANNELS,
        shape.t,
        shape.h,
        shape.w,
      ]);
      const tRun = performance.now();
      const r = await session.run({ latent: feed });
      runMs = performance.now() - tRun;
      const upsampled = r.upsampled ?? r[Object.keys(r)[0]];
      assertOrtOutput("upscaler.upsampled", upsampled, {
        type: "float16",
        dims: [1, LTX_LATENT_CHANNELS, shape.t, shape.h * 2, shape.w * 2],
      });
      const dims = upsampled.dims as number[];
      const outCount = LTX_LATENT_CHANNELS * dims[2] * dims[3] * dims[4];
      const view = assertF16View("upscaler.upsampled.data", upsampled.data, outCount);
      out = copyF16Bits(view);
      outShape = { t: dims[2], h: dims[3], w: dims[4] };
      feed.dispose?.();
      for (const k in r) (r[k] as ort.Tensor).dispose?.();
    } finally {
      await session.release();
    }

    return { latent: out, shape: outShape, loadMs, runMs };
  }
}
