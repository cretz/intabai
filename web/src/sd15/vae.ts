// SD1.5 VAE decoder. Wraps an ORT-web InferenceSession around
// vae_decoder/model.onnx (~99 MB, fp16) and exposes a single decode() call
// that turns the final denoised latent into an RGB image.
//
// Input:
//   latent_sample  float16 [B, 4, H/8, W/8]   the final denoised latent,
//                                              scaled by 1 / 0.18215 before
//                                              being fed in (this is the
//                                              SD1.5 VAE scaling factor and
//                                              is hard-coded into the model)
// Output:
//   sample         float16 [B, 3, H, W]       RGB image in the range
//                                              roughly [-1, 1]; we scale to
//                                              [0, 1] then to 8-bit
//                                              [0, 255] and pack into
//                                              RGBA ImageData
//
// The VAE encoder lives in a separate file when img2img lands; for now we
// only need the decoder. Same fp16 boundary as unet.ts: scheduler / our
// post-processing math stay in fp32, only the model boundary is fp16.
//
// RAM policy: load -> decode -> release. The VAE decoder is small (~99 MB
// weights) but the activations during decode at 512x512 are large because
// the spatial dims explode 8x. Worth releasing immediately after the
// single decode call so we don't hold the GPU memory while the user looks
// at the image.
//
// Tiled decoding: at 384x384+ on at least one mobile WebGPU stack the
// monolithic decode triggers a GPU OOM / driver crash (screen flash, then
// hung tab). Workaround: split the latent into 32x32 tiles with a 4-latent
// overlap, decode each tile separately, and stitch the resulting image
// tiles in float space using a linear-ramp feather mask. Activation memory
// is then bounded by the tile size, not the full output. Confirmed working
// at 256x256 unsplit on the same device, so 32x32 tiles (= 256x256 image
// per tile) sit safely under the ceiling. Diffusers' AutoencoderKL.tiled_decode
// is the reference for the algorithm; ours is hand-rolled for ORT-web.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { f16ToF32Array, f32ToF16Array } from "./fp16";
import { createSession, type OrtModelFile } from "./ort-helpers";

/** SD1.5 VAE latent scaling factor. Hard-coded by the trained VAE; the
 *  network was trained against latents pre-divided by this constant, so
 *  we have to multiply by 1/scale before decoding. SDXL uses a different
 *  value (0.13025); pass via VaeDecoder constructor option for that family. */
export const VAE_SCALING_FACTOR = 0.18215;
export const SDXL_VAE_SCALING_FACTOR = 0.13025;

/** Latent-space tile size for tiled decoding. 32 latent = 256-image tile,
 *  confirmed to fit on the constrained mobile WebGPU stack. */
const TILE_LATENT = 32;
/** Latent-space overlap between adjacent tiles. Image-space feather is
 *  TILE_OVERLAP * 8 = 32 pixels. Smaller overlap = sharper but more visible
 *  seams; larger = smoother but more redundant work. */
const TILE_OVERLAP = 4;

export interface DecodeOptions {
  /** Force tiled decoding even when the latent fits in a single tile.
   *  When false, tiling is used only when latent dims exceed TILE_LATENT. */
  tiled?: boolean;
  /** Called once per tile completed during a tiled decode. Lets the UI
   *  advance a progress bar across the (potentially many) tile decodes. */
  onTileProgress?: (tileIdx: number, tileCount: number) => void;
}

/** Compute tile start positions along one axis for a given latent dim. */
function tileStarts(dim: number): number[] {
  if (dim <= TILE_LATENT) return [0];
  const stride = TILE_LATENT - TILE_OVERLAP;
  const n = Math.ceil((dim - TILE_LATENT) / stride) + 1;
  const starts: number[] = [];
  for (let i = 0; i < n; i++) {
    starts.push(Math.round((i * (dim - TILE_LATENT)) / (n - 1)));
  }
  return starts;
}

/** How many VAE invocations a decode at the given latent dims will perform.
 *  Used by callers to size their progress budget. */
export function vaeTileCount(spatialH: number, spatialW: number, forceTiled: boolean): number {
  if (!forceTiled && spatialH <= TILE_LATENT && spatialW <= TILE_LATENT) {
    return 1;
  }
  return tileStarts(spatialH).length * tileStarts(spatialW).length;
}

function defaultProviders(): string[] {
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

export interface VaeDecoderOptions {
  /** Latent scaling factor. SD1.5 = 0.18215, SDXL = 0.13025. */
  scalingFactor?: number;
  /** Boundary dtype for the input latent (and the output sample). nmkd's
   *  SD1.5 fp16 export uses fp16 throughout; diffusers/optimum SDXL
   *  exports keep the boundary as fp32 even when weights are stored fp16
   *  (segmind-vega is one example). Defaults to fp16 to match SD1.5. */
  boundaryDtype?: "float16" | "float32";
}

export class VaeDecoder {
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
    options: VaeDecoderOptions = {},
  ): Promise<VaeDecoder> {
    const session = await createSession(cache, file, defaultProviders());

    if (session.inputNames.length === 0) {
      throw new Error("VAE decoder ONNX has no inputs");
    }
    const inputName = session.inputNames[0];
    let outputName = "sample";
    if (!session.outputNames.includes(outputName)) {
      outputName = session.outputNames[0];
    }

    return new VaeDecoder(
      session,
      inputName,
      outputName,
      options.scalingFactor ?? VAE_SCALING_FACTOR,
      options.boundaryDtype ?? "float16",
    );
  }

  /**
   * Decode a denoised latent into an 8-bit RGBA ImageData.
   *
   * @param latent   Float32Array length 4*H*W (latent dims, NOT image dims).
   * @param spatialH Latent height (image height / 8).
   * @param spatialW Latent width  (image width  / 8).
   * @param options  Tiling control. See DecodeOptions.
   */
  async decode(
    latent: Float32Array,
    spatialH: number,
    spatialW: number,
    options: DecodeOptions = {},
  ): Promise<ImageData> {
    if (!this.session) throw new Error("VAE decoder has been released");

    const expectedLen = 4 * spatialH * spatialW;
    if (latent.length !== expectedLen) {
      throw new Error(`latent length ${latent.length} != expected ${expectedLen}`);
    }

    const useTiling = options.tiled === true || spatialH > TILE_LATENT || spatialW > TILE_LATENT;

    if (!useTiling) {
      const chw = await this.runOnce(latent, spatialH, spatialW);
      options.onTileProgress?.(1, 1);
      return chwToImageData(chw, spatialH * 8, spatialW * 8);
    }

    return this.decodeTiled(latent, spatialH, spatialW, options);
  }

  /** Run one VAE invocation on a latent slice and return its float CHW
   *  output (image space, ~[-1, 1]). Caller handles fp16 boundary, scaling,
   *  and reshape. */
  private async runOnce(
    latent: Float32Array,
    spatialH: number,
    spatialW: number,
  ): Promise<Float32Array> {
    // Apply the VAE scaling factor (latents are stored pre-multiplied by
    // this constant; the VAE was trained against the scaled version, so
    // we have to divide by it before decoding). 0.18215 for SD1.5,
    // 0.13025 for SDXL family.
    const scaled = new Float32Array(latent.length);
    const inv = 1.0 / this.scalingFactor;
    for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] * inv;

    // Boundary dtype: nmkd's SD1.5 fp16 export uses fp16 here; diffusers
    // SDXL exports use fp32 even when weights are fp16 internally. Caller
    // picks via VaeDecoderOptions.boundaryDtype.
    const inputTensor =
      this.boundaryDtype === "float32"
        ? new ort.Tensor("float32", scaled, [1, 4, spatialH, spatialW])
        : new ort.Tensor("float16", f32ToF16Array(scaled), [1, 4, spatialH, spatialW]);

    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: inputTensor,
    };
    const results = await this.session!.run(feeds);
    const out = results[this.outputName];
    if (!out) {
      throw new Error(`VAE decoder did not produce output "${this.outputName}"`);
    }

    const imageH = spatialH * 8;
    const imageW = spatialW * 8;
    const expectedOutLen = 3 * imageH * imageW;

    // Same Float16Array detection dance as unet.ts. Modern browsers ship
    // a native Float16Array that already exposes decoded JS numbers; older
    // ones surface a Uint16Array of raw bit patterns we have to decode by
    // hand. Match by constructor name because TS lib types lag.
    const ctorName = (out.data as ArrayLike<number>).constructor.name;
    if (ctorName === "Float16Array") {
      const half = out.data as ArrayLike<number>;
      if (half.length !== expectedOutLen) {
        throw new Error(`VAE output length ${half.length} != expected ${expectedOutLen}`);
      }
      return new Float32Array(half);
    } else if (out.type === "float16") {
      const half = out.data as Uint16Array;
      if (half.length !== expectedOutLen) {
        throw new Error(`VAE output length ${half.length} != expected ${expectedOutLen}`);
      }
      return f16ToF32Array(half);
    } else {
      const data = out.data as Float32Array;
      if (data.length !== expectedOutLen) {
        throw new Error(`VAE output length ${data.length} != expected ${expectedOutLen}`);
      }
      return data;
    }
  }

  /** Tiled decode path. Splits the latent into overlapping TILE_LATENT
   *  squares, decodes each separately, and stitches them in float-image
   *  space using a per-tile linear-ramp feather mask. */
  private async decodeTiled(
    latent: Float32Array,
    spatialH: number,
    spatialW: number,
    options: DecodeOptions,
  ): Promise<ImageData> {
    const yStarts = tileStarts(spatialH);
    const xStarts = tileStarts(spatialW);
    const tileCount = yStarts.length * xStarts.length;
    const imageH = spatialH * 8;
    const imageW = spatialW * 8;
    const planeSize = imageH * imageW;

    // Float accumulator (RGB, channels-first like the VAE output) plus a
    // single-channel weight buffer. Stitching happens in float space; the
    // RGBA conversion runs once at the very end after dividing by weights.
    const accChw = new Float32Array(3 * planeSize);
    const weights = new Float32Array(planeSize);

    // Image-space feather width at each interior tile boundary. Smaller =
    // sharper / faster, larger = smoother / heavier blend.
    const blendImg = TILE_OVERLAP * 8;

    let tileIdx = 0;
    for (let yi = 0; yi < yStarts.length; yi++) {
      const ly = yStarts[yi];
      const isFirstY = yi === 0;
      const isLastY = yi === yStarts.length - 1;
      for (let xi = 0; xi < xStarts.length; xi++) {
        const lx = xStarts[xi];
        const isFirstX = xi === 0;
        const isLastX = xi === xStarts.length - 1;

        // Slice a [4, TILE_LATENT, TILE_LATENT] sub-latent out of the full
        // latent (channels-first layout: 4 planes of spatialH*spatialW).
        const tileLatent = new Float32Array(4 * TILE_LATENT * TILE_LATENT);
        for (let c = 0; c < 4; c++) {
          for (let dy = 0; dy < TILE_LATENT; dy++) {
            const srcRow = c * spatialH * spatialW + (ly + dy) * spatialW + lx;
            const dstRow = c * TILE_LATENT * TILE_LATENT + dy * TILE_LATENT;
            for (let dx = 0; dx < TILE_LATENT; dx++) {
              tileLatent[dstRow + dx] = latent[srcRow + dx];
            }
          }
        }

        const tileChw = await this.runOnce(tileLatent, TILE_LATENT, TILE_LATENT);
        const tileImg = TILE_LATENT * 8;
        const tileImgY = ly * 8;
        const tileImgX = lx * 8;

        // Build per-row and per-col ramp weights for this tile. Edge tiles
        // ramp only on their interior side; otherwise no contribution would
        // exist at the very first/last image rows.
        const vRamp = ramp1D(tileImg, blendImg, isFirstY, isLastY);
        const hRamp = ramp1D(tileImg, blendImg, isFirstX, isLastX);

        const tilePlane = tileImg * tileImg;
        for (let ty = 0; ty < tileImg; ty++) {
          const oy = tileImgY + ty;
          if (oy < 0 || oy >= imageH) continue;
          const wRow = vRamp[ty];
          for (let tx = 0; tx < tileImg; tx++) {
            const ox = tileImgX + tx;
            if (ox < 0 || ox >= imageW) continue;
            const w = wRow * hRamp[tx];
            const oIdx = oy * imageW + ox;
            const tIdx = ty * tileImg + tx;
            accChw[oIdx] += tileChw[tIdx] * w;
            accChw[planeSize + oIdx] += tileChw[tilePlane + tIdx] * w;
            accChw[2 * planeSize + oIdx] += tileChw[2 * tilePlane + tIdx] * w;
            weights[oIdx] += w;
          }
        }

        tileIdx++;
        options.onTileProgress?.(tileIdx, tileCount);
      }
    }

    // Normalize by accumulated weight, then convert CHW float -> RGBA u8.
    for (let i = 0; i < planeSize; i++) {
      const w = weights[i];
      if (w > 0) {
        accChw[i] /= w;
        accChw[planeSize + i] /= w;
        accChw[2 * planeSize + i] /= w;
      }
    }
    return chwToImageData(accChw, imageH, imageW);
  }

  async release(): Promise<void> {
    if (!this.session) return;
    await this.session.release();
    this.session = null;
  }
}

function clamp8(v: number): number {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v;
}

/** Build a 1D linear-ramp weight vector of length `len`. Ramps from 0 -> 1
 *  over `blend` pixels at the start (unless `firstEdge`) and 1 -> 0 over
 *  `blend` pixels at the end (unless `lastEdge`). Edge tiles get full
 *  weight on their image-boundary side so the boundary pixels still receive
 *  a contribution. */
function ramp1D(len: number, blend: number, firstEdge: boolean, lastEdge: boolean): Float32Array {
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    let w = 1;
    if (!firstEdge && i < blend) w *= (i + 1) / (blend + 1);
    if (!lastEdge && i >= len - blend) w *= (len - i) / (blend + 1);
    out[i] = w;
  }
  return out;
}

/** Convert a CHW float-image (3, H, W) in roughly [-1, 1] to an interleaved
 *  RGBA Uint8ClampedArray ImageData. Used by both single-pass and tiled
 *  decode paths. */
function chwToImageData(chw: Float32Array, imageH: number, imageW: number): ImageData {
  const planeSize = imageH * imageW;
  const rgba = new Uint8ClampedArray(planeSize * 4);
  for (let i = 0; i < planeSize; i++) {
    const r = chw[i];
    const g = chw[i + planeSize];
    const b = chw[i + 2 * planeSize];
    const o = i * 4;
    rgba[o] = clamp8((r * 0.5 + 0.5) * 255);
    rgba[o + 1] = clamp8((g * 0.5 + 0.5) * 255);
    rgba[o + 2] = clamp8((b * 0.5 + 0.5) * 255);
    rgba[o + 3] = 255;
  }
  return new ImageData(rgba, imageW, imageH);
}
