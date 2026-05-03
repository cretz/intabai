// Symmetric patchifier for LTX 2B distilled.
//
// Mirrors `ltx_video/models/transformers/symmetric_patchifier.py` from the
// reference space. For LTX 2B distilled the config uses patch_size=(1,1,1)
// and latent channels=128, so patchify is a pure transpose+reshape:
//   latents  [B, 128, T, H, W]  ->  hidden  [B, T*H*W, 128]
//   indices  [B, 3, T*H*W]      // (f, h, w) coords per token
//
// The (1,1,1) special case lets us skip the rearrange; if we ever bump
// patch_size we'll need to fold p1/p2/p3 into the channel axis.

import { f32ToF16Bits, f16BitsToF32 } from "../sd15/fp16";

/** LTX 2B distilled config. */
export const LTX_PATCH_SIZE_T = 1;
export const LTX_PATCH_SIZE_HW = 1;
export const LTX_LATENT_CHANNELS = 128;

/** Spatial / temporal latent compression ratios. */
export const LTX_VAE_SPATIAL_DOWNSCALE = 32;
export const LTX_VAE_TEMPORAL_DOWNSCALE = 8;

export interface LtxLatentShape {
  /** Latent frames (after VAE temporal compression). */
  t: number;
  /** Latent height (pixels / 32). */
  h: number;
  /** Latent width (pixels / 32). */
  w: number;
}

/**
 * Convert pixel-space dims to latent-space dims.
 *
 * Frame count must be 8k+1 in pixel space, which after temporal
 * compression yields k+1 latent frames (e.g. 9 px -> 2, 49 -> 7).
 * H/W must be divisible by 32.
 */
export function pixelToLatentShape(
  numFrames: number,
  height: number,
  width: number,
): LtxLatentShape {
  if ((numFrames - 1) % LTX_VAE_TEMPORAL_DOWNSCALE !== 0) {
    throw new Error(
      `LTX requires (numFrames - 1) divisible by ${LTX_VAE_TEMPORAL_DOWNSCALE}, got ${numFrames}`,
    );
  }
  if (height % LTX_VAE_SPATIAL_DOWNSCALE !== 0 || width % LTX_VAE_SPATIAL_DOWNSCALE !== 0) {
    throw new Error(
      `LTX requires H and W divisible by ${LTX_VAE_SPATIAL_DOWNSCALE}, got ${height}x${width}`,
    );
  }
  return {
    t: (numFrames - 1) / LTX_VAE_TEMPORAL_DOWNSCALE + 1,
    h: height / LTX_VAE_SPATIAL_DOWNSCALE,
    w: width / LTX_VAE_SPATIAL_DOWNSCALE,
  };
}

/** Number of patch tokens for a latent shape (B=1). */
export function latentNumTokens(shape: LtxLatentShape): number {
  return shape.t * shape.h * shape.w;
}

/**
 * Build the indices_grid tensor used as RoPE input.
 *
 * Layout: [1, 3, T*H*W] with channel order (f, h, w). Iteration order
 * matches the patchify rearrange: outer f, then h, then w.
 *
 * The reference pipeline (pipeline_ltx_video.py:1148, vae_encode.py:218)
 * passes pixel-space coordinates with two adjustments:
 *   1. spatial: latent_idx * VAE_SPATIAL_DOWNSCALE (= ×32)
 *   2. temporal causal-fix: pixel_t = max(latent_t * 8 + 1 - 8, 0)
 *      so frame 0 stays at 0 and frame i>0 lands at (i*8 - 7).
 *      Then divide by frame_rate.
 *
 * Without this scaling the values fed to RoPE are ~32x too small, and
 * because positional_embedding_max_pos = [20, 2048, 2048], every token
 * ends up at a near-zero fractional position -> RoPE rotates by ~0 -> the
 * model can't tell tokens apart and decode collapses to per-patch
 * identical content.
 */
export function buildIndicesGrid(
  shape: LtxLatentShape,
  frameRate: number,
): Float32Array {
  const { t, h, w } = shape;
  const n = t * h * w;
  const out = new Float32Array(3 * n);
  let idx = 0;
  for (let f = 0; f < t; f++) {
    const pixelT = f === 0 ? 0 : f * LTX_VAE_TEMPORAL_DOWNSCALE - (LTX_VAE_TEMPORAL_DOWNSCALE - 1);
    const fracT = pixelT / frameRate;
    for (let y = 0; y < h; y++) {
      const pixelY = y * LTX_VAE_SPATIAL_DOWNSCALE;
      for (let x = 0; x < w; x++) {
        const pixelX = x * LTX_VAE_SPATIAL_DOWNSCALE;
        out[0 * n + idx] = fracT;
        out[1 * n + idx] = pixelY;
        out[2 * n + idx] = pixelX;
        idx++;
      }
    }
  }
  return out;
}

/**
 * Patchify latent fp16 tensor `[1, 128, T, H, W]` -> hidden `[1, T*H*W, 128]`.
 *
 * For patch_size=(1,1,1) this is a CHW->HWC-style transpose: input is
 * channel-major, output is token-major with channel as the innermost axis.
 */
export function patchifyF16(
  latents: Uint16Array,
  shape: LtxLatentShape,
): Uint16Array {
  const { t, h, w } = shape;
  const n = t * h * w;
  const c = LTX_LATENT_CHANNELS;
  if (latents.length !== c * n) {
    throw new Error(`patchify: expected ${c * n} fp16 vals, got ${latents.length}`);
  }
  const out = new Uint16Array(n * c);
  // src index: ch * n + token; dst index: token * c + ch
  for (let ch = 0; ch < c; ch++) {
    const srcBase = ch * n;
    for (let i = 0; i < n; i++) {
      out[i * c + ch] = latents[srcBase + i];
    }
  }
  return out;
}

/**
 * Inverse of patchifyF16: hidden `[1, T*H*W, 128]` -> latent `[1, 128, T, H, W]`.
 */
export function unpatchifyF16(
  hidden: Uint16Array,
  shape: LtxLatentShape,
): Uint16Array {
  const { t, h, w } = shape;
  const n = t * h * w;
  const c = LTX_LATENT_CHANNELS;
  if (hidden.length !== n * c) {
    throw new Error(`unpatchify: expected ${n * c} fp16 vals, got ${hidden.length}`);
  }
  const out = new Uint16Array(c * n);
  for (let ch = 0; ch < c; ch++) {
    const dstBase = ch * n;
    for (let i = 0; i < n; i++) {
      out[dstBase + i] = hidden[i * c + ch];
    }
  }
  return out;
}

/** fp32 helpers, mostly for tests / smoke. */
export function patchifyF32(latents: Float32Array, shape: LtxLatentShape): Float32Array {
  const { t, h, w } = shape;
  const n = t * h * w;
  const c = LTX_LATENT_CHANNELS;
  const out = new Float32Array(n * c);
  for (let ch = 0; ch < c; ch++) {
    const srcBase = ch * n;
    for (let i = 0; i < n; i++) out[i * c + ch] = latents[srcBase + i];
  }
  return out;
}

/** Generate a fp16 noise tensor of the right shape, given a seeded RNG. */
export function randomLatentF16(
  shape: LtxLatentShape,
  rand01: () => number,
  scale = 1.0,
): Uint16Array {
  const n = LTX_LATENT_CHANNELS * shape.t * shape.h * shape.w;
  const out = new Uint16Array(n);
  // Box-Muller for proper N(0,1).
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rand01(), 1e-7);
    const u2 = rand01();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    out[i] = f32ToF16Bits(scale * r * Math.cos(theta));
    if (i + 1 < n) out[i + 1] = f32ToF16Bits(scale * r * Math.sin(theta));
  }
  return out;
}

/** Round-trip sanity used by tests. */
export function _testRoundTrip(shape: LtxLatentShape): boolean {
  const n = LTX_LATENT_CHANNELS * shape.t * shape.h * shape.w;
  const src = new Uint16Array(n);
  for (let i = 0; i < n; i++) src[i] = f32ToF16Bits((i % 17) - 8);
  const round = unpatchifyF16(patchifyF16(src, shape), shape);
  for (let i = 0; i < n; i++) {
    if (f16BitsToF32(round[i]) !== f16BitsToF32(src[i])) return false;
  }
  return true;
}
