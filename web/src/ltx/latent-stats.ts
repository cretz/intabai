// Per-channel latent normalize / un_normalize for the LTX 2B distilled VAE.
//
// File layout (vae/per_channel_stats.f32, written by the converter):
//   128 fp32 mean_of_means, then 128 fp32 std_of_means. Total 1024 bytes.
//
// The transformer denoises in normalized latent space:
//   normalized   = (raw - mean) / std
//   raw          = normalized * std + mean
// VAE encoder produces raw moments and the decoder takes raw latents, so
// we normalize after encode and un-normalize before decode. For T2V the
// init state is N(0,1) which is already in the normalized space, so the
// only place we touch stats is right before decode.

import type { ModelCache } from "../shared/model-cache";
import { f16BitsToF32, f32ToF16Bits } from "../sd15/fp16";
import { LTX_VAE_PER_CHANNEL_STATS_FILE } from "./models";
import { LTX_LATENT_CHANNELS } from "./patchifier";

export interface LtxLatentStats {
  mean: Float32Array; // length 128
  std: Float32Array;  // length 128
}

export async function loadLtxLatentStats(
  cache: ModelCache,
  debug?: (msg: string) => void,
): Promise<LtxLatentStats> {
  const buf = await cache.loadFile(LTX_VAE_PER_CHANNEL_STATS_FILE);
  const expected = LTX_LATENT_CHANNELS * 2 * 4;
  if (buf.byteLength !== expected) {
    throw new Error(
      `latent stats: expected ${expected} bytes, got ${buf.byteLength}`,
    );
  }
  const f32 = new Float32Array(buf);
  const mean = f32.slice(0, LTX_LATENT_CHANNELS);
  const std = f32.slice(LTX_LATENT_CHANNELS, LTX_LATENT_CHANNELS * 2);
  // Range summary so we can sanity-check the file content from the in-app
  // debug pane (browser console.log is dropped by the worklog capture).
  let mMin = Infinity, mMax = -Infinity, sMin = Infinity, sMax = -Infinity;
  for (let i = 0; i < LTX_LATENT_CHANNELS; i++) {
    if (mean[i] < mMin) mMin = mean[i];
    if (mean[i] > mMax) mMax = mean[i];
    if (std[i] < sMin) sMin = std[i];
    if (std[i] > sMax) sMax = std[i];
  }
  debug?.(
    `[ltx] stats: mean[0..3]=${[...mean.slice(0, 4)].map((v) => v.toFixed(3))} ` +
      `std[0..3]=${[...std.slice(0, 4)].map((v) => v.toFixed(3))} ` +
      `mean range [${mMin.toFixed(3)}, ${mMax.toFixed(3)}] ` +
      `std range [${sMin.toFixed(3)}, ${sMax.toFixed(3)}]`,
  );
  return { mean, std };
}

/** In-place: latent[c, *] = latent[c, *] * std[c] + mean[c]. NCDHW layout. */
export function unNormalizeLatentF16(
  latent: Uint16Array,
  stats: LtxLatentStats,
  spatial: number,
): void {
  if (latent.length !== LTX_LATENT_CHANNELS * spatial) {
    throw new Error(
      `unNormalize: expected ${LTX_LATENT_CHANNELS * spatial} fp16 vals, got ${latent.length}`,
    );
  }
  for (let c = 0; c < LTX_LATENT_CHANNELS; c++) {
    const m = stats.mean[c];
    const s = stats.std[c];
    const base = c * spatial;
    for (let i = 0; i < spatial; i++) {
      const x = f16BitsToF32(latent[base + i]);
      latent[base + i] = f32ToF16Bits(x * s + m);
    }
  }
}

/** In-place: latent[c, *] = (latent[c, *] - mean[c]) / std[c]. NCDHW. */
export function normalizeLatentF16(
  latent: Uint16Array,
  stats: LtxLatentStats,
  spatial: number,
): void {
  if (latent.length !== LTX_LATENT_CHANNELS * spatial) {
    throw new Error(
      `normalize: expected ${LTX_LATENT_CHANNELS * spatial} fp16 vals, got ${latent.length}`,
    );
  }
  for (let c = 0; c < LTX_LATENT_CHANNELS; c++) {
    const m = stats.mean[c];
    const s = stats.std[c];
    const base = c * spatial;
    for (let i = 0; i < spatial; i++) {
      const x = f16BitsToF32(latent[base + i]);
      latent[base + i] = f32ToF16Bits((x - m) / s);
    }
  }
}
