// Tiny shared helpers used by both the SD1.5 and SDXL generate functions.
// Pure functions, no DOM, no ORT.

/** Mulberry32 PRNG. Tiny, deterministic, ~2^32 period - plenty for a
 *  one-shot image generation seeding. Used so the seed input actually
 *  reproduces a generation across runs. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller for an initial random latent, driven by a seeded PRNG so the
 *  user can reproduce a generation by re-entering the seed. */
export function gaussianNoise(n: number, rand: () => number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = rand();
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1 || 1e-9));
    const theta = 2 * Math.PI * u2;
    out[i] = r * Math.cos(theta);
    if (i + 1 < n) out[i + 1] = r * Math.sin(theta);
  }
  return out;
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function formatEta(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "?";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  return `${m}m ${s}s`;
}

export interface LatentStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  nans: number;
}

export function statsOf(arr: Float32Array): LatentStats {
  let sum = 0,
    min = Infinity,
    max = -Infinity,
    nans = 0,
    finite = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isNaN(v)) {
      nans++;
      continue;
    }
    sum += v;
    finite++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = finite > 0 ? sum / finite : NaN;
  let sq = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isNaN(v)) continue;
    const d = v - mean;
    sq += d * d;
  }
  const std = finite > 0 ? Math.sqrt(sq / finite) : NaN;
  return { mean, std, min, max, nans };
}

export function fmtStats(s: LatentStats): string {
  return `[mean=${s.mean.toFixed(2)} std=${s.std.toFixed(2)} min=${s.min.toFixed(1)} max=${s.max.toFixed(1)}${s.nans > 0 ? ` NaNs=${s.nans}` : ""}]`;
}

/** Rasterize an HTMLImageElement to a fresh ImageData at (width, height).
 *  Aspect mismatch is handled by simple stretch; a fit/crop UI is a future
 *  refinement, not a correctness issue. */
export function rasterizeRefImage(img: HTMLImageElement, width: number, height: number): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("could not get 2d context for reference resample");
  ctx.drawImage(img, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}
