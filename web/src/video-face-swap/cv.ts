/**
 * Image processing utilities.
 *
 * Hand-written replacements for OpenCV.js functions we need.
 * No prebuilt OpenCV.js npm package includes imgproc (warpAffine, etc).
 * TODO: replace with a custom OpenCV.js WASM build for better performance.
 */

/**
 * Estimate a similarity transform (rotation + uniform scale + translation)
 * from source points to destination points using least squares.
 *
 * Returns a 2x3 matrix as flat array [a, -b, tx, b, a, ty] (row-major).
 */
export function estimateSimilarityTransform(src: number[][], dst: number[][]): number[] {
  const n = src.length;
  let s00 = 0,
    s02 = 0,
    s03 = 0;
  let b0 = 0,
    b1 = 0,
    b2 = 0,
    b3 = 0;

  for (let i = 0; i < n; i++) {
    const sx = src[i][0],
      sy = src[i][1];
    const dx = dst[i][0],
      dy = dst[i][1];

    s00 += sx * sx + sy * sy;
    s02 += sx;
    s03 += sy;
    b0 += sx * dx + sy * dy;
    b1 += -sy * dx + sx * dy;
    b2 += dx;
    b3 += dy;
  }

  const A = [
    [s00, 0, s02, s03],
    [0, s00, -s03, s02],
    [s02, -s03, n, 0],
    [s03, s02, 0, n],
  ];
  const rhs = [b0, b1, b2, b3];
  const x = solveLinear4x4(A, rhs);

  const a = x[0],
    bv = x[1],
    tx = x[2],
    ty = x[3];
  return [a, -bv, tx, bv, a, ty];
}

function solveLinear4x4(A: number[][], b: number[]): number[] {
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 4; col++) {
    let maxRow = col;
    for (let row = col + 1; row < 4; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) continue;
    for (let row = col + 1; row < 4; row++) {
      const f = aug[row][col] / pivot;
      for (let j = col; j <= 4; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  const x = [0, 0, 0, 0];
  for (let i = 3; i >= 0; i--) {
    x[i] = aug[i][4];
    for (let j = i + 1; j < 4; j++) x[i] -= aug[i][j] * x[j];
    x[i] /= aug[i][i];
  }
  return x;
}

/** Invert a 2x3 affine matrix [a, b, tx, c, d, ty] */
export function invertAffine(m: number[]): number[] {
  const [a, b, tx, c, d, ty] = m;
  const det = a * d - b * c;
  const id = 1 / det;
  return [d * id, -b * id, (b * ty - d * tx) * id, -c * id, a * id, (c * tx - a * ty) * id];
}

/**
 * Warp an image by a 2x3 affine matrix with bilinear interpolation.
 * Border pixels are replicated (clamped).
 */
export function warpAffine(
  src: ImageData,
  matrix: number[],
  outWidth: number,
  outHeight: number,
): ImageData {
  const inv = invertAffine(matrix);
  const dst = new ImageData(outWidth, outHeight);
  const sw = src.width,
    sh = src.height;
  const sd = src.data,
    dd = dst.data;

  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      let sx = inv[0] * x + inv[1] * y + inv[2];
      let sy = inv[3] * x + inv[4] * y + inv[5];

      // Border replicate
      sx = Math.max(0, Math.min(sw - 1, sx));
      sy = Math.max(0, Math.min(sh - 1, sy));

      // Bilinear interpolation
      const x0 = Math.floor(sx),
        y0 = Math.floor(sy);
      const x1 = Math.min(x0 + 1, sw - 1),
        y1 = Math.min(y0 + 1, sh - 1);
      const fx = sx - x0,
        fy = sy - y0;

      const i00 = (y0 * sw + x0) * 4;
      const i10 = (y0 * sw + x1) * 4;
      const i01 = (y1 * sw + x0) * 4;
      const i11 = (y1 * sw + x1) * 4;
      const di = (y * outWidth + x) * 4;

      for (let c = 0; c < 4; c++) {
        dd[di + c] =
          sd[i00 + c] * (1 - fx) * (1 - fy) +
          sd[i10 + c] * fx * (1 - fy) +
          sd[i01 + c] * (1 - fx) * fy +
          sd[i11 + c] * fx * fy;
      }
    }
  }

  return dst;
}

/**
 * Paste a warped crop back onto a frame with blending.
 * If occlusionMask is provided (from XSeg), use it combined with a box mask.
 * Otherwise fall back to a feathered elliptical mask.
 * affineMatrix is forward: frame -> crop.
 */
export function pasteBack(
  frame: ImageData,
  crop: ImageData,
  affineMatrix: number[],
  occlusionMask?: Float32Array,
): ImageData {
  const fw = frame.width,
    fh = frame.height;
  const cw = crop.width,
    ch = crop.height;
  const m = affineMatrix;

  // Use occlusion mask with box edge feathering, or fall back to elliptical mask
  let mask: Float32Array;
  if (occlusionMask) {
    const boxMask = createFeatheredMask(cw, ch, 15, 15);
    // Combine: element-wise minimum (FaceFusion convention)
    mask = new Float32Array(cw * ch);
    for (let i = 0; i < mask.length; i++) {
      mask[i] = Math.min(occlusionMask[i], boxMask[i]);
    }
  } else {
    mask = createFeatheredMask(cw, ch, 15, 15);
  }

  // Warp crop and mask back to frame space, blend as we go
  const result = new ImageData(new Uint8ClampedArray(frame.data), fw, fh);
  const rd = result.data;
  const cd = crop.data;

  for (let y = 0; y < fh; y++) {
    for (let x = 0; x < fw; x++) {
      // Map frame pixel to crop space using forward matrix
      const cx = m[0] * x + m[1] * y + m[2];
      const cy = m[3] * x + m[4] * y + m[5];

      // Skip if outside crop bounds
      if (cx < 0 || cx >= cw - 1 || cy < 0 || cy >= ch - 1) continue;

      // Bilinear sample from crop
      const x0 = Math.floor(cx),
        y0 = Math.floor(cy);
      const x1 = x0 + 1,
        y1 = y0 + 1;
      const fx = cx - x0,
        fy = cy - y0;

      const ci00 = (y0 * cw + x0) * 4;
      const ci10 = (y0 * cw + x1) * 4;
      const ci01 = (y1 * cw + x0) * 4;
      const ci11 = (y1 * cw + x1) * 4;

      // Sample mask (single channel)
      const mi00 = y0 * cw + x0;
      const mi10 = y0 * cw + x1;
      const mi01 = y1 * cw + x0;
      const mi11 = y1 * cw + x1;
      const alpha =
        mask[mi00] * (1 - fx) * (1 - fy) +
        mask[mi10] * fx * (1 - fy) +
        mask[mi01] * (1 - fx) * fy +
        mask[mi11] * fx * fy;

      if (alpha < 0.001) continue;

      const di = (y * fw + x) * 4;
      for (let c = 0; c < 3; c++) {
        const cropVal =
          cd[ci00 + c] * (1 - fx) * (1 - fy) +
          cd[ci10 + c] * fx * (1 - fy) +
          cd[ci01 + c] * (1 - fx) * fy +
          cd[ci11 + c] * fx * fy;
        rd[di + c] = rd[di + c] * (1 - alpha) + cropVal * alpha;
      }
    }
  }

  return result;
}

/**
 * Create a feathered elliptical mask.
 * Returns Float32Array of values 0-1, size w*h.
 */
export function createFeatheredMask(
  w: number,
  h: number,
  erodeX: number,
  erodeY: number,
): Float32Array {
  const mask = new Float32Array(w * h);
  const cx = w / 2,
    cy = h / 2;
  const rx = w / 2 - erodeX,
    ry = h / 2 - erodeY;
  const blurDist = Math.min(erodeX, erodeY);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist <= 1) {
        mask[y * w + x] = 1;
      } else {
        const fade = 1 - (dist - 1) * (rx / blurDist);
        mask[y * w + x] = Math.max(0, fade);
      }
    }
  }

  return mask;
}

/** Convert RGBA ImageData to RGB Float32Array in NCHW format, normalized to 0-1 */
export function rgbaToRgbFloat32(imageData: ImageData): Float32Array {
  const w = imageData.width,
    h = imageData.height;
  const pixelCount = w * h;
  const float32 = new Float32Array(3 * pixelCount);
  const data = imageData.data;

  for (let i = 0; i < pixelCount; i++) {
    float32[i] = data[i * 4] / 255.0; // R
    float32[pixelCount + i] = data[i * 4 + 1] / 255.0; // G
    float32[2 * pixelCount + i] = data[i * 4 + 2] / 255.0; // B
  }

  return float32;
}

/**
 * Convert RGBA ImageData to BGR Float32Array in NCHW, normalized to 0-1.
 * Used for models trained with cv2-loaded (BGR) images: YOLOFace, YuNet,
 * SCRFD, 2DFAN4, XSeg, etc. Browser ImageData is RGBA so we have to swap
 * the R and B channel reads to match the training distribution.
 */
export function rgbaToBgrFloat32(imageData: ImageData): Float32Array {
  const w = imageData.width,
    h = imageData.height;
  const pixelCount = w * h;
  const float32 = new Float32Array(3 * pixelCount);
  const data = imageData.data;

  for (let i = 0; i < pixelCount; i++) {
    float32[i] = data[i * 4 + 2] / 255.0; // B
    float32[pixelCount + i] = data[i * 4 + 1] / 255.0; // G
    float32[2 * pixelCount + i] = data[i * 4] / 255.0; // R
  }

  return float32;
}

// Reusable scratch canvases. OffscreenCanvas works in both window and
// worker contexts; the cv.ts helpers run in either depending on whether
// the worker mode is enabled.
let _scratchA: OffscreenCanvas | null = null;
let _scratchB: OffscreenCanvas | null = null;

function getScratchA(): [OffscreenCanvas, OffscreenCanvasRenderingContext2D] {
  if (!_scratchA) _scratchA = new OffscreenCanvas(1, 1);
  return [_scratchA, _scratchA.getContext("2d", { willReadFrequently: true })!];
}

function getScratchB(): [OffscreenCanvas, OffscreenCanvasRenderingContext2D] {
  if (!_scratchB) _scratchB = new OffscreenCanvas(1, 1);
  return [_scratchB, _scratchB.getContext("2d", { willReadFrequently: true })!];
}

/** Resize an ImageData using canvas (browser-native, good quality) */
export function resizeImageData(src: ImageData, newWidth: number, newHeight: number): ImageData {
  const [srcCanvas, srcCtx] = getScratchA();
  srcCanvas.width = src.width;
  srcCanvas.height = src.height;
  srcCtx.putImageData(src, 0, 0);

  const [dstCanvas, dstCtx] = getScratchB();
  dstCanvas.width = newWidth;
  dstCanvas.height = newHeight;
  dstCtx.drawImage(srcCanvas, 0, 0, newWidth, newHeight);
  return dstCtx.getImageData(0, 0, newWidth, newHeight);
}

/** Crop a region from ImageData */
export function cropImageData(
  src: ImageData,
  x: number,
  y: number,
  w: number,
  h: number,
): ImageData {
  const [canvas, ctx] = getScratchA();
  canvas.width = src.width;
  canvas.height = src.height;
  ctx.putImageData(src, 0, 0);
  return ctx.getImageData(x, y, w, h);
}

/**
 * Separable 1D gaussian blur for a single-channel float image. Edge pixels
 * are clamped (replicate). Returns a new Float32Array; input is untouched.
 *
 * Used for occlusion mask post-processing - cheap (a 256x256 mask with
 * sigma=5 takes about 5ms).
 */
export function gaussianBlur1f(
  src: Float32Array,
  width: number,
  height: number,
  sigma: number,
): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kSize = radius * 2 + 1;
  const kernel = new Float32Array(kSize);
  const twoSigmaSq = 2 * sigma * sigma;
  let kSum = 0;
  for (let i = 0; i < kSize; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / twoSigmaSq);
    kSum += kernel[i];
  }
  for (let i = 0; i < kSize; i++) kernel[i] /= kSum;

  const tmp = new Float32Array(width * height);
  // Horizontal pass.
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < kSize; k++) {
        let xi = x + k - radius;
        if (xi < 0) xi = 0;
        else if (xi >= width) xi = width - 1;
        acc += src[row + xi] * kernel[k];
      }
      tmp[row + x] = acc;
    }
  }
  // Vertical pass.
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < kSize; k++) {
        let yi = y + k - radius;
        if (yi < 0) yi = 0;
        else if (yi >= height) yi = height - 1;
        acc += tmp[yi * width + x] * kernel[k];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}
