// IEEE-754 binary16 (half-precision float) conversion. JS has no native
// Float16Array yet (TC39 stage 3 as of writing), so we pack/unpack manually.
// nmkd's SD1.5 ONNX export is fp16 throughout - the UNet and VAE both
// declare fp16 input/output tensors and ORT-web errors on dtype mismatch
// when fed fp32. The scheduler and our CFG math stay in fp32 because the
// rounding error from doing them at half precision compounds visibly over
// 20-50 denoising steps; we only round-trip through fp16 at the model
// boundary.
//
// Reference: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
// The encoding is sign(1) | exponent(5) | mantissa(10).
//
// The implementation uses a Float32Array view sharing memory with a
// Uint32Array view, which is the standard portable trick for getting at the
// raw bits of a JS number. This is pure CPU code, ~5 ns per element on a
// modern laptop, fast enough to convert a 16384-element latent in 0.1 ms.

const SCRATCH_BUF = new ArrayBuffer(4);
const SCRATCH_F32 = new Float32Array(SCRATCH_BUF);
const SCRATCH_U32 = new Uint32Array(SCRATCH_BUF);

/** Convert one fp32 value to its fp16 bit representation (a uint16). */
export function f32ToF16Bits(value: number): number {
  SCRATCH_F32[0] = value;
  const x = SCRATCH_U32[0];
  const sign = (x >>> 31) & 0x1;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;

  if (exp === 0xff) {
    // Inf or NaN. Preserve NaN-ness (any non-zero mant).
    return (sign << 15) | 0x7c00 | (mant !== 0 ? 0x200 : 0);
  }
  if (exp === 0) {
    // fp32 zero or subnormal -> fp16 zero (subnormal range too small to matter).
    return sign << 15;
  }

  let newExp = exp - 127 + 15;
  if (newExp >= 0x1f) {
    // Overflow -> Inf.
    return (sign << 15) | 0x7c00;
  }
  if (newExp <= 0) {
    // Result is fp16 subnormal.
    if (newExp < -10) {
      // Too small even for subnormal -> zero.
      return sign << 15;
    }
    const mantWithImplicit = mant | 0x800000;
    let shifted = mantWithImplicit >>> (1 - newExp);
    // Round to nearest, ties to even.
    if (shifted & 0x1000) shifted += 0x2000;
    return (sign << 15) | (shifted >>> 13);
  }

  // Normal fp16 number. Round to nearest, ties to even.
  let mantR = mant;
  if (mantR & 0x1000) {
    mantR += 0x2000;
    if (mantR & 0x800000) {
      // Mantissa rounded up into the next exponent.
      mantR = 0;
      newExp += 1;
      if (newExp >= 0x1f) return (sign << 15) | 0x7c00;
    }
  }
  return (sign << 15) | (newExp << 10) | (mantR >>> 13);
}

/** Convert one fp16 bit representation back to fp32. */
export function f16BitsToF32(bits: number): number {
  const sign = (bits >>> 15) & 0x1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;

  let f32Bits: number;
  if (exp === 0) {
    if (mant === 0) {
      f32Bits = sign << 31;
    } else {
      // Subnormal fp16. Normalize.
      let m = mant;
      let e = -1;
      while ((m & 0x400) === 0) {
        m <<= 1;
        e -= 1;
      }
      m &= 0x3ff;
      const newExp = -14 + e + 127;
      f32Bits = (sign << 31) | (newExp << 23) | (m << 13);
    }
  } else if (exp === 0x1f) {
    // Inf or NaN.
    f32Bits = (sign << 31) | 0x7f800000 | (mant << 13);
  } else {
    // Normal.
    const newExp = exp - 15 + 127;
    f32Bits = (sign << 31) | (newExp << 23) | (mant << 13);
  }
  SCRATCH_U32[0] = f32Bits >>> 0;
  return SCRATCH_F32[0];
}

/** Convert a Float32Array to a Uint16Array of fp16 bit patterns. Same length. */
export function f32ToF16Array(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f32ToF16Bits(src[i]);
  return out;
}

/** Convert a Uint16Array of fp16 bit patterns to a Float32Array. Same length. */
export function f16ToF32Array(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f16BitsToF32(src[i]);
  return out;
}

/** Copy fp16 bit patterns out of an ORT tensor's `.data` into a fresh
 *  Uint16Array. Handles both the legacy Uint16Array representation and the
 *  native Float16Array representation that ORT-web returns on modern
 *  browsers (Chrome 147+). Constructing `new Uint16Array(float16Array)`
 *  does a numeric round-trip (half-float values to uint16 integers) and
 *  destroys the bits, which looks like -Inf/NaN when reinterpreted as fp16.
 *  Diagnosed 2026-04-18 - this bug caused text encoder layer_00 to appear
 *  to output -Inf/NaN while the underlying math was correct. */
export function copyF16Bits(data: ArrayBufferView): Uint16Array {
  const { buffer, byteOffset, byteLength } = data;
  return new Uint16Array(buffer.slice(byteOffset, byteOffset + byteLength));
}
