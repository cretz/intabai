// Sanity tests for the hand-rolled fp16 conversion in src/sd15/fp16.ts.
// We don't need exhaustive numerical correctness - we need to know that
// the round-trip is faithful for the value ranges SD1.5 inference produces
// (latents and CFG-combined noise predictions, mostly in [-15, 15] with
// occasional excursions to ±50 in unstable conditions). If any of these
// silently produce NaN or wrong-magnitude output we'd see exactly the
// kind of garbled UNet output gate B exhibited.

import { describe, it, expect } from "vitest";

import {
  f16BitsToF32,
  f16ToF32Array,
  f32ToF16Array,
  f32ToF16Bits,
} from "../src/sd15/fp16";

function roundTrip(v: number): number {
  return f16BitsToF32(f32ToF16Bits(v));
}

describe("fp16 round-trip", () => {
  it("preserves exact integers in fp16 range", () => {
    for (const v of [0, 1, -1, 2, -2, 100, -100, 1000, -1000, 2048, -2048]) {
      expect(roundTrip(v)).toBe(v);
    }
  });

  it("preserves sign of zero", () => {
    // +0 stays +0, -0 stays -0. fp16 has a sign bit so this is a faithful
    // round-trip; we use Object.is rather than === to distinguish.
    expect(Object.is(roundTrip(0), 0)).toBe(true);
    expect(Object.is(roundTrip(-0), -0)).toBe(true);
  });

  it("approximates non-integer values within fp16 ulp", () => {
    // fp16 has ~3 decimal digits of precision; values in [-10, 10] should
    // round-trip with relative error < 0.001.
    for (const v of [0.5, -0.5, 0.18215, 7.5, -3.14159, 9.999, -7.123]) {
      const got = roundTrip(v);
      const relErr = Math.abs(got - v) / Math.max(Math.abs(v), 1e-6);
      expect(relErr).toBeLessThan(0.001);
    }
  });

  it("does not produce NaN for any normal value SD1.5 might emit", () => {
    // Sweep the range a UNet output might land in. None of these should
    // become NaN under round-trip - if our conversion has a corner-case
    // bug for some value in this range, the UNet loop will silently
    // poison itself with NaN at the first step that hits the bug.
    for (let v = -60; v <= 60; v += 0.1) {
      const got = roundTrip(v);
      expect(Number.isNaN(got)).toBe(false);
    }
  });

  it("clamps overflow to +Inf, not NaN", () => {
    // fp16 max is ~65504. Anything above that should become Inf.
    expect(roundTrip(70000)).toBe(Infinity);
    expect(roundTrip(-70000)).toBe(-Infinity);
  });

  it("array conversion length matches and preserves order", () => {
    const src = new Float32Array([0, 1, -1, 7.5, -3.14, 100]);
    const half = f32ToF16Array(src);
    expect(half.length).toBe(src.length);
    const back = f16ToF32Array(half);
    expect(back.length).toBe(src.length);
    for (let i = 0; i < src.length; i++) {
      const relErr = Math.abs(back[i] - src[i]) / Math.max(Math.abs(src[i]), 1e-6);
      expect(relErr).toBeLessThan(0.001);
    }
  });
});
