// Runtime validation helpers for the LTX pipeline.
//
// The wasm-vs-webgpu divergence chase has been chasing kernel bugs for
// weeks. Equally plausible is JS-side glue handing wrong dtypes/shapes/
// type-tags to ORT, where wasm tolerates the mistake and webgpu doesn't
// (or vice versa). These asserts catch all those at the boundary.

import * as ort from "onnxruntime-web";

export type TypedArray =
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array
  | BigInt64Array
  | BigUint64Array
  | Float32Array
  | Float64Array;

/** Throws if `arr` is not an instance of the expected typed-array ctor. */
export function assertArrayType<T extends TypedArray>(
  label: string,
  arr: unknown,
  ctor: { new (...a: never[]): T; name: string },
): asserts arr is T {
  if (!(arr instanceof ctor)) {
    const got = arr === null
      ? "null"
      : arr === undefined
        ? "undefined"
        : (arr as object).constructor?.name ?? typeof arr;
    throw new Error(`${label}: expected ${ctor.name}, got ${got}`);
  }
}

/** Throws if length mismatches. */
export function assertLength(label: string, arr: { length: number }, expected: number): void {
  if (arr.length !== expected) {
    throw new Error(`${label}: length ${arr.length} != expected ${expected}`);
  }
}

/** Throws if dims mismatch (numeric exact). */
export function assertDims(label: string, dims: readonly number[], expected: readonly number[]): void {
  if (dims.length !== expected.length) {
    throw new Error(
      `${label}: dims rank ${dims.length} != expected rank ${expected.length} ` +
        `(got [${dims.join(",")}], expected [${expected.join(",")}])`,
    );
  }
  for (let i = 0; i < dims.length; i++) {
    if (dims[i] !== expected[i]) {
      throw new Error(
        `${label}: dims[${i}] = ${dims[i]} != expected ${expected[i]} ` +
          `(got [${dims.join(",")}], expected [${expected.join(",")}])`,
      );
    }
  }
}

/** Validate an ORT output tensor: declared type and (optionally) dims.
 *  Use assertF16View / assertF32View / assertI64View on `.data` separately
 *  to lock the JS view ctor strictly. */
export function assertOrtOutput(
  label: string,
  t: ort.Tensor | undefined,
  opts: {
    type: "float16" | "float32" | "int64" | "int32";
    dims?: readonly number[];
  },
): asserts t is ort.Tensor {
  if (!t) {
    throw new Error(`${label}: missing output tensor`);
  }
  if (t.type !== opts.type) {
    throw new Error(`${label}: tensor.type = "${t.type}" != expected "${opts.type}"`);
  }
  if (opts.dims) {
    assertDims(label, t.dims as number[], opts.dims);
  }
}

/** Validate the byte-length of an fp16 typed-array view: must be exactly
 *  2 * elementCount. Catches the case where `.data` is Float32Array
 *  (4 bytes/elt) but downstream code wants 2 bytes/elt. */
export function assertF16ByteLength(label: string, view: ArrayBufferView, elementCount: number): void {
  const expected = 2 * elementCount;
  if (view.byteLength !== expected) {
    throw new Error(
      `${label}: byteLength ${view.byteLength} != expected ${expected} ` +
        `for ${elementCount} fp16 elements (likely wrong dtype)`,
    );
  }
}

/** ORT-web returns fp16 outputs as Uint16Array on legacy paths and as the
 *  native Float16Array on Chrome 147+. Within a single browser session
 *  this is deterministic - it should not flip between calls. We pin the
 *  observed ctor name on first sighting and assert every subsequent fp16
 *  output matches exactly. Mid-session drift = a bug we want surfaced.
 *
 *  Reset between unit tests via `__resetObservedF16Ctor()`. */
let _observedF16Ctor: string | null = null;

export function __resetObservedF16Ctor(): void {
  _observedF16Ctor = null;
}

export function observedF16Ctor(): string | null {
  return _observedF16Ctor;
}

/** Strict fp16 view assertion. The first call records which exact ctor
 *  ORT-web returned; later calls must match that ctor (no Uint16Array vs
 *  Float16Array drift, no surprise Float32Array). Returns the view for
 *  downstream raw-byte reads. */
export function assertF16View(
  label: string,
  data: unknown,
  elementCount: number,
): ArrayBufferView {
  const ctorName = (data as { constructor?: { name?: string } } | null)?.constructor?.name
    ?? typeof data;
  if (_observedF16Ctor === null) {
    if (ctorName !== "Uint16Array" && ctorName !== "Float16Array") {
      throw new Error(
        `${label}: expected first fp16 output to be Uint16Array or Float16Array, got ${ctorName}`,
      );
    }
    _observedF16Ctor = ctorName;
  } else if (ctorName !== _observedF16Ctor) {
    throw new Error(
      `${label}: fp16 output ctor changed mid-session. ` +
        `First saw ${_observedF16Ctor}, now ${ctorName}. ` +
        `ORT-web should be deterministic within a session.`,
    );
  }
  const view = data as ArrayBufferView;
  assertF16ByteLength(label, view, elementCount);
  return view;
}

/** Assert `data` is a Float32Array of expected length. */
export function assertF32View(
  label: string,
  data: unknown,
  elementCount: number,
): Float32Array {
  if (!(data instanceof Float32Array)) {
    const got = (data as { constructor?: { name?: string } } | null)?.constructor?.name ?? typeof data;
    throw new Error(`${label}: expected Float32Array, got ${got}`);
  }
  if (data.length !== elementCount) {
    throw new Error(`${label}: length ${data.length} != expected ${elementCount}`);
  }
  return data;
}
