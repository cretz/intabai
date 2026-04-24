// Minimal GPU-buffer / io-binding helpers for ORT-web WebGPU.
//
// ORT-web wraps a GPUBuffer as an `ort.Tensor` via the (undocumented) static
// `fromGpuBuffer(buf, { dataType, dims })`. A session.run() accepts such
// tensors as both inputs and pre-allocated output fetches; outputs land
// directly in the bound buffer with no CPU round-trip. The zimage pipeline
// has been using this pattern successfully (see zimage/generate.ts:100-159)
// - this module is a generalized version so the fastwan transformer + Wan
// VAE can share it.
//
// Why we need this: a naive FastWan run ships ~7 GB through the PCIe link
// on transformer block handoff (4 steps * 30 blocks * ~60 MB conditioning)
// plus ~56 GB on the Wan VAE cache chain (20 step iterations * 32 caches
// round-tripped). Sustained PCIe traffic at that scale is both a
// wall-time bottleneck and a stability risk on laptops with marginal
// PCIe links. Keeping intermediate tensors GPU-resident removes the
// sustained traffic and typically speeds the run.

import * as ort from "onnxruntime-web";

export type GpuDtype = "float16" | "float32" | "int64" | "int32";

/** Return the GPUDevice ORT-web lazily created on its first WebGPU session.
 *  Must be called AFTER at least one ORT WebGPU session has been created,
 *  otherwise `ort.env.webgpu.device` is undefined. */
export function getOrtGpuDevice(): GPUDevice | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (((ort.env as any).webgpu?.device) as GPUDevice | undefined) ?? null;
}

function bytesPerEl(dtype: GpuDtype): number {
  switch (dtype) {
    case "float16": return 2;
    case "int32":
    case "float32": return 4;
    case "int64": return 8;
  }
}

/** Allocate a STORAGE/COPY_SRC/COPY_DST buffer sized for the given tensor
 *  shape and wrap it as an `ort.Tensor`. The returned tensor is owned by
 *  the caller; call `destroyGpuTensor()` when done. */
export function createGpuTensor(
  device: GPUDevice,
  dtype: GpuDtype,
  dims: readonly number[],
): ort.Tensor {
  const numEl = dims.reduce((a, b) => a * b, 1);
  const bytes = Math.max(16, Math.ceil((numEl * bytesPerEl(dtype)) / 4) * 4);
  const buf = device.createBuffer({
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    size: bytes,
  });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (ort.Tensor as any).fromGpuBuffer(buf, {
    dataType: dtype,
    dims: dims.slice(),
  }) as ort.Tensor;
}

/** Extract the underlying GPUBuffer from an ORT GPU-buffer tensor. */
export function gpuBufferOf(t: ort.Tensor): GPUBuffer {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (t as any).gpuBuffer as GPUBuffer;
}

export function destroyGpuTensor(t: ort.Tensor): void {
  try { gpuBufferOf(t)?.destroy(); } catch { /* already destroyed */ }
}

/** Upload CPU bytes into a GPU-tensor's backing buffer. Caller owns the
 *  source view; bytes are copied into a fresh mapped staging buffer and
 *  submitted. */
export function writeGpuBytes(
  device: GPUDevice,
  tensor: ort.Tensor,
  src: ArrayBufferView,
): void {
  const dst = gpuBufferOf(tensor);
  const aligned = Math.ceil(src.byteLength / 4) * 4;
  const staging = device.createBuffer({
    size: aligned,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint8Array(staging.getMappedRange()).set(
    new Uint8Array(src.buffer, src.byteOffset, src.byteLength),
  );
  staging.unmap();
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(staging, 0, dst, 0, aligned);
  device.queue.submit([enc.finish()]);
  // Staging buffer is kept alive by the submitted queue until the copy
  // completes, then GC'd. Not destroyed explicitly to match the pattern
  // used in zimage/generate.ts which is known to work.
}

/** Download a GPU-tensor's backing buffer into a new ArrayBuffer of the
 *  exact requested byte length. Awaits the GPU queue. */
export async function readGpuBytes(
  device: GPUDevice,
  tensor: ort.Tensor,
  byteLength: number,
): Promise<ArrayBuffer> {
  const src = gpuBufferOf(tensor);
  const aligned = Math.ceil(byteLength / 4) * 4;
  const rb = device.createBuffer({
    size: aligned,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, aligned);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const copy = rb.getMappedRange().slice(0, byteLength);
  rb.unmap();
  rb.destroy();
  return copy;
}

/** Read an fp16-bits tensor as a Uint16Array of `count` elements. */
export async function readGpuFp16(
  device: GPUDevice,
  tensor: ort.Tensor,
  count: number,
): Promise<Uint16Array> {
  const ab = await readGpuBytes(device, tensor, count * 2);
  return new Uint16Array(ab);
}

/** Read an fp32 tensor as a Float32Array of `count` elements. */
export async function readGpuFp32(
  device: GPUDevice,
  tensor: ort.Tensor,
  count: number,
): Promise<Float32Array> {
  const ab = await readGpuBytes(device, tensor, count * 4);
  return new Float32Array(ab);
}

/** GPU-to-GPU copy between two tensors of equal byte length. Used for the
 *  Wan VAE cache ping-pong: after each step, copy cache_out buffers back
 *  into cache_in buffers so the next session.run reads the updated state
 *  without any CPU round-trip. */
export function copyGpuTensor(
  device: GPUDevice,
  srcTensor: ort.Tensor,
  dstTensor: ort.Tensor,
  byteLength: number,
): void {
  const src = gpuBufferOf(srcTensor);
  const dst = gpuBufferOf(dstTensor);
  const aligned = Math.ceil(byteLength / 4) * 4;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, dst, 0, aligned);
  device.queue.submit([enc.finish()]);
}

/** Batched GPU-to-GPU copy across many tensor pairs, coalesced into a
 *  single command encoder + submit to minimize per-submit overhead. */
export function copyGpuTensorsBatch(
  device: GPUDevice,
  pairs: ReadonlyArray<{ src: ort.Tensor; dst: ort.Tensor; byteLength: number }>,
): void {
  const enc = device.createCommandEncoder();
  for (const { src, dst, byteLength } of pairs) {
    const aligned = Math.ceil(byteLength / 4) * 4;
    enc.copyBufferToBuffer(gpuBufferOf(src), 0, gpuBufferOf(dst), 0, aligned);
  }
  device.queue.submit([enc.finish()]);
}

/** Convenience: bytes-per-element for shape + dtype size math. */
export function dtypeBytes(dtype: GpuDtype): number {
  return bytesPerEl(dtype);
}
