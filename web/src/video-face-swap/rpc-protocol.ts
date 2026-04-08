// Message protocol between main thread and pipeline-worker.
//
// ImageData and Float32Array are not themselves Transferable, but their
// underlying ArrayBuffers are. Both directions of every per-frame call
// pack/unpack via the helpers below and pass `buffer` in postMessage's
// transfer list, so frame data is moved zero-copy across the worker
// boundary instead of being structure-cloned.

import type { DetectorId } from "./models";
import type { FrameStats } from "./pipeline";

// Float32Array.buffer is typed ArrayBufferLike (which includes
// SharedArrayBuffer); the transfer list and ImageData/Float32Array
// constructors all accept it, so widen our pack types to match.
export interface PackedImage {
  width: number;
  height: number;
  buffer: ArrayBufferLike;
}

export interface PackedF32 {
  buffer: ArrayBufferLike;
  length: number;
}

export function packImage(image: ImageData): PackedImage {
  return { width: image.width, height: image.height, buffer: image.data.buffer };
}

export function unpackImage(p: PackedImage): ImageData {
  return new ImageData(new Uint8ClampedArray(p.buffer as ArrayBuffer), p.width, p.height);
}

export function packF32(arr: Float32Array): PackedF32 {
  return { buffer: arr.buffer, length: arr.length };
}

export function unpackF32(p: PackedF32): Float32Array {
  return new Float32Array(p.buffer as ArrayBuffer, 0, p.length);
}

// --- Requests (main -> worker) ---

export interface RpcInit {
  type: "init";
  useGpuPaste: boolean;
}

export interface RpcLoadModels {
  type: "loadModels";
  // Pass the set id rather than the whole ModelSet: the set's ModelFile
  // entries can carry a `transform` containing functions (e.g. xseg's patch
  // applier), which structured-clone refuses across postMessage. The worker
  // resolves the id against its own MODEL_SETS import.
  setId: string;
  enhancerId: string | null;
  detectorId: DetectorId;
}

export interface RpcExtractEmbedding {
  type: "extractEmbedding";
  image: PackedImage;
}

export interface RpcProcessFrame {
  type: "processFrame";
  frame: PackedImage;
  sourceEmbedding: PackedF32;
  useXseg: boolean;
}

export interface RpcPreviewFrame {
  type: "previewFrame";
  file: File;
  time: number;
  scale: number;
  sourceEmbedding: PackedF32;
  useXseg: boolean;
}

export interface RpcProcessVideo {
  type: "processVideo";
  file: File;
  sourceEmbedding: PackedF32;
  startTime: number;
  endTime: number;
  useXseg: boolean;
  scale: number;
}

export interface RpcAbort {
  type: "abort";
}

export interface RpcReleaseModels {
  type: "releaseModels";
}

export type RpcRequestBody =
  | RpcInit
  | RpcLoadModels
  | RpcExtractEmbedding
  | RpcProcessFrame
  | RpcPreviewFrame
  | RpcProcessVideo
  | RpcAbort
  | RpcReleaseModels;

export interface RpcRequest {
  id: number;
  body: RpcRequestBody;
}

// --- Responses + events (worker -> main) ---

export interface RpcOk {
  id: number;
  type: "ok";
  result: unknown;
}

export interface RpcError {
  id: number;
  type: "error";
  error: string;
}

export interface RpcStatsEvent {
  id: number;
  type: "event";
  kind: "stats";
  stats: FrameStats;
}

export type RpcResponse = RpcOk | RpcError | RpcStatsEvent;
