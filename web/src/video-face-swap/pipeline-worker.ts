// Web Worker entry point. Holds a single Pipeline instance and dispatches
// RPC requests from the main thread. Handles two operating modes:
//
//   - Per-frame: main thread calls processFrame(image, embedding, useXseg)
//     once per video frame; the worker just runs the compute and returns
//     the result. Used to validate that ORT-WebGPU + MediaPipe + the
//     patched models all work in a Worker context, without yet trusting
//     the in-worker mp4 demux/encode path.
//
//   - Full: main thread sends one processVideo(file, ...) request and the
//     worker runs the entire extract -> process -> encode -> mux loop
//     internally via Mp4BoxFrameProvider + the shared processVideoLoop.
//
// Both modes share init / loadModels / extractEmbedding / releaseModels /
// abort. The protocol is in rpc-protocol.ts.

/// <reference lib="webworker" />

// MediaPipe's wasm loader (vision_bundle.mjs) tries to load its glue script
// via:
//     try { importScripts(url); } catch (e) {
//       if (!(e instanceof TypeError)) throw e;
//       await self.import(url);
//     }
// In a module worker, importScripts is undefined (TypeError), and there is
// no standard `self.import` either. The glue MediaPipe loads is an
// Emscripten UMD bundle that expects classic-worker semantics: it runs
// `var Module = ...` and friends and assumes those become properties on
// the global (`self`). Dynamic import() would evaluate it as an ES module
// instead, leaving those vars module-local - which is why "ModuleFactory
// not set" pops up next. So we polyfill self.import to actually emulate
// importScripts: fetch the script and evaluate it at global scope via
// indirect eval, so its top-level `var`s land on `self`.
{
  const g = self as unknown as { import?: (url: string) => Promise<void> };
  if (typeof g.import !== "function") {
    g.import = async (url: string) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`worker.import: HTTP ${res.status} for ${url}`);
      const src = await res.text();
      // (0, eval) is indirect eval - runs the source at global scope
      // instead of in the calling module's scope, so `var X = ...` at
      // the top level becomes `self.X` like importScripts would do.
      (0, eval)(src);
    };
  }
}

import { Pipeline } from "./pipeline";
import { processVideoLoop } from "./process-video-loop";
import { Mp4BoxFrameProvider } from "./mp4box-frame-provider";
import { MODEL_SETS } from "./models";
import {
  type RpcRequest,
  type RpcResponse,
  type PackedImage,
  packImage,
  unpackImage,
  packF32,
  unpackF32,
} from "./rpc-protocol";

declare const self: DedicatedWorkerGlobalScope;

let pipeline: Pipeline | null = null;
let debugEnabled = false;

function reply(id: number, result: unknown, transfer: Transferable[] = []): void {
  const msg: RpcResponse = { id, type: "ok", result };
  self.postMessage(msg, transfer);
}

function replyError(id: number, err: unknown): void {
  const msg: RpcResponse = { id, type: "error", error: String(err) };
  self.postMessage(msg);
}

function makeLog(id: number): (msg: string) => void {
  return (msg: string) => {
    if (!debugEnabled) return;
    console.log("[video-face-swap worker] " + msg);
    const ev: RpcResponse = { id, type: "event", kind: "debug-log", msg };
    self.postMessage(ev);
  };
}

async function handle(req: RpcRequest): Promise<void> {
  const { id, body } = req;
  try {
    switch (body.type) {
      case "init": {
        pipeline = new Pipeline(body.useGpuPaste);
        debugEnabled = body.debug;
        reply(id, null);
        return;
      }
      case "loadModels": {
        if (!pipeline) throw new Error("worker not initialized");
        const set = MODEL_SETS.find((s) => s.id === body.setId);
        if (!set) throw new Error(`unknown model set: ${body.setId}`);
        await pipeline.loadModels(set, body.enhancerId, body.detectorId);
        reply(id, null);
        return;
      }
      case "extractEmbedding": {
        if (!pipeline) throw new Error("worker not initialized");
        const image = unpackImage(body.image);
        const embedding = await pipeline.extractSourceEmbeddingFromImageData(image);
        const packed = packF32(embedding);
        reply(id, packed, [packed.buffer]);
        return;
      }
      case "processFrame": {
        if (!pipeline) throw new Error("worker not initialized");
        const frame = unpackImage(body.frame);
        const sourceEmbedding = unpackF32(body.sourceEmbedding);
        const { result, timings } = await pipeline.processFrame(
          frame,
          sourceEmbedding,
          body.useXseg,
        );
        const packed: PackedImage = packImage(result);
        reply(id, { result: packed, timings }, [packed.buffer]);
        return;
      }
      case "previewFrame": {
        if (!pipeline) throw new Error("worker not initialized");
        const sourceEmbedding = unpackF32(body.sourceEmbedding);
        const log = makeLog(id);
        log(
          `previewFrame begin file.size=${body.file.size} file.type=${body.file.type || "?"} time=${body.time.toFixed(2)}s scale=${body.scale}`,
        );
        const provider = await Mp4BoxFrameProvider.create(body.file, body.scale, log);
        try {
          const image = await provider.decodeFrameAt(body.time);
          const swapped = await pipeline.swapFrame(image, sourceEmbedding, body.useXseg);
          const packed = packImage(swapped);
          reply(id, packed, [packed.buffer]);
          return;
        } finally {
          provider.dispose();
        }
      }
      case "processVideo": {
        if (!pipeline) throw new Error("worker not initialized");
        pipeline.resetAbort();
        const sourceEmbedding = unpackF32(body.sourceEmbedding);
        const log = makeLog(id);
        log(
          `processVideo begin file.size=${body.file.size} file.type=${body.file.type || "?"} range=${body.startTime.toFixed(2)}-${body.endTime.toFixed(2)}s scale=${body.scale}`,
        );
        const provider = await Mp4BoxFrameProvider.create(body.file, body.scale, log);
        try {
          const blob = await processVideoLoop({
            provider,
            processFrame: (frame) => pipeline!.processFrame(frame, sourceEmbedding, body.useXseg),
            audioFile: body.file,
            startTime: body.startTime,
            endTime: body.endTime,
            onStats: (stats) => {
              const ev: RpcResponse = { id, type: "event", kind: "stats", stats };
              self.postMessage(ev);
            },
            isAborted: () => pipeline!.isAborted,
          });
          // Blob is structured-cloneable; no transferable to extract.
          reply(id, blob);
          return;
        } finally {
          provider.dispose();
        }
      }
      case "abort": {
        pipeline?.abort();
        reply(id, null);
        return;
      }
      case "releaseModels": {
        if (pipeline) {
          await pipeline.releaseModels();
          pipeline = null;
        }
        reply(id, null);
        return;
      }
    }
  } catch (err) {
    console.error("[pipeline-worker] error handling request", body.type, err);
    replyError(id, err);
  }
}

self.onmessage = (e: MessageEvent<RpcRequest>) => {
  handle(e.data);
};
