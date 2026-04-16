// Main-thread proxy for the pipeline-worker. Owns the Worker instance,
// correlates request/response messages by id, and exposes the same
// per-method surface that main.ts uses on a Pipeline. Used by both the
// "per-frame in worker" and "full worker" modes; the difference between
// those modes is which methods main.ts calls (per-frame mode drives the
// loop locally and only RPCs processFrame; full mode calls processVideo
// and lets the worker run the whole loop).

import type { DetectorId, ModelSet } from "./models";
import type { FrameStats, FrameTimings } from "./pipeline";
import {
  type RpcRequest,
  type RpcRequestBody,
  type RpcResponse,
  packImage,
  unpackImage,
  packF32,
  unpackF32,
  type PackedImage,
} from "./rpc-protocol";

interface PendingCall {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  onEvent?: (ev: RpcResponse & { type: "event" }) => void;
}

export class WorkerHost {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, PendingCall>();
  private aborted = false;
  private log: (msg: string) => void;

  constructor(log: (msg: string) => void) {
    this.log = log;
    this.worker = new Worker(new URL("./pipeline-worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e: MessageEvent<RpcResponse>) => {
      const msg = e.data;
      const pending = this.pending.get(msg.id);
      if (!pending) {
        console.warn("[worker-host] unknown rpc id", msg.id, msg);
        return;
      }
      if (msg.type === "event") {
        if (msg.kind === "debug-log") {
          this.log("[worker] " + msg.msg);
          return;
        }
        pending.onEvent?.(msg);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.type === "ok") {
        pending.resolve(msg.result);
      } else {
        pending.reject(new Error(msg.error));
      }
    };
    this.worker.onerror = (e) => {
      console.error("[worker-host] worker error", e);
      // Reject all pending calls so callers don't hang.
      for (const [, p] of this.pending) {
        p.reject(new Error(`worker error: ${e.message}`));
      }
      this.pending.clear();
    };
  }

  private call<T>(
    body: RpcRequestBody,
    transfer: Transferable[] = [],
    onEvent?: (ev: RpcResponse & { type: "event" }) => void,
  ): Promise<T> {
    const id = this.nextId++;
    const req: RpcRequest = { id, body };
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        onEvent,
      });
      try {
        this.worker.postMessage(req, transfer);
      } catch (err) {
        this.pending.delete(id);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  async init(useGpuPaste: boolean, debug: boolean): Promise<void> {
    await this.call<null>({ type: "init", useGpuPaste, debug });
  }

  async loadModels(
    set: ModelSet,
    enhancerId: string | null,
    detectorId: DetectorId,
  ): Promise<void> {
    await this.call<null>({
      type: "loadModels",
      setId: set.id,
      enhancerId,
      detectorId,
    });
  }

  async extractEmbedding(image: ImageData): Promise<Float32Array> {
    const packed = packImage(image);
    const result = await this.call<ReturnType<typeof packF32>>(
      { type: "extractEmbedding", image: packed },
      [packed.buffer],
    );
    return unpackF32(result);
  }

  /** Per-frame RPC. Returns the swapped frame + per-step timings. */
  async processFrameRpc(
    frame: ImageData,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ): Promise<{ result: ImageData; timings: FrameTimings }> {
    const packedFrame = packImage(frame);
    // Embedding is small (~2KB) and we need to keep using it across many
    // frames, so we copy it (no transfer) on each call.
    const embeddingCopy = new Float32Array(sourceEmbedding);
    const packedEmbed = packF32(embeddingCopy);
    const out = await this.call<{ result: PackedImage; timings: FrameTimings }>(
      {
        type: "processFrame",
        frame: packedFrame,
        sourceEmbedding: packedEmbed,
        useXseg,
      },
      [packedFrame.buffer, packedEmbed.buffer],
    );
    return { result: unpackImage(out.result), timings: out.timings };
  }

  /** Full-worker preview: decode + swap one frame entirely in the worker. */
  async previewFrame(
    file: Blob,
    time: number,
    scale: number,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ): Promise<ImageData> {
    const embeddingCopy = new Float32Array(sourceEmbedding);
    const packedEmbed = packF32(embeddingCopy);
    const out = await this.call<PackedImage>(
      {
        type: "previewFrame",
        file,
        time,
        scale,
        sourceEmbedding: packedEmbed,
        useXseg,
      },
      [packedEmbed.buffer],
    );
    return unpackImage(out);
  }

  /** Full-worker run: extract + process + encode + mux entirely in the worker. */
  async processVideo(
    file: Blob,
    sourceEmbedding: Float32Array,
    startTime: number,
    endTime: number,
    useXseg: boolean,
    scale: number,
    onStats: (stats: FrameStats) => void,
  ): Promise<Blob> {
    this.aborted = false;
    const embeddingCopy = new Float32Array(sourceEmbedding);
    const packedEmbed = packF32(embeddingCopy);
    const blob = await this.call<Blob>(
      {
        type: "processVideo",
        file,
        sourceEmbedding: packedEmbed,
        startTime,
        endTime,
        useXseg,
        scale,
      },
      [packedEmbed.buffer],
      (ev) => {
        if (ev.kind === "stats") onStats(ev.stats);
      },
    );
    return blob;
  }

  abort(): void {
    this.aborted = true;
    // Best-effort: tell the worker too. Fire-and-forget, no await.
    this.call<null>({ type: "abort" }).catch(() => {});
  }

  get isAborted(): boolean {
    return this.aborted;
  }

  async releaseModels(): Promise<void> {
    await this.call<null>({ type: "releaseModels" });
  }

  dispose(): void {
    this.worker.terminate();
    for (const [, p] of this.pending) {
      p.reject(new Error("worker disposed"));
    }
    this.pending.clear();
  }
}
