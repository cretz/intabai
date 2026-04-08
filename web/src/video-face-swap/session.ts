// Session is a thin facade over the three swap-execution modes
// (inproc / per-frame in worker / full worker). main.ts picks one at the
// top of the swap handler and then uses the same method surface regardless
// of which one was chosen, so the swap-button click handler stays linear.
//
// All three sessions share the same loadModels / extractEmbedding /
// previewFrame / processVideo / abort / releaseModels surface. The
// differences are encapsulated inside each factory.

import type { DetectorId, ModelSet } from "./models";
import { Pipeline, type FrameStats } from "./pipeline";
import { WorkerHost } from "./run-host-worker";
import { grabFrame, processVideo as inprocProcessVideo } from "./run-host-inproc";
import { HtmlVideoFrameProvider } from "./html-video-frame-provider";
import { processVideoLoop } from "./process-video-loop";
export type WorkerMode = "off" | "perFrame" | "full";

export interface Session {
  loadModels(set: ModelSet, enhancerId: string | null, detectorId: DetectorId): Promise<void>;
  extractEmbedding(image: ImageData): Promise<Float32Array>;
  previewFrame(
    video: HTMLVideoElement,
    file: File,
    time: number,
    scale: number,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ): Promise<ImageData>;
  processVideo(
    video: HTMLVideoElement,
    file: File | null,
    sourceEmbedding: Float32Array,
    startTime: number,
    endTime: number,
    useXseg: boolean,
    scale: number,
    onStats: (stats: FrameStats) => void,
  ): Promise<Blob>;
  abort(): void;
  readonly isAborted: boolean;
  releaseModels(): Promise<void>;
  dispose(): void;
}

class InprocSession implements Session {
  private pipeline: Pipeline;
  constructor(useGpuPaste: boolean) {
    this.pipeline = new Pipeline(useGpuPaste);
  }
  loadModels(set: ModelSet, enhancerId: string | null, detectorId: DetectorId) {
    return this.pipeline.loadModels(set, enhancerId, detectorId);
  }
  extractEmbedding(image: ImageData) {
    return this.pipeline.extractSourceEmbeddingFromImageData(image);
  }
  async previewFrame(
    video: HTMLVideoElement,
    _file: File,
    time: number,
    scale: number,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ) {
    const frame = await grabFrame(video, time, scale);
    return this.pipeline.swapFrame(frame, sourceEmbedding, useXseg);
  }
  processVideo(
    video: HTMLVideoElement,
    file: File | null,
    sourceEmbedding: Float32Array,
    startTime: number,
    endTime: number,
    useXseg: boolean,
    scale: number,
    onStats: (stats: FrameStats) => void,
  ) {
    return inprocProcessVideo(
      this.pipeline,
      video,
      file,
      sourceEmbedding,
      startTime,
      endTime,
      useXseg,
      scale,
      onStats,
    );
  }
  abort() {
    this.pipeline.abort();
  }
  get isAborted() {
    return this.pipeline.isAborted;
  }
  releaseModels() {
    return this.pipeline.releaseModels();
  }
  dispose() {
    // Pipeline holds no top-level resources beyond what releaseModels frees.
  }
}

/** Per-frame in worker: main thread drives extract + mux, worker runs Pipeline. */
class PerFrameWorkerSession implements Session {
  private host: WorkerHost;
  private initPromise: Promise<void>;
  constructor(useGpuPaste: boolean) {
    this.host = new WorkerHost();
    this.initPromise = this.host.init(useGpuPaste);
  }
  async loadModels(set: ModelSet, enhancerId: string | null, detectorId: DetectorId) {
    await this.initPromise;
    await this.host.loadModels(set, enhancerId, detectorId);
  }
  extractEmbedding(image: ImageData) {
    return this.host.extractEmbedding(image);
  }
  async previewFrame(
    video: HTMLVideoElement,
    _file: File,
    time: number,
    scale: number,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ) {
    const frame = await grabFrame(video, time, scale);
    const out = await this.host.processFrameRpc(frame, sourceEmbedding, useXseg);
    return out.result;
  }
  async processVideo(
    video: HTMLVideoElement,
    file: File | null,
    sourceEmbedding: Float32Array,
    startTime: number,
    endTime: number,
    useXseg: boolean,
    scale: number,
    onStats: (stats: FrameStats) => void,
  ) {
    const provider = new HtmlVideoFrameProvider(video, scale);
    try {
      return await processVideoLoop({
        provider,
        processFrame: async (frame) => {
          const out = await this.host.processFrameRpc(frame, sourceEmbedding, useXseg);
          return out;
        },
        audioFile: file,
        startTime,
        endTime,
        onStats,
        isAborted: () => this.host.isAborted,
      });
    } finally {
      provider.dispose();
    }
  }
  abort() {
    this.host.abort();
  }
  get isAborted() {
    return this.host.isAborted;
  }
  releaseModels() {
    return this.host.releaseModels();
  }
  dispose() {
    this.host.dispose();
  }
}

/** Full worker: file goes into the worker; main thread is idle during run. */
class FullWorkerSession implements Session {
  private host: WorkerHost;
  private initPromise: Promise<void>;
  constructor(useGpuPaste: boolean) {
    this.host = new WorkerHost();
    this.initPromise = this.host.init(useGpuPaste);
  }
  async loadModels(set: ModelSet, enhancerId: string | null, detectorId: DetectorId) {
    await this.initPromise;
    await this.host.loadModels(set, enhancerId, detectorId);
  }
  extractEmbedding(image: ImageData) {
    return this.host.extractEmbedding(image);
  }
  previewFrame(
    _video: HTMLVideoElement,
    file: File,
    time: number,
    scale: number,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ) {
    return this.host.previewFrame(file, time, scale, sourceEmbedding, useXseg);
  }
  processVideo(
    _video: HTMLVideoElement,
    file: File | null,
    sourceEmbedding: Float32Array,
    startTime: number,
    endTime: number,
    useXseg: boolean,
    scale: number,
    onStats: (stats: FrameStats) => void,
  ) {
    if (!file) throw new Error("full worker mode requires a source file");
    return this.host.processVideo(
      file,
      sourceEmbedding,
      startTime,
      endTime,
      useXseg,
      scale,
      onStats,
    );
  }
  abort() {
    this.host.abort();
  }
  get isAborted() {
    return this.host.isAborted;
  }
  releaseModels() {
    return this.host.releaseModels();
  }
  dispose() {
    this.host.dispose();
  }
}

export function createSession(mode: WorkerMode, useGpuPaste: boolean): Session {
  switch (mode) {
    case "off":
      return new InprocSession(useGpuPaste);
    case "perFrame":
      return new PerFrameWorkerSession(useGpuPaste);
    case "full":
      return new FullWorkerSession(useGpuPaste);
  }
}
