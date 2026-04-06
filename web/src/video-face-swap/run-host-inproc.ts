// Off-mode (no worker) entry points for the face-swap pipeline. Wraps the
// shared process-video-loop with a main-thread Pipeline + an
// HtmlVideoFrameProvider, plus a tiny grabFrame helper for the preview UI.
//
// The interesting code lives in process-video-loop.ts and the frame
// providers; this file is just the wiring for the "off" mode.

import type { Pipeline } from "./pipeline";
import type { FrameStats } from "./pipeline";
import { HtmlVideoFrameProvider } from "./html-video-frame-provider";
import { processVideoLoop } from "./process-video-loop";

/**
 * Seek an HTMLVideoElement to a given time and resolve only after the
 * painted frame has caught up. The naive `currentTime = t; await onseeked`
 * pattern races on mobile - if a previous in-flight seek (e.g. from the
 * range slider's UX) is still pending, its `seeked` event can fire into
 * our newly-attached handler and resolve us on the wrong frame. Attaching
 * `seeked` BEFORE assigning currentTime avoids that. The trailing rAF
 * gives the painted buffer one frame to catch up with the metadata so
 * drawImage captures the new frame, not the previous one.
 */
async function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
  await new Promise<void>((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = time;
  });
  await new Promise<void>((r) => requestAnimationFrame(() => r()));
}

/** Grab a single frame from a video at a given time, optionally scaled. */
export async function grabFrame(
  video: HTMLVideoElement,
  time: number,
  scale = 1,
): Promise<ImageData> {
  await seekTo(video, time);
  const w = Math.round(video.videoWidth * scale);
  const h = Math.round(video.videoHeight * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(video, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/**
 * Run face swap on a video element for the given time range. Encodes
 * output as MP4 on the fly and returns a Blob. Uses the supplied Pipeline
 * for per-frame compute (in-process, on the main thread).
 */
export async function processVideo(
  pipeline: Pipeline,
  video: HTMLVideoElement,
  sourceFile: File | null,
  sourceEmbedding: Float32Array,
  startTime: number,
  endTime: number,
  useXseg: boolean,
  scale: number,
  onStats: (stats: FrameStats) => void,
): Promise<Blob> {
  pipeline.resetAbort();
  const provider = new HtmlVideoFrameProvider(video, scale);
  try {
    return await processVideoLoop({
      provider,
      processFrame: (frame) => pipeline.processFrame(frame, sourceEmbedding, useXseg),
      audioFile: sourceFile,
      startTime,
      endTime,
      onStats,
      isAborted: () => pipeline.isAborted,
    });
  } finally {
    provider.dispose();
  }
}
