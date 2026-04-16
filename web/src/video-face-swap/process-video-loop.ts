// Shared face-swap process loop. Pulls frames from a FrameProvider, runs
// each through an injected processFrame function, and writes the results to
// a WebCodecs VideoEncoder + mp4-muxer. Audio is passed through unchanged
// from the source file via the existing demuxer.
//
// This module is context-neutral (no document, no HTMLVideoElement) so it
// runs identically on the main thread or inside a Web Worker. The only two
// pieces of behavior that vary by mode are the FrameProvider implementation
// and the processFrame implementation - both injected here.
//
// The two-frame concurrency from the original main-thread loop is preserved
// implicitly by the for-await-of pattern: the provider's iterator.next()
// kicks off the next extract on the same tick that we set up the previous
// frame's process+encode promise, so extract N+1 runs concurrently with
// process N.

import { Muxer, ArrayBufferTarget } from "mp4-muxer";
import type { FrameProvider } from "./frame-provider";
import type { FrameStats, FrameTimings } from "./pipeline";
import { demuxAudio } from "./audio-passthrough";

export type ProcessFrameFn = (
  frame: ImageData,
) => Promise<{ result: ImageData; timings: FrameTimings }>;

export interface ProcessVideoOptions {
  provider: FrameProvider;
  processFrame: ProcessFrameFn;
  /** Source mp4 file for audio passthrough; null = no audio in output. */
  audioFile: Blob | null;
  startTime: number;
  endTime: number;
  onStats: (stats: FrameStats) => void;
  /** Polled between frames; truthy aborts the loop. */
  isAborted: () => boolean;
}

export async function processVideoLoop(opts: ProcessVideoOptions): Promise<Blob> {
  const { provider, processFrame, audioFile, startTime, endTime, onStats, isAborted } = opts;
  const w = provider.width;
  const h = provider.height;
  const fps = provider.fps;

  // Demux source audio (if any) before opening the muxer, since the muxer
  // needs to know up front whether there's an audio track. Streamed; only
  // the (small) encoded AAC samples for the range stay in memory.
  let audio: Awaited<ReturnType<typeof demuxAudio>> = null;
  if (audioFile) {
    try {
      audio = await demuxAudio(audioFile, startTime, endTime);
    } catch (e) {
      console.warn(`audio demux failed, output will be silent: ${e}`);
    }
  }

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: "avc", width: w, height: h },
    audio: audio
      ? {
          codec: audio.info.codec,
          sampleRate: audio.info.sampleRate,
          numberOfChannels: audio.info.numberOfChannels,
        }
      : undefined,
    fastStart: "in-memory",
    // Audio's first sample may not start exactly at t=0 (we trim to whole
    // AAC frames inside the range, so there can be up to ~23ms of leading
    // silence). cross-track-offset preserves the relative video/audio
    // offset while bypassing the strict-mode "first chunk must be 0" check.
    firstTimestampBehavior: "cross-track-offset",
  });

  // The error callback fires on a later task, so throwing here would not
  // reject the surrounding async function. Stash it and check at the
  // synchronization points (await inFlight, flush) so the loop bails out.
  let encoderError: Error | null = null;
  const throwIfEncoderError = () => {
    if (encoderError) throw encoderError;
  };
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => {
      encoderError = new Error(`encoder error: ${e.message}`);
    },
  });

  // Pick AVC level based on coded area (height rounds up to multiple of 16)
  const codedArea = w * (Math.ceil(h / 16) * 16);
  let avcLevel = "640028"; // 4.0
  if (codedArea > 2097152) avcLevel = "64002A"; // 4.2
  if (codedArea > 8912896) avcLevel = "640033"; // 5.1

  encoder.configure({
    codec: `avc1.${avcLevel}`,
    width: w,
    height: h,
    bitrate: 5_000_000,
    framerate: fps,
  });

  // The render canvas is OffscreenCanvas so this works in a worker too. The
  // current main-thread inproc path used a regular HTMLCanvasElement; both
  // are valid VideoFrame sources and produce identical pixels.
  const renderCanvas = new OffscreenCanvas(w, h);
  const renderCtx = renderCanvas.getContext("2d")!;

  // Rolling window of completion timestamps for fps calculation. With the
  // 2-frame pipeline below, per-frame wall time isn't a meaningful number
  // (extract for N+1 overlaps with process for N), so we measure end-to-end
  // throughput from the gap between consecutive frame completions.
  const completionTimes: number[] = [];
  const windowSize = 10;

  // Process + encode a single already-extracted frame. Returns when the
  // VideoFrame has been queued to the encoder. Designed to overlap with
  // the extract step of the *next* frame.
  const processAndEncodeFrame = async (
    frameData: ImageData,
    frameIndex: number,
    totalFrames: number,
  ): Promise<void> => {
    const { result: swapped, timings } = await processFrame(frameData);

    renderCtx.putImageData(swapped, 0, 0);
    const vf = new VideoFrame(renderCanvas, {
      timestamp: Math.round((frameIndex / fps) * 1_000_000),
      duration: Math.round((1 / fps) * 1_000_000),
    });
    const keyFrame = frameIndex % (fps * 2) === 0; // keyframe every 2 seconds
    encoder.encode(vf, { keyFrame });
    vf.close();

    completionTimes.push(performance.now());
    if (completionTimes.length > windowSize + 1) completionTimes.shift();
    let fpsRate = 0;
    let avgFrameMs = 0;
    if (completionTimes.length >= 2) {
      const span = completionTimes[completionTimes.length - 1] - completionTimes[0];
      avgFrameMs = span / (completionTimes.length - 1);
      fpsRate = 1000 / avgFrameMs;
    }
    const remaining = totalFrames - (frameIndex + 1);
    const etaSeconds = avgFrameMs > 0 ? (remaining * avgFrameMs) / 1000 : 0;

    onStats({
      frameIndex: frameIndex + 1,
      totalFrames,
      fps: fpsRate,
      etaSeconds,
      timings,
    });
  };

  // Two-frame concurrency. The for-await-of loop calls iterator.next() at
  // the top of each iteration, which kicks off the *next* frame's extract;
  // because we don't await `inFlight` until *after* setting up that next
  // extract, processing N overlaps with extracting N+1.
  let inFlight: Promise<void> | null = null;
  for await (const frame of provider.frames(startTime, endTime)) {
    if (isAborted()) {
      console.debug("aborted by user");
      break;
    }
    if (inFlight) await inFlight;
    throwIfEncoderError();
    inFlight = processAndEncodeFrame(frame.image, frame.index, frame.total);
  }
  if (inFlight) await inFlight;
  throwIfEncoderError();

  await encoder.flush();
  throwIfEncoderError();
  encoder.close();

  // Write audio samples (passthrough, no re-encode), trimmed to [startTime, endTime]
  // and offset so the first kept sample lands at timestamp 0.
  if (audio) {
    const startUs = Math.round(startTime * 1_000_000);
    let firstMeta = true;
    let writtenCount = 0;
    for (const s of audio.samples) {
      const tsUs = Math.round((s.cts / s.timescale) * 1_000_000);
      const durUs = Math.round((s.duration / s.timescale) * 1_000_000);
      const meta = firstMeta
        ? {
            decoderConfig: {
              codec: audio.info.codecString,
              sampleRate: audio.info.sampleRate,
              numberOfChannels: audio.info.numberOfChannels,
              description: audio.info.description,
            },
          }
        : undefined;
      firstMeta = false;
      muxer.addAudioChunkRaw(s.data, "key", tsUs - startUs, durUs, meta);
      writtenCount++;
    }
    console.debug(`audio passthrough: wrote ${writtenCount} AAC samples`);
  }

  muxer.finalize();
  return new Blob([target.buffer], { type: "video/mp4" });
}
