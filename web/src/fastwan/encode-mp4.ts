// Encode a sequence of ImageBitmap frames to an MP4 Blob using WebCodecs
// + mp4-muxer. Mirrors the pattern in video-face-swap/process-video-loop.ts
// but simplified: no audio, no streaming frame source, all frames already
// decoded in memory.
//
// AVC level picked from coded area. keyFrame every 2s. Bitrate 5 Mbps is
// generous for 480x480@16 but the file is only ~5s long so the absolute
// size is small.

import { Muxer, ArrayBufferTarget } from "mp4-muxer";

export interface EncodeMp4Options {
  frames: ImageBitmap[];
  fps: number;
  onProgress?: (done: number, total: number) => void;
}

export async function encodeFramesToMp4(opts: EncodeMp4Options): Promise<Blob> {
  const { frames, fps, onProgress } = opts;
  if (frames.length === 0) throw new Error("no frames to encode");

  const w = frames[0].width;
  const h = frames[0].height;

  // Thresholds are each AVC level's MaxFS in samples; see process-video-loop.ts.
  const codedArea = w * (Math.ceil(h / 16) * 16);
  let avcLevel = "640028"; // 4.0
  if (codedArea > 2097152) avcLevel = "64002A"; // 4.2
  if (codedArea > 2228224) avcLevel = "640032"; // 5.0
  if (codedArea > 5652480) avcLevel = "640033"; // 5.1
  if (codedArea > 9437184) avcLevel = "640034"; // 5.2

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: "avc", width: w, height: h },
    fastStart: "in-memory",
  });

  let encoderError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => {
      encoderError = new Error(`encoder error: ${e.message}`);
    },
  });
  encoder.configure({
    codec: `avc1.${avcLevel}`,
    width: w,
    height: h,
    bitrate: 5_000_000,
    framerate: fps,
  });

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("failed to get OffscreenCanvas 2d context");

  for (let i = 0; i < frames.length; i++) {
    if (encoderError) throw encoderError;
    ctx.drawImage(frames[i], 0, 0);
    const vf = new VideoFrame(canvas, {
      timestamp: Math.round((i / fps) * 1_000_000),
      duration: Math.round((1 / fps) * 1_000_000),
    });
    const keyFrame = i % (fps * 2) === 0;
    encoder.encode(vf, { keyFrame });
    vf.close();
    onProgress?.(i + 1, frames.length);
  }

  await encoder.flush();
  if (encoderError) throw encoderError;
  encoder.close();
  muxer.finalize();
  return new Blob([target.buffer], { type: "video/mp4" });
}
