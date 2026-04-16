// In-worker frame provider that demuxes an mp4 file via mp4box.js and
// decodes a time range with WebCodecs VideoDecoder. Yields frames as
// ImageData drawn through an OffscreenCanvas at the requested output size.
//
// Used by the "full worker" mode where the entire face-swap pipeline runs
// inside a Web Worker. Cannot run on the main thread (it would still work
// technically, but the main thread already has HTMLVideoElement which is
// simpler and uses the browser's built-in decoder).
//
// Strategy:
//   1. Stream the file into mp4box just far enough to parse the moov box.
//   2. Read width/height/fps and the avcC/hvcC description from the video
//      sample entry.
//   3. Walk trak.samples to find the keyframe at or before startTime, then
//      collect all samples up to the first one past endTime.
//   4. Read those sample bytes from the file in one coalesced slice (mdat
//      samples are contiguous, so this is one read regardless of count).
//   5. Feed each sample to a WebCodecs VideoDecoder, push decoded
//      VideoFrames into a small queue, drop frames before startTime,
//      draw frames in [startTime, endTime] to an OffscreenCanvas, yield
//      ImageData.
//
// Color note: drawing through OffscreenCanvas applies the browser's
// default YUV->RGB conversion, which should match the HtmlVideoFrameProvider
// path. If skin tones drift between modes, that's the place to look.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
import { createFile, MP4BoxBuffer, DataStream, type ISOFile } from "mp4box";
import type { FrameProvider, ProvidedFrame } from "./frame-provider";

const READ_CHUNK_SIZE = 1024 * 1024; // 1 MiB

type LogFn = (msg: string) => void;

const noopLog: LogFn = () => {};

interface VideoTrackInfo {
  trackId: number;
  codec: string; // WebCodecs codec string, e.g. "avc1.64001f"
  description: Uint8Array;
  srcWidth: number;
  srcHeight: number;
  fps: number;
}

interface KeptSample {
  offset: number;
  size: number;
  cts: number;
  duration: number;
  timescale: number;
  isKey: boolean;
}

async function streamFileToMp4Box(
  file: Blob,
  mp4: ISOFile,
  shouldStop: () => boolean,
  log: LogFn = noopLog,
): Promise<void> {
  // Slice-based chunked read instead of file.stream().getReader().
  // On Android Chrome the stream reader throws "TypeError: network error"
  // on the very first read when the Blob was postMessage'd from main,
  // but blob.slice(start, end).arrayBuffer() works fine (it's also the
  // same API the sample-byte read path uses further down in frames()).
  log(`streamFileToMp4Box: slice-reading blob.size=${file.size}`);
  let offset = 0;
  let readCount = 0;
  while (offset < file.size) {
    const end = Math.min(offset + READ_CHUNK_SIZE, file.size);
    let buf: ArrayBuffer;
    try {
      buf = await file.slice(offset, end).arrayBuffer();
    } catch (e) {
      const err = e as Error;
      log(
        `streamFileToMp4Box: slice(${offset}, ${end}).arrayBuffer() threw - ${err.name}: ${err.message}`,
      );
      throw e;
    }
    readCount++;
    const last = end >= file.size;
    const mp4Buf = MP4BoxBuffer.fromArrayBuffer(buf, offset);
    mp4.appendBuffer(mp4Buf, last);
    offset = end;
    if (!last && shouldStop()) {
      log(`streamFileToMp4Box: shouldStop at offset=${offset} after ${readCount} reads`);
      return;
    }
  }
  log(`streamFileToMp4Box: EOF after ${readCount} reads, ${offset} bytes`);
  mp4.flush();
}

/**
 * Extract the avcC/hvcC description bytes from the video track's sample
 * entry. WebCodecs VideoDecoder.configure() needs these as the `description`
 * field. We serialize the in-memory box back to bytes via DataStream and
 * strip the 8-byte box header (size + type) that the decoder doesn't want.
 */
function extractVideoDescription(
  mp4: ISOFile,
  trackId: number,
): { description: Uint8Array; isHevc: boolean } | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const trak: any = mp4.getTrackById(trackId);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const entries: any[] = trak?.mdia?.minf?.stbl?.stsd?.entries ?? [];
  for (const entry of entries) {
    const box = entry.avcC ?? entry.hvcC;
    if (!box) continue;
    // BIG_ENDIAN = 1 (per mp4box Endianness enum)
    const stream = new DataStream(undefined, 0, 1);
    box.write(stream);
    const full = new Uint8Array(stream.buffer);
    // Strip the 8-byte box header (4 bytes size + 4 bytes type).
    return { description: full.subarray(8), isHevc: !!entry.hvcC };
  }
  return null;
}

export class Mp4BoxFrameProvider implements FrameProvider {
  readonly width: number;
  readonly height: number;
  readonly fps: number;

  private constructor(
    private readonly file: Blob,
    private readonly track: VideoTrackInfo,
    private readonly samples: KeptSample[],
    width: number,
    height: number,
    private readonly log: LogFn,
  ) {
    this.width = width;
    this.height = height;
    this.fps = track.fps;
  }

  /**
   * Parse the moov, build the sample table, and return a ready-to-iterate
   * provider. Does not start decoding - that happens lazily in frames().
   */
  static async create(
    file: Blob,
    scale: number,
    log: LogFn = noopLog,
  ): Promise<Mp4BoxFrameProvider> {
    const mp4 = createFile(false);
    let info: VideoTrackInfo | null = null;
    let allSamples: KeptSample[] = [];
    let parseError: string | null = null;
    let ready = false;

    mp4.onError = (_m, msg) => {
      parseError = msg;
    };
    mp4.onReady = (movie) => {
      const video = movie.videoTracks[0];
      if (!video || !video.video) {
        parseError = "no video track";
        ready = true;
        return;
      }
      const desc = extractVideoDescription(mp4, video.id);
      if (!desc) {
        parseError = "could not extract video codec description";
        ready = true;
        return;
      }
      const fps =
        video.nb_samples > 0 && video.movie_duration > 0
          ? (video.nb_samples * movie.timescale) / video.movie_duration
          : 30;
      info = {
        trackId: video.id,
        codec: video.codec,
        description: desc.description,
        srcWidth: video.video.width,
        srcHeight: video.video.height,
        fps,
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const trak: any = mp4.getTrackById(video.id);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const samples: any[] = trak?.samples ?? [];
      if (samples.length === 0) {
        parseError = "video track has empty sample list";
        ready = true;
        return;
      }
      for (const s of samples) {
        allSamples.push({
          offset: s.offset,
          size: s.size,
          cts: s.cts,
          duration: s.duration,
          timescale: s.timescale,
          isKey: !!s.is_sync,
        });
      }
      ready = true;
    };

    try {
      await streamFileToMp4Box(file, mp4, () => ready, log);
    } catch (e) {
      throw new Error(`mp4 demux read error: ${e}`);
    }
    if (parseError) throw new Error(`mp4 demux error: ${parseError}`);
    if (!info) throw new Error("mp4 demux: no video track info");

    const trackInfo: VideoTrackInfo = info;
    // H.264 requires even dimensions
    const width = Math.round((trackInfo.srcWidth * scale) / 2) * 2;
    const height = Math.round((trackInfo.srcHeight * scale) / 2) * 2;
    log(
      `Mp4BoxFrameProvider: ready codec=${trackInfo.codec} src=${trackInfo.srcWidth}x${trackInfo.srcHeight} out=${width}x${height} fps=${trackInfo.fps.toFixed(2)} samples=${allSamples.length}`,
    );
    return new Mp4BoxFrameProvider(file, trackInfo, allSamples, width, height, log);
  }

  /**
   * Decode and return a single frame at or after `time`. Used for the preview
   * path, where we want "the frame the user is asking to see" rather than "any
   * frames in a tight window" - the latter is fragile because the first decoded
   * frame's timestamp can be shifted by an edit list or B-frame reordering.
   */
  async decodeFrameAt(time: number): Promise<ImageData> {
    // 2s lookahead covers edit-list shifts and B-frame reordering with
    // plenty of slack, while bounding the mdat slice we coalesce into
    // memory - frames() reads the entire window's bytes at once.
    for await (const f of this.frames(time, time + 2)) {
      return f.image;
    }
    throw new Error(`decodeFrameAt(${time}): no frame produced`);
  }

  async *frames(startTime: number, endTime: number): AsyncGenerator<ProvidedFrame> {
    const samples = this.samples;
    const ts = samples[0].timescale;
    const startTs = startTime * ts;
    const endTs = endTime * ts;

    // Find keyframe at or before startTs, then collect through endTs.
    let firstIdx = 0;
    for (let i = 0; i < samples.length; i++) {
      if (samples[i].cts <= startTs && samples[i].isKey) firstIdx = i;
      if (samples[i].cts > startTs) break;
    }
    let lastIdx = samples.length - 1;
    for (let i = firstIdx; i < samples.length; i++) {
      if (samples[i].cts >= endTs) {
        lastIdx = i;
        break;
      }
    }
    const window = samples.slice(firstIdx, lastIdx + 1);
    this.log(
      `frames: start=${startTime.toFixed(3)}s end=${endTime.toFixed(3)}s firstIdx=${firstIdx} lastIdx=${lastIdx} window=${window.length} fps=${this.fps.toFixed(2)}`,
    );
    if (window.length === 0) return;

    // Coalesced read: one slice covering all sample bytes in the window.
    // Mdat samples are contiguous in fast-start mp4s, so this is typically
    // one big read of (window_duration * bitrate / 8) bytes.
    const byteStart = window[0].offset;
    const byteEnd = window[window.length - 1].offset + window[window.length - 1].size;
    this.log(`frames: slice bytes=${byteStart}..${byteEnd} (${byteEnd - byteStart}B)`);
    const blob = await this.file.slice(byteStart, byteEnd).arrayBuffer();
    const mdat = new Uint8Array(blob);

    // Decoder + queue. The decoder fires `output` from a separate task, so
    // we use a tiny promise-based queue to bridge into the async generator.
    const queue: VideoFrame[] = [];
    let decoderError: Error | null = null;
    let stopped = false;
    let waiter: (() => void) | null = null;
    const wake = () => {
      const w = waiter;
      waiter = null;
      if (w) w();
    };

    let outputCount = 0;
    let firstOutputTs: number | null = null;
    const decoder = new VideoDecoder({
      output: (vf) => {
        outputCount++;
        if (firstOutputTs === null) {
          firstOutputTs = vf.timestamp;
          this.log(`decoder: first output ts=${vf.timestamp}us`);
        }
        // If the consumer has already stopped iterating, don't queue
        // the frame - just close it immediately to avoid the leak.
        if (stopped) {
          vf.close();
          return;
        }
        queue.push(vf);
        wake();
      },
      error: (e) => {
        const err = e instanceof Error ? e : new Error(String(e));
        this.log(`decoder.error: ${err.name}: ${err.message}`);
        decoderError = err;
        wake();
      },
    });
    try {
      decoder.configure({
        codec: this.track.codec,
        description: this.track.description,
        optimizeForLatency: false,
      });
      this.log(`decoder.configure ok codec=${this.track.codec}`);
    } catch (e) {
      const err = e as Error;
      this.log(`decoder.configure threw: ${err.name}: ${err.message}`);
      throw e;
    }

    const totalRangeFrames = Math.ceil((endTime - startTime) * this.fps);
    const canvas = new OffscreenCanvas(this.width, this.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

    // Pump samples into the decoder. Run as a parallel task so the
    // generator body can drain `queue` while feed continues.
    const feed = (async () => {
      try {
        for (const s of window) {
          // Backpressure: if the queue is getting long, give the consumer
          // a chance to drain. 8 frames ~= 250ms at 30fps - small memory
          // hit, hides any per-frame jitter from processFrame.
          while (queue.length >= 8 && !decoderError && !stopped) {
            await new Promise<void>((r) => {
              waiter = r;
            });
          }
          if (decoderError || stopped) return;
          const data = mdat.subarray(s.offset - byteStart, s.offset - byteStart + s.size);
          decoder.decode(
            new EncodedVideoChunk({
              type: s.isKey ? "key" : "delta",
              timestamp: Math.round((s.cts / s.timescale) * 1_000_000),
              duration: Math.round((s.duration / s.timescale) * 1_000_000),
              data,
            }),
          );
        }
        this.log(`feed: decoded samples=${window.length}, flushing...`);
        await decoder.flush();
        this.log(`feed: flush complete, outputCount=${outputCount}, queue=${queue.length}`);
      } finally {
        wake();
      }
    })();

    let yielded = 0;
    const startUs = Math.round(startTime * 1_000_000);
    const endUs = Math.round(endTime * 1_000_000);
    try {
      while (true) {
        if (decoderError) throw decoderError;
        if (queue.length === 0) {
          // Decoder still working? wait for next output.
          // If feed is done and decoder has flushed, queue stays empty - exit.
          let done = false;
          await Promise.race([
            new Promise<void>((r) => {
              waiter = r;
            }),
            feed.then(() => {
              done = true;
            }),
          ]);
          if (queue.length === 0 && done) break;
          continue;
        }
        const vf = queue.shift()!;
        const tUs = vf.timestamp;
        if (tUs < startUs) {
          vf.close();
          continue;
        }
        if (tUs >= endUs) {
          vf.close();
          // Drain remaining queue then exit.
          for (const rest of queue) rest.close();
          queue.length = 0;
          break;
        }
        ctx.drawImage(vf, 0, 0, this.width, this.height);
        vf.close();
        const image = ctx.getImageData(0, 0, this.width, this.height);
        yield { image, index: yielded, total: totalRangeFrames };
        yielded++;
      }
    } finally {
      // Tell feed + decoder.output to stop producing more frames, then
      // close the decoder, drain any frames the feed task pushed between
      // the loop break and now, and let feed unwind.
      stopped = true;
      wake();
      try {
        decoder.close();
      } catch {
        // already closed
      }
      await feed.catch(() => {});
      for (const vf of queue) vf.close();
      queue.length = 0;
    }
  }

  dispose(): void {
    // Per-iteration resources are released inside frames(); nothing to do.
  }
}
