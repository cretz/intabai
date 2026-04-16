// Demuxes the audio track from a source MP4 file so we can pass through the
// already-encoded AAC samples to mp4-muxer without re-encoding (no quality
// loss). Only AAC-LC in MP4 is supported - other codecs/containers return null
// and the output will be silent.
//
// Reads the source file in chunks via file.stream() rather than loading the
// whole thing into memory at once. Probing stops as soon as the moov box has
// been parsed (typically after a few MB for fast-start MP4s).

import { createFile, MP4BoxBuffer, type ISOFile } from "mp4box";

/**
 * A single AAC frame ready to feed mp4-muxer's addAudioChunkRaw. Built from
 * the source's sample table (offset/size/cts) plus the bytes read directly
 * from the file via slice() - no streaming demux needed.
 */
export interface AudioSample {
  data: Uint8Array;
  /** Composition time in source timescale units. */
  cts: number;
  duration: number;
  timescale: number;
}

export interface AudioPassthroughInfo {
  /** Codec accepted by mp4-muxer. Currently always "aac". */
  codec: "aac";
  sampleRate: number;
  numberOfChannels: number;
  /** AudioSpecificConfig bytes for the muxer's decoderConfig.description. */
  description: Uint8Array;
  /** Source codec string (e.g. "mp4a.40.2") for diagnostics. */
  codecString: string;
}

export interface AudioPassthroughResult {
  info: AudioPassthroughInfo;
  /** Audio samples within the requested range, in decode order. */
  samples: AudioSample[];
}

export interface AudioProbeResult {
  /** Reason audio cannot be preserved, or null if it can. */
  unsupported: string | null;
  /** Codec string from the source, if any. */
  codecString?: string;
  /** True if the file has any audio track at all. */
  hasAudio: boolean;
}

const READ_CHUNK_SIZE = 1024 * 1024; // 1 MiB

/**
 * Extract the AudioSpecificConfig bytes from the audio track's esds box.
 * This is the real config the source encoder used (works for AAC-LC,
 * HE-AAC, HE-AACv2, etc) - much more reliable than synthesizing it.
 */
function extractAudioSpecificConfig(mp4: ISOFile, trackId: number): Uint8Array | null {
  const trak = mp4.getTrackById(trackId);
  if (!trak) return null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const entry: any = trak.mdia?.minf?.stbl?.stsd?.entries?.[0];
  const esds = entry?.esds;
  if (!esds?.esd) return null;
  // ES_Descriptor (0x03) > DecoderConfigDescriptor (0x04) > DecoderSpecificInfo (0x05)
  const dcd = esds.esd.findDescriptor(0x04);
  if (!dcd) return null;
  const dsi = dcd.findDescriptor(0x05);
  if (!dsi?.data) return null;
  return new Uint8Array(dsi.data);
}

/**
 * Stream bytes from a file into mp4box in chunks.
 * Returns when the file is exhausted or shouldStop returns true.
 */
async function streamFileToMp4Box(
  file: Blob,
  mp4: ISOFile,
  shouldStop: () => boolean,
): Promise<void> {
  const reader = file.stream().getReader();
  let offset = 0;
  // Buffer up to 1 MiB before flushing to mp4box, to limit appendBuffer overhead
  let pending: Uint8Array[] = [];
  let pendingBytes = 0;

  const flush = (last: boolean) => {
    if (pendingBytes === 0 && !last) return;
    const merged = new Uint8Array(pendingBytes);
    let p = 0;
    for (const c of pending) {
      merged.set(c, p);
      p += c.length;
    }
    pending = [];
    pendingBytes = 0;
    const buf = MP4BoxBuffer.fromArrayBuffer(merged.buffer, offset);
    offset += merged.length;
    mp4.appendBuffer(buf, last);
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending.push(value);
      pendingBytes += value.length;
      if (pendingBytes >= READ_CHUNK_SIZE) {
        flush(false);
        if (shouldStop()) {
          await reader.cancel();
          return;
        }
      }
    }
    flush(true);
    mp4.flush();
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // already released
    }
  }
}

/**
 * Quickly probe a video file to determine whether its audio can be passed
 * through to the output unchanged. Stops reading the file as soon as the
 * moov box has been parsed.
 */
export async function probeAudioPassthrough(file: File): Promise<AudioProbeResult> {
  const mp4 = createFile(false);
  let result: AudioProbeResult | null = null;
  let parseError: string | null = null;

  mp4.onError = (_m, msg) => {
    parseError = msg;
  };
  mp4.onReady = (info) => {
    const audio = info.audioTracks[0];
    if (!audio) {
      result = { unsupported: null, hasAudio: false };
      return;
    }
    const codecString = audio.codec;
    // Accept any AAC variant (mp4a.40.x). The real AudioSpecificConfig is
    // extracted from the source's esds box at demux time.
    if (codecString.startsWith("mp4a.40.")) {
      result = { unsupported: null, hasAudio: true, codecString };
    } else {
      result = {
        unsupported: `audio codec ${codecString} not supported for passthrough`,
        hasAudio: true,
        codecString,
      };
    }
  };

  try {
    await streamFileToMp4Box(file, mp4, () => result !== null);
  } catch (e) {
    return { unsupported: `read error: ${e}`, hasAudio: false };
  }

  if (result) return result;
  if (parseError) return { unsupported: `mp4 parse error: ${parseError}`, hasAudio: false };
  return { unsupported: "mp4 had no moov box", hasAudio: false };
}

interface SampleTableEntry {
  offset: number;
  size: number;
  cts: number;
  duration: number;
}

/**
 * Demux the source file's audio track for the given time range. Streams just
 * enough of the file for mp4box to parse the moov header (a few MB), then
 * walks the audio sample table directly to find which sample byte ranges
 * overlap [startTime, endTime] and reads only those bytes via file.slice().
 *
 * Memory cost: ~moov size + (audio bitrate * range duration). For a 5-sec
 * slice of a 43-min HE-AAC file that is well under a megabyte total.
 *
 * Returns null if there is no audio track or the codec is unsupported.
 */
export async function demuxAudio(
  file: Blob,
  startTime: number = 0,
  endTime: number = Infinity,
): Promise<AudioPassthroughResult | null> {
  const mp4: ISOFile = createFile(false);
  let info: AudioPassthroughInfo | null = null;
  let parseError: string | null = null;
  let ready = false;
  let entries: SampleTableEntry[] = [];
  let timescale = 0;

  mp4.onError = (_m, msg) => {
    parseError = msg;
  };
  mp4.onReady = (movie) => {
    const audio = movie.audioTracks[0];
    if (!audio || !audio.audio) {
      ready = true;
      return;
    }
    const codecString = audio.codec;
    if (!codecString.startsWith("mp4a.40.")) {
      ready = true;
      return;
    }
    const description = extractAudioSpecificConfig(mp4, audio.id);
    if (!description) {
      parseError = "could not find AudioSpecificConfig in esds box";
      ready = true;
      return;
    }
    info = {
      codec: "aac",
      sampleRate: audio.audio.sample_rate,
      numberOfChannels: audio.audio.channel_count,
      description,
      codecString,
    };

    // mp4box has already populated trak.samples with offset/size/cts/duration
    // for every sample in the moov. Walk it, filter by range, and remember
    // which byte ranges to read.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const trak: any = mp4.getTrackById(audio.id);
    const allSamples = trak?.samples ?? [];
    if (allSamples.length === 0) {
      parseError = "audio track has empty sample list";
      ready = true;
      return;
    }
    timescale = allSamples[0].timescale;
    const startTs = startTime * timescale;
    const endTs = endTime * timescale;
    for (const s of allSamples) {
      // Only keep samples that lie ENTIRELY inside [startTs, endTs]. This
      // guarantees zero audio/video drift in the output - the cost is up to
      // ~one AAC frame (~23ms) of silence at the start and end of the range.
      if (s.cts < startTs) continue;
      if (s.cts + s.duration > endTs) break;
      entries.push({
        offset: s.offset,
        size: s.size,
        cts: s.cts,
        duration: s.duration,
      });
    }
    ready = true;
  };

  // Stream just enough of the file for mp4box to parse the moov and fire
  // onReady. For fast-start MP4s this stops after a few MB; for moov-at-end
  // files it has to read the whole file (unavoidable without HTTP range).
  try {
    await streamFileToMp4Box(file, mp4, () => ready);
  } catch (e) {
    throw new Error(`mp4 demux read error: ${e}`);
  }

  if (parseError) throw new Error(`mp4 demux error: ${parseError}`);
  if (!info) return null;
  if (entries.length === 0) {
    return { info, samples: [] };
  }

  // Read each selected sample's bytes directly from the file. We coalesce
  // adjacent (or near-adjacent) ranges into a single slice read to amortize
  // I/O - audio samples in mp4 are usually packed contiguously in the mdat.
  const samples: AudioSample[] = await readSampleBytes(file, entries, timescale);
  return { info, samples };
}

/**
 * Read sample bytes from the file. Adjacent samples in the sample table are
 * almost always contiguous in the mdat, so we coalesce them into a single
 * slice + arrayBuffer call per run.
 */
async function readSampleBytes(
  file: Blob,
  entries: SampleTableEntry[],
  timescale: number,
): Promise<AudioSample[]> {
  const out: AudioSample[] = Array.from({ length: entries.length });

  let runStart = 0;
  while (runStart < entries.length) {
    let runEnd = runStart + 1;
    let runByteStart = entries[runStart].offset;
    let runByteEnd = runByteStart + entries[runStart].size;
    // Extend run while next sample is contiguous (or adjacent) in the file
    while (runEnd < entries.length) {
      const next = entries[runEnd];
      if (next.offset !== runByteEnd) break;
      runByteEnd = next.offset + next.size;
      runEnd++;
    }
    const buf = new Uint8Array(await file.slice(runByteStart, runByteEnd).arrayBuffer());
    let cursor = 0;
    for (let i = runStart; i < runEnd; i++) {
      const e = entries[i];
      out[i] = {
        data: buf.subarray(cursor, cursor + e.size),
        cts: e.cts,
        duration: e.duration,
        timescale,
      };
      cursor += e.size;
    }
    runStart = runEnd;
  }
  return out;
}
