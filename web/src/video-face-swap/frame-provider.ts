// Provider abstraction for the per-frame source feeding the face-swap
// process loop. Two implementations exist:
//
//   - HtmlVideoFrameProvider: seeks an <video> element on the main thread.
//     Used for "off" and "per-frame in worker" modes.
//   - Mp4BoxFrameProvider: parses an mp4 file and decodes via WebCodecs
//     VideoDecoder inside a worker. Used for "full worker" mode.
//
// The shared process-video-loop only depends on this interface, so the same
// extract -> process -> encode -> mux pipeline runs identically regardless
// of which thread the provider lives on.

export interface ProvidedFrame {
  /** Frame pixels at the provider's output (width x height) size. */
  image: ImageData;
  /** 0-based index within the requested time range. */
  index: number;
  /** Total number of frames the provider expects to yield for this range. */
  total: number;
}

export interface FrameProvider {
  /** Output width, already rounded to a multiple of 2 (H.264 requirement). */
  readonly width: number;
  /** Output height, already rounded to a multiple of 2. */
  readonly height: number;
  /** Frames per second the provider will yield. */
  readonly fps: number;

  /**
   * Async iterable that yields frames in order for [startTime, endTime].
   * The shared loop relies on the iterator's `next()` call kicking off the
   * next extract concurrently with the previous frame's processing - i.e.
   * implementations should be safe to call `next()` while the consumer is
   * still working on the previous yielded frame.
   */
  frames(startTime: number, endTime: number): AsyncIterable<ProvidedFrame>;

  /** Release any resources held by the provider. */
  dispose(): void;
}
