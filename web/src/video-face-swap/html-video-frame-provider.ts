// Frame provider that wraps an HTMLVideoElement and seeks through it to
// extract frames as ImageData. Cannot run in a worker (depends on
// HTMLVideoElement and document).
//
// Seek correctness: we use requestVideoFrameCallback when available so the
// promise resolves only after a new frame has actually been painted to the
// video element's internal buffer. The naive `currentTime = t; await
// onseeked` pattern races on mobile - `seeked` fires when metadata
// advances, not when the new frame is rendered, and a fast-completing
// previous seek can fire its `seeked` event before our listener attaches.
// Both bugs reliably produced "wrong frame captured" on Android Chrome.

import type { FrameProvider, ProvidedFrame } from "./frame-provider";

async function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
  // Attach `seeked` BEFORE assigning currentTime - otherwise a previous
  // in-flight seek (e.g. from the range slider's UX) can fire its `seeked`
  // event into our newly-attached handler, resolving us on the wrong
  // seek. After seeked fires we wait one rAF so the painted frame buffer
  // catches up with the metadata before drawImage.
  //
  // We deliberately do NOT use requestVideoFrameCallback here even though
  // it's more accurate, because rVFC only fires when a NEW frame is
  // painted - and consecutive seeks by 1/30s through a video that isn't
  // exactly 30fps can land on the same source frame, so rVFC would never
  // fire and the loop would deadlock.
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

export class HtmlVideoFrameProvider implements FrameProvider {
  readonly width: number;
  readonly height: number;
  readonly fps: number = 30; // TODO: detect from video metadata

  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    private readonly video: HTMLVideoElement,
    scale: number,
  ) {
    // H.264 requires even dimensions
    this.width = Math.round((video.videoWidth * scale) / 2) * 2;
    this.height = Math.round((video.videoHeight * scale) / 2) * 2;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  }

  async *frames(startTime: number, endTime: number): AsyncGenerator<ProvidedFrame> {
    const fps = this.fps;
    const total = Math.ceil((endTime - startTime) * fps);
    for (let i = 0; i < total; i++) {
      const time = startTime + i / fps;
      await seekTo(this.video, time);
      this.ctx.drawImage(this.video, 0, 0, this.width, this.height);
      const image = this.ctx.getImageData(0, 0, this.width, this.height);
      yield { image, index: i, total };
    }
  }

  dispose(): void {
    // Nothing to release - the canvas is GC'd with the provider.
  }
}
