// Progress video: renders a progress ring, stats, and status text into
// a <video> element via canvas captureStream. The video is the primary
// on-page progress display during generation.
//
// A sub-audible audio track (19 kHz, gain 0.01) is added to the stream
// so Chrome considers the tab as playing audio, which protects it from
// tab discarding for ~1 minute after playback.
//
// The caller can optionally request PiP via enterPip() to keep the GPU
// alive when switching apps on mobile (Android Chrome reclaims GPU
// context from background tabs).

const CANVAS_SIZE = 200;
const RING_RADIUS = 60;
const RING_WIDTH = 14;
const TITLE_FONT_SIZE = 16;
const TITLE_Y = 22;
const DRAW_INTERVAL_MS = 250;

export class ProgressVideo {
  private ctx: CanvasRenderingContext2D | null = null;
  private stream: MediaStream | null = null;
  private audioCtx: AudioContext | null = null;
  private intervalId = 0;
  private title = "";
  private active = false;
  private log: (msg: string) => void = () => {};

  private progressPct = 0;
  private statusText = "";
  private statsText = "";

  constructor(private readonly video: HTMLVideoElement) {
    // Re-play after PiP exit or visibility change (screen lock/unlock).
    video.addEventListener("leavepictureinpicture", () => {
      this.log(
        "[progress-video] leavepictureinpicture, paused=" +
          video.paused +
          " readyState=" +
          video.readyState,
      );
      // Resume even after stop() so the final frame stays visible inline.
      video.play().catch((e) => this.log("[progress-video] play() after leavepip failed: " + e));
      this.tick();
    });
    video.addEventListener("pause", () => {
      this.log("[progress-video] pause event");
      // Resume even after stop() so the video doesn't go black.
      if (video.srcObject) {
        video.play().catch((e) => this.log("[progress-video] play() after pause failed: " + e));
      }
    });
    document.addEventListener("visibilitychange", () => {
      this.log(
        "[progress-video] visibilitychange: " +
          document.visibilityState +
          " paused=" +
          video.paused,
      );
      if (this.active) {
        video
          .play()
          .catch((e) => this.log("[progress-video] play() after visibility failed: " + e));
      }
    });
  }

  /** Begin a generation session. Must be called during a user gesture
   *  (for AudioContext.resume). */
  async start(title: string, log?: (msg: string) => void): Promise<void> {
    this.log = log ?? (() => {});
    this.title = title;
    if (this.active) return;

    // Stop tracks from any previous stream to avoid leaking MediaStreamTracks
    // across repeated generations in the same session.
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      this.log("[progress-video] failed to get 2d context");
      return;
    }
    this.ctx = ctx;

    this.progressPct = 0;
    this.statusText = "";
    this.statsText = "";
    this.draw();

    const stream = canvas.captureStream(0);

    // Sub-audible audio: 19 kHz at gain 0.01. Chrome detects this as
    // audio playback, protecting the tab from discard when backgrounded.
    try {
      const audioCtx = new AudioContext();
      if (audioCtx.state === "suspended") await audioCtx.resume();
      const oscillator = audioCtx.createOscillator();
      oscillator.frequency.value = 19000;
      const gain = audioCtx.createGain();
      gain.gain.value = 0.01;
      oscillator.connect(gain);
      const dest = audioCtx.createMediaStreamDestination();
      gain.connect(dest);
      oscillator.start();
      stream.addTrack(dest.stream.getAudioTracks()[0]);
      this.audioCtx = audioCtx;
      this.log("[progress-video] audio track added (19 kHz, gain 0.01)");
    } catch (err) {
      this.log("[progress-video] failed to create audio: " + (err as Error).message);
    }

    this.video.srcObject = stream;
    this.stream = stream;

    if ("mediaSession" in navigator) {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: this.title,
        artist: "intabai",
        album: "in-browser AI",
      });
      navigator.mediaSession.playbackState = "playing";
      navigator.mediaSession.setPositionState({
        duration: 3600,
        playbackRate: 1.0,
        position: 0,
      });
      try {
        navigator.mediaSession.setActionHandler("play", () => {});
      } catch {
        /* unsupported */
      }
      try {
        navigator.mediaSession.setActionHandler("pause", () => {});
      } catch {
        /* unsupported */
      }
    }

    try {
      await this.video.play();
      this.log("[progress-video] playing");
    } catch (err) {
      this.log("[progress-video] failed to play: " + (err as Error).message);
      this.stop();
      return;
    }

    this.intervalId = window.setInterval(() => this.tick(), DRAW_INTERVAL_MS);
    this.active = true;
  }

  /** Request PiP on the video. Must be called during a user gesture. */
  async enterPip(): Promise<void> {
    try {
      await this.video.requestPictureInPicture();
      this.log("[progress-video] entered PiP");
    } catch (err) {
      this.log("[progress-video] PiP failed: " + (err as Error).message);
    }
  }

  setProgress(pct: number): void {
    this.progressPct = Number.isFinite(pct) ? Math.min(100, Math.max(0, pct)) : 0;
  }

  setStatus(text: string): void {
    this.statusText = text;
  }

  setStats(text: string): void {
    this.statsText = text;
  }

  /** Stop updating. Draws one final frame and leaves it visible.
   *  The stream stays connected so the video doesn't go blank.
   *  Audio and PiP are torn down; start() replaces the stream next time. */
  stop(): void {
    if (!this.active) return;
    this.active = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = 0;
    }
    // Push one last frame so the video shows the final state.
    this.tick();
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {});
      this.audioCtx = null;
    }
    if ("mediaSession" in navigator) {
      navigator.mediaSession.playbackState = "none";
    }
  }

  isActive(): boolean {
    return this.active;
  }

  private tick(): void {
    this.draw();
    const track = this.stream?.getVideoTracks()[0] as
      | (MediaStreamTrack & { requestFrame?: () => void })
      | undefined;
    if (track?.requestFrame) {
      track.requestFrame();
    }
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const size = CANVAS_SIZE;
    const cx = size / 2;
    const cy = size / 2 + 6;

    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, size, size);

    if (this.title) {
      ctx.fillStyle = "#ccc";
      ctx.font = `${TITLE_FONT_SIZE}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(this.title, size / 2, TITLE_Y);
    }

    ctx.beginPath();
    ctx.arc(cx, cy, RING_RADIUS, 0, Math.PI * 2);
    ctx.lineWidth = RING_WIDTH;
    ctx.strokeStyle = "#333";
    ctx.stroke();

    const frac = this.progressPct / 100;
    if (frac > 0) {
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + frac * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(cx, cy, RING_RADIUS, startAngle, endAngle);
      ctx.lineWidth = RING_WIDTH;
      ctx.strokeStyle = "#4a9eff";
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.lineCap = "butt";
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (this.statsText) {
      const maxWidth = RING_RADIUS * 2 - 16;
      ctx.font = "13px sans-serif";
      ctx.fillStyle = "#ddd";
      const lines = this.wrapText(ctx, this.statsText, maxWidth);
      const lineHeight = 16;
      const startY = cy - ((lines.length - 1) * lineHeight) / 2;
      for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], cx, startY + i * lineHeight);
      }
    }

    if (this.statusText) {
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#999";
      ctx.textBaseline = "bottom";
      ctx.fillText(
        this.statusText.length > 30 ? this.statusText.slice(0, 29) + "\u2026" : this.statusText,
        size / 2,
        size - 6,
      );
    }
  }

  private wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    const segments = text.split(/\s*\|\s*/);
    const lines: string[] = [];
    for (const seg of segments) {
      const words = seg.split(" ");
      let line = "";
      for (const word of words) {
        const test = line ? line + " " + word : word;
        if (ctx.measureText(test).width > maxWidth && line) {
          lines.push(line);
          line = word;
        } else {
          line = test;
        }
      }
      if (line) lines.push(line);
    }
    return lines;
  }
}
