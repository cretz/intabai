import noUiSlider, { type API as NoUiSliderAPI } from "nouislider";
import "nouislider/dist/nouislider.css";

export interface VideoRangeElements {
  details: HTMLDetailsElement;
  slider: HTMLDivElement;
  startInput: HTMLInputElement;
  previewInput: HTMLInputElement;
  previewLabel: HTMLElement;
  previewCheck: HTMLInputElement;
  endInput: HTMLInputElement;
  limitCheck: HTMLInputElement;
}

export class VideoInput {
  private video: HTMLVideoElement;
  private statusEl: HTMLElement;
  private onLoadCallback: (() => void) | null = null;

  private slider: NoUiSliderAPI | null = null;
  private previewMarker: HTMLDivElement | null = null;
  private rangeStart = 0;
  private rangePreview = 0;
  private rangeEnd = 0;
  private file: File | null = null;
  private currentObjectUrl: string | null = null;

  constructor(
    private fileInput: HTMLInputElement,
    private previewContainer: HTMLDivElement,
    private rangeEls: VideoRangeElements,
  ) {
    this.video = document.createElement("video");
    this.video.controls = true;
    this.video.loop = true;
    this.video.style.maxWidth = "100%";
    this.video.style.display = "none";
    this.previewContainer.appendChild(this.video);

    this.statusEl = document.createElement("p");
    this.statusEl.style.display = "none";
    this.previewContainer.appendChild(this.statusEl);

    this.fileInput.addEventListener("change", () => this.onFileChange());

    this.rangeEls.startInput.addEventListener("change", () =>
      this.setRangeFromInput("start", this.rangeEls.startInput),
    );
    this.rangeEls.endInput.addEventListener("change", () =>
      this.setRangeFromInput("end", this.rangeEls.endInput),
    );
    this.rangeEls.previewInput.addEventListener("change", () => this.setPreviewFromInput());
    this.rangeEls.previewCheck.addEventListener("change", () => this.applyPreviewIndependence());
    this.rangeEls.limitCheck.addEventListener("change", () => this.applyRangeLimit());

    // Loop within range when limit is on; otherwise the native `loop` attr loops
    // the whole video.
    this.video.addEventListener("timeupdate", () => {
      if (!this.rangeEls.limitCheck.checked) return;
      if (this.video.currentTime < this.rangeStart) this.video.currentTime = this.rangeStart;
      if (this.video.currentTime >= this.rangeEnd) {
        this.video.currentTime = this.rangeStart;
      }
    });
  }

  private onFileChange(): void {
    const file = this.fileInput.files?.[0];
    if (!file) return;
    this.file = file;
    if (this.currentObjectUrl) URL.revokeObjectURL(this.currentObjectUrl);
    this.currentObjectUrl = URL.createObjectURL(file);
    this.video.src = this.currentObjectUrl;
    this.video.style.display = "block";
    this.statusEl.style.display = "none";

    this.video.onloadedmetadata = () => {
      const w = this.video.videoWidth;
      const h = this.video.videoHeight;
      const dur = this.video.duration;
      this.updateStatus(`${w}x${h}, ${dur.toFixed(1)}s`);
      this.initRangeSlider();
      this.onLoadCallback?.();
    };

    // Audio probing is deferred to swap time: on Android, the picked File's
    // content-URI permission is fragile and each read risks burning it.
    // demuxAudio runs inside processVideoLoop from a fresh read at swap time.

    this.video.onerror = () => {
      this.statusEl.textContent = "failed to load video";
      this.statusEl.style.color = "red";
      this.statusEl.style.display = "";
    };
  }

  // --- Range slider ---

  private initRangeSlider(): void {
    const duration = this.video.duration;
    if (!isFinite(duration) || duration <= 0) return;

    this.rangeStart = 0;
    this.rangeEnd = duration;
    this.rangePreview = 0;
    this.rangeEls.previewCheck.checked = false;

    if (this.slider) {
      this.slider.destroy();
      this.slider = null;
      this.previewMarker = null;
    }

    this.slider = noUiSlider.create(this.rangeEls.slider, {
      start: [this.rangeStart, this.rangeEnd],
      range: { min: 0, max: duration },
      connect: [false, true, false],
      behaviour: "drag-tap",
      step: 0.01,
    });

    // Non-interactive preview marker overlaid on the slider track
    const marker = document.createElement("div");
    marker.className = "range-preview-marker";
    marker.innerHTML = `<div class="range-preview-line"></div><div class="range-preview-tag">preview</div>`;
    this.rangeEls.slider.appendChild(marker);
    this.previewMarker = marker;

    this.slider.on("update", (values) => {
      this.rangeStart = Number(values[0]);
      this.rangeEnd = Number(values[1]);
      if (!this.rangeEls.previewCheck.checked) {
        this.rangePreview = this.rangeStart;
      } else {
        if (this.rangePreview < this.rangeStart) this.rangePreview = this.rangeStart;
        if (this.rangePreview > this.rangeEnd) this.rangePreview = this.rangeEnd;
      }
      this.rangeEls.startInput.value = formatTime(this.rangeStart);
      this.rangeEls.endInput.value = formatTime(this.rangeEnd);
      this.rangeEls.previewInput.value = formatTime(this.rangePreview);
      this.updatePreviewMarker();
      this.updateRangeSummary();
    });

    // Detect range-bar drag (both handles move together) vs single-handle slide.
    // When the whole range drags, also shift the preview by the same delta.
    let dragStartValues: number[] | null = null;
    this.slider.on("start", (values) => {
      dragStartValues = values.map(Number);
    });
    this.slider.on("slide", (values, handle) => {
      const t = Number(values[handle]);
      if (isFinite(t)) this.video.currentTime = t;

      if (dragStartValues && this.rangeEls.previewCheck.checked && values.length === 2) {
        const newStart = Number(values[0]);
        const newEnd = Number(values[1]);
        const startDelta = newStart - dragStartValues[0];
        const endDelta = newEnd - dragStartValues[1];
        // Both handles moved by the same delta -> range bar drag
        if (Math.abs(startDelta - endDelta) < 0.001 && Math.abs(startDelta) > 0) {
          this.rangePreview += startDelta;
          if (this.rangePreview < this.rangeStart) this.rangePreview = this.rangeStart;
          if (this.rangePreview > this.rangeEnd) this.rangePreview = this.rangeEnd;
          this.rangeEls.previewInput.value = formatTime(this.rangePreview);
          this.updatePreviewMarker();
          dragStartValues = [newStart, newEnd];
        }
      }
    });
    this.slider.on("end", () => {
      dragStartValues = null;
    });

    this.applyPreviewIndependence();
    this.rangeEls.details.style.display = "";
    this.updateRangeSummary();
    this.updatePreviewMarker();
    this.applyRangeLimit();
  }

  private applyPreviewIndependence(): void {
    const independent = this.rangeEls.previewCheck.checked;
    this.rangeEls.previewLabel.style.display = independent ? "" : "none";
    this.rangeEls.previewInput.style.display = independent ? "" : "none";
    if (independent) {
      // Default to middle of current range when first turned on
      if (this.rangePreview <= this.rangeStart || this.rangePreview >= this.rangeEnd) {
        this.rangePreview = (this.rangeStart + this.rangeEnd) / 2;
      }
    } else {
      this.rangePreview = this.rangeStart;
    }
    this.rangeEls.previewInput.value = formatTime(this.rangePreview);
    this.updatePreviewMarker();
  }

  private updatePreviewMarker(): void {
    if (!this.previewMarker) return;
    const duration = this.video.duration;
    if (!isFinite(duration) || duration <= 0) return;
    const pct = (this.rangePreview / duration) * 100;
    this.previewMarker.style.left = `${pct}%`;
    this.previewMarker.style.display = this.rangeEls.previewCheck.checked ? "" : "none";
  }

  private updateRangeSummary(): void {
    const summary = this.rangeEls.details.querySelector("summary");
    if (!summary) return;
    const isFull = this.rangeStart <= 0.001 && this.rangeEnd >= this.video.duration - 0.001;
    summary.textContent = isFull
      ? "video subset (full video)"
      : `video subset (${formatTime(this.rangeStart)} - ${formatTime(this.rangeEnd)})`;
  }

  private setRangeFromInput(which: "start" | "end", input: HTMLInputElement): void {
    if (!this.slider) return;
    const t = parseTime(input.value);
    if (t == null) {
      input.value = formatTime(which === "start" ? this.rangeStart : this.rangeEnd);
      return;
    }
    if (which === "start") {
      this.slider.set([t, null]);
    } else {
      this.slider.set([null, t]);
    }
    this.video.currentTime = which === "start" ? this.rangeStart : this.rangeEnd;
  }

  private setPreviewFromInput(): void {
    const t = parseTime(this.rangeEls.previewInput.value);
    if (t == null) {
      this.rangeEls.previewInput.value = formatTime(this.rangePreview);
      return;
    }
    let clamped = t;
    if (clamped < this.rangeStart) clamped = this.rangeStart;
    if (clamped > this.rangeEnd) clamped = this.rangeEnd;
    this.rangePreview = clamped;
    this.rangeEls.previewInput.value = formatTime(this.rangePreview);
    this.updatePreviewMarker();
    this.video.currentTime = this.rangePreview;
  }

  private applyRangeLimit(): void {
    if (!this.rangeEls.limitCheck.checked) return;
    if (this.video.currentTime < this.rangeStart) this.video.currentTime = this.rangeStart;
    if (this.video.currentTime > this.rangeEnd) {
      this.video.currentTime = this.rangeStart;
    }
  }

  // --- Public API ---

  onLoad(cb: () => void): void {
    this.onLoadCallback = cb;
  }

  getFile(): File | null {
    return this.file;
  }

  private updateStatus(base: string): void {
    this.baseStatus = base;
    this.refreshStatus();
  }

  private baseStatus = "";

  private refreshStatus(): void {
    this.statusEl.textContent = this.baseStatus;
    this.statusEl.style.display = "";
  }

  hasVideo(): boolean {
    return this.video.src !== "" && this.video.readyState >= 1;
  }

  getVideo(): HTMLVideoElement | null {
    return this.hasVideo() ? this.video : null;
  }

  getVideoWidth(): number {
    return this.video.videoWidth;
  }

  getVideoHeight(): number {
    return this.video.videoHeight;
  }

  getDuration(): number {
    return this.video.duration;
  }

  getRangeStart(): number {
    return this.rangeStart;
  }

  getRangeEnd(): number {
    return this.rangeEnd;
  }

  getPreviewTime(): number {
    return this.rangePreview;
  }
}

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

function parseTime(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const colonMatch = trimmed.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (colonMatch) {
    return Number(colonMatch[1]) * 60 + Number(colonMatch[2]);
  }
  const n = Number(trimmed);
  return isFinite(n) ? n : null;
}
