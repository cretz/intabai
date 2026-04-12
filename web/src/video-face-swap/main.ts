import * as ort from "onnxruntime-web";
import { VideoInput } from "./video-input";
import { FaceInput } from "./face-input";
import { ModelManager } from "./model-manager";
import { type FrameTimings } from "./pipeline";
import { createSession, type WorkerMode } from "./session";
import { MODEL_SETS, type DetectorId } from "./models";
import {
  PersistedSettings as PersistedSettingsStore,
  isMobile,
} from "../shared/persisted-settings";
import { initThemeSelect } from "../shared/theme";

{
  const sel = document.getElementById("theme-select");
  if (sel instanceof HTMLSelectElement) initThemeSelect(sel);
}

// Persisted UI settings: everything that isn't per-video or per-face (so no
// file inputs, no time ranges) so that the next visit can pick up where the
// last one left off. Mobile-friendly defaults are applied on first visit so
// the tool is usable out of the box on phones.
interface PersistedSettings {
  swapModel: string;
  detector: string;
  enhancer: string;
  useXseg: boolean;
  doPreview: boolean;
  downscale: string;
  rangeLimit: boolean;
  separatePreview: boolean;
  rangePanelOpen: boolean;
  advancedPanelOpen: boolean;
  profilePreview: boolean;
  gpuPaste: boolean;
  workerMode: WorkerMode;
}

const settingsStore = new PersistedSettingsStore<PersistedSettings>(
  "intabai-video-face-swap-settings",
);

/** Mobile-tuned defaults: skip slow CPU work, use the fast detector. */
function getMobileDefaults(): Partial<PersistedSettings> {
  return {
    detector: "scrfd_500m",
    enhancer: "",
    useXseg: false,
    doPreview: true,
    downscale: "720", // 720p target if available
    rangeLimit: false,
    separatePreview: false,
    rangePanelOpen: false,
    advancedPanelOpen: false,
    profilePreview: false,
    gpuPaste: true,
  };
}

// Check browser compatibility
{
  const browserError = document.getElementById("browser-error")!;
  if (typeof VideoEncoder === "undefined") {
    browserError.style.display = "";
    browserError.textContent =
      "This tool requires WebCodecs (VideoEncoder) which is not supported in your browser.";
    document.getElementById("swap-btn")!.setAttribute("disabled", "true");
  }
}

// Check WebGPU support and show warning if unavailable
{
  const gpuWarning = document.getElementById("gpu-warning")!;
  let reason = "";
  if (!navigator.gpu) {
    reason = "your browser does not support WebGPU";
  } else {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        reason = "no GPU adapter found";
      }
    } catch (e) {
      reason = `GPU adapter request failed: ${e}`;
    }
  }
  if (reason) {
    gpuWarning.style.display = "";
    gpuWarning.textContent = `WebGPU not available (${reason}). Processing will use WASM (CPU) and be significantly slower.`;
  }
}

// UI elements
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const enhancerSelect = document.getElementById("enhancer-select") as HTMLSelectElement;
const detectorSelect = document.getElementById("detector-select") as HTMLSelectElement;
const swapBtn = document.getElementById("swap-btn") as HTMLButtonElement;
const processingFieldset = document.getElementById("processing") as HTMLFieldSetElement;
const progressBar = document.getElementById("progress") as HTMLProgressElement;
const statusLine = document.getElementById("status-line") as HTMLElement;
const frameLine = document.getElementById("frame-line") as HTMLDivElement;
const timingLine = document.getElementById("timing-line") as HTMLDivElement;
const errorLine = document.getElementById("error-line") as HTMLDivElement;
const previewCheck = document.getElementById("preview-check") as HTMLInputElement;
const xsegCheck = document.getElementById("xseg-check") as HTMLInputElement;
const outputDiv = document.getElementById("output") as HTMLDivElement;
const downscaleContainer = document.getElementById("video-downscale") as HTMLDivElement;
const downscaleSelect = document.getElementById("downscale-select") as HTMLSelectElement;
const resetBtn = document.getElementById("reset-options-btn") as HTMLButtonElement;
const mobileNotice = document.getElementById("mobile-defaults-notice") as HTMLElement;
const rangeDetails = document.getElementById("video-range") as HTMLDetailsElement;
const rangeLimitCheck = document.getElementById("range-limit-check") as HTMLInputElement;
const rangePreviewCheck = document.getElementById("range-preview-check") as HTMLInputElement;
const cancelBtn = document.getElementById("cancel-btn") as HTMLButtonElement;
const advancedSection = document.getElementById("advanced-section") as HTMLDetailsElement;
const profilePreviewCheck = document.getElementById("profile-preview-check") as HTMLInputElement;
const gpuPasteCheck = document.getElementById("gpu-paste-check") as HTMLInputElement;
const workerModeSelect = document.getElementById("worker-mode-select") as HTMLSelectElement;

let lastOutputUrl: string | null = null;

function setStatus(text: string): void {
  statusLine.textContent = text;
}

function formatEta(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "?";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function formatMs(ms: number): string {
  if (ms < 1) return "0ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

const modelManager = new ModelManager(
  document.getElementById("model-manager") as HTMLDivElement,
  enhancerSelect,
);

const videoInput = new VideoInput(
  document.getElementById("video-input") as HTMLInputElement,
  document.getElementById("video-preview") as HTMLDivElement,
  {
    details: document.getElementById("video-range") as HTMLDetailsElement,
    slider: document.getElementById("range-slider") as HTMLDivElement,
    startInput: document.getElementById("range-start-input") as HTMLInputElement,
    previewInput: document.getElementById("range-preview-input") as HTMLInputElement,
    previewLabel: document.getElementById("range-preview-label") as HTMLElement,
    previewCheck: document.getElementById("range-preview-check") as HTMLInputElement,
    endInput: document.getElementById("range-end-input") as HTMLInputElement,
    limitCheck: document.getElementById("range-limit-check") as HTMLInputElement,
  },
);

const faceInput = new FaceInput(
  document.getElementById("face-input") as HTMLInputElement,
  document.getElementById("face-preview") as HTMLDivElement,
);

// Set by applySettings before a video is loaded so updateDownscaleOptions
// can pick the right option once the dropdown is populated. Must be declared
// before the top-level await below because applyInitialSettings reads it.
let pendingDownscaleHeight: string | null = null;

await modelManager.init();
updateModelSelect();
applyInitialSettings();
updateSwapButton();
wireSettingsPersistence();

document.getElementById("face-input")!.addEventListener("change", updateSwapButton);
modelSelect.addEventListener("change", updateSwapButton);

videoInput.onLoad(() => {
  updateDownscaleOptions();
  updateSwapButton();
});

document.getElementById("video-input")!.addEventListener("change", () => {
  swapBtn.disabled = true;
});

const RESOLUTION_STEPS = [
  { label: "2160p (4K)", height: 2160 },
  { label: "1440p", height: 1440 },
  { label: "1080p", height: 1080 },
  { label: "720p", height: 720 },
  { label: "480p", height: 480 },
];

function updateDownscaleOptions(): void {
  const h = videoInput.getVideoHeight();
  if (h <= 0) {
    downscaleContainer.style.display = "none";
    return;
  }

  const options = RESOLUTION_STEPS.filter((r) => r.height <= h);
  if (options.length <= 1) {
    downscaleContainer.style.display = "none";
    return;
  }

  downscaleSelect.innerHTML = "";
  const nativeOpt = document.createElement("option");
  nativeOpt.value = "1";
  nativeOpt.dataset.height = "native";
  nativeOpt.textContent = `native (${videoInput.getVideoWidth()}x${h})`;
  downscaleSelect.appendChild(nativeOpt);

  for (const res of options) {
    if (res.height >= h) continue;
    const scale = res.height / h;
    const scaledW = Math.round(videoInput.getVideoWidth() * scale);
    const opt = document.createElement("option");
    opt.value = String(scale);
    opt.dataset.height = String(res.height);
    opt.textContent = `${res.label} (${scaledW}x${res.height})`;
    downscaleSelect.appendChild(opt);
  }

  // Apply persisted/default downscale by target height (not raw scale, since
  // scale depends on the source video). Falls back to 1080p if no preference.
  const desiredHeight = pendingDownscaleHeight ?? "1080";
  const match = Array.from(downscaleSelect.options).find((o) => o.dataset.height === desiredHeight);
  if (match) {
    downscaleSelect.value = match.value;
  } else if (h > 1080) {
    const default1080 = Array.from(downscaleSelect.options).find(
      (o) => o.dataset.height === "1080",
    );
    if (default1080) downscaleSelect.value = default1080.value;
  }

  downscaleContainer.style.display = "";
  // Persist whatever ended up selected (in case the saved one wasn't available)
  persistCurrentSettings();
}

function updateModelSelect(): void {
  const previous = modelSelect.value;
  const readyIds = new Set(modelManager.getReadySetIds());
  modelSelect.innerHTML = "";
  modelSelect.disabled = false;

  for (const set of MODEL_SETS) {
    const opt = document.createElement("option");
    opt.value = set.id;
    const cached = readyIds.has(set.id);
    opt.textContent = cached ? set.name : `${set.name} - download model to enable`;
    opt.disabled = !cached;
    modelSelect.appendChild(opt);
  }

  if (previous && readyIds.has(previous)) {
    modelSelect.value = previous;
  } else {
    // Pick first cached model, or leave on first option (which is disabled)
    const firstReady = MODEL_SETS.find((s) => readyIds.has(s.id));
    modelSelect.value = firstReady ? firstReady.id : "";
  }
}

function updateSwapButton(): void {
  const hasVideo = videoInput.hasVideo();
  const hasFace = faceInput.getImage() !== null;
  const hasModel = modelSelect.value !== "";
  swapBtn.disabled = !(hasVideo && hasFace && hasModel);
}

const observer = new MutationObserver(() => {
  updateModelSelect();
  updateSwapButton();
});
observer.observe(document.getElementById("model-manager")!, {
  childList: true,
  subtree: true,
});

function captureCurrentSettings(): PersistedSettings {
  const downscaleHeight = downscaleSelect.selectedOptions[0]?.dataset.height ?? "native";
  return {
    swapModel: modelSelect.value,
    detector: detectorSelect.value,
    enhancer: enhancerSelect.value,
    useXseg: xsegCheck.checked,
    doPreview: previewCheck.checked,
    downscale: downscaleHeight,
    rangeLimit: rangeLimitCheck.checked,
    separatePreview: rangePreviewCheck.checked,
    rangePanelOpen: rangeDetails.open,
    advancedPanelOpen: advancedSection.open,
    profilePreview: profilePreviewCheck.checked,
    gpuPaste: gpuPasteCheck.checked,
    workerMode: workerModeSelect.value as WorkerMode,
  };
}

function persistCurrentSettings(): void {
  settingsStore.save(captureCurrentSettings());
}

/**
 * Apply a settings object to the form. Missing fields are left at whatever
 * the form currently shows. Selects only accept values that exist in the
 * dropdown - otherwise the field stays at its current value.
 */
function applySettings(s: Partial<PersistedSettings>): void {
  if (s.swapModel !== undefined) {
    const opt = Array.from(modelSelect.options).find((o) => o.value === s.swapModel && !o.disabled);
    if (opt) modelSelect.value = s.swapModel;
  }
  if (s.detector !== undefined) {
    const opt = Array.from(detectorSelect.options).find((o) => o.value === s.detector);
    if (opt) detectorSelect.value = s.detector;
  }
  if (s.enhancer !== undefined) {
    const opt = Array.from(enhancerSelect.options).find(
      (o) => o.value === s.enhancer && !o.disabled,
    );
    if (opt) enhancerSelect.value = s.enhancer;
  }
  if (s.useXseg !== undefined) xsegCheck.checked = s.useXseg;
  if (s.doPreview !== undefined) previewCheck.checked = s.doPreview;
  if (s.rangeLimit !== undefined) rangeLimitCheck.checked = s.rangeLimit;
  if (s.separatePreview !== undefined) rangePreviewCheck.checked = s.separatePreview;
  if (s.rangePanelOpen !== undefined) rangeDetails.open = s.rangePanelOpen;
  if (s.advancedPanelOpen !== undefined) advancedSection.open = s.advancedPanelOpen;
  if (s.profilePreview !== undefined) profilePreviewCheck.checked = s.profilePreview;
  if (s.gpuPaste !== undefined) gpuPasteCheck.checked = s.gpuPaste;
  if (s.workerMode !== undefined) {
    const opt = Array.from(workerModeSelect.options).find((o) => o.value === s.workerMode);
    if (opt) workerModeSelect.value = s.workerMode;
  }
  // Downscale: stash the desired height for updateDownscaleOptions to pick up
  // when the video is loaded (since the dropdown is empty until then).
  if (s.downscale !== undefined) {
    pendingDownscaleHeight = s.downscale;
    // If a video is already loaded, re-apply now too.
    if (videoInput.hasVideo()) {
      const match = Array.from(downscaleSelect.options).find(
        (o) => o.dataset.height === s.downscale,
      );
      if (match) downscaleSelect.value = match.value;
    }
  }
}

/**
 * On startup: restore saved settings if present, otherwise apply mobile
 * defaults (and show the notice) if we're on a touch device.
 */
function applyInitialSettings(): void {
  const saved = settingsStore.load();
  if (saved) {
    applySettings(saved);
    mobileNotice.style.display = "none";
  } else if (isMobile()) {
    applySettings(getMobileDefaults());
    mobileNotice.style.display = "inline";
  } else {
    mobileNotice.style.display = "none";
  }
}

/** Wire change/input handlers on every persisted control. */
function wireSettingsPersistence(): void {
  const onChange = () => persistCurrentSettings();
  modelSelect.addEventListener("change", onChange);
  detectorSelect.addEventListener("change", onChange);
  enhancerSelect.addEventListener("change", onChange);
  xsegCheck.addEventListener("change", onChange);
  previewCheck.addEventListener("change", onChange);
  downscaleSelect.addEventListener("change", onChange);
  rangeLimitCheck.addEventListener("change", onChange);
  rangePreviewCheck.addEventListener("change", onChange);
  rangeDetails.addEventListener("toggle", onChange);
  advancedSection.addEventListener("toggle", onChange);
  profilePreviewCheck.addEventListener("change", onChange);
  gpuPasteCheck.addEventListener("change", onChange);
  workerModeSelect.addEventListener("change", onChange);

  resetBtn.addEventListener("click", () => {
    settingsStore.clear();
    pendingDownscaleHeight = null;
    if (isMobile()) {
      applySettings(getMobileDefaults());
      mobileNotice.style.display = "inline";
    } else {
      // Reload-friendly defaults: re-pick first cached model/no enhancer/etc.
      // Easiest way is just to reload, but we can also reset in-place.
      modelSelect.dispatchEvent(new Event("change"));
      detectorSelect.value = "scrfd_500m";
      enhancerSelect.value = "";
      xsegCheck.checked = true;
      previewCheck.checked = true;
      rangeLimitCheck.checked = false;
      rangePreviewCheck.checked = false;
      rangeDetails.open = false;
      advancedSection.open = false;
      profilePreviewCheck.checked = false;
      gpuPasteCheck.checked = false;
      workerModeSelect.value = "off";
      pendingDownscaleHeight = "1080";
      mobileNotice.style.display = "none";
    }
    if (videoInput.hasVideo()) updateDownscaleOptions();
    updateSwapButton();
    // Don't persist immediately - hasSettings() should return false until
    // the user actively changes something again.
  });
}

function disableForm(): void {
  document
    .querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>(
      "input, select, button",
    )
    .forEach((el) => (el.disabled = true));
}

function enableForm(): void {
  document
    .querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>(
      "input, select, button",
    )
    .forEach((el) => (el.disabled = false));
  updateSwapButton();
}

function showPreviewImage(imageData: ImageData): void {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.style.maxWidth = "100%";
  canvas.getContext("2d")!.putImageData(imageData, 0, 0);
  outputDiv.innerHTML = "";
  outputDiv.appendChild(canvas);
}

function waitForConfirmation(): Promise<boolean> {
  return new Promise((resolve) => {
    const div = document.createElement("div");
    div.style.marginTop = "8px";
    const continueBtn = document.createElement("button");
    continueBtn.textContent = "looks good, process full video";
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "cancel";
    cancelBtn.style.marginLeft = "8px";
    div.appendChild(continueBtn);
    div.appendChild(cancelBtn);
    outputDiv.appendChild(div);

    continueBtn.addEventListener("click", () => {
      div.remove();
      resolve(true);
    });
    cancelBtn.addEventListener("click", () => {
      div.remove();
      resolve(false);
    });
  });
}

swapBtn.addEventListener("click", async () => {
  const video = videoInput.getVideo();
  const setId = modelSelect.value;
  if (!video || !setId) return;

  const set = MODEL_SETS.find((s) => s.id === setId);
  if (!set) return;

  const enhancerId = modelManager.getSelectedEnhancerId();
  const enhance = enhancerId != null;
  const detectorId = detectorSelect.value as DetectorId;
  const useXseg = xsegCheck.checked;
  const doPreview = previewCheck.checked;
  const scale = Number(downscaleSelect.value) || 1;

  disableForm();
  processingFieldset.style.display = "";
  progressBar.value = 0;
  statusLine.textContent = "";
  frameLine.textContent = "";
  timingLine.textContent = "";
  errorLine.textContent = "";
  outputDiv.innerHTML = "";

  const session = createSession(workerModeSelect.value as WorkerMode, gpuPasteCheck.checked);
  const sourceFile = videoInput.getFile();
  const onCancelClick = () => {
    session.abort();
    cancelBtn.disabled = true;
    cancelBtn.textContent = "cancelling...";
  };
  // Cancel button stays hidden during model load + preview - the preview
  // step has its own confirm/cancel UI. Shown once we kick off the full
  // video encode below.
  cancelBtn.disabled = false;
  cancelBtn.textContent = "cancel";
  cancelBtn.style.display = "none";
  cancelBtn.addEventListener("click", onCancelClick);
  const swapStartTime = performance.now();

  try {
    setStatus("loading face image...");
    const faceImageData = faceInput.getImageData();
    if (!faceImageData) {
      throw new Error("failed to read face image");
    }

    setStatus("loading models...");
    await session.loadModels(set, enhancerId, detectorId);

    setStatus("extracting source embedding...");
    const sourceEmbedding = await session.extractEmbedding(faceImageData);

    const startTime = videoInput.getRangeStart();
    const endTime = videoInput.getRangeEnd();
    const previewTime = videoInput.getPreviewTime();

    if (doPreview) {
      setStatus(`previewing frame at ${previewTime.toFixed(1)}s...`);
      const profiling = profilePreviewCheck.checked;
      if (profiling) {
        console.log("[profile] enabling WebGPU profiling for preview frame");
        (ort.env.webgpu as { profiling?: { mode: string } }).profiling = { mode: "default" };
      }
      const tPreview = performance.now();
      if (!sourceFile) throw new Error("source video file unavailable");
      const previewResult = await session.previewFrame(
        video,
        sourceFile,
        previewTime,
        scale,
        sourceEmbedding,
        useXseg,
      );
      const previewMs = performance.now() - tPreview;
      if (profiling) {
        (ort.env.webgpu as { profiling?: { mode: string } }).profiling = { mode: "off" };
        console.log(`[profile] preview frame swap wall time: ${previewMs.toFixed(1)} ms`);
      }
      showPreviewImage(previewResult);
      setStatus("preview complete - waiting for confirmation");

      const confirmed = await waitForConfirmation();
      if (!confirmed) {
        setStatus("cancelled");
        return;
      }
    }

    // Clear preview before full processing
    outputDiv.innerHTML = "";
    setStatus("encoding video...");
    cancelBtn.style.display = "";

    // Rolling window for smoothing per-step timings
    const timingWindow: FrameTimings[] = [];
    const windowSize = 10;

    const videoBlob = await session.processVideo(
      video,
      sourceFile,
      sourceEmbedding,
      startTime,
      endTime,
      useXseg,
      scale,
      (stats) => {
        progressBar.value = (stats.frameIndex / stats.totalFrames) * 100;

        timingWindow.push(stats.timings);
        if (timingWindow.length > windowSize) timingWindow.shift();
        const avg: FrameTimings = {
          detect: 0,
          landmarks: 0,
          swap: 0,
          xseg: 0,
          paste: 0,
          enhance: 0,
        };
        for (const t of timingWindow) {
          avg.detect += t.detect;
          avg.landmarks += t.landmarks;
          avg.swap += t.swap;
          avg.xseg += t.xseg;
          avg.paste += t.paste;
          avg.enhance += t.enhance;
        }
        const n = timingWindow.length;
        for (const k of Object.keys(avg) as (keyof FrameTimings)[]) avg[k] /= n;

        const pct = ((stats.frameIndex / stats.totalFrames) * 100).toFixed(0);
        frameLine.textContent =
          `frame ${stats.frameIndex}/${stats.totalFrames} (${pct}%) | ` +
          `${stats.fps.toFixed(1)} fps | ~${formatEta(stats.etaSeconds)} left`;

        const parts = [
          `detect ${formatMs(avg.detect)}`,
          `landmarks ${formatMs(avg.landmarks)}`,
          `swap ${formatMs(avg.swap)}`,
          `xseg ${formatMs(avg.xseg)}`,
          `paste ${formatMs(avg.paste)}`,
        ];
        if (enhance) parts.push(`enhance ${formatMs(avg.enhance)}`);
        timingLine.textContent = parts.join(" | ");
      },
    );

    if (session.isAborted) {
      setStatus("cancelled");
      return;
    }

    setStatus("finalizing...");
    if (lastOutputUrl) URL.revokeObjectURL(lastOutputUrl);
    const url = URL.createObjectURL(videoBlob);
    lastOutputUrl = url;
    const sizeMB = (videoBlob.size / 1024 / 1024).toFixed(1);
    const elapsedSec = (performance.now() - swapStartTime) / 1000;
    setStatus(`done - ${sizeMB} MB - took ${formatEta(elapsedSec)}`);

    const outputVideo = document.createElement("video");
    outputVideo.src = url;
    outputVideo.controls = true;
    outputVideo.style.maxWidth = "100%";
    outputDiv.appendChild(outputVideo);

    const dl = document.createElement("a");
    dl.href = url;
    dl.download = "face-swap.mp4";
    dl.textContent = `download (${sizeMB} MB)`;
    dl.style.display = "block";
    dl.style.marginTop = "8px";
    outputDiv.appendChild(dl);
  } catch (err) {
    console.error(err);
    errorLine.textContent = `ERROR: ${err}`;
  } finally {
    cancelBtn.removeEventListener("click", onCancelClick);
    cancelBtn.style.display = "none";
    try {
      await session.releaseModels();
    } catch (e) {
      console.warn("releaseModels failed:", e);
    }
    session.dispose();
    enableForm();
  }
});
