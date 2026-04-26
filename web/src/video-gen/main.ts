// video-gen tool DOM shell. Minimal UI: cached-models panel + prompt +
// (seed, debug-log) advanced options + generate + progress + result
// canvas. FastWan 2.2 config is otherwise fixed (4 UniPC steps, 480x480,
// 5s @ 16fps), so there is no resolution/steps/CFG control here.
//
// Backend dispatch: main calls generateFastwan directly because that's
// the only backend right now. When we add a second model, replace the
// direct call with a discriminator on VideoModelEntry.backend.

import { initThemeSelect } from "../shared/theme";
import { PersistedSettings } from "../shared/persisted-settings";
import { ProgressVideo } from "../shared/progress-video";
import { VideoGenModelManager } from "./model-manager";
import { VIDEO_GEN_MODELS } from "./models";
import {
  generateFastwan,
  type ProgressInfo,
} from "../fastwan/generate";
import { encodeFramesToMp4 } from "../fastwan/encode-mp4";

interface VideoGenSettings {
  debugLog: boolean;
  seed: string;
  modelId: string;
  prompt: string;
  advancedOpen: boolean;
}

const DEFAULT_SETTINGS: VideoGenSettings = {
  debugLog: false,
  seed: "",
  modelId: "",
  prompt: "",
  advancedOpen: false,
};

const persisted = new PersistedSettings<VideoGenSettings>("intabai:video-gen:settings");

{
  const sel = document.getElementById("theme-select");
  if (sel instanceof HTMLSelectElement) initThemeSelect(sel);
}

// ---- DOM refs --------------------------------------------------------------

const modelManagerContainer = document.getElementById("model-manager") as HTMLDivElement;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const promptInput = document.getElementById("prompt") as HTMLTextAreaElement;
const seedInput = document.getElementById("seed-input") as HTMLInputElement;
const debugLogCheck = document.getElementById("debug-log-check") as HTMLInputElement;
const advancedSection = document.getElementById("advanced-section") as HTMLDetailsElement;
const debugLogPane = document.getElementById("debug-log-pane") as HTMLTextAreaElement;
const generateBtn = document.getElementById("generate-btn") as HTMLButtonElement;
const processing = document.getElementById("processing") as HTMLFieldSetElement;
const cancelBtn = document.getElementById("cancel-btn") as HTMLButtonElement;
const previewSection = document.getElementById("preview-section") as HTMLFieldSetElement;
const previewCanvas = document.getElementById("preview-canvas") as HTMLCanvasElement;
const previewControls = document.getElementById("preview-controls") as HTMLDivElement;
const resultSection = document.getElementById("result-section") as HTMLFieldSetElement;
const resultVideo = document.getElementById("result-video") as HTMLVideoElement;
const resultControls = document.getElementById("result-controls") as HTMLDivElement;
const pipBtn = document.getElementById("pip-btn") as HTMLButtonElement;
const pv = new ProgressVideo(document.getElementById("progress-video") as HTMLVideoElement);

// ---- Settings --------------------------------------------------------------

function loadSettings(): VideoGenSettings {
  const s = persisted.load();
  return { ...DEFAULT_SETTINGS, ...(s ?? {}) };
}

function applySettings(s: VideoGenSettings): void {
  debugLogCheck.checked = s.debugLog;
  seedInput.value = s.seed;
  promptInput.value = s.prompt;
  advancedSection.open = s.advancedOpen;
  // modelSelect is filled by refreshModelSelect; setting .value before the
  // options exist is a no-op, so the preferred id is read again after
  // refreshModelSelect() picks option state.
  preferredModelId = s.modelId;
}

function currentSettings(): VideoGenSettings {
  return {
    debugLog: debugLogCheck.checked,
    seed: seedInput.value,
    modelId: modelSelect.value,
    prompt: promptInput.value,
    advancedOpen: advancedSection.open,
  };
}

function persistSettings(): void {
  persisted.save(currentSettings());
}

// ---- Model manager ---------------------------------------------------------

let preferredModelId = "";

const modelManager = new VideoGenModelManager(modelManagerContainer, () => {
  refreshModelSelect();
  updateGenerateButton();
});

function refreshModelSelect(): void {
  const previous = modelSelect.value;
  modelSelect.innerHTML = "";
  for (const m of VIDEO_GEN_MODELS) {
    const opt = document.createElement("option");
    opt.value = m.id;
    const cached = modelManager.isReady(m.id);
    opt.textContent = cached ? m.name : `${m.name} - download to enable`;
    opt.disabled = !cached;
    modelSelect.appendChild(opt);
  }
  // Priority: previous live selection > preferred id from saved settings
  // > first cached model.
  if (previous && modelManager.isReady(previous)) {
    modelSelect.value = previous;
  } else if (preferredModelId && modelManager.isReady(preferredModelId)) {
    modelSelect.value = preferredModelId;
  } else {
    const firstReady = VIDEO_GEN_MODELS.find((m) => modelManager.isReady(m.id));
    modelSelect.value = firstReady ? firstReady.id : "";
  }
}

function selectedModel() {
  const id = modelSelect.value;
  if (!id || !modelManager.isReady(id)) return undefined;
  return VIDEO_GEN_MODELS.find((m) => m.id === id);
}

function updateGenerateButton(): void {
  const ready = selectedModel() !== undefined;
  const hasPrompt = promptInput.value.trim().length > 0;
  generateBtn.disabled = !(ready && hasPrompt);
}

promptInput.addEventListener("input", () => {
  updateGenerateButton();
  persistSettings();
});
modelSelect.addEventListener("change", () => {
  updateGenerateButton();
  persistSettings();
});
seedInput.addEventListener("change", persistSettings);
debugLogCheck.addEventListener("change", persistSettings);
advancedSection.addEventListener("toggle", persistSettings);

// ---- Preview + result ------------------------------------------------------
//
// Preview (intermediate decode between denoising steps): canvas with a
// setInterval loop, because we only have ImageBitmaps and we want to
// show them immediately without paying the MP4-encode cost.
// Result (final output): MP4 Blob via WebCodecs, loaded into a <video>
// with native controls + a download link.

let previewTimer: number | null = null;

function stopPreview(): void {
  if (previewTimer !== null) {
    clearInterval(previewTimer);
    previewTimer = null;
  }
}

function renderPreview(frames: ImageBitmap[], fps: number): void {
  stopPreview();
  if (frames.length === 0) return;
  previewCanvas.width = frames[0].width;
  previewCanvas.height = frames[0].height;
  const ctx = previewCanvas.getContext("2d");
  if (!ctx) return;
  let i = 0;
  const draw = () => {
    ctx.drawImage(frames[i], 0, 0);
    i = (i + 1) % frames.length;
  };
  draw();
  previewTimer = window.setInterval(draw, 1000 / fps);

  previewControls.innerHTML = "";
  const info = document.createElement("small");
  info.textContent = `${frames.length} frames @ ${fps} fps (intermediate preview, still denoising)`;
  previewControls.appendChild(info);
}

/** Track the current result blob URL so we can revoke it on re-run. */
let currentResultUrl: string | null = null;

async function renderFinalResult(
  frames: ImageBitmap[],
  fps: number,
  seed: number,
): Promise<void> {
  stopPreview();
  previewSection.style.display = "none";

  // Encoding an 81-frame 480x480 MP4 via WebCodecs takes ~1s on desktop.
  // Show a brief status while it runs.
  pv.setStatus("encoding mp4...");
  const encodeStart = performance.now();
  const blob = await encodeFramesToMp4({ frames, fps });
  const encodeMs = performance.now() - encodeStart;

  if (currentResultUrl) URL.revokeObjectURL(currentResultUrl);
  currentResultUrl = URL.createObjectURL(blob);
  resultVideo.src = currentResultUrl;
  resultVideo.loop = true;
  resultVideo.play().catch(() => {});
  resultSection.style.display = "";

  resultControls.innerHTML = "";
  const dl = document.createElement("a");
  dl.href = currentResultUrl;
  dl.download = `video-gen-${seed}.mp4`;
  dl.textContent = "download mp4";
  resultControls.appendChild(dl);

  const info = document.createElement("small");
  info.style.marginLeft = "8px";
  const sizeMb = (blob.size / 1e6).toFixed(1);
  info.textContent = `${frames.length} frames @ ${fps} fps, ${sizeMb} MB (encoded in ${encodeMs.toFixed(0)} ms)`;
  resultControls.appendChild(info);
}

// ---- Debug log -------------------------------------------------------------

function startDebugPane(enabled: boolean): ((msg: string) => void) | undefined {
  if (!enabled) {
    debugLogPane.style.display = "none";
    debugLogPane.value = "";
    return undefined;
  }
  debugLogPane.style.display = "block";
  debugLogPane.value = "";
  return (msg: string) => {
    debugLogPane.value += msg + "\n";
    debugLogPane.scrollTop = debugLogPane.scrollHeight;
  };
}

// ---- Generate --------------------------------------------------------------

let currentAbort: AbortController | null = null;

async function onGenerate(): Promise<void> {
  const model = selectedModel();
  if (!model) return;

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  persistSettings();
  const settings = currentSettings();
  const seedTrimmed = settings.seed.trim();
  const seed = seedTrimmed === "" ? undefined : Number(seedTrimmed);
  if (seed !== undefined && (!Number.isFinite(seed) || seed < 0)) {
    console.error("[video-gen] seed must be a non-negative number");
    return;
  }

  currentAbort = new AbortController();
  generateBtn.disabled = true;
  cancelBtn.style.display = "";
  cancelBtn.disabled = false;
  processing.style.display = "";
  resultSection.style.display = "none";
  previewSection.style.display = "none";
  const onDebug = startDebugPane(settings.debugLog);

  // Start the progress video. Must be in the generate-click gesture so
  // AudioContext.resume + video.play are allowed. Expose the PiP button
  // so mobile users can background the tab without losing the GPU.
  await pv.start("generating video", onDebug);
  pipBtn.style.display = "";
  const onPipClick = () => pv.enterPip();
  pipBtn.addEventListener("click", onPipClick);

  const runStart = performance.now();
  try {
    if (model.backend !== "fastwan") {
      throw new Error(`unknown backend: ${model.backend}`);
    }
    const result = await generateFastwan({
      cache: modelManager.cache,
      prompt,
      seed,
      transformerPrecision: model.transformerPrecision,
      textEncoderPrecision:
        new URLSearchParams(location.search).get("textencoderfp16") === "1"
          ? "fp16"
          : "q4f16",
      resolution: model.resolution,
      signal: currentAbort.signal,
      onPreview: (frames) => {
        previewSection.style.display = "";
        renderPreview(frames, 16);
      },
      onProgress: (info: ProgressInfo) => {
        const elapsed = performance.now() - runStart;
        // ETA only once we're into denoise - earlier stages are a tiny
        // fraction of total work and extrapolating from ~4% progress
        // after the text encoder gives a misleadingly-short estimate.
        let eta = "";
        if ((info.stage === "denoise" || info.stage === "vae") && info.pct > 0.05) {
          const etaMs = (elapsed / info.pct) * (1 - info.pct);
          eta = ` ETA ${formatDuration(etaMs)}`;
        }
        pv.setProgress(info.pct * 100);
        pv.setStatus(`${info.stage}${eta}`);
        pv.setStats(info.message);
      },
      onDebug,
    });
    await renderFinalResult(result.frames, result.fps, result.seed);
    pv.setProgress(100);
    pv.setStatus(`done, seed ${result.seed}`);
    pv.setStats("");
  } catch (err) {
    if (currentAbort?.signal.aborted) {
      pv.setStatus("cancelled");
    } else {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[video-gen]", err);
      pv.setStatus(`failed: ${msg}`);
    }
  } finally {
    cancelBtn.style.display = "none";
    pipBtn.removeEventListener("click", onPipClick);
    pipBtn.style.display = "none";
    pv.stop();
    currentAbort = null;
    updateGenerateButton();
  }
}

generateBtn.addEventListener("click", onGenerate);
cancelBtn.addEventListener("click", () => {
  currentAbort?.abort();
  cancelBtn.disabled = true;
});

function formatDuration(ms: number): string {
  if (!isFinite(ms) || ms < 0) return "?";
  const sec = Math.round(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s}s`;
}

// ---- Init ------------------------------------------------------------------

applySettings(loadSettings());
refreshModelSelect();
void modelManager.init();
