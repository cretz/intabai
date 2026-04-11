// Image-gen tool entry point. UI controller only - the actual generation
// pipelines live in web/src/sd15/generate.ts and web/src/sdxl/generate.ts
// and are dispatched by ModelSet.family. Adding a new model family is
// "implement a new generate function with the GenerateFn shape, register
// it in PIPELINES, add a ModelSet entry pointing at the right files".

import * as ort from "onnxruntime-web/webgpu";
import { IMAGE_GEN_MODELS, type ModelSet } from "../sd15/models";
import { sd15GenerateFn } from "../sd15/generate";
import { sdxlGenerateFn } from "../sdxl/generate";

// Prefer the discrete GPU on dual-GPU systems (e.g. laptop with Intel iGPU +
// NVIDIA dGPU). Must be set before the first ORT session is created.
(ort.env.webgpu as unknown as Record<string, unknown>).powerPreference = "high-performance";
// NOTE: janus is loaded via dynamic import below, NOT a static import. The
// janus pipeline pulls in @huggingface/transformers (which ships as a
// pre-bundled file with its own ORT-web inlined, ~432 KB minified). Static-
// importing it would force every image-gen page load to drag transformers.js
// + a second ORT instance into the bundle even for users who never touch
// Janus. The dynamic import keeps SD1.5/SDXL users on a single ORT instance
// (our `onnxruntime-web` dep) and only loads transformers.js when a Janus
// model is actually selected and the user clicks generate.
import type { GenerateCallbacks, GenerateFn, GenerateInput, RefImageInput } from "./generate-types";
import { ImageGenModelManager } from "./model-manager";
import {
  PersistedSettings as PersistedSettingsStore,
  isMobile,
} from "../shared/persisted-settings";

/** Per-model settings. Undefined means "use model default" (shown as
 *  placeholder text in the input, not an actual value). */
interface PerModelSettings {
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  /** Empty string = pick a fresh random seed every run. */
  seed?: string;
  tileVae?: boolean;
  refStrength?: number;
}

/** Top-level persisted settings. Global fields are shared across models;
 *  per-model fields are saved/restored when switching the model dropdown. */
interface ImageGenSettings {
  model: string;
  prompt: string;
  debugLog: boolean;
  advancedOpen: boolean;
  refImageOpen: boolean;
  /** Per-model settings keyed by model id. */
  perModel: Record<string, PerModelSettings>;
}

/** Sync pipelines that are statically imported (small/no extra deps). */
const STATIC_PIPELINES: Partial<Record<ModelSet["family"], GenerateFn>> = {
  sd15: sd15GenerateFn,
  sdxl: sdxlGenerateFn,
};

/** Lazy-loaded pipelines. The loader returns a GenerateFn the first time
 *  it is called and caches the resolved module so subsequent generates do
 *  not re-evaluate the import. */
const LAZY_PIPELINES: Partial<Record<ModelSet["family"], () => Promise<GenerateFn>>> = {
  janus: async () => (await import("../janus/generate")).janusGenerateFn,
  zimage: async () => (await import("../zimage/generate")).zimageGenerateFn,
};

async function resolvePipeline(family: ModelSet["family"]): Promise<GenerateFn> {
  const sync = STATIC_PIPELINES[family];
  if (sync) return sync;
  const lazy = LAZY_PIPELINES[family];
  if (lazy) return await lazy();
  throw new Error(`no pipeline for family: ${family}`);
}

/** Resolved per-model defaults from the model definition. Used as
 *  placeholder values for inputs that the user hasn't explicitly set. */
function modelDefaults(set: ModelSet): PerModelSettings {
  const d = set.defaults;
  const mobile = isMobile();
  const square = mobile ? d.mobileResolution : d.width;
  return {
    width: square,
    height: square,
    steps: d.steps,
    cfg: d.cfg,
    seed: "",
    tileVae: mobile,
    refStrength: 0.6,
  };
}

/** Resolve a per-model setting: use the user's explicit value if set,
 *  otherwise fall back to the model default. */
function resolvePerModel(saved: PerModelSettings | undefined, set: ModelSet): Required<PerModelSettings> {
  const d = modelDefaults(set);
  return {
    width: saved?.width ?? d.width!,
    height: saved?.height ?? d.height!,
    steps: saved?.steps ?? d.steps!,
    cfg: saved?.cfg ?? d.cfg!,
    seed: saved?.seed ?? d.seed!,
    tileVae: saved?.tileVae ?? d.tileVae!,
    refStrength: saved?.refStrength ?? d.refStrength!,
  };
}

function defaultGlobalSettings(): ImageGenSettings {
  return {
    model: "",
    prompt: "",
    debugLog: false,
    advancedOpen: false,
    refImageOpen: false,
    perModel: {},
  };
}

const settingsStore = new PersistedSettingsStore<ImageGenSettings>("intabai-image-gen-settings");

const modelManagerContainer = document.getElementById("model-manager") as HTMLDivElement;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const debugLogCheck = document.getElementById("debug-log-check") as HTMLInputElement;
const debugLogPane = document.getElementById("debug-log-pane") as HTMLTextAreaElement;
const promptInput = document.getElementById("prompt") as HTMLTextAreaElement;
const advancedSection = document.getElementById("advanced-section") as HTMLDetailsElement;
const widthInput = document.getElementById("width-input") as HTMLInputElement;
const heightInput = document.getElementById("height-input") as HTMLInputElement;
const stepsInput = document.getElementById("steps-input") as HTMLInputElement;
const cfgInput = document.getElementById("cfg-input") as HTMLInputElement;
const seedInput = document.getElementById("seed-input") as HTMLInputElement;
const tileVaeCheck = document.getElementById("tile-vae-check") as HTMLInputElement;
const resetBtn = document.getElementById("reset-options-btn") as HTMLButtonElement;
const generateBtn = document.getElementById("generate-btn") as HTMLButtonElement;
const processingFieldset = document.getElementById("processing") as HTMLFieldSetElement;
const progressBar = document.getElementById("progress") as HTMLProgressElement;
const statusLine = document.getElementById("status-line") as HTMLElement;
const statsLine = document.getElementById("stats-line") as HTMLElement;
const cancelBtn = document.getElementById("cancel-btn") as HTMLButtonElement;
const resultSection = document.getElementById("result-section") as HTMLFieldSetElement;
const resultCanvas = document.getElementById("result-canvas") as HTMLCanvasElement;
const resultDownloads = document.getElementById("result-downloads") as HTMLDivElement;
const browserError = document.getElementById("browser-error") as HTMLDivElement;
const gpuWarning = document.getElementById("gpu-warning") as HTMLDivElement;
const refImageSection = document.getElementById("ref-image-section") as HTMLDetailsElement;
const refImageModelHelp = document.getElementById("ref-image-modelhelp") as HTMLDivElement;
const refImageInput = document.getElementById("ref-image-input") as HTMLInputElement;
const refImageClear = document.getElementById("ref-image-clear") as HTMLButtonElement;
const refImagePreviewWrap = document.getElementById("ref-image-preview-wrap") as HTMLDivElement;
const refImagePreview = document.getElementById("ref-image-preview") as HTMLImageElement;
const refStrengthWrap = document.getElementById("ref-image-strength-wrap") as HTMLDivElement;
const refStrengthInput = document.getElementById("ref-strength-input") as HTMLInputElement;
const refStrengthLabel = document.getElementById("ref-strength-label") as HTMLLabelElement;
const refStrengthValue = document.getElementById("ref-strength-value") as HTMLSpanElement;
const resolutionHelp = document.getElementById("resolution-help") as HTMLElement;
const stepsHelp = document.getElementById("steps-help") as HTMLElement;
const cfgHelp = document.getElementById("cfg-help") as HTMLElement;

let refImageElement: HTMLImageElement | null = null;

if (!("storage" in navigator) || !("getDirectory" in navigator.storage)) {
  browserError.style.display = "block";
  browserError.textContent =
    "Your browser does not support OPFS, which is required to cache the model files. Try a recent Chrome, Edge, or Safari.";
}
if (!("gpu" in navigator)) {
  gpuWarning.style.display = "block";
  gpuWarning.textContent =
    "WebGPU is not available. Inference will fall back to CPU (wasm), which will be very slow.";
}

function refreshGenerateEnabled(): void {
  const ready = manager.getSelectedSetId() !== null;
  generateBtn.disabled = !ready || promptInput.value.trim().length === 0;
}

const manager = new ImageGenModelManager(
  modelManagerContainer,
  modelSelect,
  refreshGenerateEnabled,
);

/** Capture per-model input values. Empty inputs = undefined (use default). */
function capturePerModel(): PerModelSettings {
  const w = widthInput.value.trim() ? parseInt(widthInput.value, 10) : undefined;
  const h = heightInput.value.trim() ? parseInt(heightInput.value, 10) : undefined;
  const st = stepsInput.value.trim() ? parseInt(stepsInput.value, 10) : undefined;
  const cf = cfgInput.value.trim() ? parseFloat(cfgInput.value) : undefined;
  return {
    width: w !== undefined && !isNaN(w) ? w : undefined,
    height: h !== undefined && !isNaN(h) ? h : undefined,
    steps: st !== undefined && !isNaN(st) ? st : undefined,
    cfg: cf !== undefined && !isNaN(cf) ? cf : undefined,
    seed: seedInput.value.trim() || undefined,
    tileVae: tileVaeCheck.checked,
    refStrength: parseFloat(refStrengthInput.value) || undefined,
  };
}

function captureSettings(): ImageGenSettings {
  const modelId = modelSelect.value;
  // Start from previously saved perModel map, then update current model.
  const prev = settingsStore.load();
  const perModel: Record<string, PerModelSettings> = prev?.perModel ?? {};
  if (modelId) perModel[modelId] = capturePerModel();
  return {
    model: modelId,
    prompt: promptInput.value,
    debugLog: debugLogCheck.checked,
    advancedOpen: advancedSection.open,
    refImageOpen: refImageSection.open,
    perModel,
  };
}

/** Apply per-model settings to the form. Values that are undefined use
 *  placeholders from the model defaults instead of setting input.value. */
function applyPerModelSettings(saved: PerModelSettings | undefined, set: ModelSet): void {
  const d = modelDefaults(set);
  const setOrPlaceholder = (
    input: HTMLInputElement,
    val: number | undefined,
    fallback: number,
  ) => {
    if (val !== undefined) {
      input.value = String(val);
      input.placeholder = "";
    } else {
      input.value = "";
      input.placeholder = String(fallback);
    }
  };
  setOrPlaceholder(widthInput, saved?.width, d.width!);
  setOrPlaceholder(heightInput, saved?.height, d.height!);
  setOrPlaceholder(stepsInput, saved?.steps, d.steps!);
  setOrPlaceholder(cfgInput, saved?.cfg, d.cfg!);
  seedInput.value = saved?.seed ?? "";
  tileVaeCheck.checked = saved?.tileVae ?? d.tileVae!;
  const rs = saved?.refStrength ?? d.refStrength!;
  refStrengthInput.value = String(rs);
  refStrengthValue.textContent = rs.toFixed(2);
}

function applyGlobalSettings(s: ImageGenSettings): void {
  promptInput.value = s.prompt;
  debugLogCheck.checked = s.debugLog;
  advancedSection.open = s.advancedOpen;
  refImageSection.open = s.refImageOpen;
}

function persistSettings(): void {
  settingsStore.save(captureSettings());
}

/** Apply a model's defaults to help text, input bounds, and disable
 *  inputs that are fixed by the model architecture. */
function applyModelDefaultsToHelp(set: ModelSet): void {
  const d = set.defaults;
  resolutionHelp.textContent = `- ${d.resolutionHelp}`;
  stepsHelp.textContent = `- ${d.stepsHelp}`;
  cfgHelp.textContent = `- ${d.cfgHelp}`;
  widthInput.max = String(d.maxResolution);
  heightInput.max = String(d.maxResolution);
  // Disable inputs that are fixed by the model architecture.
  widthInput.disabled = !!d.fixedResolution;
  heightInput.disabled = !!d.fixedResolution;
  stepsInput.disabled = !!d.fixedSteps;
}

promptInput.addEventListener("input", () => {
  refreshGenerateEnabled();
  persistSettings();
});
modelSelect.addEventListener("change", () => {
  refreshGenerateEnabled();
  const m = currentModelOrNull();
  if (m) {
    applyModelDefaultsToHelp(m);
    // Load this model's saved settings (or defaults via placeholders).
    const saved = settingsStore.load();
    applyPerModelSettings(saved?.perModel?.[m.id], m);
  }
  refreshRefImageUi();
  persistSettings();
});
for (const el of [widthInput, heightInput, stepsInput, cfgInput, seedInput]) {
  el.addEventListener("input", persistSettings);
}
tileVaeCheck.addEventListener("change", persistSettings);
debugLogCheck.addEventListener("change", persistSettings);
advancedSection.addEventListener("toggle", persistSettings);
refImageSection.addEventListener("toggle", persistSettings);
refStrengthInput.addEventListener("input", () => {
  refStrengthValue.textContent = parseFloat(refStrengthInput.value).toFixed(2);
  persistSettings();
});

refImageInput.addEventListener("change", () => {
  const file = refImageInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    refImageElement = img;
    refImagePreview.src = url;
    refImagePreviewWrap.style.display = "";
    refImageClear.style.display = "";
    refreshRefImageUi();
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    alert("Failed to load image");
  };
  img.src = url;
});

refImageClear.addEventListener("click", () => {
  refImageElement = null;
  refImageInput.value = "";
  refImagePreview.removeAttribute("src");
  refImagePreviewWrap.style.display = "none";
  refImageClear.style.display = "none";
  refreshRefImageUi();
});

resetBtn.addEventListener("click", () => {
  if (!confirm("Reset all options to defaults?")) return;
  const m = currentModelOrNull();
  // Clear this model's per-model settings (revert to placeholders).
  if (m) {
    const prev = settingsStore.load();
    if (prev?.perModel?.[m.id]) {
      delete prev.perModel[m.id];
      settingsStore.save(prev);
    }
    applyPerModelSettings(undefined, m);
    applyModelDefaultsToHelp(m);
  }
  refreshRefImageUi();
  refreshGenerateEnabled();
});

function currentModelOrNull(): ModelSet | null {
  const id = manager.getSelectedSetId();
  if (!id) return null;
  return IMAGE_GEN_MODELS.find((m) => m.id === id) ?? null;
}

function currentModel(): ModelSet {
  const m = currentModelOrNull();
  if (!m) throw new Error("no model selected");
  return m;
}

/** Update visibility + per-model help text for the reference-image
 *  section. Called whenever the model changes or a reference image is
 *  added/cleared. */
function refreshRefImageUi(): void {
  const set = currentModelOrNull();
  if (!set) {
    refImageModelHelp.innerHTML = "<small>Select a model first.</small>";
    refImageInput.disabled = true;
    refStrengthWrap.style.display = "none";
    return;
  }
  const supports = set.capabilities.img2img && !!set.img2img;
  if (!supports) {
    refImageModelHelp.innerHTML = "<small>This model does not support reference images.</small>";
    refImageInput.disabled = true;
    refStrengthWrap.style.display = "none";
    return;
  }
  refImageInput.disabled = false;
  const help = set.img2img!;
  refImageModelHelp.innerHTML = `<small><strong>${escapeHtml(set.name)}:</strong> ${escapeHtml(help.description)}</small>`;
  if (refImageElement && help.strengthLabel) {
    refStrengthLabel.textContent = help.strengthLabel;
    refStrengthWrap.style.display = "";
  } else {
    refStrengthWrap.style.display = "none";
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

let aborted = false;

const onCancelClick = () => {
  aborted = true;
  cancelBtn.disabled = true;
  cancelBtn.textContent = "cancelling...";
};

class CancelledError extends Error {
  constructor() {
    super("cancelled");
  }
}

generateBtn.addEventListener("click", async () => {
  const set = currentModel();
  const settings = captureSettings();
  const pm = resolvePerModel(settings.perModel?.[set.id], set);
  // Snap dimensions to multiples of 8 (latent is image/8) and clamp to a
  // safe range. Round up so the generated image is at least as big as the
  // user asked for.
  const maxRes = set.defaults.maxResolution;
  const width = Math.max(64, Math.min(maxRes, Math.ceil(pm.width / 8) * 8));
  const height = Math.max(64, Math.min(maxRes, Math.ceil(pm.height / 8) * 8));
  const numSteps = Math.max(1, Math.min(200, pm.steps));
  const cfg = pm.cfg;
  const seedNum =
    pm.seed === "" ? Math.floor(Math.random() * 0x100000000) : Number(pm.seed) >>> 0;

  const refImage: RefImageInput | null =
    refImageElement && set.capabilities.img2img && set.img2img
      ? { image: refImageElement, strength: pm.refStrength }
      : null;

  const input: GenerateInput = {
    set,
    cache: manager.getCache(),
    prompt: promptInput.value,
    width,
    height,
    steps: numSteps,
    cfg,
    seed: seedNum,
    refImage,
    tileVae: pm.tileVae,
  };

  let pipeline: GenerateFn;
  try {
    pipeline = await resolvePipeline(set.family);
  } catch (err) {
    statusLine.textContent = (err as Error).message;
    return;
  }
  const { totalUnits } = pipeline.estimate(input);

  generateBtn.disabled = true;
  processingFieldset.style.display = "block";
  progressBar.value = 0;
  statsLine.textContent = "";

  const debugEnabled = settings.debugLog;
  if (debugEnabled) {
    debugLogPane.value = "";
    debugLogPane.style.display = "block";
  } else {
    debugLogPane.style.display = "none";
  }
  const appendDebug = (msg: string) => {
    debugLogPane.value += msg + "\n";
    debugLogPane.scrollTop = debugLogPane.scrollHeight;
  };
  const log = debugEnabled
    ? (msg: string) => {
      console.log("[image-gen] " + msg);
      appendDebug(msg);
    }
    : (_: string) => {};
  if (debugEnabled && "gpu" in navigator) {
    navigator.gpu.requestAdapter().then(a => {
      const name = a?.info?.description || a?.info?.device || a?.info?.vendor || "unknown";
      console.info(`[image-gen] WebGPU adapter: ${name}`);
    }).catch(() => {});
  }
  log(`pipeline=${set.family} model=${set.id}`);

  // Tweened progress: between explicit advance() calls, ease the bar
  // from the current unit toward the next using the prior unit's wall-
  // clock duration as the estimate. Capped at 95% so completion still
  // produces a visible snap.
  let unitsDone = 0;
  let lastUnitStart = performance.now();
  let lastUnitDurationMs = 0;
  let rafHandle = 0;
  const tickProgress = () => {
    const target = ((unitsDone + 1) / totalUnits) * 100;
    const base = (unitsDone / totalUnits) * 100;
    if (lastUnitDurationMs > 0) {
      const frac = Math.min((performance.now() - lastUnitStart) / lastUnitDurationMs, 0.95);
      progressBar.value = base + (target - base) * frac;
    }
    rafHandle = requestAnimationFrame(tickProgress);
  };
  const advance = () => {
    const now = performance.now();
    lastUnitDurationMs = now - lastUnitStart;
    lastUnitStart = now;
    unitsDone++;
    progressBar.value = (unitsDone / totalUnits) * 100;
  };
  rafHandle = requestAnimationFrame(tickProgress);

  aborted = false;
  cancelBtn.disabled = false;
  cancelBtn.textContent = "cancel";
  cancelBtn.style.display = "";
  cancelBtn.addEventListener("click", onCancelClick);

  let wakeLock: WakeLockSentinel | null = null;
  try {
    if ("wakeLock" in navigator) {
      wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch {
    // ignore
  }

  resultSection.style.display = "none";
  resultDownloads.innerHTML = "";

  const cb: GenerateCallbacks = {
    log,
    status: (msg) => {
      statusLine.textContent = msg;
    },
    stats: (msg) => {
      statsLine.textContent = msg;
    },
    advance,
    checkAborted: () => {
      if (aborted) throw new CancelledError();
    },
  };

  try {
    const imageData = await pipeline.run(input, cb);

    resultCanvas.width = imageData.width;
    resultCanvas.height = imageData.height;
    const ctx = resultCanvas.getContext("2d");
    if (!ctx) throw new Error("could not get 2d context on result canvas");
    ctx.putImageData(imageData, 0, 0);

    resultDownloads.innerHTML = "";
    const makeDownloadLink = (label: string, filename: string, mime: string, quality?: number) => {
      const link = document.createElement("a");
      link.href = "#";
      link.textContent = label;
      link.style.marginRight = "12px";
      link.addEventListener("click", (ev) => {
        ev.preventDefault();
        resultCanvas.toBlob(
          (blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
          },
          mime,
          quality,
        );
      });
      resultDownloads.appendChild(link);
    };
    makeDownloadLink("download PNG", "image-gen.png", "image/png");
    makeDownloadLink("download JPG", "image-gen.jpg", "image/jpeg", 0.92);

    resultSection.style.display = "";
    statusLine.textContent = "done";
  } catch (err) {
    if (err instanceof CancelledError) {
      statusLine.textContent = "cancelled";
    } else {
      statusLine.textContent = `failed: ${(err as Error).message}`;
      console.error("[image-gen]", err);
    }
  } finally {
    cancelAnimationFrame(rafHandle);
    cancelBtn.removeEventListener("click", onCancelClick);
    cancelBtn.style.display = "none";
    if (wakeLock) {
      try {
        await wakeLock.release();
      } catch {
        // ignore
      }
    }
    refreshGenerateEnabled();
  }
});

// Apply persisted settings before kicking off the model manager so the
// initial model-select refresh sees the right preferred id.
const saved = settingsStore.load();
const initial: ImageGenSettings = { ...defaultGlobalSettings(), ...(saved ?? {}) };
applyGlobalSettings(initial);
// Hand the saved model id to the manager BEFORE init so its first
// refreshModelSelect prefers it directly.
manager.setPreferredId(saved?.model ?? null);

manager
  .init()
  .then(() => {
    const m = currentModelOrNull();
    if (m) {
      applyModelDefaultsToHelp(m);
      applyPerModelSettings(initial.perModel?.[m.id], m);
    }
    refreshRefImageUi();
  })
  .catch(console.error);
