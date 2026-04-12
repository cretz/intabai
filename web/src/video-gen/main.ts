// Video-gen tool - UI shell. Backend implementations live in `backends/`.
// This file only knows about the VideoGenBackend interface; it does not
// reference any specific model implementation directly.

import { listModels, getModel, type ModelEntry } from "./models";
import type { ReferenceFrame, VideoGenCapabilities } from "./pipeline";
import { initThemeSelect } from "../shared/theme";

{
  const sel = document.getElementById("theme-select");
  if (sel instanceof HTMLSelectElement) initThemeSelect(sel);
}

// --- DOM refs -----------------------------------------------------------

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

const modelSelect = $<HTMLSelectElement>("model-select");
const resolutionSelect = $<HTMLSelectElement>("resolution-select");

const negativePromptCheck = $<HTMLInputElement>("negative-prompt-check");
const negativePromptField = $<HTMLTextAreaElement>("negative-prompt");

const referenceFramesSection = $<HTMLDetailsElement>("reference-frames-section");
const refFile = $<HTMLInputElement>("ref-file");
const refFrameIndex = $<HTMLInputElement>("ref-frame-index");
const addRefButton = $<HTMLButtonElement>("add-ref-button");
const refList = $<HTMLDivElement>("ref-list");

const numFramesInput = $<HTMLInputElement>("num-frames");
const framesHint = $<HTMLElement>("frames-hint");
const numStepsInput = $<HTMLInputElement>("num-steps");
const stepsHint = $<HTMLElement>("steps-hint");
const seedInput = $<HTMLInputElement>("seed");
const resetOptionsBtn = $<HTMLButtonElement>("reset-options-btn");

const generateButton = $<HTMLButtonElement>("generate-button");
const processingFieldset = $<HTMLFieldSetElement>("processing");
const progressBar = $<HTMLProgressElement>("progress");
const statusLine = $<HTMLElement>("status-line");
const frameStrip = $<HTMLDivElement>("frame-strip");
const errorLine = $<HTMLDivElement>("error-line");
const cancelButton = $<HTMLButtonElement>("cancel-button");

// --- State --------------------------------------------------------------

const referenceFrames: ReferenceFrame[] = [];
let currentModel: ModelEntry | undefined;

// --- Model dropdown -----------------------------------------------------

function populateModelSelect(): void {
  modelSelect.innerHTML = "";
  for (const m of listModels()) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    modelSelect.appendChild(opt);
  }
}

function applyCapabilities(caps: VideoGenCapabilities): void {
  // Resolution dropdown
  resolutionSelect.innerHTML = "";
  for (const r of caps.resolutions) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    if (r === caps.defaultResolution) opt.selected = true;
    resolutionSelect.appendChild(opt);
  }

  // Frame count
  numFramesInput.min = String(caps.minFrames);
  numFramesInput.max = String(caps.maxFrames);
  numFramesInput.value = String(caps.defaultFrames);
  framesHint.textContent = `${caps.minFrames}-${caps.maxFrames} @ ${caps.nativeFps} fps`;

  // Steps
  numStepsInput.min = String(Math.min(...caps.stepOptions));
  numStepsInput.max = String(Math.max(...caps.stepOptions));
  numStepsInput.value = String(caps.defaultSteps);
  stepsHint.textContent = `recommended: ${caps.stepOptions.join(", ")}`;

  // Negative prompt: hide entire control if unsupported
  const negParent = negativePromptCheck.parentElement;
  if (negParent) negParent.style.display = caps.supportsNegativePrompt ? "" : "none";
  if (!caps.supportsNegativePrompt) {
    negativePromptCheck.checked = false;
    negativePromptField.style.display = "none";
  }

  // Reference frames section visibility
  const showRefs = caps.supportsI2V || caps.supportsArbitraryReferenceFrames;
  referenceFramesSection.style.display = showRefs ? "" : "none";

  // If only frame 0 is allowed, lock the frame index input
  if (caps.supportsI2V && !caps.supportsArbitraryReferenceFrames) {
    refFrameIndex.value = "0";
    refFrameIndex.disabled = true;
    refFrameIndex.max = "0";
  } else {
    refFrameIndex.disabled = false;
    refFrameIndex.max = String(caps.maxFrames - 1);
  }
}

function onModelChange(): void {
  const id = modelSelect.value;
  currentModel = getModel(id);
  if (!currentModel) return;
  applyCapabilities(currentModel.capabilities);
  // Drop any reference frames that no longer fit the new model's range.
  for (let i = referenceFrames.length - 1; i >= 0; i--) {
    if (referenceFrames[i].frameIndex >= currentModel.capabilities.maxFrames) {
      referenceFrames.splice(i, 1);
    }
  }
  renderRefList();
  // TODO: check OPFS cache, enable/disable generate button
}

// --- Negative prompt toggle ---------------------------------------------

negativePromptCheck.addEventListener("change", () => {
  negativePromptField.style.display = negativePromptCheck.checked ? "" : "none";
});

// --- Reference frame list -----------------------------------------------

function renderRefList(): void {
  refList.innerHTML = "";
  referenceFrames.sort((a, b) => a.frameIndex - b.frameIndex);
  referenceFrames.forEach((rf, i) => {
    const row = document.createElement("div");
    row.className = "ref-row";
    const thumb = document.createElement("canvas");
    thumb.width = rf.image.width;
    thumb.height = rf.image.height;
    const ctx = thumb.getContext("2d");
    ctx?.drawImage(rf.image, 0, 0);
    const label = document.createElement("span");
    label.textContent = `frame ${rf.frameIndex}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "remove";
    remove.onclick = () => {
      referenceFrames.splice(i, 1);
      renderRefList();
    };
    row.appendChild(thumb);
    row.appendChild(label);
    row.appendChild(remove);
    refList.appendChild(row);
  });
  // Update <summary> with count
  const summary = referenceFramesSection.querySelector("summary");
  if (summary) {
    const n = referenceFrames.length;
    summary.textContent = `reference frames (${n === 0 ? "none" : n})`;
  }
}

addRefButton.onclick = async () => {
  const file = refFile.files?.[0];
  if (!file) {
    statusLine.textContent = "pick an image first";
    return;
  }
  const bitmap = await createImageBitmap(file);
  referenceFrames.push({
    frameIndex: parseInt(refFrameIndex.value, 10) || 0,
    image: bitmap,
  });
  refFile.value = "";
  renderRefList();
};

// --- Reset options ------------------------------------------------------

resetOptionsBtn.onclick = () => {
  if (currentModel) applyCapabilities(currentModel.capabilities);
  seedInput.value = "";
  negativePromptField.value = "";
  negativePromptCheck.checked = false;
  negativePromptField.style.display = "none";
};

// --- Generate (stub) ----------------------------------------------------

generateButton.onclick = () => {
  processingFieldset.style.display = "";
  statusLine.textContent = "generate not implemented yet (spike scaffold)";
  errorLine.textContent = "";
  progressBar.value = 0;
  frameStrip.innerHTML = "";
};

cancelButton.onclick = () => {
  statusLine.textContent = "cancelled";
};

// --- Init ---------------------------------------------------------------

modelSelect.addEventListener("change", onModelChange);
populateModelSelect();
onModelChange();
