// Image-gen model manager. Mirrors video-face-swap/model-manager.ts in
// behavior: renders one row per ModelSet with cached/download/delete state,
// drives a "model select" dropdown that disables non-cached entries, and
// provides a clear-all-cached button.
//
// Each row shows total-bytes progress while downloading (the underlying
// ModelCache emits per-file progress events; we sum the known per-file
// sizes to compute an overall percentage so the bar reflects bytes, not
// file count - the 1.7 GB UNet shouldn't progress at the same speed as
// the 524 KB merges.txt).

import { ModelCache, type DownloadProgress } from "../shared/model-cache";
import { IMAGE_GEN_MODELS, modelSetFiles, type ModelSet } from "../sd15/models";

type ModelSection = "recommended" | "lower-quality";

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

function modelSection(set: ModelSet): ModelSection {
  return set.id === "zimage_turbo" || set.id === "zimage_turbo_sharded"
    ? "recommended"
    : "lower-quality";
}

function cacheListLabel(set: ModelSet): string {
  const linkedName = set.hfRepoUrl
    ? `<a href="${set.hfRepoUrl}" target="_blank" rel="noopener noreferrer"><strong>${set.name}</strong></a>`
    : `<strong>${set.name}</strong>`;
  if (set.id === "zimage_turbo") {
    return `${linkedName} <small>recommended for desktop</small>`;
  }
  if (set.id === "zimage_turbo_sharded") {
    return `${linkedName} <small>recommended for mobile</small>`;
  }
  return linkedName;
}

function totalSetBytes(set: ModelSet): number {
  let n = 0;
  for (const f of modelSetFiles(set)) n += f.sizeBytes;
  return n;
}

export class ImageGenModelManager {
  private readonly cache: ModelCache;
  private readonly readySets = new Set<string>();
  /** The model id the user last had selected (loaded from persisted
   *  settings). refreshModelSelect prefers this over the "first cached"
   *  fallback so a refresh restores the previous choice without needing
   *  a post-init restoration step that races with input events. */
  private preferredId: string | null = null;
  /** The fieldset legend element whose parenthetical we update. */
  private readonly legendEl: HTMLElement | null;

  constructor(
    private readonly container: HTMLElement,
    private readonly modelSelect: HTMLSelectElement,
    private readonly onReadyChange: () => void,
  ) {
    this.cache = new ModelCache({
      opfsDirName: "intabai-image-gen",
      legacyDirNames: ["intabai-sd15"],
    });
    // Find the parent fieldset's legend so we can update the count.
    this.legendEl = this.container.closest("fieldset")?.querySelector("legend") ?? null;
  }

  /** Hint the manager about a previously-selected model id. Must be called
   *  before init() so the first refreshModelSelect picks it up. */
  setPreferredId(id: string | null): void {
    this.preferredId = id;
  }

  getCache(): ModelCache {
    return this.cache;
  }

  isReady(setId: string): boolean {
    return this.readySets.has(setId);
  }

  getSelectedSetId(): string | null {
    const v = this.modelSelect.value;
    return v && this.readySets.has(v) ? v : null;
  }

  async init(): Promise<void> {
    this.container.innerHTML = "";
    this.readySets.clear();

    const header = document.createElement("small");
    header.textContent = "recommended models first. all downloads stay in this browser.";
    header.style.display = "block";
    header.style.marginBottom = "8px";
    this.container.appendChild(header);

    const recommendedSection = this.createSection(
      "recommended",
      "best current options: full Z-Image for desktop, sharded Z-Image for mobile.",
    );
    const lowerQualitySection = this.createSection(
      "lower quality / fallback",
      "older, smaller, or weaker models that are still available if you want them.",
    );
    this.container.appendChild(recommendedSection);
    this.container.appendChild(lowerQualitySection);

    for (const set of IMAGE_GEN_MODELS) {
      const parent =
        modelSection(set) === "recommended"
          ? recommendedSection
          : lowerQualitySection;
      await this.renderSet(set, parent);
    }

    const clearWrap = document.createElement("div");
    clearWrap.style.marginTop = "8px";
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.textContent = "clear all cached models";
    clearBtn.addEventListener("click", async () => {
      if (
        !confirm(
          "Delete all cached models from this browser? You'll need to re-download them next time.",
        )
      ) {
        return;
      }
      clearBtn.disabled = true;
      clearBtn.textContent = "clearing...";
      await this.cache.clearAll();
      await this.init();
      this.refreshModelSelect();
      this.onReadyChange();
    });
    clearWrap.appendChild(clearBtn);
    this.container.appendChild(clearWrap);

    this.refreshModelSelect();
    this.refreshLegend();
    this.onReadyChange();
  }

  private createSection(title: string, description: string): HTMLDivElement {
    const wrap = document.createElement("div");
    wrap.style.margin = "10px 0";
    wrap.style.padding = "6px 8px";
    wrap.style.border = "1px solid";

    const heading = document.createElement("div");
    heading.style.fontSize = "12px";
    heading.style.fontWeight = "bold";
    heading.style.letterSpacing = "0.06em";
    heading.style.marginBottom = "2px";
    heading.textContent = title.toUpperCase();
    wrap.appendChild(heading);

    const help = document.createElement("small");
    help.textContent = description;
    help.style.display = "block";
    help.style.marginBottom = "6px";
    wrap.appendChild(help);

    return wrap;
  }

  private async renderSet(set: ModelSet, parent: HTMLElement): Promise<void> {
    const files = modelSetFiles(set);
    const cached = await this.cache.areAllCached(files);
    if (cached) this.readySets.add(set.id);

    const div = document.createElement("div");
    div.id = `model-set-${set.id}`;
    this.fillSetDiv(div, set, cached);
    parent.appendChild(div);
  }

  private fillSetDiv(div: HTMLElement, set: ModelSet, cached: boolean): void {
    const totalSize = formatBytes(totalSetBytes(set));
    const label = cacheListLabel(set);
    div.title = set.description;
    div.style.display = "flex";
    div.style.alignItems = "baseline";
    div.style.gap = "8px";
    div.style.padding = "2px 0";
    if (cached) {
      div.innerHTML =
        `<span style="flex:1">${label}</span>` +
        `<small>${totalSize}</small>` +
        `<small style="color:green">cached</small> ` +
        `<button class="delete-btn">delete</button>`;
      div.querySelector(".delete-btn")!.addEventListener("click", () => this.onDeleteSet(set, div));
    } else {
      div.innerHTML =
        `<span style="flex:1">${label}</span>` +
        `<small>${totalSize}</small>` +
        `<button class="download-btn">download</button>` +
        `<small class="progress-text"></small>`;
      div
        .querySelector(".download-btn")!
        .addEventListener("click", () => this.onDownloadSet(set, div));
    }
  }

  private async onDownloadSet(set: ModelSet, div: HTMLElement): Promise<void> {
    const btn = div.querySelector(".download-btn") as HTMLButtonElement;
    const progressText = div.querySelector(".progress-text") as HTMLElement;
    btn.disabled = true;

    const files = modelSetFiles(set);
    // Total-bytes progress: ModelCache now downloads multiple files in
    // parallel, so progress events from different files interleave. We keep
    // a per-fileId map of "latest bytesLoaded" and sum across all entries to
    // compute the overall bar. Files already cached at start are seeded with
    // their full declared size so the bar reflects real progress on resume.
    const totalBytes = totalSetBytes(set);
    const bytesByFile = new Map<string, number>();
    for (const f of files) {
      bytesByFile.set(f.id, (await this.cache.isFileCached(f)) ? f.sizeBytes : 0);
    }

    try {
      await this.cache.downloadFiles(files, (p: DownloadProgress) => {
        bytesByFile.set(p.fileId, p.bytesLoaded);
        let overallBytes = 0;
        for (const v of bytesByFile.values()) overallBytes += v;
        const pct = ((overallBytes / totalBytes) * 100).toFixed(1);
        progressText.textContent = ` ${pct}% (${formatBytes(overallBytes)} / ${formatBytes(totalBytes)})`;
      });

      this.readySets.add(set.id);
      this.fillSetDiv(div, set, true);
      this.refreshModelSelect();
      this.refreshLegend();
      this.onReadyChange();
    } catch (err) {
      progressText.textContent = ` error: ${err}`;
      btn.disabled = false;
    }
  }

  private async onDeleteSet(set: ModelSet, div: HTMLElement): Promise<void> {
    await this.cache.deleteFiles(modelSetFiles(set));
    this.readySets.delete(set.id);
    this.fillSetDiv(div, set, false);
    this.refreshModelSelect();
    this.refreshLegend();
    this.onReadyChange();
  }

  private refreshLegend(): void {
    if (!this.legendEl) return;
    const n = this.readySets.size;
    const label =
      n === 0
        ? "models (at least 1 needed)"
        : `models (${n} cached)`;
    this.legendEl.textContent = label;
  }

  private refreshModelSelect(): void {
    const previous = this.modelSelect.value;
    this.modelSelect.innerHTML = "";

    for (const set of IMAGE_GEN_MODELS) {
      const opt = document.createElement("option");
      opt.value = set.id;
      const cached = this.readySets.has(set.id);
      opt.textContent = cached ? set.name : `${set.name} - download to enable`;
      opt.disabled = !cached;
      this.modelSelect.appendChild(opt);
    }

    // Priority: previous selection (if still cached) > preferred id from
    // persisted settings (if cached) > first cached fallback.
    if (previous && this.readySets.has(previous)) {
      this.modelSelect.value = previous;
    } else if (this.preferredId && this.readySets.has(this.preferredId)) {
      this.modelSelect.value = this.preferredId;
    } else {
      const firstReady = IMAGE_GEN_MODELS.find((m) => this.readySets.has(m.id));
      this.modelSelect.value = firstReady ? firstReady.id : "";
    }
  }
}
