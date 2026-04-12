import { ModelCache, type DownloadProgress } from "../shared/model-cache";
import {
  ENHANCERS,
  MODEL_SETS,
  type ModelFile,
  type ModelSet,
  allFiles,
  depsSize,
  findEnhancer,
  formatBytes,
} from "./models";

/** Wrap a display name in <strong>, optionally as an external link when the
 *  model has a known source page. Links open in a new tab. */
function linkedStrong(name: string, href: string | undefined): string {
  if (!href) return `<strong>${name}</strong>`;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer"><strong>${name}</strong></a>`;
}

export class ModelManager {
  private cache: ModelCache;
  private readySets = new Set<string>();
  private enhancerCached = new Set<string>();

  constructor(
    private container: HTMLDivElement,
    private enhancerSelect: HTMLSelectElement,
  ) {
    this.cache = new ModelCache({
      opfsDirName: "intabai-video-face-swap",
      legacyDirNames: ["intabai-models"],
    });
  }

  async init(): Promise<void> {
    this.container.innerHTML = "";
    this.readySets.clear();
    this.enhancerCached.clear();

    const swapHeader = document.createElement("div");
    swapHeader.innerHTML = "<small>--- swap models ---</small>";
    swapHeader.style.margin = "12px 0 6px";
    this.container.appendChild(swapHeader);
    for (const set of MODEL_SETS) {
      await this.renderSet(set);
    }

    const enhancerHeader = document.createElement("div");
    enhancerHeader.innerHTML = "<small>--- face enhancers (optional) ---</small>";
    enhancerHeader.style.margin = "16px 0 6px";
    this.container.appendChild(enhancerHeader);
    for (const enhancer of ENHANCERS) {
      await this.renderEnhancer(enhancer);
    }
    this.refreshEnhancerSelect();

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
    });
    clearWrap.appendChild(clearBtn);
    this.container.appendChild(clearWrap);
  }

  // --- Swap model sets ---

  private async renderSet(set: ModelSet): Promise<void> {
    const cached = await this.cache.areAllCached(allFiles(set));
    if (cached) this.readySets.add(set.id);

    const div = document.createElement("div");
    div.id = `model-set-${set.id}`;
    this.fillSetDiv(div, set, cached);
    this.container.appendChild(div);
  }

  private fillSetDiv(div: HTMLElement, set: ModelSet, cached: boolean): void {
    const totalSize = formatBytes(set.primary.sizeBytes + depsSize(set));
    div.title = set.description;
    const linkedName = linkedStrong(set.name, set.primary.hfRepoUrl);
    if (cached) {
      div.innerHTML =
        `${linkedName} <small>(cached)</small> ` + `<button class="delete-btn">delete</button>`;
      div.querySelector(".delete-btn")!.addEventListener("click", () => this.onDeleteSet(set, div));
    } else {
      div.innerHTML =
        `${linkedName} <small>(${totalSize})</small> ` +
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

    // Downloads now run in parallel, so per-file events interleave. Track
    // latest bytesLoaded per fileId and sum for an overall progress bar.
    const files = allFiles(set);
    const totalBytes = set.primary.sizeBytes + depsSize(set);
    const bytesByFile = new Map<string, number>();
    for (const f of files) {
      bytesByFile.set(f.id, (await this.cache.isFileCached(f)) ? f.sizeBytes : 0);
    }

    try {
      await this.cache.downloadFiles(files, (p: DownloadProgress) => {
        bytesByFile.set(p.fileId, p.bytesLoaded);
        let overallBytes = 0;
        for (const v of bytesByFile.values()) overallBytes += v;
        const pct = ((overallBytes / totalBytes) * 100).toFixed(0);
        progressText.textContent = ` ${pct}% (${formatBytes(overallBytes)} / ${formatBytes(totalBytes)})`;
      });

      this.readySets.add(set.id);
      this.fillSetDiv(div, set, true);
    } catch (err) {
      progressText.textContent = ` error: ${err}`;
      btn.disabled = false;
    }
  }

  private async onDeleteSet(set: ModelSet, div: HTMLElement): Promise<void> {
    await this.cache.deleteFiles(allFiles(set));
    this.readySets.delete(set.id);
    this.fillSetDiv(div, set, false);
  }

  // --- Optional enhancers ---

  private async renderEnhancer(enhancer: ModelFile): Promise<void> {
    const cached = await this.cache.isFileCached(enhancer);
    if (cached) this.enhancerCached.add(enhancer.id);

    const div = document.createElement("div");
    div.id = `model-enhancer-${enhancer.id}`;
    this.fillEnhancerDiv(div, enhancer, cached);
    this.container.appendChild(div);
  }

  private fillEnhancerDiv(div: HTMLElement, enhancer: ModelFile, cached: boolean): void {
    div.title = "optional face enhancement";
    const linkedName = linkedStrong(enhancer.name, enhancer.hfRepoUrl);
    if (cached) {
      div.innerHTML =
        `${linkedName} <small>(cached)</small> ` + `<button class="delete-btn">delete</button>`;
      div
        .querySelector(".delete-btn")!
        .addEventListener("click", () => this.onDeleteEnhancer(enhancer, div));
    } else {
      div.innerHTML =
        `${linkedName} <small>(${formatBytes(enhancer.sizeBytes)})</small> ` +
        `<button class="download-btn">download</button>` +
        `<small class="progress-text"></small>`;
      div
        .querySelector(".download-btn")!
        .addEventListener("click", () => this.onDownloadEnhancer(enhancer, div));
    }
  }

  private async onDownloadEnhancer(enhancer: ModelFile, div: HTMLElement): Promise<void> {
    const btn = div.querySelector(".download-btn") as HTMLButtonElement;
    const progressText = div.querySelector(".progress-text") as HTMLSpanElement;
    btn.disabled = true;

    try {
      await this.cache.downloadFile(enhancer, 0, 1, (p: DownloadProgress) => {
        const pct = ((p.bytesLoaded / p.bytesTotal) * 100).toFixed(0);
        progressText.textContent = ` ${pct}%`;
      });

      this.enhancerCached.add(enhancer.id);
      this.fillEnhancerDiv(div, enhancer, true);
      this.refreshEnhancerSelect();
    } catch (err) {
      progressText.textContent = ` error: ${err}`;
      btn.disabled = false;
    }
  }

  private async onDeleteEnhancer(enhancer: ModelFile, div: HTMLElement): Promise<void> {
    await this.cache.deleteFile(enhancer);
    this.enhancerCached.delete(enhancer.id);
    this.fillEnhancerDiv(div, enhancer, false);
    this.refreshEnhancerSelect();
  }

  private refreshEnhancerSelect(): void {
    const previous = this.enhancerSelect.value;
    this.enhancerSelect.innerHTML = "";

    const noneOpt = document.createElement("option");
    noneOpt.value = "";
    noneOpt.textContent = "(none) - faster";
    this.enhancerSelect.appendChild(noneOpt);

    for (const enhancer of ENHANCERS) {
      const opt = document.createElement("option");
      opt.value = enhancer.id;
      const cached = this.enhancerCached.has(enhancer.id);
      opt.textContent = cached ? enhancer.name : `${enhancer.name} - download model to enable`;
      opt.disabled = !cached;
      this.enhancerSelect.appendChild(opt);
    }

    // Restore prior selection if still valid, otherwise default to none
    if (previous && this.enhancerCached.has(previous) && findEnhancer(previous)) {
      this.enhancerSelect.value = previous;
    } else {
      this.enhancerSelect.value = "";
    }
  }

  // --- Public API ---

  isReady(setId: string): boolean {
    return this.readySets.has(setId);
  }

  getSelectedEnhancerId(): string | null {
    const v = this.enhancerSelect.value;
    return v && this.enhancerCached.has(v) ? v : null;
  }

  getReadySetIds(): string[] {
    return [...this.readySets];
  }

  getCache(): ModelCache {
    return this.cache;
  }

  getAvailableFiles(setId: string): ReturnType<typeof allFiles> | null {
    const set = MODEL_SETS.find((s) => s.id === setId);
    return set ? allFiles(set) : null;
  }
}
