// Minimal model-cache UI for the video-gen tool. Renders one compact
// row per VideoModelEntry inside section containers, mirroring
// image-gen/model-manager.ts. The single section today is "experimental
// (quality issues)" since FastWan output does not yet match the HF Space
// reference. Add new sections as more backends land.

import { ModelCache, type DownloadProgress, type ModelFile } from "../shared/model-cache";
import { VIDEO_GEN_MODELS, type VideoModelEntry } from "./models";

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

function totalBytes(files: ModelFile[]): number {
  let n = 0;
  for (const f of files) n += f.sizeBytes;
  return n;
}

function rowLabel(model: VideoModelEntry): string {
  const linkedName = model.hfRepoUrl
    ? `<a href="${model.hfRepoUrl}" target="_blank" rel="noopener noreferrer"><strong>${model.name}</strong></a>`
    : `<strong>${model.name}</strong>`;
  return `${linkedName} <small>${model.resolutionLabel}, ${model.clipLabel}</small>`;
}

export class VideoGenModelManager {
  readonly cache: ModelCache;
  private readonly ready = new Set<string>();
  private readonly legendEl: HTMLElement | null;

  constructor(
    private readonly container: HTMLElement,
    private readonly onReadyChange: () => void,
  ) {
    this.cache = new ModelCache({ opfsDirName: "intabai-video-gen" });
    this.legendEl = container.closest("fieldset")?.querySelector("legend") ?? null;
  }

  isReady(id: string): boolean {
    return this.ready.has(id);
  }

  readyIds(): string[] {
    return VIDEO_GEN_MODELS.filter((m) => this.ready.has(m.id)).map((m) => m.id);
  }

  async init(): Promise<void> {
    this.container.innerHTML = "";
    this.ready.clear();

    const experimentalSection = this.createSection(
      "experimental (quality issues)",
      "FastWan runs end-to-end but output is blocky and topically off vs the reference HF Space. See the README's known-issues section.",
    );
    this.container.appendChild(experimentalSection);

    for (const model of VIDEO_GEN_MODELS) {
      await this.renderModel(model, experimentalSection);
    }

    const clearWrap = document.createElement("div");
    clearWrap.style.marginTop = "8px";
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.textContent = "clear all cached models";
    clearBtn.addEventListener("click", async () => {
      if (
        !confirm(
          "Delete all cached video-gen models? You'll need to re-download them next time.",
        )
      )
        return;
      clearBtn.disabled = true;
      clearBtn.textContent = "clearing...";
      await this.cache.clearAll();
      await this.init();
      this.onReadyChange();
    });
    clearWrap.appendChild(clearBtn);
    this.container.appendChild(clearWrap);

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

  private async renderModel(model: VideoModelEntry, parent: HTMLElement): Promise<void> {
    const cached = await this.cache.areAllCached(model.files);
    if (cached) this.ready.add(model.id);

    const div = document.createElement("div");
    div.id = `model-${model.id}`;
    this.fillRow(div, model, cached);
    parent.appendChild(div);
  }

  private fillRow(div: HTMLElement, model: VideoModelEntry, cached: boolean): void {
    const total = formatBytes(totalBytes(model.files));
    const label = rowLabel(model);
    div.title = model.description;
    div.style.display = "flex";
    div.style.alignItems = "baseline";
    div.style.gap = "8px";
    div.style.padding = "2px 0";
    if (cached) {
      div.innerHTML =
        `<span style="flex:1">${label}</span>` +
        `<small>${total}</small>` +
        `<small class="cached-label">cached</small> ` +
        `<button class="delete-btn">delete</button>`;
      div
        .querySelector(".delete-btn")!
        .addEventListener("click", () => this.deleteModel(model, div));
    } else {
      div.innerHTML =
        `<span style="flex:1">${label}</span>` +
        `<small>${total}</small>` +
        `<button class="download-btn">download</button>` +
        `<small class="progress-text"></small>`;
      div
        .querySelector(".download-btn")!
        .addEventListener("click", () => this.downloadModel(model, div));
    }
  }

  private async downloadModel(model: VideoModelEntry, div: HTMLElement): Promise<void> {
    const btn = div.querySelector(".download-btn") as HTMLButtonElement;
    const progressText = div.querySelector(".progress-text") as HTMLElement;
    btn.disabled = true;

    const total = totalBytes(model.files);
    const bytesByFile = new Map<string, number>();
    for (const f of model.files) {
      bytesByFile.set(f.id, (await this.cache.isFileCached(f)) ? f.sizeBytes : 0);
    }
    try {
      await this.cache.downloadFiles(model.files, (p: DownloadProgress) => {
        bytesByFile.set(p.fileId, p.bytesLoaded);
        let loaded = 0;
        for (const v of bytesByFile.values()) loaded += v;
        const pct = ((loaded / total) * 100).toFixed(1);
        progressText.textContent = ` ${pct}% (${formatBytes(loaded)} / ${formatBytes(total)})`;
      });
      this.ready.add(model.id);
      this.fillRow(div, model, true);
      this.refreshLegend();
      this.onReadyChange();
    } catch (err) {
      progressText.textContent = ` error: ${err}`;
      btn.disabled = false;
    }
  }

  private async deleteModel(model: VideoModelEntry, div: HTMLElement): Promise<void> {
    await this.cache.deleteFiles(model.files);
    this.ready.delete(model.id);
    this.fillRow(div, model, false);
    this.refreshLegend();
    this.onReadyChange();
  }

  private refreshLegend(): void {
    if (!this.legendEl) return;
    const n = this.ready.size;
    this.legendEl.textContent =
      n === 0 ? "models (download one to enable generate)" : `models (${n} cached)`;
  }
}
