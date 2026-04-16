// Minimal model-cache UI for the video-gen tool. Renders one row per
// VideoModelEntry with download / cached / delete state and a byte
// progress bar. Mirrors image-gen's ImageGenModelManager but simplified:
// no recommended/fallback sections (only one model for now), and no
// persisted model-select restoration (no select to restore).
//
// As we add more video backends, extend this in the same direction
// image-gen went - promote a VIDEO_GEN_MODELS iteration into sections,
// surface a model dropdown, track ready-state per bundle.

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

  /** Is this model bundle fully downloaded? */
  isReady(id: string): boolean {
    return this.ready.has(id);
  }

  /** Ids of every bundle that's fully cached, in registry order. */
  readyIds(): string[] {
    return VIDEO_GEN_MODELS.filter((m) => this.ready.has(m.id)).map((m) => m.id);
  }

  async init(): Promise<void> {
    this.container.innerHTML = "";
    this.ready.clear();

    const header = document.createElement("small");
    header.textContent = "all downloads stay in this browser.";
    header.style.display = "block";
    header.style.marginBottom = "8px";
    this.container.appendChild(header);

    for (const model of VIDEO_GEN_MODELS) {
      await this.renderModel(model);
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

  private async renderModel(model: VideoModelEntry): Promise<void> {
    const cached = await this.cache.areAllCached(model.files);
    if (cached) this.ready.add(model.id);

    const wrap = document.createElement("div");
    wrap.id = `model-${model.id}`;
    wrap.style.margin = "10px 0";
    wrap.style.padding = "6px 8px";
    wrap.style.border = "1px solid";

    const heading = document.createElement("div");
    heading.style.fontSize = "12px";
    heading.style.fontWeight = "bold";
    heading.style.letterSpacing = "0.06em";
    heading.textContent = model.name.toUpperCase();
    if (model.hfRepoUrl) {
      const link = document.createElement("a");
      link.href = model.hfRepoUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = model.name;
      heading.textContent = "";
      heading.appendChild(link);
    }
    wrap.appendChild(heading);

    const desc = document.createElement("small");
    desc.style.display = "block";
    desc.style.margin = "4px 0 6px";
    desc.textContent = `${model.resolutionLabel}, ${model.clipLabel}. ${model.description}`;
    wrap.appendChild(desc);

    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "baseline";
    row.style.gap = "8px";
    wrap.appendChild(row);

    this.fillRow(row, model, cached);
    this.container.appendChild(wrap);
  }

  private fillRow(row: HTMLElement, model: VideoModelEntry, cached: boolean): void {
    const total = formatBytes(totalBytes(model.files));
    row.innerHTML = "";
    const sizeLabel = document.createElement("small");
    sizeLabel.style.flex = "1";
    sizeLabel.textContent = `total ${total}`;
    row.appendChild(sizeLabel);

    if (cached) {
      const cachedLabel = document.createElement("small");
      cachedLabel.className = "cached-label";
      cachedLabel.textContent = "cached";
      row.appendChild(cachedLabel);
      const del = document.createElement("button");
      del.type = "button";
      del.textContent = "delete";
      del.addEventListener("click", () => this.deleteModel(model, row));
      row.appendChild(del);
      return;
    }

    const dl = document.createElement("button");
    dl.type = "button";
    dl.textContent = "download";
    const progressText = document.createElement("small");
    progressText.className = "progress-text";
    dl.addEventListener("click", () => this.downloadModel(model, row, dl, progressText));
    row.appendChild(dl);
    row.appendChild(progressText);
  }

  private async downloadModel(
    model: VideoModelEntry,
    row: HTMLElement,
    btn: HTMLButtonElement,
    progressText: HTMLElement,
  ): Promise<void> {
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
      this.fillRow(row, model, true);
      this.refreshLegend();
      this.onReadyChange();
    } catch (err) {
      progressText.textContent = ` error: ${err}`;
      btn.disabled = false;
    }
  }

  private async deleteModel(model: VideoModelEntry, row: HTMLElement): Promise<void> {
    await this.cache.deleteFiles(model.files);
    this.ready.delete(model.id);
    this.fillRow(row, model, false);
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
