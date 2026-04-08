import { type ModelFile, type ModelSet, allFiles } from "./models";
import { PatchApplier, sha256Hex } from "./model-patch";

export interface DownloadProgress {
  fileId: string;
  fileName: string;
  bytesLoaded: number;
  bytesTotal: number;
  fileIndex: number;
  fileCount: number;
}

export type ProgressCallback = (progress: DownloadProgress) => void;

const OPFS_DIR = "intabai-models";

/**
 * Storage key for a file in OPFS. For patched files we tack on a short
 * fingerprint of the patch's expected output hash, so changing the patch
 * (or removing it) naturally invalidates the old cache entry instead of
 * silently reusing a stale or unpatched blob.
 */
function cacheKey(file: ModelFile): string {
  if (!file.patch) return file.id;
  return `${file.id}.patched.${file.patch.dstSha256.slice(0, 8)}`;
}

export class ModelCache {
  private dirHandle: FileSystemDirectoryHandle | null = null;

  private async dir(): Promise<FileSystemDirectoryHandle> {
    if (!this.dirHandle) {
      const root = await navigator.storage.getDirectory();
      this.dirHandle = await root.getDirectoryHandle(OPFS_DIR, {
        create: true,
      });
    }
    return this.dirHandle;
  }

  async isFileCached(file: ModelFile): Promise<boolean> {
    try {
      const dir = await this.dir();
      await dir.getFileHandle(cacheKey(file));
      return true;
    } catch {
      return false;
    }
  }

  async getCachedStatus(set: ModelSet): Promise<Map<string, boolean>> {
    const status = new Map<string, boolean>();
    for (const file of allFiles(set)) {
      status.set(file.id, await this.isFileCached(file));
    }
    return status;
  }

  async isSetCached(set: ModelSet): Promise<boolean> {
    const status = await this.getCachedStatus(set);
    return [...status.values()].every(Boolean);
  }

  async downloadSet(set: ModelSet, onProgress: ProgressCallback): Promise<void> {
    const files = allFiles(set);
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (await this.isFileCached(file)) continue;
      await this.downloadFile(file, i, files.length, onProgress);
    }
  }

  async downloadFile(
    file: ModelFile,
    fileIndex: number,
    fileCount: number,
    onProgress: ProgressCallback,
  ): Promise<void> {
    const response = await fetch(file.url);
    if (!response.ok) {
      throw new Error(`Failed to download ${file.name}: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error(`No response body for ${file.name}`);
    }

    const contentLength = Number(response.headers.get("content-length")) || file.sizeBytes;
    const dir = await this.dir();
    const key = cacheKey(file);
    const handle = await dir.getFileHandle(key, { create: true });
    const writable = await handle.createWritable();
    const applier = file.patch ? new PatchApplier(file.patch) : null;
    const write = (bytes: Uint8Array) => writable.write(bytes as Uint8Array<ArrayBuffer>);
    let loaded = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (applier) {
          await applier.apply(value, write);
        } else {
          await write(value);
        }
        loaded += value.length;
        onProgress({
          fileId: file.id,
          fileName: file.name,
          bytesLoaded: loaded,
          bytesTotal: contentLength,
          fileIndex,
          fileCount,
        });
      }
      if (applier) await applier.finish(write);
      await writable.close();
    } catch (err) {
      try {
        await writable.close();
      } catch {
        // ignore
      }
      try {
        await dir.removeEntry(key);
      } catch {
        // ignore
      }
      throw err;
    }

    if (file.patch) {
      // Verify the patched output matches the expected hash. Reads the file
      // back from OPFS into memory once (~70-200ms for our patched models)
      // and runs SubtleCrypto SHA-256 over it. If it ever bothers us — e.g.
      // when we patch a much larger model — switch to a streaming JS hasher
      // fed during download instead. See worklog TODO.
      const buf = await (await handle.getFile()).arrayBuffer();
      const got = await sha256Hex(buf);
      if (got !== file.patch.dstSha256) {
        try {
          await dir.removeEntry(key);
        } catch {
          // ignore
        }
        throw new Error(
          `${file.name}: patched hash mismatch (got ${got}, expected ${file.patch.dstSha256})`,
        );
      }
    }
  }

  async loadFile(file: ModelFile): Promise<ArrayBuffer> {
    const dir = await this.dir();
    const handle = await dir.getFileHandle(cacheKey(file));
    const blob = await handle.getFile();
    return blob.arrayBuffer();
  }

  async deleteFile(file: ModelFile): Promise<void> {
    const dir = await this.dir();
    await dir.removeEntry(cacheKey(file));
  }

  async deleteSet(set: ModelSet): Promise<void> {
    for (const file of allFiles(set)) {
      try {
        await this.deleteFile(file);
      } catch {
        // already gone
      }
    }
  }

  /**
   * Wipe everything inside our OPFS directory, including stale entries from
   * earlier cache key schemes (e.g. unpatched `xseg_1` blobs left behind when
   * the patched cache key took over). Belt-and-suspenders: walks every entry
   * in the dir and removes it, then removes the dir itself so it gets
   * recreated fresh on next use.
   */
  async clearAll(): Promise<void> {
    let dir: FileSystemDirectoryHandle;
    try {
      dir = await this.dir();
    } catch {
      return;
    }
    // FileSystemDirectoryHandle is async-iterable in supporting browsers.
    const iter = (dir as unknown as { keys(): AsyncIterable<string> }).keys();
    for await (const name of iter) {
      try {
        await dir.removeEntry(name, { recursive: true });
      } catch {
        // best-effort
      }
    }
    try {
      const root = await navigator.storage.getDirectory();
      await root.removeEntry(OPFS_DIR, { recursive: true });
    } catch {
      // best-effort
    }
    this.dirHandle = null;
  }
}
