// Generic OPFS-backed model file cache. Tool-agnostic: it knows about
// ModelFile (id, url, size) and nothing about which tool the file belongs to,
// what shape a "model set" takes, or how files are grouped. Each tool
// constructs its own ModelCache with its own OPFS subdirectory so tools can
// be wiped independently.
//
// History: video-face-swap originally had its own model-cache.ts entangled
// with its PatchApplier and ModelSet shape. This module is the generic
// replacement; both image-gen and face-swap now consume it. Patch support
// lives behind the optional ModelFileTransform hook below so the cache itself
// stays patch-agnostic.

export interface ModelFile {
  id: string;
  name: string;
  url: string;
  /** Approximate size for download progress UI. Not used for verification. */
  sizeBytes: number;
  /**
   * Optional public source page (typically a Hugging Face repo) shown from
   * the model list UI so users can click through to see what they're
   * downloading. Not used by the cache itself.
   */
  hfRepoUrl?: string;
  /**
   * Optional transform applied to the bytes as they stream in from the
   * network, before being written to OPFS. Lets a tool (e.g. face-swap) patch
   * an ONNX file on the way down without the cache needing to know what a
   * patch is.
   */
  transform?: ModelFileTransform;
}

/**
 * A streaming transform plugged into a download. The cache calls
 * `createApplier()` once per download, feeds every chunk through `apply()`,
 * calls `finish()` after the network reader closes, and then (if defined)
 * `verify()` against the fully-written cached bytes.
 */
export interface ModelFileTransform {
  /**
   * Suffix appended to `file.id` to form the OPFS storage key. Used so that
   * changing the transform (e.g. updating a patch) naturally invalidates the
   * old cache entry instead of silently reusing a stale blob. Should be
   * stable across runs but unique per transform configuration. Empty string
   * means "use file.id verbatim".
   */
  cacheKeySuffix: string;
  createApplier(): ModelFileApplier;
}

/**
 * Write callback passed to applier methods. Call with just bytes to write to
 * the primary output file. Pass an optional sidecarFileId to direct bytes to
 * a separate OPFS file instead (opened lazily on first use, closed when the
 * download completes, cleaned up on error).
 */
export type TransformWrite = (bytes: Uint8Array, sidecarFileId?: string) => Promise<void>;

export interface ModelFileApplier {
  apply(chunk: Uint8Array, write: TransformWrite): Promise<void>;
  finish(write: TransformWrite): Promise<void>;
  /**
   * Optional post-write verification. Receives the full cached file bytes
   * after `writable.close()`. Throw to signal a corrupt/mismatched result;
   * the cache will then delete the entry and rethrow.
   */
  verify?(bytes: ArrayBuffer): Promise<void>;
}

function cacheKey(file: ModelFile): string {
  return file.transform ? file.id + file.transform.cacheKeySuffix : file.id;
}

export interface DownloadProgress {
  fileId: string;
  fileName: string;
  bytesLoaded: number;
  bytesTotal: number;
  fileIndex: number;
  fileCount: number;
}

export type ProgressCallback = (progress: DownloadProgress) => void;

export interface ModelCacheOptions {
  /**
   * Name of the OPFS subdirectory this cache owns. Must be unique per tool
   * so tools can be cleared independently (e.g. "intabai-image-gen").
   */
  opfsDirName: string;
  /**
   * Old OPFS directory names this tool used in prior releases. They are NOT
   * read from, but `clearAll()` wipes them too so users can reclaim the
   * space when they hit "clear all cached models". Add to this list when
   * renaming, never remove from it.
   */
  legacyDirNames?: string[];
}

export class ModelCache {
  private dirHandle: FileSystemDirectoryHandle | null = null;
  private readonly opfsDirName: string;
  private readonly legacyDirNames: string[];

  constructor(options: ModelCacheOptions) {
    this.opfsDirName = options.opfsDirName;
    this.legacyDirNames = options.legacyDirNames ?? [];
  }

  private async dir(): Promise<FileSystemDirectoryHandle> {
    if (!this.dirHandle) {
      const root = await navigator.storage.getDirectory();
      this.dirHandle = await root.getDirectoryHandle(this.opfsDirName, {
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

  /** Map of file id -> cached?. Files passed in any order. */
  async getCachedStatus(files: ModelFile[]): Promise<Map<string, boolean>> {
    const status = new Map<string, boolean>();
    for (const file of files) {
      status.set(file.id, await this.isFileCached(file));
    }
    return status;
  }

  async areAllCached(files: ModelFile[]): Promise<boolean> {
    for (const file of files) {
      if (!(await this.isFileCached(file))) return false;
    }
    return true;
  }

  /** Download every file in `files` that is not already cached. Files
   *  already present are skipped silently.
   *
   *  Up to `concurrency` files are fetched in parallel. This matters most on
   *  the local dev-server proxy (where each file is served from disk and
   *  serial transfers leave the link idle) but is also a speedup for HF
   *  multi-file bundles. The browser's per-origin connection pool (commonly
   *  6 sockets) naturally throttles when multiple files share a host, so it
   *  is safe to pick a number and let the network layer handle it.
   *
   *  Progress events still fire per-file, so multiple files will be emitting
   *  `bytesLoaded` concurrently. Callers that aggregate into a total-bytes
   *  bar must track in-flight bytes per `fileId` (not just add the latest
   *  event), because progress callbacks from different files will interleave. */
  async downloadFiles(
    files: ModelFile[],
    onProgress: ProgressCallback,
    concurrency = 4,
  ): Promise<void> {
    let nextIndex = 0;
    const worker = async (): Promise<void> => {
      while (true) {
        const i = nextIndex++;
        if (i >= files.length) return;
        const file = files[i];
        if (await this.isFileCached(file)) continue;
        await this.downloadFile(file, i, files.length, onProgress);
      }
    };
    const workerCount = Math.max(1, Math.min(concurrency, files.length));
    await Promise.all(Array.from({ length: workerCount }, () => worker()));
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
    const applier = file.transform ? file.transform.createApplier() : null;
    let loaded = 0;

    // Sidecar file handles opened lazily on first write, keyed by fileId.
    const sidecars = new Map<string, FileSystemWritableFileStream>();
    const sidecarKeys: string[] = [];

    const write: TransformWrite = async (bytes, sidecarFileId?) => {
      if (!sidecarFileId) {
        await writable.write(bytes as Uint8Array<ArrayBuffer>);
        return;
      }
      let stream = sidecars.get(sidecarFileId);
      if (!stream) {
        const h = await dir.getFileHandle(sidecarFileId, { create: true });
        stream = await h.createWritable();
        sidecars.set(sidecarFileId, stream);
        sidecarKeys.push(sidecarFileId);
      }
      await stream.write(bytes as Uint8Array<ArrayBuffer>);
    };

    const closeSidecars = async () => {
      for (const stream of sidecars.values()) {
        try {
          await stream.close();
        } catch {
          /* ignore */
        }
      }
    };

    const removeSidecars = async () => {
      for (const k of sidecarKeys) {
        try {
          await dir.removeEntry(k);
        } catch {
          /* ignore */
        }
      }
    };

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
      await closeSidecars();
    } catch (err) {
      try {
        await writable.close();
      } catch {
        /* ignore */
      }
      try {
        await closeSidecars();
      } catch {
        /* ignore */
      }
      try {
        await dir.removeEntry(key);
      } catch {
        /* ignore */
      }
      await removeSidecars();
      throw err;
    }

    if (applier?.verify) {
      // Read the cached bytes back and let the transform validate them.
      // For our current consumer (face-swap's xseg patch) this is ~70-200ms.
      // If a future transform makes this prohibitive, switch to a streaming
      // hasher fed inside the loop above instead.
      const buf = await (await handle.getFile()).arrayBuffer();
      try {
        await applier.verify(buf);
      } catch (err) {
        try {
          await dir.removeEntry(key);
        } catch {
          /* ignore */
        }
        await removeSidecars();
        throw err;
      }
    }
  }

  /** Load a cached file as raw bytes. Use for ONNX weights handed to ORT. */
  async loadFile(file: ModelFile): Promise<ArrayBuffer> {
    const dir = await this.dir();
    const handle = await dir.getFileHandle(cacheKey(file));
    const blob = await handle.getFile();
    return blob.arrayBuffer();
  }

  /** Load a cached file by raw OPFS key. Use for sidecar files produced by
   *  split transforms, which don't have a corresponding ModelFile entry. */
  async loadFileByKey(key: string): Promise<ArrayBuffer> {
    const dir = await this.dir();
    const handle = await dir.getFileHandle(key);
    const blob = await handle.getFile();
    return blob.arrayBuffer();
  }

  /** Load a cached file as UTF-8 text. Use for tokenizer vocab.json /
   *  merges.txt and other text-shaped artifacts. */
  async loadFileText(file: ModelFile): Promise<string> {
    const dir = await this.dir();
    const handle = await dir.getFileHandle(cacheKey(file));
    const blob = await handle.getFile();
    return blob.text();
  }

  /**
   * Load a cached file as a blob URL. Use this for very large ONNX models
   * (the SD1.5 UNet is 1.72 GB) where loading as an ArrayBuffer would
   * blow ORT-web's wasm 32-bit linear memory cap (4 GB) by forcing the
   * whole file to be copied into the JS heap before being copied again
   * into the wasm allocator.
   *
   * Returned object has a revoke() that the caller MUST call once the
   * URL is no longer needed (typically right after the InferenceSession
   * has been created and is holding its own copy of the bytes).
   */
  async loadFileAsBlobUrl(file: ModelFile): Promise<{ url: string; revoke: () => void }> {
    const dir = await this.dir();
    const handle = await dir.getFileHandle(cacheKey(file));
    const blob = await handle.getFile();
    const url = URL.createObjectURL(blob);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }

  /** Return the size in bytes of a cached file. Cheap (no read). */
  async getFileSize(file: ModelFile): Promise<number> {
    const dir = await this.dir();
    const handle = await dir.getFileHandle(cacheKey(file));
    const blob = await handle.getFile();
    return blob.size;
  }

  async deleteFile(file: ModelFile): Promise<void> {
    const dir = await this.dir();
    await dir.removeEntry(cacheKey(file));
  }

  async deleteFiles(files: ModelFile[]): Promise<void> {
    for (const file of files) {
      try {
        await this.deleteFile(file);
      } catch {
        // already gone
      }
    }
  }

  /** Wipe everything in this cache's OPFS subdirectory. */
  async clearAll(): Promise<void> {
    let dir: FileSystemDirectoryHandle;
    try {
      dir = await this.dir();
    } catch {
      return;
    }
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
      await root.removeEntry(this.opfsDirName, { recursive: true });
    } catch {
      // best-effort
    }
    // Free space from any prior OPFS dir names this tool used. We never read
    // from these, but a user upgrading across the rename needs a way to
    // reclaim the disk - this is it.
    for (const legacy of this.legacyDirNames) {
      try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(legacy, { recursive: true });
      } catch {
        // best-effort: dir may not exist on this device
      }
    }
    this.dirHandle = null;
  }
}
