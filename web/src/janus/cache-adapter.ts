// Adapter that exposes our OPFS-backed ModelCache to transformers.js via its
// `env.customCache` hook (CacheInterface in @huggingface/transformers).
//
// Why: transformers.js owns its own model loading flow (it knows which ONNX
// files MultiModalityCausalLM needs, in which order, and how to wire their
// inputs together). We do not want to fight that. But we DO want every byte
// to flow through the same OPFS dir + cached badge + clear-all UX that
// sd15 and sdxl already use. The customCache hook is the seam: transformers.js
// calls cache.match(url) before fetching, and if we return a Response built
// from OPFS bytes, it never touches the network.
//
// The model manager pre-downloads every URL in JANUS_PRO_1B_FILES via our
// existing ModelCache.downloadFiles() so that by the time generate() runs,
// every file transformers.js will ask for is already on disk. The adapter's
// match() builds a fresh Response from the cached ArrayBuffer for each call.
//
// put() is a no-op. The pre-declared file list in janus/models.ts covers
// every URL transformers.js needs (7 configs + 6 ONNX components). If
// something unexpected ever fires put(), the warning says so and we add
// the URL to JANUS_PRO_1B_FILES. Letting put() write into our OPFS cache
// would require either reaching into ModelCache's private dir handle or
// promoting putBytes() to a public method - neither is worth doing for a
// safety net we expect to never trigger.

import type { ModelCache, ModelFile } from "../shared/model-cache";

/** Extract the path portion after /resolve/<revision>/ from an HF URL.
 *  e.g. "https://huggingface.co/org/repo/resolve/main/onnx/foo.onnx" -> "onnx/foo.onnx"
 *       "https://huggingface.co/org/repo/resolve/abc123/tokenizer.json" -> "tokenizer.json" */
function extractHfPath(url: string): string | undefined {
  const m = url.match(/\/resolve\/[^/]+\/(.+)$/);
  return m?.[1];
}

/** Shape transformers.js expects from `env.customCache`. Mirrors
 *  CacheInterface in @huggingface/transformers/utils/cache.d.ts but typed
 *  loosely so we do not have to import a deep type from the lib. */
export interface TransformersCacheInterface {
  match(request: string): Promise<Response | undefined>;
  put(
    request: string,
    response: Response,
    progress_callback?: (data: { progress: number; loaded: number; total: number }) => void,
  ): Promise<void>;
  delete?(request: string): Promise<boolean>;
}

export class JanusCacheAdapter implements TransformersCacheInterface {
  // Map from the path portion after /resolve/*/ to the ModelFile.
  // transformers.js may request with /resolve/main/ while our URLs use a
  // pinned commit hash, so we match on the path suffix (e.g. "tokenizer.json"
  // or "onnx/language_model_q4f16.onnx") rather than the full URL.
  private readonly byPath = new Map<string, ModelFile>();
  // Path suffix -> OPFS cache key for sidecar files produced by split
  // transforms. When transformers.js resolves an external_data reference
  // (e.g. requesting "...language_model_q4f16.onnx_data"), we serve it
  // from the sidecar file the split wrote into OPFS during download.
  private readonly sidecarByPath = new Map<string, string>();

  constructor(
    private readonly cache: ModelCache,
    knownFiles: ModelFile[],
  ) {
    for (const f of knownFiles) {
      const path = extractHfPath(f.url);
      if (path) this.byPath.set(path, f);
    }
  }

  /** Register a sidecar file so match() can serve it. `pathSuffix` is the
   *  path after /resolve/ that transformers.js will request (e.g.
   *  "onnx/language_model_q4f16.onnx_data"). `cacheKey` is the OPFS file ID
   *  the split transform wrote to. */
  addSidecar(pathSuffix: string, cacheKey: string): void {
    this.sidecarByPath.set(pathSuffix, cacheKey);
  }

  async match(request: string): Promise<Response | undefined> {
    const reqPath = extractHfPath(request);
    if (!reqPath) {
      // Not an HF URL (e.g. ORT wasm runtime files from jsdelivr).
      return undefined;
    }

    // Check pre-declared model files first.
    const file = this.byPath.get(reqPath);
    if (file) {
      const buf = await this.cache.loadFile(file);
      return new Response(buf);
    }
    // Check sidecar files produced by split transforms.
    const sidecarKey = this.sidecarByPath.get(reqPath);
    if (sidecarKey) {
      const buf = await this.cache.loadFileByKey(sidecarKey);
      return new Response(buf);
    }
    // Every HF file must be pre-declared or registered as a sidecar.
    // If we get here, the file list is incomplete - fail loudly rather
    // than silently downloading from HF.
    throw new Error(
      `[janus-cache] undeclared HF path: ${reqPath} (from ${request}). ` +
        `Add it to JANUS_PRO_1B_FILES or register it as a sidecar.`,
    );
  }

  async put(_request: string, _response: Response): Promise<void> {
    // No-op. transformers.js calls put() after a successful fetch to cache
    // the response. Since we never return undefined for HF URLs (we throw
    // instead), put() should only fire for non-HF URLs (ORT wasm runtime
    // files from jsdelivr, handled by browser HTTP cache).
  }

  async delete(request: string): Promise<boolean> {
    const reqPath = extractHfPath(request);
    const file = reqPath ? this.byPath.get(reqPath) : undefined;
    if (!file) return false;
    try {
      await this.cache.deleteFile(file);
      return true;
    } catch {
      return false;
    }
  }
}
