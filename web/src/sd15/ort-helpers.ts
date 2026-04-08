// Shared helpers for loading ORT-web InferenceSessions out of a ModelCache.
//
// SD1.5 nmkd ships every file as a monolithic ONNX (the small ones) plus
// one external-data UNet (graph + weights.pb sidecar). Segmind-Vega and
// every diffusers / optimum SDXL export ships ALL components as external-
// data (graph + model.onnx_data sidecar). The two layouts are otherwise
// identical at the ORT API level: external-data is wired via the
// sessionOptions.externalData option, which maps a path string referenced
// inside the .onnx graph to a blob URL pointing at the cached sidecar.
//
// This helper hides the difference behind a single load function so the
// per-component wrappers (TextEncoder, Unet, VaeDecoder, VaeEncoder) can
// stay focused on their pre/post-processing math.

import * as ort from "onnxruntime-web";

import type { ModelCache, ModelFile } from "../shared/model-cache";

/** A loadable ONNX model. Either a single monolithic file, or a graph
 *  file plus an external-data sidecar. The dataPath string must match the
 *  filename the .onnx graph internally references; "weights.pb" for nmkd
 *  SD1.5 exports, "model.onnx_data" for diffusers/optimum SDXL exports. */
export type OrtModelFile = ModelFile | { graph: ModelFile; data: ModelFile; dataPath: string };

/** Type guard. */
export function isExternalData(
  m: OrtModelFile,
): m is { graph: ModelFile; data: ModelFile; dataPath: string } {
  return (m as { graph?: unknown }).graph !== undefined;
}

/** Flatten an OrtModelFile to the list of underlying ModelFile entries.
 *  Used by the bundle file enumerator (modelSetFiles). */
export function ortModelFiles(m: OrtModelFile): ModelFile[] {
  if (isExternalData(m)) return [m.graph, m.data];
  return [m];
}

/**
 * Create an ORT InferenceSession from a model that may or may not have an
 * external-data sidecar. Handles blob URL bookkeeping and externalData
 * wiring transparently.
 */
export async function createSession(
  cache: ModelCache,
  model: OrtModelFile,
  providers: string[],
  extraOptions: Partial<ort.InferenceSession.SessionOptions> = {},
): Promise<ort.InferenceSession> {
  if (!isExternalData(model)) {
    const buffer = await cache.loadFile(model);
    return ort.InferenceSession.create(buffer, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      ...extraOptions,
    });
  }

  // External-data layout. Stream both pieces as blob URLs (avoids copying
  // the multi-GB sidecar through the wasm allocator) and wire them via
  // sessionOptions.externalData.
  const { url: graphUrl, revoke: revokeGraph } = await cache.loadFileAsBlobUrl(model.graph);
  const { url: dataUrl, revoke: revokeData } = await cache.loadFileAsBlobUrl(model.data);
  try {
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      ...extraOptions,
    };
    // ORT-web 1.18+ accepts externalData on SessionOptions. Cast through
    // unknown because the .d.ts in some versions has not caught up.
    (
      sessionOptions as unknown as {
        externalData: Array<{ path: string; data: string }>;
      }
    ).externalData = [{ path: model.dataPath, data: dataUrl }];
    return await ort.InferenceSession.create(graphUrl, sessionOptions);
  } finally {
    revokeGraph();
    revokeData();
  }
}
