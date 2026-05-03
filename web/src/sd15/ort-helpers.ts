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

/** A loadable ONNX model. One of:
 *  - monolithic single file (small ONNX with weights inline);
 *  - graph + one external-data sidecar (`dataPath` matches the filename
 *    the .onnx internally references, e.g. "weights.pb" for nmkd SD1.5,
 *    "model.onnx_data" for diffusers/optimum SDXL exports);
 *  - graph + N external-data shards (used by LTX transformer dynamo
 *    export: each shard.bin is referenced by the graph at its own path
 *    via the location/offset rewrite from shard_external_data.py). */
export type OrtModelFile =
  | ModelFile
  | { graph: ModelFile; data: ModelFile; dataPath: string }
  | { graph: ModelFile; shards: Array<{ file: ModelFile; pathInGraph: string }> };

type SingleSidecar = { graph: ModelFile; data: ModelFile; dataPath: string };
type ShardSidecar = { graph: ModelFile; shards: Array<{ file: ModelFile; pathInGraph: string }> };

/** Type guard: any external-data layout (single sidecar or multi-shard). */
export function isExternalData(m: OrtModelFile): m is SingleSidecar | ShardSidecar {
  return (m as { graph?: unknown }).graph !== undefined;
}

function isShardSidecar(m: SingleSidecar | ShardSidecar): m is ShardSidecar {
  return (m as { shards?: unknown }).shards !== undefined;
}

/** Flatten an OrtModelFile to the list of underlying ModelFile entries.
 *  Used by the bundle file enumerator (modelSetFiles). */
export function ortModelFiles(m: OrtModelFile): ModelFile[] {
  if (!isExternalData(m)) return [m];
  if (isShardSidecar(m)) return [m.graph, ...m.shards.map((s) => s.file)];
  return [m.graph, m.data];
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

  // External-data layout. Stream sidecars as blob URLs (avoids copying
  // multi-GB files through the wasm allocator) and wire them via
  // sessionOptions.externalData.
  const { url: graphUrl, revoke: revokeGraph } = await cache.loadFileAsBlobUrl(model.graph);
  const sidecars: Array<{ pathInGraph: string; file: ModelFile }> = isShardSidecar(model)
    ? model.shards.map((s) => ({ pathInGraph: s.pathInGraph, file: s.file }))
    : [{ pathInGraph: model.dataPath, file: model.data }];
  const loaded = await Promise.all(
    sidecars.map(async (s) => {
      const { url, revoke } = await cache.loadFileAsBlobUrl(s.file);
      return { path: s.pathInGraph, data: url, revoke };
    }),
  );
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
    ).externalData = loaded.map((l) => ({ path: l.path, data: l.data }));
    return await ort.InferenceSession.create(graphUrl, sessionOptions);
  } finally {
    revokeGraph();
    for (const l of loaded) l.revoke();
  }
}
