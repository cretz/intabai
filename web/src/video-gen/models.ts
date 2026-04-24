// video-gen model registry. Backend-agnostic: each entry binds a user-
// facing model identity to a backend discriminator and a flat list of
// ModelFiles the cache needs to download. Mirrors the image-gen/sd15
// pattern so the model-manager UI can iterate without knowing how any
// specific backend works internally.
//
// Current lineup: one model (FastWan 2.2 TI2V 5B). Add further entries
// here as we port more backends; each new backend gets its own
// `src/<family>/` directory that mirrors the fastwan/ layout and exposes
// a `generate()` entry point plus a `xxxAllFiles()` manifest.

import type { ModelFile } from "../shared/model-cache";
import {
  fastwanAllFiles,
  type FastwanTransformerPrecision,
} from "../fastwan/models";
import type { FastwanResolution } from "../fastwan/transformer";

export type VideoBackend = "fastwan";

export interface VideoModelEntry {
  id: string;
  name: string;
  description: string;
  /** Flat list of cache files for this bundle. */
  files: ModelFile[];
  backend: VideoBackend;
  /** Backend-specific precision tag. Passed through to the generator so the
   *  same backend can expose multiple quant tiers without code branches in
   *  the caller. */
  transformerPrecision?: FastwanTransformerPrecision;
  /** Output resolution. Drives the transformer/VAE asset selection and the
   *  pipeline shape. */
  resolution: FastwanResolution;
  /** Optional public model page. Undefined if the repo is not yet
   *  published. */
  hfRepoUrl?: string;
  /** Fixed output resolution label for display. */
  resolutionLabel: string;
  /** Fixed output clip length label for display. */
  clipLabel: string;
}

export const VIDEO_GEN_MODELS: VideoModelEntry[] = [
  {
    id: "fastwan_22_ti2v_5b_480",
    name: "FastWan 2.2 TI2V 5B (480×480)",
    description:
      "DMD-distilled, 4-step flow-matching text-to-video. " +
      "480×480 at 16 fps, 5 second clips. q4f16 transformer + UMT5 text " +
      "encoder + LightTAE VAE. ~6.5 GB total download. Desktop and mobile.",
    files: fastwanAllFiles("q4f16", 480),
    backend: "fastwan",
    transformerPrecision: "q4f16",
    resolution: 480,
    resolutionLabel: "480×480",
    clipLabel: "5 s @ 16 fps",
  },
  {
    id: "fastwan_22_ti2v_5b_480_fp16",
    name: "FastWan 2.2 TI2V 5B (480×480, fp16, desktop)",
    description:
      "480×480 with fp16 transformer blocks (no int4 dequant on every " +
      "matmul). ~25% faster per step at the cost of ~9.4 GB more download. " +
      "Desktop GPUs only - per-block memory budget too large for mobile.",
    files: fastwanAllFiles("fp16", 480),
    backend: "fastwan",
    transformerPrecision: "fp16",
    resolution: 480,
    resolutionLabel: "480×480",
    clipLabel: "5 s @ 16 fps",
  },
  {
    id: "fastwan_22_ti2v_5b_576",
    name: "FastWan 2.2 TI2V 5B (576×576)",
    description:
      "576×576 variant. ~2x slower per step than 480×480 (attention is " +
      "O(N²) and seq grows from 4725 to 6804). Same q4f16 quant + UMT5 + " +
      "LightTAE. Desktop and mobile.",
    files: fastwanAllFiles("q4f16", 576),
    backend: "fastwan",
    transformerPrecision: "q4f16",
    resolution: 576,
    resolutionLabel: "576×576",
    clipLabel: "5 s @ 16 fps",
  },
  {
    id: "fastwan_22_ti2v_5b_576_fp16",
    name: "FastWan 2.2 TI2V 5B (576×576, fp16, desktop)",
    description:
      "576×576 with fp16 transformer blocks. Desktop GPUs only.",
    files: fastwanAllFiles("fp16", 576),
    backend: "fastwan",
    transformerPrecision: "fp16",
    resolution: 576,
    resolutionLabel: "576×576",
    clipLabel: "5 s @ 16 fps",
  },
];

export function getModel(id: string): VideoModelEntry | undefined {
  return VIDEO_GEN_MODELS.find((m) => m.id === id);
}
