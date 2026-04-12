// Model registry for the video-gen tool. Each entry binds a user-facing
// model identity to a backend implementation and a list of files to
// download. URLs are pinned to specific HuggingFace revisions so we get
// reproducible behavior across users.
//
// File URLs marked TODO are placeholders until the AnimateDiff spike
// validates the actual ONNX exports we need.

import type { VideoGenCapabilities } from "./pipeline";

export interface ModelFile {
  id: string;
  name: string;
  url: string;
  sizeBytes: number;
  sha256?: string;
}

export interface ModelEntry {
  id: string;
  /** Human-facing display name. */
  name: string;
  /** Which backend implementation to instantiate. */
  backend: "animate-diff" | "wan" | "ltx";
  /** Short description for the UI. */
  description: string;
  /** License string to surface to users. */
  license: string;
  /** Files to download into OPFS cache. */
  files: ModelFile[];
  capabilities: VideoGenCapabilities;
}

export const MODELS: Record<string, ModelEntry> = {
  "animate-diff-lightning-4step": {
    id: "animate-diff-lightning-4step",
    name: "AnimateDiff Lightning (4-step, SD1.5)",
    backend: "animate-diff",
    description:
      "Temporal motion module on top of Stable Diffusion 1.5. " +
      "Generates ~2 second clips at 8 fps. Fastest browser path. " +
      "Best for short clips on modest hardware.",
    license: "CreativeML OpenRAIL-M",
    files: [
      // TODO: real URLs after spike. Need ONNX-exported variants of:
      //   - SD1.5 UNet (fp16)
      //   - AnimateDiff Lightning 4-step motion module
      //   - SD1.5 VAE decoder
      //   - CLIP ViT-L/14 text encoder
      //   - SparseCtrl conditioning model (for reference frames)
      // Decision pending: pre-export to ONNX ourselves vs. find a community
      // export. Likely the former since AnimateDiff motion module ONNX
      // exports are not common.
    ],
    capabilities: {
      minFrames: 8,
      maxFrames: 32,
      defaultFrames: 16,
      nativeFps: 8,
      stepOptions: [1, 2, 4, 8],
      defaultSteps: 4,
      supportsT2V: true,
      supportsI2V: true, // via SparseCtrl
      supportsArbitraryReferenceFrames: true, // via SparseCtrl
      supportsNegativePrompt: true,
      resolutions: ["512x512", "512x768", "768x512"],
      defaultResolution: "512x512",
      approxGpuMemoryMB: 2800,
    },
  },
};

export function listModels(): ModelEntry[] {
  return Object.values(MODELS);
}

export function getModel(id: string): ModelEntry | undefined {
  return MODELS[id];
}
