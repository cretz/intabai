// Shared types for the per-family generate functions in
// web/src/sd15/generate.ts and web/src/sdxl/generate.ts. The image-gen UI
// (web/src/image-gen/main.ts) builds one of these per "generate" click and
// dispatches by ModelSet.family.
//
// Decoupling generation from the UI lets us add new model families (Janus,
// SDXL-Turbo, Z-Image, etc.) by writing a new generate function with the
// same shape, without growing main.ts into a giant per-family if-tree.

import type { ModelCache } from "../shared/model-cache";
import type { ModelSet } from "../sd15/models";

/** Reference-image input. The pipeline rasterizes the HTMLImageElement to
 *  the requested generation width/height before encoding it through the
 *  VAE encoder. */
export interface RefImageInput {
  image: HTMLImageElement;
  /** Strength in [0, 1]. Lower keeps the reference, higher drifts away. */
  strength: number;
}

/** All inputs the user provides for one generation. */
export interface GenerateInput {
  set: ModelSet;
  cache: ModelCache;
  prompt: string;
  width: number;
  height: number;
  /** Effective denoise step count the UI requested. The pipeline may
   *  inflate the scheduler's internal step count for img2img so the
   *  effective count after partial-start trimming still equals this. */
  steps: number;
  cfg: number;
  seed: number;
  refImage: RefImageInput | null;
  /** Force tiled VAE decode even when the latent fits in a single tile. */
  tileVae: boolean;
}

/** Callbacks the UI hooks into the pipeline. All are optional from the
 *  pipeline's POV; we provide every callback by default in main.ts but
 *  individual ones can no-op for tests. */
export interface GenerateCallbacks {
  /** Verbose log message (only emitted when the user enabled debug log). */
  log: (msg: string) => void;
  /** Coarse-grained status update; the UI puts this in its status line. */
  status: (msg: string) => void;
  /** Per-step stats line for the UI's stats line (avg time, ETA, etc.). */
  stats: (msg: string) => void;
  /** A discrete progress unit completed. The UI advances its progress bar
   *  by 1 / totalUnits. */
  advance: () => void;
  /** Throw if the user clicked cancel. The pipeline calls this between
   *  long-running phases so cancellation actually takes effect. */
  checkAborted: () => void;
}

/** Tells the UI how many discrete progress units the run will produce, so
 *  the bar can be sized before the first advance() call. */
export interface PipelineEstimate {
  totalUnits: number;
}

/** A pipeline implementation. One per model family. */
export interface GenerateFn {
  estimate(input: GenerateInput): PipelineEstimate;
  run(input: GenerateInput, cb: GenerateCallbacks): Promise<ImageData>;
}
