// Backend-agnostic interface for video generation models. Each model
// (AnimateDiff, Wan, LTX, ...) implements VideoGenBackend. The UI talks to
// this interface only, so adding a new model is a new file in `backends/`
// plus an entry in `models.ts`.

export interface VideoGenCapabilities {
  /** Minimum number of frames the model accepts in a single call. */
  minFrames: number;
  /** Maximum number of frames the model can generate in a single call. */
  maxFrames: number;
  /** Default frame count to show in the UI. */
  defaultFrames: number;
  /** Native frame rate the model was trained at. */
  nativeFps: number;
  /** Allowed/recommended denoising step counts. UI shows these as options. */
  stepOptions: number[];
  /** Default step count. */
  defaultSteps: number;
  /** Whether the backend can run pure text-to-video (no reference frame). */
  supportsT2V: boolean;
  /** Whether the backend accepts a reference frame at frame 0 (classic I2V). */
  supportsI2V: boolean;
  /**
   * Whether the backend supports reference frames at arbitrary positions
   * in the sequence (e.g. SparseCtrl-style keyframe conditioning).
   */
  supportsArbitraryReferenceFrames: boolean;
  /** Whether the backend uses negative prompts (most diffusion models do). */
  supportsNegativePrompt: boolean;
  /** Output resolution(s) the model is comfortable with, "WxH". */
  resolutions: string[];
  /** Default resolution. */
  defaultResolution: string;
  /** Approximate GPU memory required in megabytes. */
  approxGpuMemoryMB: number;
}

export interface ReferenceFrame {
  /** Index in the output sequence (0-based) where this reference applies. */
  frameIndex: number;
  image: ImageBitmap;
}

export interface GenerateRequest {
  prompt: string;
  negativePrompt?: string;
  numFrames: number;
  steps: number;
  /** If undefined, the backend should pick a random seed. */
  seed?: number;
  /** Empty = pure T2V. */
  referenceFrames: ReferenceFrame[];
}

export interface GenerateResult {
  frames: ImageBitmap[];
  fps: number;
  seed: number;
}

export type GenerateProgress = (info: {
  step: number;
  totalSteps: number;
  message: string;
}) => void;

export type FrameCallback = (frameIndex: number, frame: ImageBitmap) => void;

export interface VideoGenBackend {
  readonly id: string;
  readonly capabilities: VideoGenCapabilities;

  /** Load all model weights into memory. Idempotent. */
  load(onProgress: (loaded: number, total: number, msg: string) => void): Promise<void>;

  /** Free model resources. */
  unload(): Promise<void>;

  /**
   * Run a generation. Calls `onFrame` as each frame becomes available
   * (for backends that can stream; others fire all frames at the end).
   * Honors `signal` for cancellation.
   */
  generate(
    req: GenerateRequest,
    onProgress: GenerateProgress,
    onFrame: FrameCallback,
    signal: AbortSignal,
  ): Promise<GenerateResult>;
}
