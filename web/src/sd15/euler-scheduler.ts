// Euler Discrete scheduler for SDXL-Turbo (and other ADD/LCM distilled
// models). Same interface shape as DdimScheduler so the generate pipeline
// can swap between them based on the model bundle.
//
// SDXL-Turbo was distilled with Adversarial Diffusion Distillation (ADD)
// using the Euler Discrete scheduler with timestep_spacing="trailing" and
// prediction_type="epsilon". The MS WebNN reference implementation
// (webnn/sdxl-turbo/tools/create_scheduler_model.py) hardcodes sigma=14.6146
// for the 1-step case, which corresponds to sqrt((1-alpha_bar_999)/alpha_bar_999)
// from the standard SDXL beta schedule.
//
// For multi-step generation (2-4 steps), we compute the full sigma schedule
// from the training alpha_bar values with trailing timestep spacing.

const NUM_TRAIN_TIMESTEPS = 1000;
const BETA_START = 0.00085;
const BETA_END = 0.012;

function buildAlphasCumprod(): Float64Array {
  const sqrtStart = Math.sqrt(BETA_START);
  const sqrtEnd = Math.sqrt(BETA_END);
  const out = new Float64Array(NUM_TRAIN_TIMESTEPS);
  let acc = 1.0;
  for (let t = 0; t < NUM_TRAIN_TIMESTEPS; t++) {
    const lerp = sqrtStart + (sqrtEnd - sqrtStart) * (t / (NUM_TRAIN_TIMESTEPS - 1));
    const beta = lerp * lerp;
    acc *= 1.0 - beta;
    out[t] = acc;
  }
  return out;
}

export interface EulerSchedulerOptions {
  numInferenceSteps: number;
  guidanceScale: number;
}

export class EulerScheduler {
  readonly timesteps: Int32Array;
  readonly numInferenceSteps: number;
  readonly guidanceScale: number;
  /** Initial noise sigma. For Euler, noise is scaled by sigma[0] (e.g.
   *  ~14.6146 for SDXL-Turbo 1-step from t=999). */
  readonly initNoiseSigma: number;
  /** Sigma schedule: one sigma per timestep plus a trailing 0. Length =
   *  numInferenceSteps + 1. */
  private readonly sigmas: Float64Array;

  constructor(opts: EulerSchedulerOptions) {
    if (opts.numInferenceSteps < 1 || opts.numInferenceSteps > NUM_TRAIN_TIMESTEPS) {
      throw new Error(
        `numInferenceSteps must be 1..${NUM_TRAIN_TIMESTEPS}, got ${opts.numInferenceSteps}`,
      );
    }
    this.numInferenceSteps = opts.numInferenceSteps;
    this.guidanceScale = opts.guidanceScale;

    const alphasCumprod = buildAlphasCumprod();

    // Trailing timestep spacing (matches diffusers EulerDiscreteScheduler
    // with timestep_spacing="trailing"):
    //   timesteps = arange(N, 0, -1) * (1000 / N) - 1, clamped to [0, 999]
    // For N=1: [999]. For N=4: [999, 749, 499, 249].
    this.timesteps = new Int32Array(opts.numInferenceSteps);
    for (let i = 0; i < opts.numInferenceSteps; i++) {
      const t =
        Math.round((opts.numInferenceSteps - i) * (NUM_TRAIN_TIMESTEPS / opts.numInferenceSteps)) -
        1;
      this.timesteps[i] = Math.max(0, Math.min(NUM_TRAIN_TIMESTEPS - 1, t));
    }

    // Sigma schedule: sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t).
    // Plus a trailing 0 for the final step destination.
    this.sigmas = new Float64Array(opts.numInferenceSteps + 1);
    for (let i = 0; i < opts.numInferenceSteps; i++) {
      const t = this.timesteps[i];
      const ab = alphasCumprod[t];
      this.sigmas[i] = Math.sqrt((1 - ab) / ab);
    }
    this.sigmas[opts.numInferenceSteps] = 0; // final destination

    this.initNoiseSigma = this.sigmas[0];
  }

  applyCfg(uncondNoise: Float32Array, condNoise: Float32Array): Float32Array {
    if (uncondNoise.length !== condNoise.length) {
      throw new Error("CFG: uncond and cond noise lengths differ");
    }
    const out = new Float32Array(uncondNoise.length);
    const s = this.guidanceScale;
    for (let i = 0; i < out.length; i++) {
      out[i] = uncondNoise[i] + s * (condNoise[i] - uncondNoise[i]);
    }
    return out;
  }

  addNoise(cleanLatent: Float32Array, noise: Float32Array, t: number): Float32Array {
    if (cleanLatent.length !== noise.length) {
      throw new Error("addNoise: clean latent and noise lengths differ");
    }
    if (t < 0 || t >= NUM_TRAIN_TIMESTEPS) {
      throw new Error(`addNoise: t out of range: ${t}`);
    }
    // Same DDPM forward process as DdimScheduler.
    const alphasCumprod = buildAlphasCumprod();
    const alphaBar = alphasCumprod[t];
    const sqrtAlpha = Math.sqrt(alphaBar);
    const sqrtOneMinus = Math.sqrt(1.0 - alphaBar);
    const out = new Float32Array(cleanLatent.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = sqrtAlpha * cleanLatent[i] + sqrtOneMinus * noise[i];
    }
    return out;
  }

  getImg2ImgTimesteps(strength: number): { startIndex: number; tStart: number } {
    if (strength < 0 || strength > 1) {
      throw new Error(`strength must be in [0, 1], got ${strength}`);
    }
    const initTimestep = Math.min(
      Math.round(this.numInferenceSteps * strength),
      this.numInferenceSteps,
    );
    const startIndex = Math.max(this.numInferenceSteps - initTimestep, 0);
    const tStart = startIndex < this.timesteps.length ? this.timesteps[startIndex] : 0;
    return { startIndex, tStart };
  }

  /**
   * Scale the model input before passing to the UNet. Diffusers'
   * EulerDiscreteScheduler.scale_model_input divides by sqrt(sigma^2 + 1)
   * at each step. Without this the UNet gets out-of-distribution inputs
   * (values ~sigma times too large) and produces garbage.
   */
  scaleModelInput(latent: Float32Array, stepIndex: number): Float32Array {
    const sigma = this.sigmas[stepIndex];
    const scale = 1 / Math.sqrt(sigma * sigma + 1);
    const out = new Float32Array(latent.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = latent[i] * scale;
    }
    return out;
  }

  /**
   * One Euler step. Given the current noisy latent at sigma[stepIndex] and
   * the UNet's predicted noise (epsilon), advance to sigma[stepIndex+1].
   *
   * Euler Discrete (prediction_type="epsilon"):
   *   pred_original_sample = sample - sigma * model_output
   *   derivative = (sample - pred_original_sample) / sigma = model_output
   *   dt = sigma_next - sigma
   *   prev_sample = sample + derivative * dt
   *
   * When sigma_next = 0 (final step): prev_sample = pred_original_sample.
   */
  step(stepIndex: number, latent: Float32Array, noisePred: Float32Array): Float32Array {
    if (latent.length !== noisePred.length) {
      throw new Error("Euler step: latent and noisePred lengths differ");
    }
    const sigma = this.sigmas[stepIndex];
    const sigmaNext = this.sigmas[stepIndex + 1];
    const dt = sigmaNext - sigma; // negative (stepping toward 0)

    const out = new Float32Array(latent.length);
    for (let i = 0; i < out.length; i++) {
      // derivative = model_output (for epsilon prediction)
      out[i] = latent[i] + noisePred[i] * dt;
    }
    return out;
  }
}
