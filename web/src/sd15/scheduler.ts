// DDIM scheduler for SD1.5. Pure CPU math, no ORT, no GPU.
//
// The scheduler owns:
//   1. The forward noise schedule (alphas/betas) the UNet was trained against
//   2. The reverse-process timestep sequence we step through during sampling
//   3. The state update that turns (latent_t, predicted_noise_t) into
//      latent_{t-1}, optionally with classifier-free guidance applied
//
// We implement DDIM (deterministic, eta=0). Reasons:
// - It is the simplest stable sampler that gives reasonable quality at
//   low step counts (~20-50), which is the budget for in-browser inference.
// - It has no internal state beyond the timestep index, so it composes
//   trivially with the UNet inference loop.
// - It matches the diffusers reference exactly when constructed with
//   beta_schedule="scaled_linear" and the standard SD1.5 hyperparameters,
//   which is what every SD1.5 finetune is trained against. Using a
//   different sampler at inference time is fine for SD1.5 (samplers are
//   training-independent), but DDIM is the safest "definitely correct"
//   default. We can add Euler/DPM++ later as a quality option.
//
// Math reference: Song et al, "Denoising Diffusion Implicit Models" (2020),
// equation (12). diffusers/schedulers/scheduling_ddim.py is the cross-check.
//
// SD1.5 hyperparameters (hard-coded by the trained UNet, do not change):
//   num_train_timesteps = 1000
//   beta_start          = 0.00085
//   beta_end            = 0.012
//   beta_schedule       = "scaled_linear"  (betas = linspace(sqrt(start), sqrt(end), N) ** 2)
//   prediction_type     = "epsilon"        (UNet predicts noise, not x0)

const NUM_TRAIN_TIMESTEPS = 1000;
const BETA_START = 0.00085;
const BETA_END = 0.012;

/** Build the cumulative product of alphas (alpha_bar in DDPM notation).
 *  Length is NUM_TRAIN_TIMESTEPS. alphasCumprod[t] = prod_{s<=t}(1 - beta_s).
 *  Computed once per scheduler instance; cheap (~1000 multiplies). */
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

export interface DdimSchedulerOptions {
  /** Number of denoising steps the user wants to run. Typical: 20-50. */
  numInferenceSteps: number;
  /** Classifier-free guidance scale. 1.0 disables CFG; 7.5 is the SD1.5
   *  default. Higher values follow the prompt more strictly at the cost of
   *  diversity and image quality. */
  guidanceScale: number;
}

export class DdimScheduler {
  private readonly alphasCumprod: Float64Array;
  /** Inference timesteps in descending order (e.g. [981, 961, ..., 1]). */
  readonly timesteps: Int32Array;
  readonly numInferenceSteps: number;
  readonly guidanceScale: number;
  /** Initial noise scale. SD1.5 latents are scaled by sqrt(alpha_bar_T)
   *  before the first step; we expose this so the caller can multiply the
   *  random initial latent by it. */
  readonly initNoiseSigma: number;

  constructor(opts: DdimSchedulerOptions) {
    if (opts.numInferenceSteps < 1 || opts.numInferenceSteps > NUM_TRAIN_TIMESTEPS) {
      throw new Error(
        `numInferenceSteps must be 1..${NUM_TRAIN_TIMESTEPS}, got ${opts.numInferenceSteps}`,
      );
    }
    this.numInferenceSteps = opts.numInferenceSteps;
    this.guidanceScale = opts.guidanceScale;
    this.alphasCumprod = buildAlphasCumprod();

    // Linspace timestep selection (matches diffusers DDIM with
    // timestep_spacing="linspace"). Produces N evenly-spaced integer
    // timesteps in [0, NUM_TRAIN_TIMESTEPS-1] in descending order.
    this.timesteps = new Int32Array(opts.numInferenceSteps);
    const stepRatio = (NUM_TRAIN_TIMESTEPS - 1) / (opts.numInferenceSteps - 1 || 1);
    for (let i = 0; i < opts.numInferenceSteps; i++) {
      const t = Math.round((opts.numInferenceSteps - 1 - i) * stepRatio);
      this.timesteps[i] = t;
    }

    // For DDIM with eta=0 starting from pure Gaussian noise, the standard
    // diffusers convention is initNoiseSigma = 1.0 (the random latent is
    // not pre-scaled). We expose it as a constant so the caller's code
    // doesn't have to know which sampler convention applies.
    this.initNoiseSigma = 1.0;
  }

  /** Returns the alpha_bar value at integer timestep t. Caller never needs
   *  this directly; exposed for tests and debugging. */
  alphaBar(t: number): number {
    return this.alphasCumprod[t];
  }

  /**
   * Apply classifier-free guidance to a pair of noise predictions.
   * The UNet is run twice per step (once with the unconditional embedding,
   * once with the conditional embedding) and the two outputs are combined:
   *
   *     guided = uncond + scale * (cond - uncond)
   *
   * Output is written into a fresh Float32Array.
   */
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

  /**
   * Forward-process noising. Given a clean latent (e.g. the output of the
   * VAE encoder, in pipeline / "model" space) and a Gaussian noise tensor
   * of the same shape, return the noisy latent at integer timestep t:
   *
   *     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
   *
   * This is the standard DDPM forward process and is what diffusers uses
   * for img2img init: encode the reference image, sample fresh noise, and
   * call addNoise() with the partial-start timestep returned by
   * getImg2ImgTimesteps(strength). The same primitive is what AnimateDiff
   * I2V conditioning needs (encode reference frame -> partial-noise to a
   * mid-schedule timestep -> hand to the temporal UNet).
   */
  addNoise(cleanLatent: Float32Array, noise: Float32Array, t: number): Float32Array {
    if (cleanLatent.length !== noise.length) {
      throw new Error("addNoise: clean latent and noise lengths differ");
    }
    if (t < 0 || t >= NUM_TRAIN_TIMESTEPS) {
      throw new Error(`addNoise: t out of range: ${t}`);
    }
    const alphaBar = this.alphasCumprod[t];
    const sqrtAlpha = Math.sqrt(alphaBar);
    const sqrtOneMinus = Math.sqrt(1.0 - alphaBar);
    const out = new Float32Array(cleanLatent.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = sqrtAlpha * cleanLatent[i] + sqrtOneMinus * noise[i];
    }
    return out;
  }

  /**
   * Diffusers-style img2img schedule trimming. Given a strength in [0, 1],
   * returns the index into this.timesteps where the denoise loop should
   * start, AND the integer training timestep at which the input latent
   * should be partial-noised via addNoise().
   *
   *   strength = 1.0 -> start from pure noise (index 0, full schedule),
   *                     equivalent to txt2img
   *   strength = 0.0 -> start from the clean image (skip the loop entirely)
   *   strength = 0.5 -> noise to the midpoint and run the back half of the
   *                     schedule
   *
   * The returned `tStart` is the timestep at this.timesteps[startIndex];
   * caller passes it to addNoise() to produce the initial latent.
   */
  getImg2ImgTimesteps(strength: number): { startIndex: number; tStart: number } {
    if (strength < 0 || strength > 1) {
      throw new Error(`strength must be in [0, 1], got ${strength}`);
    }
    // Same convention as diffusers StableDiffusionImg2ImgPipeline:
    //   init_timestep = min(round(num_inference_steps * strength), num_inference_steps)
    //   t_start       = max(num_inference_steps - init_timestep, 0)
    const initTimestep = Math.min(
      Math.round(this.numInferenceSteps * strength),
      this.numInferenceSteps,
    );
    const startIndex = Math.max(this.numInferenceSteps - initTimestep, 0);
    // Edge case: strength == 0 leaves no steps to run; tStart is the
    // smallest valid timestep so addNoise() is effectively a no-op even if
    // the caller invokes it (it will not, because startIndex == numInferenceSteps).
    const tStart = startIndex < this.timesteps.length ? this.timesteps[startIndex] : 0;
    return { startIndex, tStart };
  }

  /**
   * One DDIM step. Given the current noisy latent at timestep t and the
   * UNet's predicted noise (already CFG-combined), return the latent at
   * the next timestep.
   *
   * @param stepIndex Index into this.timesteps. 0 is the first (highest-t)
   *                  step, numInferenceSteps-1 is the last.
   * @param latent    Current latent x_t, length = 4 * H/8 * W/8.
   * @param noisePred Noise prediction eps_theta(x_t, t), same length.
   * @returns         Next latent x_{t-1}, fresh Float32Array.
   */
  step(stepIndex: number, latent: Float32Array, noisePred: Float32Array): Float32Array {
    if (latent.length !== noisePred.length) {
      throw new Error("DDIM step: latent and noisePred lengths differ");
    }
    const t = this.timesteps[stepIndex];
    const tPrev = stepIndex + 1 < this.timesteps.length ? this.timesteps[stepIndex + 1] : -1;

    const alphaBarT = this.alphasCumprod[t];
    // For the final step, x_{t-1} corresponds to t = -1 / "x_0"; the
    // diffusers DDIM uses alpha_bar_prev = 1 in this case.
    const alphaBarPrev = tPrev >= 0 ? this.alphasCumprod[tPrev] : 1.0;

    const sqrtAlphaBarT = Math.sqrt(alphaBarT);
    const sqrtOneMinusAlphaBarT = Math.sqrt(1.0 - alphaBarT);
    const sqrtAlphaBarPrev = Math.sqrt(alphaBarPrev);
    const sqrtOneMinusAlphaBarPrev = Math.sqrt(1.0 - alphaBarPrev);

    // DDIM update with eta = 0 (deterministic):
    //   pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    //   dir_xt  = sqrt(1 - alpha_bar_prev) * eps
    //   x_prev  = sqrt(alpha_bar_prev) * pred_x0 + dir_xt
    const out = new Float32Array(latent.length);
    for (let i = 0; i < out.length; i++) {
      const eps = noisePred[i];
      const predX0 = (latent[i] - sqrtOneMinusAlphaBarT * eps) / sqrtAlphaBarT;
      const dirXt = sqrtOneMinusAlphaBarPrev * eps;
      out[i] = sqrtAlphaBarPrev * predX0 + dirXt;
    }
    return out;
  }
}
