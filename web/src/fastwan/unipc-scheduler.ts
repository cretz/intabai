/** UniPCMultistepScheduler port for FastWan 2.2 TI2V.
 *
 *  Matches diffusers' UniPCMultistepScheduler with the exact config shipped
 *  in `scheduler/scheduler_config.json`:
 *    use_flow_sigmas=true, predict_x0=true, prediction_type="flow_prediction",
 *    solver_order=2, solver_type="bh2", final_sigmas_type="zero",
 *    lower_order_final=true, disable_corrector=[].
 *
 *  Replaces the previous DMD direct-x0 + re-noise loop. That loop was our
 *  own interpretation of FastVideo's config; the reference HF space
 *  (KingNish/wan2-2-fast) uses stock UniPC with flow_shift=8.0 and 4 steps,
 *  which is what we match here.
 *
 *  Reference: huggingface/diffusers src/diffusers/schedulers/scheduling_unipc_multistep.py
 *  License: Apache 2.0 (upstream).
 *
 *  Everything runs in fp32 on Float32Arrays. Coefficients are plain scalars.
 *  Tensor shape is opaque to the scheduler - it just needs Float32Arrays of
 *  matching length.
 */

function expm1(x: number): number {
  return Math.expm1(x);
}

/** tensor_out = a * scale_a + b * scale_b (new array). */
function addScaled(
  a: Float32Array,
  scaleA: number,
  b: Float32Array,
  scaleB: number,
): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * scaleA + b[i] * scaleB;
  return out;
}

/** In-place: a += b * scale. */
function addScaledInPlace(a: Float32Array, b: Float32Array, scale: number): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i] * scale;
}

/** (a - b) / denom, returned as new array. */
function subDiv(a: Float32Array, b: Float32Array, denom: number): Float32Array {
  const out = new Float32Array(a.length);
  const inv = 1 / denom;
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - b[i]) * inv;
  return out;
}

export interface UniPCConfig {
  /** Number of denoising steps (e.g. 4 to match HF space). */
  numInferenceSteps: number;
  /** Flow-matching sigma shift. HF space uses 8.0; config file default 5.0. */
  flowShift: number;
  /** Solver order. Default 2. */
  solverOrder?: number;
  /** Whether to drop to lower order on the final step. Default true. */
  lowerOrderFinal?: boolean;
  /** num_train_timesteps. Default 1000. */
  numTrainTimesteps?: number;
}

export class UniPCFlowScheduler {
  readonly sigmas: Float32Array;
  readonly timesteps: Float32Array;
  readonly numInferenceSteps: number;
  readonly solverOrder: number;
  readonly lowerOrderFinal: boolean;

  /** History of converted (x0-predicted) model outputs, length solverOrder.
   *  Most recent at index solverOrder-1. */
  private modelOutputs: (Float32Array | null)[];
  /** Sample before the most recent step (for corrector). */
  private lastSample: Float32Array | null = null;
  /** How many outputs have been stored so far (caps at solverOrder). */
  private lowerOrderNums = 0;
  /** The order selected for the *previous* step's predictor, reused by
   *  the *current* step's corrector. Diffusers calls this `self.this_order`. */
  private lastOrder = 1;
  private stepIndex = 0;

  constructor(config: UniPCConfig) {
    const numSteps = config.numInferenceSteps;
    const shift = config.flowShift;
    const numTrain = config.numTrainTimesteps ?? 1000;
    this.numInferenceSteps = numSteps;
    this.solverOrder = config.solverOrder ?? 2;
    this.lowerOrderFinal = config.lowerOrderFinal ?? true;

    // flow sigmas: np.linspace(1, 1/num_train_timesteps, numSteps+1)[:-1]
    // That is numSteps values from 1.0 down to just above 1/numTrain.
    const rawSigmas = new Float32Array(numSteps);
    const endFrac = 1 / numTrain;
    for (let i = 0; i < numSteps; i++) {
      const t = i / numSteps;
      rawSigmas[i] = 1 + t * (endFrac - 1);
    }
    // apply shift: sigma = shift*s / (1 + (shift-1)*s)
    // final_sigmas_type="zero" appends 0.0, so array length is numSteps+1.
    const sigmas = new Float32Array(numSteps + 1);
    for (let i = 0; i < numSteps; i++) {
      const s = rawSigmas[i];
      sigmas[i] = (shift * s) / (1 + (shift - 1) * s);
    }
    sigmas[numSteps] = 0;
    this.sigmas = sigmas;

    // timesteps (for logging / onDebug): sigma * num_train_timesteps, skip last.
    const ts = new Float32Array(numSteps);
    for (let i = 0; i < numSteps; i++) ts[i] = sigmas[i] * numTrain;
    this.timesteps = ts;

    this.modelOutputs = new Array(this.solverOrder).fill(null);
  }

  /** One UniPC step. Takes the model's raw noise_pred + current sample,
   *  returns the next sample. `modelOutput` and `sample` must be fp32 of
   *  identical length. Caller owns lifecycle - this returns a new buffer. */
  step(modelOutput: Float32Array, sample: Float32Array): Float32Array {
    if (this.stepIndex >= this.numInferenceSteps) {
      throw new Error(`UniPC: step called past end (${this.stepIndex} >= ${this.numInferenceSteps})`);
    }

    const sigma_cur = this.sigmas[this.stepIndex];
    // flow_prediction + predict_x0: x0 = sample - sigma * noise_pred
    const m_converted = addScaled(sample, 1, modelOutput, -sigma_cur);

    // Corrector (if we have history). Uses this.lastOrder (set by previous
    // step's predictor), and must run BEFORE we shift history.
    let correctedSample = sample;
    const useCorrector =
      this.stepIndex > 0 && this.lastSample !== null;
    if (useCorrector) {
      correctedSample = this.correctorUpdate(
        m_converted,
        this.lastSample!,
        this.lastOrder,
      );
    }

    // Shift history: [a, b] -> [b, m_converted]
    for (let i = 0; i < this.solverOrder - 1; i++) {
      this.modelOutputs[i] = this.modelOutputs[i + 1];
    }
    this.modelOutputs[this.solverOrder - 1] = m_converted;

    // Determine predictor order for this step.
    let thisOrder: number;
    if (this.lowerOrderFinal) {
      thisOrder = Math.min(this.solverOrder, this.numInferenceSteps - this.stepIndex);
    } else {
      thisOrder = this.solverOrder;
    }
    thisOrder = Math.min(thisOrder, this.lowerOrderNums + 1);
    this.lastOrder = thisOrder;

    const prevSample = this.predictorUpdate(m_converted, correctedSample, thisOrder);

    if (this.lowerOrderNums < this.solverOrder) this.lowerOrderNums++;
    this.lastSample = sample;
    this.stepIndex++;
    return prevSample;
  }

  /** multistep_uni_p_bh_update for predict_x0=true, flow_sigmas=true, bh2.
   *  Moves `sample` at sigmas[stepIndex] to the next sigma sigmas[stepIndex+1]. */
  private predictorUpdate(
    m0: Float32Array,
    sample: Float32Array,
    order: number,
  ): Float32Array {
    const sigma_t = this.sigmas[this.stepIndex + 1];
    const sigma_s0 = this.sigmas[this.stepIndex];
    const alpha_t = 1 - sigma_t;
    const alpha_s0 = 1 - sigma_s0;

    // Log-SNR lambdas. At sigma_t = 0 (final step), alpha_t=1, log(sigma_t) = -inf,
    // lambda_t = +inf, h = +inf, hh = -inf, expm1(-inf) = -1. For predict_x0:
    //   x_t_ = (sigma_t/sigma_s0) * x - alpha_t * h_phi_1 * m0
    //        = 0 - 1 * (-1) * m0 = m0  (i.e. the predicted x0 is returned)
    // which is the correct flow-match endpoint. We rely on JS arithmetic
    // producing these limits naturally.
    const lambda_t = Math.log(alpha_t) - Math.log(sigma_t);
    const lambda_s0 = Math.log(alpha_s0) - Math.log(sigma_s0);
    const h = lambda_t - lambda_s0;
    const hh = -h; // predict_x0 branch
    const h_phi_1 = expm1(hh);
    const B_h = expm1(hh); // bh2

    // Base term: x_t_ = (sigma_t/sigma_s0) * sample - alpha_t * h_phi_1 * m0
    const xT = addScaled(sample, sigma_t / sigma_s0, m0, -alpha_t * h_phi_1);

    if (order >= 2) {
      // rks[0] = (lambda_s1 - lambda_s0) / h, D1s[0] = (m1 - m0) / rks[0]
      // For order=2, rhos_p = [0.5] (hardcoded in diffusers).
      // pred_res = 0.5 * D1s[0]
      // x_t = x_t_ - alpha_t * B_h * pred_res
      const sigma_s1 = this.sigmas[this.stepIndex - 1];
      const alpha_s1 = 1 - sigma_s1;
      const lambda_s1 = Math.log(alpha_s1) - Math.log(sigma_s1);
      const rk0 = (lambda_s1 - lambda_s0) / h;
      const m1 = this.modelOutputs[this.solverOrder - 2]!;
      const D10 = subDiv(m1, m0, rk0);
      addScaledInPlace(xT, D10, -alpha_t * B_h * 0.5);
    }

    return xT;
  }

  /** multistep_uni_c_bh_update for predict_x0=true, flow_sigmas=true, bh2.
   *  Refines `thisSample` (the current step's predicted sample) using the
   *  previous step's sample and current model output. */
  private correctorUpdate(
    thisModelOutput: Float32Array,
    lastSample: Float32Array,
    order: number,
  ): Float32Array {
    // At entry: history still holds previous step's m0 at modelOutputs[-1];
    // thisModelOutput is the just-computed m for stepIndex.
    const sigma_t = this.sigmas[this.stepIndex];
    const sigma_s0 = this.sigmas[this.stepIndex - 1];
    const alpha_t = 1 - sigma_t;
    const alpha_s0 = 1 - sigma_s0;
    const lambda_t = Math.log(alpha_t) - Math.log(sigma_t);
    const lambda_s0 = Math.log(alpha_s0) - Math.log(sigma_s0);
    const h = lambda_t - lambda_s0;
    const hh = -h; // predict_x0
    const h_phi_1 = expm1(hh);
    const B_h = expm1(hh);

    const m0 = this.modelOutputs[this.solverOrder - 1]!;
    const D1_t = addScaled(thisModelOutput, 1, m0, -1);

    // Base: x_t_ = (sigma_t/sigma_s0)*last - alpha_t*h_phi_1*m0
    const xT = addScaled(lastSample, sigma_t / sigma_s0, m0, -alpha_t * h_phi_1);

    if (order >= 2) {
      // Add older D1s term with rhos_c[0] coefficient. For order=2, rhos_c = [0.5, 0.5].
      // D1s[0] = (m_prev_prev - m0) / rk0 where rk0 uses sigma at step_index-2.
      const sigma_s1 = this.sigmas[this.stepIndex - 2];
      const alpha_s1 = 1 - sigma_s1;
      const lambda_s1 = Math.log(alpha_s1) - Math.log(sigma_s1);
      const rk0 = (lambda_s1 - lambda_s0) / h;
      const m1 = this.modelOutputs[this.solverOrder - 2]!;
      const D10 = subDiv(m1, m0, rk0);
      addScaledInPlace(xT, D10, -alpha_t * B_h * 0.5);
      // + rhos_c[-1] * D1_t, rhos_c[-1] = 0.5 for order 2.
      addScaledInPlace(xT, D1_t, -alpha_t * B_h * 0.5);
    } else {
      // Order 1: rhos_c = [0.5], only the D1_t term.
      addScaledInPlace(xT, D1_t, -alpha_t * B_h * 0.5);
    }

    return xT;
  }
}
