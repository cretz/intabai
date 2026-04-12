// Flow matching scheduler for Z-Image-Turbo. Based on the MS WebNN
// reference implementation (webnn-developer-preview/demos/z-image-turbo/
// scheduler.js). Uses a shifted sigma schedule and Euler steps.
//
// The MS reference bakes the Euler step into a tiny ONNX model
// (scheduler_step_model_f16.onnx) that takes step_info = [i, numSteps].
// We implement it in TypeScript to avoid an extra ORT session and to
// keep the math transparent.

const NUM_TRAIN_TIMESTEPS = 1000;

/** Flow matching shift factor. Higher values push sigmas toward 1 (more
 *  denoising per step). Z-Image-Turbo uses 3.0. */
const SHIFT = 3.0;

/** Compute the sigma schedule and transformed timesteps for a given
 *  number of inference steps. Returns:
 *  - timesteps: the values passed to the transformer's timestep input.
 *    Transformed as (1000 - schedulerTimestep) / 1000, last clamped to 1.0.
 *  - sigmas: length numSteps + 1, with a trailing 0 for the final step.
 */
export function buildSchedule(numSteps: number): {
  timesteps: Float32Array;
  sigmas: Float64Array;
} {
  // Compute shifted sigmas over the full training schedule, then select
  // evenly-spaced inference points.
  //
  // sigma(t) = shift * (t/1000) / (1 + (shift - 1) * (t/1000))
  //
  // Inference timesteps are linspaced from sigma_max to sigma_min in
  // sigma space, then converted back to timestep space.

  // sigma_max and sigma_min from the full schedule
  const sMax = (SHIFT * 1.0) / (1 + (SHIFT - 1) * 1.0); // t=1000
  const sMin = (SHIFT * (1 / NUM_TRAIN_TIMESTEPS)) / (1 + (SHIFT - 1) * (1 / NUM_TRAIN_TIMESTEPS)); // t=1

  const tStart = sMax * NUM_TRAIN_TIMESTEPS;
  const tEnd = sMin * NUM_TRAIN_TIMESTEPS;

  // Linspace inference timesteps in t-space
  const rawTimesteps = new Float64Array(numSteps);
  if (numSteps === 1) {
    rawTimesteps[0] = tStart;
  } else {
    const step = (tEnd - tStart) / (numSteps - 1);
    for (let i = 0; i < numSteps; i++) {
      rawTimesteps[i] = tStart + step * i;
    }
  }

  // Convert to shifted sigmas + trailing 0
  const sigmas = new Float64Array(numSteps + 1);
  for (let i = 0; i < numSteps; i++) {
    const s = rawTimesteps[i] / NUM_TRAIN_TIMESTEPS;
    sigmas[i] = (SHIFT * s) / (1 + (SHIFT - 1) * s);
  }
  sigmas[numSteps] = 0; // final destination

  // Transformer timestep input: (1000 - schedulerTimestep) / 1000
  // where schedulerTimestep = sigma * 1000
  const timesteps = new Float32Array(numSteps);
  for (let i = 0; i < numSteps; i++) {
    timesteps[i] = (NUM_TRAIN_TIMESTEPS - sigmas[i] * NUM_TRAIN_TIMESTEPS) / NUM_TRAIN_TIMESTEPS;
  }
  // Clamp last to 1.0 (matches MS reference)
  timesteps[numSteps - 1] = 1.0;

  return { timesteps, sigmas };
}

/** One Euler step in flow matching sigma space.
 *
 *  prev = latent + modelOutput * (sigma_next - sigma_curr)
 *
 *  When sigma_next = 0 (final step): prev = latent - sigma * modelOutput
 *  which is the fully denoised sample.
 */
export function eulerStep(
  latent: Float32Array,
  modelOutput: Float32Array,
  sigmaCurr: number,
  sigmaNext: number,
): Float32Array {
  const dt = sigmaNext - sigmaCurr;
  const out = new Float32Array(latent.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = latent[i] + modelOutput[i] * dt;
  }
  return out;
}
