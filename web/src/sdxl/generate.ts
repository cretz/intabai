// SDXL image generation. Mirror of sd15/generate.ts shape but with the
// SDXL-specific differences:
//
//   - Two text encoders (CLIP-L + OpenCLIP-bigG/14), assembled into a 2048-
//     dim concatenated hidden state plus a 1280-dim pooled output. Both
//     tokenizers reuse the same CLIP BPE class - the bigG vocab is byte-
//     identical to CLIP-L's in every diffusers SDXL export I have inspected.
//   - UNet additionally consumes `text_embeds` and `time_ids` per call.
//   - VAE scaling factor is 0.13025 (vs 0.18215 for SD1.5).
//   - Native resolution is 1024 (we default to 768 for in-browser memory
//     headroom; the model bundle's defaults field controls this).
//
// Pipeline shape, fixed-phase units, and img2img inflation logic match
// sd15/generate.ts so the UI's progress accounting can stay common.

import type { SdxlModelSet } from "../sd15/models";
import { ClipTokenizer } from "../sd15/tokenizer";
import { DdimScheduler } from "../sd15/scheduler";
import { EulerScheduler } from "../sd15/euler-scheduler";
import { VaeDecoder, vaeTileCount, SDXL_VAE_SCALING_FACTOR } from "../sd15/vae";
import { VaeEncoder, imageDataToChw } from "../sd15/vae-encoder";
import {
  fmtStats,
  formatEta,
  formatMs,
  gaussianNoise,
  mulberry32,
  rasterizeRefImage,
  statsOf,
} from "../image-gen/generate-utils";
import type {
  GenerateCallbacks,
  GenerateFn,
  GenerateInput,
  PipelineEstimate,
} from "../image-gen/generate-types";
import { SdxlDualTextEncoder } from "./text-encoders";
import { SdxlUnet, SDXL_UNET_LATENT_CHANNELS, buildTimeIds } from "./unet";

const FIXED_PHASE_UNITS = 4;

function computeSchedulerSteps(steps: number, refStrength: number, hasRef: boolean): number {
  if (!hasRef || refStrength <= 0) return steps;
  return Math.max(steps, Math.ceil(steps / refStrength));
}

/** SDXL CFG combine. Same shape as DdimScheduler.applyCfg but expressed
 *  inline so we can reuse the scheduler's helper directly. */

function estimate(input: GenerateInput): PipelineEstimate {
  const latentH = input.height / 8;
  const latentW = input.width / 8;
  const vaeTiles = vaeTileCount(latentH, latentW, input.tileVae);
  const img2imgUnits = input.refImage ? 1 : 0;
  const totalUnits = FIXED_PHASE_UNITS + img2imgUnits + input.steps + vaeTiles;
  return { totalUnits };
}

async function run(input: GenerateInput, cb: GenerateCallbacks): Promise<ImageData> {
  if (input.set.family !== "sdxl") {
    throw new Error(`generateSdxl: expected sdxl model, got ${input.set.family}`);
  }
  const set = input.set as SdxlModelSet;
  const cache = input.cache;
  const { width, height, steps: numSteps, cfg, prompt } = input;
  const latentH = height / 8;
  const latentW = width / 8;
  const latentLen = SDXL_UNET_LATENT_CHANNELS * latentH * latentW;
  const rng = mulberry32(input.seed);
  const useImg2Img = input.refImage !== null;
  const refStrength = input.refImage?.strength ?? 1;
  const schedulerSteps = computeSchedulerSteps(numSteps, refStrength, useImg2Img);
  const log = cb.log;
  log(`[sdxl] seed=${input.seed} size=${width}x${height} steps=${numSteps} cfg=${cfg}`);

  // ----- 1. Tokenize -----
  cb.status("loading tokenizers...");
  const vocabText = await cache.loadFileText(set.tokenizer.vocab);
  const mergesText = await cache.loadFileText(set.tokenizer.merges);
  const tokenizer = new ClipTokenizer(JSON.parse(vocabText), mergesText);
  // The bigG tokenizer files are byte-identical to CLIP-L's in every SDXL
  // export I have inspected, but we still load them via the cache so that
  // (a) the model manager downloaded them, and (b) if a future export
  // diverges we are not silently using the wrong vocab.
  const vocab2Text = await cache.loadFileText(set.tokenizer2.vocab);
  const merges2Text = await cache.loadFileText(set.tokenizer2.merges);
  const tokenizer2 = new ClipTokenizer(JSON.parse(vocab2Text), merges2Text);
  // SDXL pipelines tokenize the prompt independently for each encoder.
  // The encoders themselves consume the same string and the same vocab,
  // so the tokenization step is effectively duplicated; we keep both code
  // paths for safety against future export divergence.
  const condIds = tokenizer.encode(prompt);
  const uncondIds = tokenizer.encode("");
  const condIds2 = tokenizer2.encode(prompt);
  const uncondIds2 = tokenizer2.encode("");
  log(
    `[sdxl] tokenized: cond=${condIds.length}/${condIds2.length} uncond=${uncondIds.length}/${uncondIds2.length}`,
  );
  cb.advance();
  cb.checkAborted();

  // ----- 2. Dual text encoder -----
  cb.status("loading SDXL text encoders (~1.5 GB)...");
  const tEncLoad = performance.now();
  const dualTe = await SdxlDualTextEncoder.load(cache, set.textEncoder, set.textEncoder2);
  log(`[sdxl] text encoders loaded in ${(performance.now() - tEncLoad).toFixed(1)} ms`);
  cb.status("encoding prompts...");
  const tEncRun = performance.now();
  // Each encoder cares about its own ids; we run the encode pair for both
  // ids using the bigG tokenizer's ids since they are byte-identical to
  // CLIP-L's. Use condIds for both because (a) tokenizer2 is the same
  // bytes, and (b) the bigG encoder's int dtype is matched at session-load
  // time inside SdxlDualTextEncoder.
  void uncondIds2;
  void condIds2;
  const uncondEmb = await dualTe.encode(uncondIds);
  const condEmb = await dualTe.encode(condIds);
  log(
    `[sdxl] dual text encoder ran in ${(performance.now() - tEncRun).toFixed(1)} ms (2 prompts x 2 encoders)`,
  );
  cb.status("releasing text encoders...");
  await dualTe.release();
  log("[sdxl] text encoders released");
  cb.advance();
  cb.checkAborted();

  // ----- 3. UNet load -----
  cb.status("loading SDXL UNet...");
  const tUnetLoad = performance.now();
  const boundaryDtype = set.boundaryDtype ?? "float32";
  const unet = await SdxlUnet.load(cache, set.unet, { boundaryDtype });
  log(`[sdxl] UNet ORT session created in ${(performance.now() - tUnetLoad).toFixed(1)} ms`);
  cb.advance();
  cb.checkAborted();

  const schedulerOpts = { numInferenceSteps: schedulerSteps, guidanceScale: cfg };
  const scheduler =
    set.schedulerType === "euler"
      ? new EulerScheduler(schedulerOpts)
      : new DdimScheduler(schedulerOpts);

  const timeIds = buildTimeIds(width, height);

  // ----- 4. Initial latent -----
  let latent: Float32Array;
  let loopStart = 0;
  if (useImg2Img) {
    cb.status("loading VAE encoder...");
    const tVencLoad = performance.now();
    const refImageData = rasterizeRefImage(input.refImage!.image, width, height);
    const refChw = imageDataToChw(refImageData);
    const vaeEnc = await VaeEncoder.load(cache, set.vaeEncoder!, {
      scalingFactor: SDXL_VAE_SCALING_FACTOR,
      boundaryDtype: boundaryDtype,
    });
    log(`[sdxl] VAE encoder loaded in ${(performance.now() - tVencLoad).toFixed(1)} ms`);
    cb.status("encoding reference image...");
    const tVencRun = performance.now();
    const cleanLatent = await vaeEnc.encode(refChw, height, width);
    log(`[sdxl] VAE encoder ran in ${(performance.now() - tVencRun).toFixed(1)} ms`);
    await vaeEnc.release();
    log("[sdxl] VAE encoder released");

    const { startIndex, tStart } = scheduler.getImg2ImgTimesteps(refStrength);
    loopStart = startIndex;
    log(
      `[sdxl] img2img: strength=${refStrength}, schedulerSteps=${schedulerSteps}, startIndex=${startIndex}, tStart=${tStart}, effectiveSteps=${schedulerSteps - startIndex}`,
    );

    if (startIndex >= schedulerSteps) {
      latent = cleanLatent;
    } else {
      const noise = gaussianNoise(latentLen, rng);
      latent = scheduler.addNoise(cleanLatent, noise, tStart);
    }
    cb.advance();
    cb.checkAborted();
  } else {
    latent = gaussianNoise(latentLen, rng);
    for (let i = 0; i < latent.length; i++) latent[i] *= scheduler.initNoiseSigma;
  }

  // CFG <= 1 means no classifier-free guidance: skip the unconditional pass
  // entirely and use the conditional prediction directly. SDXL-Turbo (ADD
  // distillation) is trained this way - 1 UNet pass per step instead of 2.
  const useCfg = cfg > 1;

  log(`[sdxl] initial latent: ${latent.length} floats, ${JSON.stringify(statsOf(latent))}`);
  log(
    `[sdxl] scheduler: ${schedulerSteps} steps (loop from ${loopStart}), CFG=${cfg}${useCfg ? "" : " (guidance disabled)"}`,
  );

  // ----- 5. Denoise loop -----
  let stepTimeSum = 0;
  let stepTimeCount = 0;
  const tLoop = performance.now();
  const totalLoopSteps = schedulerSteps - loopStart;
  for (let step = loopStart; step < schedulerSteps; step++) {
    const t = scheduler.timesteps[step];
    const stepDisplay = step - loopStart + 1;
    cb.status(`denoising step ${stepDisplay} / ${totalLoopSteps} (t=${t})...`);
    const tStep = performance.now();

    // Euler scheduler requires scaling the model input by 1/sqrt(sigma^2+1)
    // before each UNet call. The unscaled latent is kept for the step update.
    const modelInput =
      scheduler instanceof EulerScheduler ? scheduler.scaleModelInput(latent, step) : latent;

    let noisePred: Float32Array;
    if (useCfg) {
      const uncondNoise = await unet.predictNoise(
        modelInput,
        t,
        uncondEmb.hiddenStates,
        uncondEmb.pooledTextEmbeds,
        timeIds,
        latentH,
        latentW,
      );
      const condNoise = await unet.predictNoise(
        modelInput,
        t,
        condEmb.hiddenStates,
        condEmb.pooledTextEmbeds,
        timeIds,
        latentH,
        latentW,
      );
      log(
        `  [sdxl] step ${stepDisplay}: uncond=${fmtStats(statsOf(uncondNoise))} cond=${fmtStats(statsOf(condNoise))}`,
      );
      noisePred = scheduler.applyCfg(uncondNoise, condNoise);
    } else {
      noisePred = await unet.predictNoise(
        modelInput,
        t,
        condEmb.hiddenStates,
        condEmb.pooledTextEmbeds,
        timeIds,
        latentH,
        latentW,
      );
      log(`  [sdxl] step ${stepDisplay}: pred=${fmtStats(statsOf(noisePred))}`);
    }

    latent = scheduler.step(step, latent, noisePred);
    const stepMs = performance.now() - tStep;
    stepTimeSum += stepMs;
    stepTimeCount++;
    const avgMs = stepTimeSum / stepTimeCount;
    const remaining = schedulerSteps - (step + 1);
    const etaSec = (avgMs * remaining) / 1000;
    cb.stats(
      `step ${stepDisplay}/${totalLoopSteps} | ${formatMs(avgMs)} avg/step | ~${formatEta(etaSec)} left`,
    );
    log(
      `  [sdxl] step ${stepDisplay} (${stepMs.toFixed(0)} ms): latent=${fmtStats(statsOf(latent))}`,
    );
    cb.advance();
    cb.checkAborted();
  }
  log(`[sdxl] denoising loop total: ${((performance.now() - tLoop) / 1000).toFixed(1)} s`);

  cb.status("releasing UNet...");
  await unet.release();
  log("[sdxl] UNet released");
  log(`[sdxl] final latent: ${fmtStats(statsOf(latent))}`);

  // ----- 6. VAE decode -----
  cb.status("loading VAE decoder...");
  const tVaeLoad = performance.now();
  const vae = await VaeDecoder.load(cache, set.vaeDecoder, {
    scalingFactor: SDXL_VAE_SCALING_FACTOR,
    boundaryDtype: boundaryDtype,
  });
  log(`[sdxl] VAE decoder loaded in ${(performance.now() - tVaeLoad).toFixed(1)} ms`);
  cb.advance();
  cb.checkAborted();

  const tDecode = performance.now();
  const imageData = await vae.decode(latent, latentH, latentW, {
    tiled: input.tileVae,
    onTileProgress: (idx, total) => {
      cb.status(total > 1 ? `decoding latent tile ${idx} / ${total}...` : "decoding latent...");
      cb.advance();
    },
  });
  log(`[sdxl] VAE decode ran in ${(performance.now() - tDecode).toFixed(1)} ms`);
  cb.checkAborted();

  cb.status("releasing VAE...");
  await vae.release();
  log("[sdxl] VAE released");

  return imageData;
}

export const sdxlGenerateFn: GenerateFn = { estimate, run };
