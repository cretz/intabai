// SD1.5 image generation. Extracted from web/src/image-gen/main.ts so the
// pipeline can be reused once we add a worker mode and so the SDXL pipeline
// has a peer to dispatch to.
//
// Inputs / outputs / callbacks: see web/src/image-gen/generate-types.ts.
//
// Pipeline shape (txt2img):
//   1. Tokenize prompt + empty (uncond)
//   2. Load text encoder, encode both prompts, release
//   3. Load UNet
//   4. Build initial latent (Gaussian noise)
//   5. Run the denoise loop (uncond + cond per step, CFG-combine, DDIM step)
//   6. Release UNet
//   7. Load VAE decoder, decode, release
//   8. Return ImageData
//
// img2img mode inserts steps 0a (load VAE encoder, encode reference image,
// release) and modifies step 4 (partial-noise the encoded latent at the
// strength-implied timestep instead of starting from pure noise) and the
// loop bounds (start from `loopStart` instead of 0). The scheduler step
// count is inflated upstream so that after diffusers-style trimming the
// effective step count still equals what the user asked for.

import type { Sd15ModelSet } from "./models";
import { ClipTokenizer } from "./tokenizer";
import { TextEncoder } from "./text-encoder";
import { DdimScheduler } from "./scheduler";
import { Unet, UNET_LATENT_CHANNELS } from "./unet";
import { VaeDecoder, vaeTileCount } from "./vae";
import { VaeEncoder, imageDataToChw } from "./vae-encoder";
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

// Fixed-phase progress unit budget. Phases that always run regardless of
// txt2img vs img2img mode: tokenize+encode, unet load, vae load. (The
// per-tile vae decode units are added separately, replacing what would
// have been a single "decode" unit.)
const FIXED_PHASE_UNITS = 4;

function computeSchedulerSteps(steps: number, refStrength: number, hasRef: boolean): number {
  // Inflate the scheduler step count so that after partial-start trimming
  // there are still `steps` actual UNet passes left to run. Matches A1111
  // / user intuition: the steps slider says 20, generation runs 20 UNet
  // forwards regardless of strength.
  if (!hasRef || refStrength <= 0) return steps;
  return Math.max(steps, Math.ceil(steps / refStrength));
}

function estimate(input: GenerateInput): PipelineEstimate {
  const latentH = input.height / 8;
  const latentW = input.width / 8;
  const vaeTiles = vaeTileCount(latentH, latentW, input.tileVae);
  const img2imgUnits = input.refImage ? 1 : 0;
  // The denoise loop is `steps` units regardless of img2img inflation,
  // because the actual loop only runs `steps` iterations.
  const totalUnits = FIXED_PHASE_UNITS + img2imgUnits + input.steps + vaeTiles;
  return { totalUnits };
}

async function run(input: GenerateInput, cb: GenerateCallbacks): Promise<ImageData> {
  if (input.set.family !== "sd15") {
    throw new Error(`generateSd15: expected sd15 model, got ${input.set.family}`);
  }
  const set = input.set as Sd15ModelSet;
  const cache = input.cache;
  const { width, height, steps: numSteps, cfg, prompt } = input;
  const latentH = height / 8;
  const latentW = width / 8;
  const latentLen = UNET_LATENT_CHANNELS * latentH * latentW;
  const rng = mulberry32(input.seed);
  const useImg2Img = input.refImage !== null;
  const refStrength = input.refImage?.strength ?? 1;
  const schedulerSteps = computeSchedulerSteps(numSteps, refStrength, useImg2Img);
  const log = cb.log;
  log(`seed=${input.seed} size=${width}x${height} steps=${numSteps} cfg=${cfg}`);

  // ----- 1. Tokenize -----
  cb.status("loading tokenizer...");
  const vocabText = await cache.loadFileText(set.tokenizer.vocab);
  const mergesText = await cache.loadFileText(set.tokenizer.merges);
  const tokenizer = new ClipTokenizer(JSON.parse(vocabText), mergesText);
  const condIds = tokenizer.encode(prompt);
  const uncondIds = tokenizer.encode("");
  log(`tokenized: cond=${condIds.length} tokens, uncond=${uncondIds.length} tokens`);
  cb.advance();
  cb.checkAborted();

  // ----- 2. Text encoder -----
  cb.status("loading text encoder...");
  const encoder = await TextEncoder.load(cache, set.textEncoder);
  cb.status("encoding prompts...");
  const tEnc = performance.now();
  const uncondEmb = await encoder.encode(uncondIds);
  const condEmb = await encoder.encode(condIds);
  log(`text encoder ran in ${(performance.now() - tEnc).toFixed(1)} ms (2 passes)`);
  cb.status("releasing text encoder...");
  await encoder.release();
  log("text encoder released");
  cb.advance();
  cb.checkAborted();

  // ----- 3. UNet load -----
  cb.status("loading UNet (this is the big one)...");
  const tUnetLoad = performance.now();
  const unet = await Unet.load(cache, set.unet);
  log(`UNet ORT session created in ${(performance.now() - tUnetLoad).toFixed(1)} ms`);
  cb.advance();
  cb.checkAborted();

  const scheduler = new DdimScheduler({
    numInferenceSteps: schedulerSteps,
    guidanceScale: cfg,
  });

  // ----- 4. Initial latent -----
  let latent: Float32Array;
  let loopStart = 0;
  if (useImg2Img) {
    cb.status("loading VAE encoder...");
    const tEncLoad = performance.now();
    const refImageData = rasterizeRefImage(input.refImage!.image, width, height);
    const refChw = imageDataToChw(refImageData);
    const vaeEnc = await VaeEncoder.load(cache, set.vaeEncoder!);
    log(`VAE encoder ORT session created in ${(performance.now() - tEncLoad).toFixed(1)} ms`);
    cb.status("encoding reference image...");
    const tEncRun = performance.now();
    const cleanLatent = await vaeEnc.encode(refChw, height, width);
    log(`VAE encoder ran in ${(performance.now() - tEncRun).toFixed(1)} ms`);
    await vaeEnc.release();
    log("VAE encoder released");

    const { startIndex, tStart } = scheduler.getImg2ImgTimesteps(refStrength);
    loopStart = startIndex;
    log(
      `img2img: strength=${refStrength}, schedulerSteps=${schedulerSteps}, startIndex=${startIndex}, tStart=${tStart}, effectiveSteps=${schedulerSteps - startIndex}`,
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

  log(`initial latent: ${latent.length} floats, ${JSON.stringify(statsOf(latent))}`);
  log(`scheduler: ${schedulerSteps} steps (loop from ${loopStart}), CFG=${cfg}`);

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
    const uncondNoise = await unet.predictNoise(latent, t, uncondEmb, latentH, latentW);
    const condNoise = await unet.predictNoise(latent, t, condEmb, latentH, latentW);
    const uncondStats = statsOf(uncondNoise);
    const condStats = statsOf(condNoise);
    const guided = scheduler.applyCfg(uncondNoise, condNoise);
    latent = scheduler.step(step, latent, guided);
    const latentStats = statsOf(latent);
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
      `  step ${stepDisplay} (${stepMs.toFixed(0)} ms): ` +
        `uncond=${fmtStats(uncondStats)} cond=${fmtStats(condStats)} latent=${fmtStats(latentStats)}`,
    );
    cb.advance();
    cb.checkAborted();
  }
  log(`denoising loop total: ${((performance.now() - tLoop) / 1000).toFixed(1)} s`);

  cb.status("releasing UNet...");
  await unet.release();
  log("UNet released");
  log(`final latent: ${fmtStats(statsOf(latent))}`);

  // ----- 6. VAE decode -----
  cb.status("loading VAE decoder...");
  const tVaeLoad = performance.now();
  const vae = await VaeDecoder.load(cache, set.vaeDecoder);
  log(`VAE decoder ORT session created in ${(performance.now() - tVaeLoad).toFixed(1)} ms`);
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
  log(`VAE decode ran in ${(performance.now() - tDecode).toFixed(1)} ms`);
  cb.checkAborted();

  cb.status("releasing VAE...");
  await vae.release();
  log("VAE released");

  return imageData;
}

export const sd15GenerateFn: GenerateFn = { estimate, run };
