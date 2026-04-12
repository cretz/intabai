// Z-Image-Turbo image generation pipeline. S3-DiT architecture with a
// Qwen2-class text encoder and flow matching scheduler.
//
// Key differences from the SD/SDXL pipeline:
//   - Text encoder is a Qwen2-class LM tokenized via transformers.js
//     AutoTokenizer with a chat template (not CLIP BPE).
//   - 16-channel latents in 5D [B, 16, 1, H/8, W/8] (not 4-ch 4D).
//   - Flow matching scheduler (shift=3.0) instead of DDIM/Euler-DDPM.
//   - No classifier-free guidance (single pass per step).
//   - VAE pre-process: squeeze dim-2 and scale (inline, no ONNX helper).
//   - VAE decoder outputs [-1,1] range, not [0,1].
//
// IMPORTANT: Z-Image MUST use the onnxruntime-web/webgpu build, not the
// default onnxruntime-web bundle. The default bundle's WebGPU EP has
// different kernel implementations that produce spatially incoherent
// transformer output for this model's S3-DiT architecture. Other models
// (SD1.5, SDXL, segmind-vega) work fine with the default bundle and
// BREAK with the webgpu bundle (different fused ops like BiasSplitGelu).
//
// Sequential session lifecycle: each ORT session is created, run, then
// released before the next one loads. Peak GPU memory = max(component),
// not sum. The text encoder (~2.2 GB) and transformer (~3.7 GB) are
// never resident simultaneously.

import * as ort from "onnxruntime-web/webgpu";

import type { ZImageModelSet, ZImageShardedModelSet } from "../sd15/models";
import type { ModelCache } from "../shared/model-cache";
import { isExternalData, type OrtModelFile } from "../sd15/ort-helpers";
import { buildSchedule } from "./scheduler";
import {
  formatEta,
  formatMs,
  gaussianNoise,
  mulberry32,
} from "../image-gen/generate-utils";
import type {
  GenerateCallbacks,
  GenerateFn,
  GenerateInput,
  PipelineEstimate,
} from "../image-gen/generate-types";

// Session creation using the webgpu-specific ORT build. Duplicated from
// ort-helpers.ts because that module imports the default onnxruntime-web
// bundle which uses different (incompatible) WebGPU kernels.
async function createZImageSession(
  cache: ModelCache,
  model: OrtModelFile,
  providers: string[],
  extraOptions: Partial<ort.InferenceSession.SessionOptions> = {},
): Promise<ort.InferenceSession> {
  if (!isExternalData(model)) {
    const buffer = await cache.loadFile(model);
    return ort.InferenceSession.create(buffer, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      ...extraOptions,
    });
  }
  const { url: graphUrl, revoke: revokeGraph } = await cache.loadFileAsBlobUrl(model.graph);
  const { url: dataUrl, revoke: revokeData } = await cache.loadFileAsBlobUrl(model.data);
  try {
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      ...extraOptions,
    };
    (sessionOptions as unknown as { externalData: Array<{ path: string; data: string }> })
      .externalData = [{ path: model.dataPath, data: dataUrl }];
    return await ort.InferenceSession.create(graphUrl, sessionOptions);
  } finally {
    revokeGraph();
    revokeData();
  }
}

// Transformers.js AutoTokenizer, loaded lazily on first generate.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let cachedTokenizer: any = null;

const LATENT_CHANNELS = 16;

// Fixed phase units: tokenize + text encode + load transformer + load vae
const FIXED_PHASE_UNITS = 4;

function estimate(input: GenerateInput): PipelineEstimate {
  const totalUnits = FIXED_PHASE_UNITS + input.steps; // + 1 vae tile (no tiling for zimage)
  return { totalUnits };
}

function defaultProviders(): string[] {
  // wasm fallback is required for perf: ~66 transformer nodes (mostly
  // shape ops) and ~47 VAE-decoder nodes always end up on a CPU EP
  // regardless. Listing wasm puts them on the threaded SIMD WASM EP
  // (~1s/step on desktop); dropping wasm forces single-threaded scalar
  // CPU and balloons each step to ~3.5s. Verified 2026-04-11 during
  // the chat-template artifact hunt.
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

// --- GPU buffer helpers ---

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getGpuDevice(): GPUDevice | undefined {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (ort.env as any).webgpu?.device as GPUDevice | undefined;
}

function createGpuBuf(gpuDevice: GPUDevice, dataType: string, dims: readonly number[]): ort.Tensor {
  const numEl = dims.reduce((a, b) => a * b, 1);
  const bpe = dataType === "int64" ? 8 : dataType === "float16" ? 2 : 4;
  const size = Math.max(16, Math.ceil(numEl * bpe / 4) * 4);
  const buf = gpuDevice.createBuffer({
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    size,
  });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (ort.Tensor as any).fromGpuBuffer(buf, { dataType, dims }) as ort.Tensor;
}

function writeGpu(gpuDevice: GPUDevice, tensor: ort.Tensor, data: Float32Array): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const buf = (tensor as any).gpuBuffer as GPUBuffer;
  const aligned = Math.ceil(data.byteLength / 4) * 4;
  const enc = gpuDevice.createCommandEncoder();
  const tmp = gpuDevice.createBuffer({ size: aligned, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
  new Uint8Array(tmp.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  tmp.unmap();
  enc.copyBufferToBuffer(tmp, 0, buf, 0, aligned);
  gpuDevice.queue.submit([enc.finish()]);
}

async function readGpu(gpuDevice: GPUDevice, tensor: ort.Tensor, count: number): Promise<Float32Array> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const buf = (tensor as any).gpuBuffer as GPUBuffer;
  const bytes = count * 4;
  const aligned = Math.ceil(bytes / 4) * 4;
  const rb = gpuDevice.createBuffer({ size: aligned, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = gpuDevice.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, aligned);
  gpuDevice.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(rb.getMappedRange().slice(0, bytes));
  rb.unmap();
  rb.destroy();
  return result;
}

// --- Monolithic transformer loop (original, unchanged behavior) ---

async function runMonolithicTransformerLoop(
  set: ZImageModelSet, cache: ModelCache, numSteps: number,
  initialLatent: Float32Array, encoderHiddenState: Float32Array,
  latentH: number, latentW: number, latentLen: number, sequenceLength: number,
  cb: GenerateCallbacks, log: (msg: string) => void,
): Promise<Float32Array> {
  cb.status("loading transformer (~3.7 GB)...");
  const tXfmrLoad = performance.now();
  const xfmrSess = await createZImageSession(cache, set.transformer, defaultProviders());
  log(`[zimage] transformer loaded in ${(performance.now() - tXfmrLoad).toFixed(1)} ms`);

  const schedSess = await createZImageSession(cache, set.schedulerStep, defaultProviders());
  log("[zimage] scheduler step model loaded");
  cb.advance();
  cb.checkAborted();

  // Capture GPU device AFTER session creation (ORT may init/change the device)
  const gpuDevice = getGpuDevice();
  if (!gpuDevice) log("[zimage] WARNING: no GPU device");

  const schedule = buildSchedule(numSteps);
  log(`[zimage] schedule: ${numSteps} steps`);

  const hiddenDims = [1, LATENT_CHANNELS, 1, latentH, latentW] as const;
  const noiseDims = [LATENT_CHANNELS, 1, latentH, latentW] as const;
  const gpuHidden = createGpuBuf(gpuDevice!, "float32", hiddenDims);
  const gpuTimestep = createGpuBuf(gpuDevice!, "float32", [1]);
  const gpuEncHidden = createGpuBuf(gpuDevice!, "float32", [1, sequenceLength, set.hiddenDim]);
  const gpuNoisePred = createGpuBuf(gpuDevice!, "float32", noiseDims);
  const gpuStepInfo = createGpuBuf(gpuDevice!, "float32", [2]);
  const gpuLatentsOut = createGpuBuf(gpuDevice!, "float32", hiddenDims);

  writeGpu(gpuDevice!, gpuEncHidden, encoderHiddenState);
  writeGpu(gpuDevice!, gpuHidden, initialLatent);

  const xfmrFeed: Record<string, ort.Tensor> = {
    hidden_states: gpuHidden,
    timestep: gpuTimestep,
    encoder_hidden_states: gpuEncHidden,
  };
  const xfmrFetches: Record<string, ort.Tensor> = { unified_results: gpuNoisePred };

  const schedFeed: Record<string, ort.Tensor> = {
    noise_pred: gpuNoisePred,
    latents: gpuHidden,
    step_info: gpuStepInfo,
  };
  const schedFetches: Record<string, ort.Tensor> = { latents_out: gpuLatentsOut };

  log("[zimage] GPU IO binding ready");

  let stepTimeSum = 0;
  let stepTimeCount = 0;
  const tLoop = performance.now();
  for (let step = 0; step < numSteps; step++) {
    const stepDisplay = step + 1;
    cb.status(`DiT step ${stepDisplay} / ${numSteps}...`);
    const tStep = performance.now();

    writeGpu(gpuDevice!, gpuTimestep, Float32Array.from([schedule.timesteps[step]]));
    await xfmrSess.run(xfmrFeed, xfmrFetches);

    writeGpu(gpuDevice!, gpuStepInfo, Float32Array.from([step, numSteps]));
    await schedSess.run(schedFeed, schedFetches);

    // Ping-pong: scheduler output becomes next transformer input
    const nextInput = schedFetches.latents_out;
    const nextOutput = schedFeed.latents;
    schedFeed.latents = nextInput;
    schedFetches.latents_out = nextOutput;
    xfmrFeed.hidden_states = nextInput;

    const stepMs = performance.now() - tStep;
    stepTimeSum += stepMs;
    stepTimeCount++;
    const avgMs = stepTimeSum / stepTimeCount;
    const remaining = numSteps - stepDisplay;
    const etaSec = (avgMs * remaining) / 1000;
    cb.stats(
      `step ${stepDisplay}/${numSteps} | ${formatMs(avgMs)} avg/step | ~${formatEta(etaSec)} left`,
    );
    log(`  [zimage] step ${stepDisplay} (${stepMs.toFixed(0)} ms)`);
    cb.advance();
    cb.checkAborted();
  }
  log(`[zimage] DiT loop total: ${((performance.now() - tLoop) / 1000).toFixed(1)} s`);

  const latent = await readGpu(gpuDevice!, xfmrFeed.hidden_states, latentLen);

  cb.status("releasing transformer + scheduler...");
  await xfmrSess.release();
  await schedSess.release();
  log("[zimage] transformer + scheduler released");
  return latent;
}

// --- Sharded transformer loop ---
// Runs N shard sessions per step with create/run overlap. Inter-shard tensors
// are passed as CPU tensors; each shard session is released immediately after
// its run so ORT reclaims weight memory. The shared external data file is
// loaded once as a blob URL and reused across every shard create in the loop.
//
// ORT-web's WebGPU EP serializes InferenceSession.create() with a mutex
// ("another WebGPU EP inference session is being created"), so only one
// create can ever be in flight. What we CAN overlap is the RUN of shard N
// with the CREATE of shard N+1. Implementation: a single-slot promise
// ("create queue") holds the next shard's in-flight create. On each
// iteration we pop (await) the slot to get the current shard, immediately
// refill the slot by kicking off the next shard's create, then run +
// release the current shard. The "refill happens before run" ordering is
// what produces the overlap — by the time the next iteration pops the
// slot, ORT has been working on its create during the previous run.
//
// Effective per-step cost drops from `nShards * (create + run)` (strict
// sequential) toward `create + nShards * run` (first shard pays full create,
// each subsequent shard's create hides behind the prior run), assuming ORT
// does not also serialize run-vs-create across sessions.

// Number of shard session creates allowed to be in flight IN ADDITION to
// the currently-running shard. The create queue intentionally trails the
// data queue by one slot: with DATA_PREFETCH_DEPTH=2 and this set to 1, one
// future shard can already be in `InferenceSession.create()` while another
// future shard's data buffer is still waiting behind it in dataQueue.
const SHARD_PREFETCH_DEPTH = 0;

// Number of shard *data* buffers to keep pre-loaded in RAM ahead of the
// currently-executing shard. Session create/run/release remain strictly
// sequential (SHARD_PREFETCH_DEPTH=0 above). Data loading is the only
// thing we overlap: shard N+k's bytes can be coming off OPFS while shard N
// is creating / running / releasing, so the OPFS read latency does not
// stack onto the critical path.
//
// Peak JS heap from data buffers: ~DATA_PREFETCH_DEPTH * shard_data_size.
// With the onnx-600 experiment, shard data is closer to ~338-356 MB, so keep
// the queue shallow until we know mobile can tolerate more than one read
// ahead.
const DATA_PREFETCH_DEPTH = 1;

async function runShardedTransformerLoop(
  set: ZImageShardedModelSet, cache: ModelCache, numSteps: number,
  initialLatent: Float32Array, encoderHiddenState: Float32Array,
  latentH: number, latentW: number, _latentLen: number, sequenceLength: number,
  cb: GenerateCallbacks, log: (msg: string) => void,
): Promise<Float32Array> {
  const shards = set.transformerShards;
  const nShards = shards.length;
  log(`[zimage-sharded] ${nShards} shards, ${numSteps} steps, shardPrefetch=${SHARD_PREFETCH_DEPTH}, dataPrefetch=${DATA_PREFETCH_DEPTH}`);

  // Keep the scheduler on WASM so the shard sessions are the only live
  // WebGPU sessions during the DiT loop. ORT-web's WebGPU allocator/cache
  // appears to defer freeing buffers until all sessions are gone; keeping a
  // persistent scheduler WebGPU session alive may prevent shard memory from
  // actually being reclaimed on mobile between shard releases.
  const schedSess = await createZImageSession(cache, set.schedulerStep, ["wasm"]);
  log("[zimage-sharded] scheduler step model loaded (wasm)");

  const gpuDevice = getGpuDevice();
  if (!gpuDevice) throw new Error("Sharded Z-Image requires WebGPU");

  const schedule = buildSchedule(numSteps);
  const hiddenDims = [1, LATENT_CHANNELS, 1, latentH, latentW] as const;
  const noiseDims = [LATENT_CHANNELS, 1, latentH, latentW] as const;

  // Shards still use GPU IO for their final noise prediction upload path.
  const gpuNoisePred = createGpuBuf(gpuDevice, "float32", noiseDims);

  // Parallel CPU copy of the current latent, used as `hidden_states` input to
  // shard 0 of the next step. After each step, the WASM scheduler produces
  // the next latent directly on CPU.
  let cachedHiddenData: Float32Array = initialLatent;

  // Graph-level inputs as CPU tensors (re-used every step)
  const cpuEncHidden = new ort.Tensor("float32", encoderHiddenState, [1, sequenceLength, set.hiddenDim]);

  cb.advance();
  cb.checkAborted();

  // Load a shard's external data file into a fresh Uint8Array. Used by the
  // dataQueue prefetcher below so that OPFS reads can overlap with the
  // current shard's create/run/release phases instead of blocking at the
  // start of the next shard's create.
  const loadShardData = (shardIdx: number): Promise<Uint8Array> =>
    cache.loadFile(shards[shardIdx].data).then((buf) => new Uint8Array(buf));

  // Create one shard InferenceSession using pre-loaded data bytes. Splitting
  // data load from create is what lets us overlap OPFS reads with the
  // previous shard's run (via the dataQueue below) while still keeping the
  // create itself strictly sequential (so there is a zero-session window
  // between shards for mobile WebGPU buffer reclaim).
  //
  // ORT-web's createSession calls loadFile on whatever we pass for
  // externalData.data. loadFile short-circuits on a Uint8Array and returns
  // it directly (ort.all.mjs:25437); mountExternalData is O(1) (just stores
  // a reference in a Map). So passing our pre-read Uint8Array here makes
  // create skip the fetch entirely - it just mounts, wasm copies in, unmount.
  //
  // The graph blob URL is still created and revoked per shard (the graph
  // file is ~170 MB and only touched on its own shard's create).
  const createShardSession = async (shardIdx: number, dataBytes: Uint8Array): Promise<ort.InferenceSession> => {
    const shard = shards[shardIdx];
    const { url: graphUrl, revoke: revokeGraph } =
      await cache.loadFileAsBlobUrl(shard.graph);
    try {
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: defaultProviders(),
        graphOptimizationLevel: "all",
      };
      // ORT's type for externalData.data is `string`, but the runtime
      // accepts Uint8Array / Blob / File too (see loadFile in ort.all.mjs).
      (sessionOptions as unknown as { externalData: Array<{ path: string; data: Uint8Array }> })
        .externalData = [{ path: shard.dataPath, data: dataBytes }];
      return await ort.InferenceSession.create(graphUrl, sessionOptions);
    } finally {
      revokeGraph();
    }
  };

  // FIFO of in-flight shard data buffers. Capacity is DATA_PREFETCH_DEPTH:
  // at steady state this many shard-data OPFS reads are either in flight or
  // completed-and-pending-consumption. Primed at loop entry with the first
  // DEPTH shards so iter 0's create doesn't pay the full OPFS read latency.
  //
  // A new data load is pushed at the end of each iteration (after we've
  // null'd our local reference to the current shard's bytes, so peak JS
  // heap doesn't include both the current shard's data AND the refilled
  // tail of the queue).
  const totalFlat = numSteps * nShards;
  const dataQueue: Array<Promise<Uint8Array>> = [];
  const primeCount = Math.min(DATA_PREFETCH_DEPTH, totalFlat);
  for (let i = 0; i < primeCount; i++) {
    dataQueue.push(loadShardData(i % nShards));
  }

  type PendingShardSession = {
    flatIdx: number;
    shardIdx: number;
    session: ort.InferenceSession;
  };

  const effectiveCreatePrefetchDepth = Math.min(
    SHARD_PREFETCH_DEPTH,
    Math.max(0, DATA_PREFETCH_DEPTH - 1),
  );
  const createQueue: Array<Promise<PendingShardSession>> = [];

  const enqueueCreate = (flatIdx: number): void => {
    const shardIdx = flatIdx % nShards;
    const dataPromise = dataQueue.shift();
    if (!dataPromise) throw new Error(`data queue empty while enqueueing create at flatIdx ${flatIdx}`);
    createQueue.push((async () => {
      const dataBytes = await dataPromise;
      const session = await createShardSession(shardIdx, dataBytes);
      return { flatIdx, shardIdx, session };
    })());
  };

  for (let i = 0; i < Math.min(effectiveCreatePrefetchDepth, totalFlat); i++) {
    enqueueCreate(i);
  }

  // Drain pending data loads on error/abort. These are just Uint8Arrays;
  // nothing to release, just await to avoid unhandled-rejection noise.
  const drainDataQueue = async (): Promise<void> => {
    while (dataQueue.length > 0) {
      try { await dataQueue.shift(); } catch { /* best-effort */ }
    }
  };

  const drainCreateQueue = async (): Promise<void> => {
    while (createQueue.length > 0) {
      try {
        const pending = await createQueue.shift();
        await pending?.session.release();
      } catch {
        /* best-effort */
      }
    }
  };

  // ETA is derived from rate = wall-clock / bytes processed, with the first
  // shard of the first step excluded from the baseline. Rationale: shard 0 is
  // small on Z-Image-Turbo-sharded (~221 MB vs ~900 MB for the middle shards)
  // and pays the full first-time session-create overhead, so its ms/byte is
  // not representative of steady-state throughput. Including it pins the ETA
  // to a distorted rate until enough big shards have completed to wash it
  // out. Instead we start the rate clock when shard 0 finishes and measure
  // everything after that as "time-over-size" from a clean baseline.
  const shardWeights =
    set.progress?.repeatedUnitWeightBytes?.length === nShards
      ? set.progress.repeatedUnitWeightBytes
      : shards.map((shard) => shard.graph.sizeBytes + shard.data.sizeBytes);
  const totalWeightPerStep = shardWeights.reduce((sum, weight) => sum + weight, 0);
  let rateStartMs = 0;  // set at end of shard 0 / step 0
  let bytesSinceRateStart = 0;
  const tLoop = performance.now();

  try {
    for (let step = 0; step < numSteps; step++) {
      const stepDisplay = step + 1;
      const tStep = performance.now();

      const cpuTimestep = new ort.Tensor("float32", Float32Array.from([schedule.timesteps[step]]), [1]);
      // hidden_states for shard 0 comes from the cached scheduler output (or
      // initial latent for step 0). We never read this from GPU because the
      // sharded path keeps inter-shard values on CPU.
      const cpuHidden = new ort.Tensor("float32", cachedHiddenData, hiddenDims);

      // Accumulate inter-shard tensors. Shard 0 gets the 3 graph inputs;
      // subsequent shards get the union of all prior shard outputs.
      const tensorPool: Record<string, ort.Tensor> = {
        hidden_states: cpuHidden,
        timestep: cpuTimestep,
        encoder_hidden_states: cpuEncHidden,
      };

      for (let si = 0; si < nShards; si++) {
        cb.status(`step ${stepDisplay}/${numSteps} shard ${si + 1}/${nShards}...`);
        const flatIdx = step * nShards + si;
        const tShard = performance.now();

        let sess: ort.InferenceSession;
        let dataWaitMs = 0;
        let createMs = 0;
        if (effectiveCreatePrefetchDepth > 0) {
          const tCreateWait = performance.now();
          const pendingCreate = createQueue.shift();
          if (!pendingCreate) throw new Error(`create queue empty at flatIdx ${flatIdx}`);
          const ready = await pendingCreate;
          if (ready.flatIdx !== flatIdx || ready.shardIdx !== si) {
            throw new Error(`create queue mismatch at flatIdx ${flatIdx}: got flatIdx=${ready.flatIdx}, shard=${ready.shardIdx}`);
          }
          sess = ready.session;
          createMs = performance.now() - tCreateWait;
        } else {
          // --- Phase: await head of data FIFO ---
          // If prefetch kept up, this await is ~0 because the OPFS read for
          // this shard finished while the previous shard was running. If the
          // read is slower than run+release, we pay the residual here.
          const tDataWait = performance.now();
          const dataPromise = dataQueue.shift();
          if (!dataPromise) throw new Error(`data queue empty at flatIdx ${flatIdx}`);
          const dataBytes = await dataPromise;
          dataWaitMs = performance.now() - tDataWait;

          // --- Phase: create (sequential) ---
          const tCreate = performance.now();
          sess = await createShardSession(si, dataBytes);
          createMs = performance.now() - tCreate;
        }

        // --- Phase: refill data queue ---
        // Push the next shard's data load. With DEPTH=2 this means one
        // shard's worth of data is always being fetched in the background
        // while we run+release the current shard, and another is already
        // sitting at the head of the queue ready for the next iteration.
        const nextDataFlat = flatIdx + DATA_PREFETCH_DEPTH;
        if (nextDataFlat < totalFlat) {
          dataQueue.push(loadShardData(nextDataFlat % nShards));
        }

        // --- Phase: refill create queue ---
        // Consume the current head of dataQueue so one create stays one slot
        // behind the data queue. That keeps one raw data buffer in reserve
        // behind the in-flight create.
        const nextCreateFlat = flatIdx + effectiveCreatePrefetchDepth;
        if (effectiveCreatePrefetchDepth > 0 && nextCreateFlat < totalFlat) {
          enqueueCreate(nextCreateFlat);
        }

        // Build feed from the tensor pool - only pass inputs this shard needs
        const feed: Record<string, ort.Tensor> = {};
        for (const name of sess.inputNames) {
          const t = tensorPool[name];
          if (!t) throw new Error(`shard ${si} needs input "${name}" not in tensor pool`);
          feed[name] = t;
        }

        // --- Phase: run ---
        const tRun = performance.now();
        const results = await sess.run(feed);
        const runMs = performance.now() - tRun;

        // Copy data out of ORT's result tensors into the pool, then dispose
        // all results so ORT's internal GPU buffers are freed before release.
        for (const key in results) {
          const rt = results[key];
          const src = rt.data;
          // Copy using the same typed array constructor to preserve dtype
          const copy = new (src.constructor as { new(a: ArrayLike<never>): typeof src })(
            src as unknown as ArrayLike<never>,
          );
          tensorPool[key] = new ort.Tensor(rt.type, copy, rt.dims);
          rt.dispose();
        }

        // --- Phase: release + GPU flush ---
        const tRelease = performance.now();
        await sess.release();
        gpuDevice.queue.submit([]);  // empty submit to flush pending frees
        await gpuDevice.queue.onSubmittedWorkDone();
        const releaseMs = performance.now() - tRelease;

        const shardMs = performance.now() - tShard;
        const flatIdxJustCompleted = step * nShards + si;
        if (flatIdxJustCompleted === 0) {
          // End of shard 0: start the rate clock here. The small-shard +
          // session-create overhead of shard 0 is intentionally NOT folded
          // into the rate measurement.
          rateStartMs = performance.now();
        } else {
          bytesSinceRateStart += shardWeights[si];
        }
        const remainingThisStepWeight =
          totalWeightPerStep - shardWeights.slice(0, si + 1).reduce((sum, weight) => sum + weight, 0);
        const remainingFutureStepsWeight = (numSteps - stepDisplay) * totalWeightPerStep;
        const remainingWeight = remainingThisStepWeight + remainingFutureStepsWeight;
        let etaLabel: string;
        if (bytesSinceRateStart > 0) {
          const msPerByte = (performance.now() - rateStartMs) / bytesSinceRateStart;
          const etaSec = (msPerByte * remainingWeight) / 1000;
          etaLabel = `~${formatEta(etaSec)} left`;
        } else {
          etaLabel = "estimating...";
        }
        cb.stats(
          `step ${stepDisplay}/${numSteps} shard ${si + 1}/${nShards} | ${etaLabel}`,
        );
        log(`  [zimage-sharded] step ${stepDisplay} shard ${si}: dataWait ${dataWaitMs.toFixed(0)}ms, create ${createMs.toFixed(0)}ms, run ${runMs.toFixed(0)}ms, release ${releaseMs.toFixed(0)}ms (total ${shardMs.toFixed(0)}ms)`);
      }

      // Final shard output: unified_results -> noise_pred for scheduler
      const noisePredTensor = tensorPool["unified_results"];
      if (!noisePredTensor) throw new Error("final shard did not produce unified_results");

      // Preserve the existing GPU upload here as a lightweight validation that
      // the final shard output is still representable in the same boundary form
      // the monolithic path uses, but run the scheduler itself on CPU/WASM.
      const noisePredData = noisePredTensor.data as Float32Array;
      writeGpu(gpuDevice, gpuNoisePred, noisePredData);

      // Scheduler Euler step on CPU/WASM. This avoids keeping a persistent
      // WebGPU session alive across the shard create/release churn.
      const schedResults = await schedSess.run({
        noise_pred: new ort.Tensor("float32", noisePredData, noiseDims),
        latents: new ort.Tensor("float32", cachedHiddenData, hiddenDims),
        step_info: new ort.Tensor("float32", Float32Array.from([step, numSteps]), [2]),
      });
      const nextLatents = schedResults["latents_out"];
      if (!nextLatents) throw new Error('scheduler did not produce "latents_out"');
      cachedHiddenData = new Float32Array(nextLatents.data as Float32Array);
      nextLatents.dispose();

      const stepMs = performance.now() - tStep;
      log(`  [zimage-sharded] step ${stepDisplay} total (${stepMs.toFixed(0)} ms)`);
      cb.advance();
      cb.checkAborted();
    }
  } catch (err) {
    await drainCreateQueue();
    await drainDataQueue();
    throw err;
  }
  log(`[zimage-sharded] DiT loop total: ${((performance.now() - tLoop) / 1000).toFixed(1)} s`);

  const latent = cachedHiddenData;
  await drainCreateQueue();
  await schedSess.release();
  log("[zimage-sharded] scheduler released");
  return latent;
}

async function run(input: GenerateInput, cb: GenerateCallbacks): Promise<ImageData> {
  if (input.set.family !== "zimage") {
    throw new Error(`generateZImage: expected zimage model, got ${input.set.family}`);
  }
  const isSharded = "transformerShards" in input.set;
  const set = input.set as ZImageModelSet | ZImageShardedModelSet;
  const cache = input.cache;
  const { width, height, prompt } = input;
  const numSteps = input.steps;
  const latentH = height / 8;
  const latentW = width / 8;
  const latentLen = LATENT_CHANNELS * 1 * latentH * latentW;
  const rng = mulberry32(input.seed);
  const log = cb.log;
  log(`[zimage] seed=${input.seed} size=${width}x${height} steps=${numSteps}`);

  // ----- 1. Tokenize with Qwen2 tokenizer -----
  cb.status("loading tokenizer...");
  const tokenizer = await loadTokenizer(cache, set);
  const { inputIds, attentionMask, sequenceLength, usedTemplate } = tokenizePrompt(
    tokenizer,
    prompt,
  );
  log(`[zimage] tokenized: ${sequenceLength} tokens (${usedTemplate ? "chat template" : "manual fallback"})`);
  cb.advance();
  cb.checkAborted();

  // ----- 2. Text encoder (load, run, release) -----
  cb.status("loading text encoder (~2.2 GB)...");
  const tEncLoad = performance.now();
  const teSess = await createZImageSession(cache, set.textEncoder, defaultProviders());
  log(`[zimage] text encoder loaded in ${(performance.now() - tEncLoad).toFixed(1)} ms`);

  cb.status("encoding prompt...");
  const tEncRun = performance.now();
  const teFeeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor("int64", inputIds, [1, sequenceLength]),
    attention_mask: new ort.Tensor("int64", attentionMask, [1, sequenceLength]),
  };
  const teResults = await teSess.run(teFeeds);
  // Output: encoder_hidden_state [B, seq_len, 2560]
  const hiddenStateKey =
    teResults["encoder_hidden_state"] ? "encoder_hidden_state" :
    teResults["encoder_hidden_states"] ? "encoder_hidden_states" :
    teSess.outputNames[0];
  const teOutTensor = teResults[hiddenStateKey];
  const teOutData = teOutTensor.data;

  // Safe readback: handle Float16Array, Uint16Array (fp16 polyfill), or Float32Array
  let encoderHiddenState: Float32Array;
  if (teOutData.constructor.name === "Float16Array" || teOutData instanceof Float32Array) {
    // Float16Array[i] returns decoded JS numbers; new Float32Array(it) copies correctly.
    // Float32Array -> Float32Array is a plain copy.
    encoderHiddenState = new Float32Array(teOutData as unknown as ArrayLike<number>);
  } else if (teOutData instanceof Uint16Array) {
    // fp16 polyfill: raw bit patterns, need manual decode
    encoderHiddenState = new Float32Array(teOutData.length);
    const view = new DataView(teOutData.buffer, teOutData.byteOffset, teOutData.byteLength);
    for (let i = 0; i < teOutData.length; i++) {
      const bits = view.getUint16(i * 2, true);
      const sign = (bits >> 15) & 1;
      const exp = (bits >> 10) & 0x1f;
      const frac = bits & 0x3ff;
      let val: number;
      if (exp === 0) val = (frac / 1024) * Math.pow(2, -14);
      else if (exp === 31) val = frac === 0 ? Infinity : NaN;
      else val = Math.pow(2, exp - 15) * (1 + frac / 1024);
      encoderHiddenState[i] = sign ? -val : val;
    }
    log(`[zimage] WARNING: text encoder returned Uint16Array (fp16 polyfill), decoded manually`);
  } else {
    // Unknown type, try direct copy
    encoderHiddenState = new Float32Array(teOutData as unknown as ArrayLike<number>);
    log(`[zimage] WARNING: text encoder returned unexpected data type ${teOutData.constructor.name}`);
  }

  log(`[zimage] text encoder ran in ${(performance.now() - tEncRun).toFixed(1)} ms`);

  cb.status("releasing text encoder...");
  await teSess.release();
  log("[zimage] text encoder released");
  cb.advance();
  cb.checkAborted();

  // ----- 3. Transformer + scheduler step (N-step DiT loop) -----
  let latent: Float32Array;
  if (isSharded) {
    latent = await runShardedTransformerLoop(
      set as ZImageShardedModelSet, cache, numSteps,
      gaussianNoise(latentLen, rng), encoderHiddenState,
      latentH, latentW, latentLen, sequenceLength, cb, log,
    );
  } else {
    latent = await runMonolithicTransformerLoop(
      set as ZImageModelSet, cache, numSteps,
      gaussianNoise(latentLen, rng), encoderHiddenState,
      latentH, latentW, latentLen, sequenceLength, cb, log,
    );
  }

  // ----- 4. VAE pre-process + decode (load, run, release) -----
  cb.status("loading VAE...");
  const tVaeLoad = performance.now();
  const vaePreSess = await createZImageSession(cache, set.vaePreProcess, defaultProviders());
  const vaeSess = await createZImageSession(cache, set.vaeDecoder, defaultProviders());
  log(`[zimage] VAE loaded in ${(performance.now() - tVaeLoad).toFixed(1)} ms`);
  cb.advance();
  cb.checkAborted();

  cb.status("decoding latent...");
  const tDecode = performance.now();

  // VAE pre-process: [B, 16, 1, H/8, W/8] -> [B, 16, H/8, W/8] (squeeze + scale)
  const preResults = await vaePreSess.run({
    latents: new ort.Tensor("float32", new Float32Array(latent), [1, LATENT_CHANNELS, 1, latentH, latentW]),
  });
  const scaledKey = preResults["scaled_latents"] ? "scaled_latents" : vaePreSess.outputNames[0];
  const scaledLatent = new Float32Array(preResults[scaledKey].data as Float32Array);

  // VAE decode: [B, 16, H/8, W/8] -> [B, 3, H, W]
  const vaeResults = await vaeSess.run({
    latent_sample: new ort.Tensor("float32", scaledLatent, [1, LATENT_CHANNELS, latentH, latentW]),
  });
  const sampleKey = vaeResults["sample"] ? "sample" : vaeSess.outputNames[0];
  const pixels = new Float32Array(vaeResults[sampleKey].data as Float32Array);
  log(`[zimage] VAE decode ran in ${(performance.now() - tDecode).toFixed(1)} ms`);

  cb.status("releasing VAE...");
  await vaePreSess.release();
  await vaeSess.release();
  log("[zimage] VAE released");

  // ----- 5. Convert pixels to ImageData -----
  // VAE output is NCHW [1, 3, H, W] in [-1, 1]. Map to [0, 255].
  const channelSize = height * width;
  const rgba = new Uint8ClampedArray(channelSize * 4);
  for (let j = 0; j < channelSize; j++) {
    const r = pixels[j];
    const g = pixels[j + channelSize];
    const b = pixels[j + 2 * channelSize];
    rgba[j * 4 + 0] = (r / 2 + 0.5) * 255;
    rgba[j * 4 + 1] = (g / 2 + 0.5) * 255;
    rgba[j * 4 + 2] = (b / 2 + 0.5) * 255;
    rgba[j * 4 + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

// ---- Tokenizer helpers ----

async function loadTokenizer(
  cache: ModelCache,
  set: ZImageModelSet | ZImageShardedModelSet,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Promise<any> {
  if (cachedTokenizer) return cachedTokenizer;

  // Load tokenizer files from OPFS cache into memory.
  const tokenizerJson = await cache.loadFileText(set.tokenizer.vocab);
  const tokenizerConfigJson = await cache.loadFileText(set.tokenizer.config);

  // Build a map of filename -> content for the cache adapter.
  const fileMap = new Map<string, string>();
  fileMap.set("tokenizer.json", tokenizerJson);
  fileMap.set("tokenizer_config.json", tokenizerConfigJson);

  // Construct the tokenizer directly from the cached JSON. We can't use
  // AutoTokenizer.from_pretrained because transformers.js captures fetch
  // at module load time and we can't intercept it reliably (the module
  // may already be loaded by Janus). Direct construction works; we just
  // need to set the chat_template manually since tokenizer_config.json
  // doesn't include one.
  const transformers = await import("@huggingface/transformers");
  const tokJson = JSON.parse(tokenizerJson);
  const tokConfig = JSON.parse(tokenizerConfigJson);

  const TokClass =
    (transformers as Record<string, unknown>)["Qwen2Tokenizer"] ??
    (transformers as Record<string, unknown>)["PreTrainedTokenizer"];
  if (!TokClass || typeof TokClass !== "function") {
    throw new Error("transformers.js does not export Qwen2Tokenizer or PreTrainedTokenizer");
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tokenizer = new (TokClass as any)(tokJson, tokConfig);

  // Chat template matching the MS reference demo (xenova AutoTokenizer
  // default for Qwen2): no auto-injected system prompt. Verified by
  // diffing tokenized input_ids against MS's pipeline for the same
  // prompt - the auto-injected "You are a helpful assistant" prefix
  // was conditioning the transformer on extra tokens MS never sends,
  // which made the model invent decorative marks not in the prompt.
  tokenizer.chat_template =
    `{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}`;

  cachedTokenizer = tokenizer;
  return tokenizer;
}

function tokenizePrompt(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tokenizer: any,
  prompt: string,
  maxLength = 512,
): { inputIds: BigInt64Array; attentionMask: BigInt64Array; sequenceLength: number; usedTemplate: boolean } {
  let formatted: string;
  let usedTemplate = false;
  try {
    const messages = [{ role: "user", content: prompt }];
    formatted = tokenizer.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: true,
    });
    usedTemplate = true;
  } catch {
    // Mirrors the chat_template above (no system prompt, no <think>)
    // so a fallback path produces the same tokens MS's pipeline does.
    formatted =
      `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
  }

  const inputs = tokenizer([formatted], {
    padding: false,
    max_length: maxLength,
    truncation: true,
    return_tensor: false,
  });

  const ids: number[] = inputs.input_ids[0];
  const mask: number[] = inputs.attention_mask[0];
  const inputIds = new BigInt64Array(ids.map(BigInt));
  const attentionMask = new BigInt64Array(mask.map(BigInt));

  return { inputIds, attentionMask, sequenceLength: ids.length, usedTemplate };
}

export const zimageGenerateFn: GenerateFn = { estimate, run };
