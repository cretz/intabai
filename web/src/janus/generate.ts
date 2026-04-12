// Janus-Pro-1B image generation pipeline.
//
// Implemented as a thin adapter over @huggingface/transformers'
// MultiModalityCausalLM.generate_images. We do NOT hand-roll the
// autoregressive loop, the LLaMA tokenizer, the KV cache, or the CFG
// batching - that is exactly the gnarly code transformers.js already gets
// right, and the cost of pulling it in (a separate bundled ORT instance,
// ~432 KB minified) is acceptable for one model. See web/src/janus/models.ts
// for the rationale and web/src/janus/cache-adapter.ts for how we route the
// library's file loads through our existing OPFS cache.
//
// What we own here:
//   1. Wiring transformers.js's `env.customCache` to a JanusCacheAdapter so
//      every byte flows from OPFS, no network.
//   2. Translating the GenerateInput shape (prompt, cfg, seed) into the
//      conversation + generate_images options that transformers.js wants.
//   3. A BaseStreamer subclass that maps each generated image token to one
//      `cb.advance()` call, so the existing image-gen progress bar / ETA /
//      stats line all work without any UI changes.
//   4. A StoppingCriteria subclass that returns "stop" when the user hits
//      the cancel button.
//   5. Converting the resulting RawImage (RGB Uint8) to an ImageData (RGBA
//      Uint8ClampedArray) for the canvas put.
//
// What we explicitly do NOT support yet:
//   - img2img / reference image (Janus's image-conditioning path is a
//     separate code path in transformers.js and we have not wired it up;
//     ModelSet.capabilities.img2img is false)
//   - Custom width/height (the image_decode head is hard-wired to 384x384;
//     the input width/height are ignored)
//   - Custom step count (574 image tokens is fixed by the architecture;
//     we pass numImageTokens through to min/max_new_tokens)

import {
  AutoProcessor,
  MultiModalityCausalLM,
  BaseStreamer,
  StoppingCriteria,
  Tensor,
  env,
  type RawImage,
} from "@huggingface/transformers";

import type {
  GenerateCallbacks,
  GenerateFn,
  GenerateInput,
  PipelineEstimate,
} from "../image-gen/generate-types";
import { JanusCacheAdapter } from "./cache-adapter";
import { JANUS_LANGUAGE_MODEL_SIDECAR_ID, JANUS_PRO_1B_FILES } from "./models";

class CancelledError extends Error {
  constructor() {
    super("cancelled");
  }
}

/** BaseStreamer subclass that fires `cb.advance()` once per generated image
 *  token. Janus produces 576 tokens per image (one per 16x16 patch of the
 *  24x24 latent grid), so the progress bar advances 576 times during the
 *  decode loop and once more for the final image_decode pass. */
class JanusProgressStreamer extends BaseStreamer {
  private tokensSoFar = 0;
  private readonly start = performance.now();

  constructor(
    private readonly cb: GenerateCallbacks,
    private readonly numImageTokens: number,
  ) {
    super();
  }

  // transformers.js calls put() with `bigint[][]` of shape [batch, k] each
  // step. With Janus's CFG batching that batch is 2 (cond + uncond) and k=1
  // (one new token per step). We do not look at the values - we just count
  // calls.
  put(_value: bigint[][]): void {
    this.tokensSoFar++;
    this.cb.advance();
    // Avoid spamming the stats line every single step on the slow path -
    // every 8 tokens is enough resolution for "X / 576, ETA Ys".
    if (this.tokensSoFar % 8 === 0 || this.tokensSoFar === this.numImageTokens) {
      const elapsedMs = performance.now() - this.start;
      const avgMs = elapsedMs / this.tokensSoFar;
      const remaining = this.numImageTokens - this.tokensSoFar;
      const etaMs = remaining * avgMs;
      this.cb.stats(
        `${this.tokensSoFar} / ${this.numImageTokens} tokens - avg ${avgMs.toFixed(0)} ms/token - ETA ${(etaMs / 1000).toFixed(0)}s`,
      );
      this.cb.stepProgress(this.tokensSoFar, this.numImageTokens, etaMs / 1000);
    }
    // Throwing inside a streamer would be swallowed by transformers.js, so
    // cancellation is handled by the StoppingCriteria below instead.
  }

  end(): void {
    // No-op. The image_decode + RawImage->ImageData advance happens in
    // run() after generate_images returns.
  }
}

/** StoppingCriteria subclass that flips to "stop" when the user clicks
 *  cancel. transformers.js polls this between every decode step, so it
 *  takes effect within one token-generation cycle (~hundreds of ms on a
 *  desktop iGPU, faster on discrete GPU). */
class CancelStoppingCriteria extends StoppingCriteria {
  constructor(private readonly cb: GenerateCallbacks) {
    super();
  }

  // The base type is `(input_ids: number[][], scores: number[][]) => boolean[]`.
  // The boolean[] return shape is "one entry per batch sequence". Janus runs
  // a batch of 2 for CFG so we return 2 trues on cancel.
  _call(input_ids: number[][], _scores: number[][]): boolean[] {
    let cancelled = false;
    try {
      this.cb.checkAborted();
    } catch {
      cancelled = true;
    }
    return Array.from({ length: input_ids.length }, () => cancelled);
  }
}

// Cached across runs: we only build the processor + model once per page
// load. The first generate() pays the cold-start cost (~3-5s on desktop
// for the small components, dominated by the language_model session create
// at ~700 MB), every subsequent run reuses the resident sessions. We do
// NOT release between runs because Janus's KV cache and the gen_head /
// gen_img_embeds sessions are tiny enough to keep resident, and the
// language_model release-and-reload cost would dominate the per-run budget.
// (This is the "release between phases pattern does not apply to Janus"
// rule from the worklog.)
let cachedProcessor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>> | null = null;
let cachedModel: MultiModalityCausalLM | null = null;
let cachedModelId: string | null = null;

async function loadOnce(
  hfModelId: string,
  cb: GenerateCallbacks,
): Promise<{
  processor: NonNullable<typeof cachedProcessor>;
  model: MultiModalityCausalLM;
}> {
  if (cachedProcessor && cachedModel && cachedModelId === hfModelId) {
    return { processor: cachedProcessor, model: cachedModel };
  }
  cb.status("loading processor...");
  const processor = await AutoProcessor.from_pretrained(hfModelId);
  cb.status("loading model...");
  //
  // Per-component dtype map: most components use q4f16 (smallest viable
  // bundle, ~1.93 GB total). gen_head is forced to fp16 because its
  // q4f16 export has an fp32 input boundary on `hidden_states` while
  // language_model outputs fp16, and transformers.js does not cast
  // between sessions. fp16 gen_head is 75 MB vs 21 MB for q4f16 - a
  // 54 MB cost. If a NEW dtype mismatch surfaces on a different component
  // (lm_head, gen_img_embeds, etc.), apply the same pattern: swap that
  // one component's URL in JANUS_PRO_1B_FILES and update the map below.
  //
  // Per-component device map: CRITICAL. When you pass a per-component
  // dtype map to transformers.js, it stops honoring the top-level
  // `device` option and defaults each component's device per its own
  // (undocumented) heuristic - which lands at least one session on wasm.
  // ORT-web's wasm session.run is SYNCHRONOUS and blocks the main thread
  // for the entire forward pass; with a 698 MB language_model that means
  // Chrome's tab becomes unresponsive (the 2026-04-08 mid-stream wedge
  // that locked the browser). WebGPU session.run is async and keeps the
  // UI thread free. Forcing every component onto webgpu explicitly is
  // the only reliable way to combine a per-component dtype map with
  // async execution. The earlier vision_tower Conv-zero WebGPU bug only
  // fires when pixel_values is in the inputs - and the Janus processor
  // with chat_template "text_to_image" does NOT emit pixel_values (the
  // actual keys are input_ids, attention_mask, images_seq_mask,
  // images_emb_mask) - so the vision tower never runs and the
  // dispatch-zero bug never triggers.
  //
  // session_options.graphOptimizationLevel = 'disabled': ORT-web's graph
  // optimizer crashes inside SimplifiedLayerNormFusion on Janus's
  // language_model.onnx with "Attempting to get index by a name which
  // does not exist: InsertedPrecisionFreeCast_/vision_model/.../
  // norm1/Constant_output_0". Same class of bug the worklog already
  // documented for segmind-vega's text encoders - an earlier optimizer
  // pass removes a Constant node that a later pass references.
  // Disabling all graph opts is a blunt fix but ORT-web does not expose
  // per-pass disabling at the JS level.
  const dtypeMap: Record<string, string> = {
    prepare_inputs_embeds: "q4f16",
    embed_tokens: "q4f16",
    language_model: "q4f16",
    lm_head: "q4f16",
    gen_head: "fp16",
    gen_img_embeds: "q4f16",
    image_decode: "q4f16",
  };
  const deviceMap: Record<string, string> = {
    prepare_inputs_embeds: "webgpu",
    embed_tokens: "webgpu",
    language_model: "webgpu",
    lm_head: "webgpu",
    gen_head: "webgpu",
    gen_img_embeds: "webgpu",
    image_decode: "webgpu",
  };
  const model = (await MultiModalityCausalLM.from_pretrained(hfModelId, {
    dtype: dtypeMap as unknown as "q4f16",
    device: deviceMap as unknown as "webgpu",
    // language_model was split into graph + external-data sidecar during
    // download (see split transform in janus/models.ts). Tell transformers.js
    // to fetch the .onnx_data sidecar and pass it as externalData to ORT.
    use_external_data_format: { language_model: true } as unknown as boolean,
    session_options: {
      graphOptimizationLevel: "disabled",
    },
  })) as MultiModalityCausalLM;
  cachedProcessor = processor;
  cachedModel = model;
  cachedModelId = hfModelId;
  return { processor, model };
}

export const janusGenerateFn: GenerateFn = {
  estimate(input: GenerateInput): PipelineEstimate {
    if (input.set.family !== "janus") {
      throw new Error(`janusGenerateFn called with non-janus family: ${input.set.family}`);
    }
    // Progress units = N image tokens (one advance() per streamer.put) plus
    // 2 fixed phases (model load on cold start, image_decode + RawImage
    // post-processing). The cold-start unit is also counted on warm runs
    // because the UI prefers a stable totalUnits over a more accurate but
    // model-state-dependent number.
    return { totalUnits: input.set.numImageTokens };
  },

  async run(input: GenerateInput, cb: GenerateCallbacks): Promise<ImageData> {
    if (input.set.family !== "janus") {
      throw new Error(`janusGenerateFn called with non-janus family: ${input.set.family}`);
    }
    const set = input.set;

    // Wire transformers.js's customCache to our OPFS adapter. This must
    // happen before from_pretrained() so the very first file fetch hits
    // our cache. Setting env.* is global mutation but we own the only
    // transformers.js consumer in the project.
    env.useCustomCache = true;
    const cacheAdapter = new JanusCacheAdapter(input.cache, JANUS_PRO_1B_FILES);
    cacheAdapter.addSidecar("onnx/language_model_q4f16.onnx_data", JANUS_LANGUAGE_MODEL_SIDECAR_ID);
    env.customCache = cacheAdapter;
    // Stop the library from trying to also fetch from local file paths
    // (it would warn about node-only paths in the browser otherwise).
    env.allowLocalModels = false;
    env.allowRemoteModels = true;

    cb.checkAborted();

    const { processor, model } = await loadOnce(set.hfModelId, cb);
    cb.checkAborted();

    cb.status("preparing inputs...");
    // The chat template comes from tokenizer_config.json. The
    // 'text_to_image' template is the one Janus's processor exposes for
    // image generation; it injects the right special tokens around the
    // user prompt to switch the LM into image-token-generation mode.
    const conversation = [
      {
        role: "<|User|>",
        content: input.prompt,
      },
    ];
    // The processor returns a dict of tensors keyed by what
    // generate_images expects (input_ids, attention_mask, etc).
    //
    // chat_template "text_to_image" is a named template defined in
    // tokenizer_config.json. CRITICAL: it only emits the
    // `<begin_of_image>` special token when `add_generation_prompt: true`
    // is passed. Without that token the model has no cue to start
    // generating image tokens and produces structureless gradient noise.
    // The README example is missing this option but it appears to be
    // load-bearing for v4.x of @huggingface/transformers.
    const procInputs = await (
      processor as unknown as {
        (
          conversation: Array<{ role: string; content: string }>,
          options: { chat_template: string; add_generation_prompt: boolean },
        ): Promise<Record<string, unknown>>;
      }
    )(conversation, {
      chat_template: "text_to_image",
      add_generation_prompt: true,
    });
    cb.checkAborted();

    // Inject non-empty dummy `pixel_values` so the SigLIP vision tower
    // (baked into language_model.onnx) does not crash on a zero-batch
    // Conv. Background:
    //
    // 1. The processor with chat_template "text_to_image" emits
    //    images_emb_mask with dims [1, 0, 576] - middle 0 means "0 images
    //    per sample, no image embeddings to fuse". transformers.js's
    //    MultiModalityCausalLM.forward() then synthesizes pixel_values
    //    with shape [0, 0, 3, 384, 384] (batch=0, num_images=0). The
    //    leading zero dimension flows into SigLIP's first Conv and
    //    ORT-web's WebGPU EP throws "Invalid dispatch group size
    //    (W, H, 0)" because it cannot dispatch a workgroup grid with a
    //    zero axis. Wasm has the same problem AND additionally pegs the
    //    main thread on synchronous session.run, hanging Chrome.
    //
    // 2. The workaround: replace ONLY pixel_values with a non-empty
    //    [1, 1, 3, 384, 384] of zeros (one black image, one batch).
    //    The vision tower runs harmlessly on the dummy. CRITICALLY we
    //    leave images_emb_mask AS-IS at the processor's [1, 0, 576] -
    //    that mask is what tells `prepare_inputs_embeds` "0 images
    //    present, ignore the vision tower's output entirely". An earlier
    //    iteration of this workaround rewrote the mask to [1, 1, 576] of
    //    false ("1 image present but every embedding token masked out")
    //    and the result was gradient noise: the model was running in
    //    image-conditioning mode and `prepare_inputs_embeds` was fusing
    //    the SigLIP output of the zero image into the sequence,
    //    drowning out the prompt. The right answer is to keep the mask
    //    at "no images" and let the vision tower's wasted output be
    //    ignored at the fusion step.
    //
    // images_seq_mask also stays as the processor produced it ([1, 40]
    // of zeros = "no image positions in the input sequence").
    //
    // Note: input_ids gets doubled to [2, 40] inside forward() for CFG
    // batching, but the image-related tensors stay batch=1. This
    // batching mismatch is something MultiModalityCausalLM handles
    // internally and is not our concern.
    //
    // Tensor shapes from the 2026-04-08 ground-truth diagnostic:
    //   processor procInputs.images_emb_mask = [1, 0, 576] bool
    //   forward() synthesized pixel_values  = [0, 0, 3, 384, 384] float32
    const SIGLIP_SIZE = 384;
    const dummyPixelValues = new Tensor(
      "float32",
      new Float32Array(1 * 1 * 3 * SIGLIP_SIZE * SIGLIP_SIZE),
      [1, 1, 3, SIGLIP_SIZE, SIGLIP_SIZE],
    );
    const augmentedInputs: Record<string, unknown> = {
      ...procInputs,
      pixel_values: dummyPixelValues,
    };
    cb.status("generating image tokens...");
    const streamer = new JanusProgressStreamer(cb, set.numImageTokens);
    const stoppingCriteria = new CancelStoppingCriteria(cb);

    // generate_images returns RawImage[]. With do_sample: true the result
    // varies between runs; we forward the user-supplied seed by setting it
    // on transformers.js's global RNG (see env.seed below). guidance_scale
    // maps to Janus's CFG.
    // Note: the lib reads the global RNG seed via the `random` util it
    // exports. We import it indirectly through env to keep this file's
    // import surface small.
    let outputs: RawImage[];
    try {
      // The generate_images options type is declared narrowly (streamer
      // typed as TextStreamer) but the runtime accepts any BaseStreamer
      // and any StoppingCriteria subclass. Cast through unknown to bypass
      // the over-constrained type rather than subclass TextStreamer (which
      // would require importing a tokenizer just to satisfy the type).
      // Intentionally NOT passing guidance_scale: Janus's generation_config.json
      // does not specify one, the README example does not pass one, and
      // transformers.js's MultiModalityCausalLM.generate_images has its own
      // internal default for Janus that we should not second-guess.
      // Overriding with our own value (5.0 or 7.5 inherited from the SD1.5
      // CFG slider) produced gradient-stripe noise even when the prompt
      // was clearly reaching the model. Same logic for temperature/top_p:
      // generation_config.json sets them and transformers.js picks them
      // up automatically.
      outputs = await model.generate_images({
        ...augmentedInputs,
        min_new_tokens: set.numImageTokens,
        max_new_tokens: set.numImageTokens,
        do_sample: true,
        streamer,
        stopping_criteria: stoppingCriteria,
        // Seed: transformers.js threads this through to its sampling RNG
        // when do_sample is true. >>> 0 normalizes to a 32-bit unsigned int.
        seed: input.seed >>> 0,
      } as unknown as Parameters<MultiModalityCausalLM["generate_images"]>[0]);
    } catch (err) {
      // The cancel path comes through as a generic error from inside the
      // generate loop because StoppingCriteria does not raise. But if the
      // user already clicked cancel, surface it as the same CancelledError
      // the rest of the UI uses.
      try {
        cb.checkAborted();
      } catch {
        throw new CancelledError();
      }
      throw err;
    }

    // Even if the streamer fired the full count, double-check the cancel
    // flag in case the cancel landed during image_decode.
    cb.checkAborted();

    if (outputs.length === 0) {
      throw new Error("Janus generate_images returned no images");
    }
    const raw = outputs[0];
    // Convert RawImage (RGB or RGBA, Uint8) -> ImageData (RGBA,
    // Uint8ClampedArray). RawImage.rgba() converts in place and returns
    // the same instance.
    const rgba = raw.rgba();
    // Copy into a fresh Uint8ClampedArray<ArrayBuffer>. The RawImage's
    // backing buffer is typed as ArrayBufferLike (could be a SharedArrayBuffer
    // in some library configurations) which the ImageData constructor
    // refuses, so a copy is the simplest way to satisfy both the type
    // checker and the runtime.
    const data = new Uint8ClampedArray(rgba.data.length);
    data.set(rgba.data);
    cb.status("done");
    return new ImageData(data, rgba.width, rgba.height);
  },
};
