// Janus-Pro-1B model bundle for the image-gen tool.
//
// Janus is a fundamentally different beast from sd15 / sdxl: it is an
// autoregressive multimodal LM, not a diffusion model. There is no UNet,
// no scheduler, no CFG-as-doubled-batch in our code - the inference loop
// is a 24-layer transformer that generates 576 image tokens sequentially,
// each token feeding back into the next step's KV cache, then a final
// image_decode pass turns the token stream into a raster image.
//
// We do NOT hand-roll that loop. transformers.js (@huggingface/transformers)
// already implements MultiModalityCausalLM.generate_images correctly,
// including the LLaMA-style BPE tokenizer with the Janus chat template,
// per-component KV cache plumbing, sampling, and CFG batching. We import
// it as a dependency and use it for Janus only.
//
// Cost of the dep: it ships as a pre-bundled single file (transformers.web.js,
// ~432 KB minified) with its own ORT-web inlined. It is NOT tree-shakeable
// and our `onnxruntime-web` dep cannot be deduped against it via Vite alias
// because their ORT is bundled-in, not an external import. Net result:
// Janus runs on transformers.js's bundled ORT instance, sd15/sdxl run on
// our own. The two coexist cleanly because they are partitioned by family.
//
// What this file declares: the list of files transformers.js will request
// from HuggingFace for the q4f16 variant of Janus-Pro-1B-ONNX. Our existing
// ModelCache (shared with sd15/sdxl in the same OPFS dir) downloads them
// via the same model-manager UI everything else uses. At generation time,
// `web/src/janus/cache-adapter.ts` exposes those cached bytes to
// transformers.js via env.customCache so transformers.js never touches the
// network and our "cached" badge / clear-all / byte-progress UX all keep
// working unchanged.

import type { ModelFile } from "../shared/model-cache";
import { patchTransform, type Patch } from "../shared/model-patch";
import languageModelSplit from "./patches/language_model_q4f16.split.json";

const HF = "https://huggingface.co";
const JANUS_REPO_ID = "onnx-community/Janus-Pro-1B-ONNX";
// Pin to a specific commit so the byte-level split patch stays valid.
const JANUS_REV = "04efdf2e36cb07a034b0d94f7322356b292f0418";
const JANUS_BASE = `${HF}/${JANUS_REPO_ID}/resolve/${JANUS_REV}`;

// -----------------------------------------------------------------------------
// Tokenizer + processor + config files. transformers.js fetches all of these
// during AutoProcessor.from_pretrained() and MultiModalityCausalLM.from_pretrained().
// Sizes from the HF tree listing 2026-04-08.
// -----------------------------------------------------------------------------

const JANUS_TOKENIZER_JSON: ModelFile = {
  id: "janus_pro_1b_tokenizer_json",
  name: "Janus-Pro-1B tokenizer.json",
  url: `${JANUS_BASE}/tokenizer.json`,
  sizeBytes: 4_720_000,
};

const JANUS_TOKENIZER_CONFIG: ModelFile = {
  id: "janus_pro_1b_tokenizer_config",
  name: "Janus-Pro-1B tokenizer_config.json",
  url: `${JANUS_BASE}/tokenizer_config.json`,
  sizeBytes: 1_360,
};

const JANUS_SPECIAL_TOKENS: ModelFile = {
  id: "janus_pro_1b_special_tokens_map",
  name: "Janus-Pro-1B special_tokens_map.json",
  url: `${JANUS_BASE}/special_tokens_map.json`,
  sizeBytes: 344,
};

const JANUS_CONFIG: ModelFile = {
  id: "janus_pro_1b_config",
  name: "Janus-Pro-1B config.json",
  url: `${JANUS_BASE}/config.json`,
  sizeBytes: 1_890,
};

const JANUS_PROCESSOR_CONFIG: ModelFile = {
  id: "janus_pro_1b_processor_config",
  name: "Janus-Pro-1B processor_config.json",
  url: `${JANUS_BASE}/processor_config.json`,
  sizeBytes: 288,
};

const JANUS_PREPROCESSOR_CONFIG: ModelFile = {
  id: "janus_pro_1b_preprocessor_config",
  name: "Janus-Pro-1B preprocessor_config.json",
  url: `${JANUS_BASE}/preprocessor_config.json`,
  sizeBytes: 346,
};

const JANUS_GENERATION_CONFIG: ModelFile = {
  id: "janus_pro_1b_generation_config",
  name: "Janus-Pro-1B generation_config.json",
  url: `${JANUS_BASE}/generation_config.json`,
  sizeBytes: 167,
};

// -----------------------------------------------------------------------------
// ONNX components, q4f16 variant. Picked because:
//   - Smallest viable bundle (~1.34 GB) of any Janus dtype
//   - Worklog 2026-04-08 smoke proved q4f16 weight decompression kernels work
//     in stock ORT-web 1.24.3 WebGPU (Z-Image-Turbo finding)
//   - All files except language_model are monolithic (no .onnx_data sidecars)
//   - language_model (698 MB) is split into a ~77 MB graph + ~621 MB
//     external-data sidecar during download via a pre-computed patch, so
//     ORT-web allocates weights as many small GPU buffers instead of one
//     monolithic 698 MB allocation (which crashes mobile Chrome)
//
// Why six components and not the diffusion-style three (text encoder / unet /
// vae): Janus's autoregressive generation path is
//   embed_tokens(prompt) -> language_model.prefill -> loop {
//     gen_head(hidden) -> sample image_token -> gen_img_embeds(token) ->
//     language_model.step(embed) -> next hidden
//   } x 576 -> image_decode(all_576_tokens) -> RGB
// lm_head is the text-side head used for the language path; we still need it
// in the bundle because transformers.js's MultiModalityCausalLM loads it
// alongside the image-gen heads even when only generate_images is called.
// -----------------------------------------------------------------------------

// prepare_inputs_embeds: a 596 MB component that fuses text and image
// embeddings before the language_model prefill. transformers.js requests
// this on first run; we missed it from the README example because the
// example only shows the high-level generate_images call. The cache
// adapter's "undeclared URL" warning surfaced it.
const JANUS_PREPARE_INPUTS_EMBEDS: ModelFile = {
  id: "janus_pro_1b_prepare_inputs_embeds_q4f16",
  name: "Janus-Pro-1B prepare_inputs_embeds (q4f16)",
  url: `${JANUS_BASE}/onnx/prepare_inputs_embeds_q4f16.onnx`,
  sizeBytes: 596_699_315,
};

const JANUS_EMBED_TOKENS: ModelFile = {
  id: "janus_pro_1b_embed_tokens_q4f16",
  name: "Janus-Pro-1B embed_tokens (q4f16)",
  url: `${JANUS_BASE}/onnx/embed_tokens_q4f16.onnx`,
  sizeBytes: 419_000_000,
};

const JANUS_LANGUAGE_MODEL: ModelFile = {
  id: "janus_pro_1b_language_model_q4f16",
  name: "Janus-Pro-1B language_model (q4f16)",
  url: `${JANUS_BASE}/onnx/language_model_q4f16.onnx`,
  sizeBytes: 698_000_000,
  transform: patchTransform(languageModelSplit as Patch),
};

const JANUS_LM_HEAD: ModelFile = {
  id: "janus_pro_1b_lm_head_q4f16",
  name: "Janus-Pro-1B lm_head (q4f16)",
  url: `${JANUS_BASE}/onnx/lm_head_q4f16.onnx`,
  sizeBytes: 118_000_000,
};

// gen_head: fp16 variant, NOT q4f16. The q4f16 export of gen_head has an
// fp32 input boundary on its `hidden_states` input, but Janus's
// language_model outputs fp16 hidden states (likely because language_model
// has fp16 boundaries even though it's q4f16-quantized). transformers.js
// does not auto-cast between sessions, so feeding gen_head_q4f16 erros at
// runtime with "Unexpected input data type. Actual: float16, expected:
// float32". Switching gen_head to fp16 puts both ends of the boundary at
// the same dtype. Cost: 75.5 MB instead of 21.3 MB (+54 MB to the bundle),
// which keeps total well under the mobile-feasible threshold.
const JANUS_GEN_HEAD: ModelFile = {
  id: "janus_pro_1b_gen_head_fp16",
  name: "Janus-Pro-1B gen_head (fp16)",
  url: `${JANUS_BASE}/onnx/gen_head_fp16.onnx`,
  sizeBytes: 75_536_078,
};

const JANUS_GEN_IMG_EMBEDS: ModelFile = {
  id: "janus_pro_1b_gen_img_embeds_q4f16",
  name: "Janus-Pro-1B gen_img_embeds (q4f16)",
  url: `${JANUS_BASE}/onnx/gen_img_embeds_q4f16.onnx`,
  sizeBytes: 2_670_000,
};

const JANUS_IMAGE_DECODE: ModelFile = {
  id: "janus_pro_1b_image_decode_q4f16",
  name: "Janus-Pro-1B image_decode (q4f16)",
  url: `${JANUS_BASE}/onnx/image_decode_q4f16.onnx`,
  sizeBytes: 85_300_000,
};

/** Every file the Janus pipeline needs in OPFS before generation can run.
 *  Order does not matter for the cache layer; the model manager downloads
 *  them in array order so users see the small config files complete first
 *  and the big language_model last. */
export const JANUS_PRO_1B_FILES: ModelFile[] = [
  JANUS_TOKENIZER_JSON,
  JANUS_TOKENIZER_CONFIG,
  JANUS_SPECIAL_TOKENS,
  JANUS_CONFIG,
  JANUS_PROCESSOR_CONFIG,
  JANUS_PREPROCESSOR_CONFIG,
  JANUS_GENERATION_CONFIG,
  JANUS_GEN_IMG_EMBEDS,
  JANUS_GEN_HEAD,
  JANUS_IMAGE_DECODE,
  JANUS_LM_HEAD,
  JANUS_EMBED_TOKENS,
  JANUS_PREPARE_INPUTS_EMBEDS,
  JANUS_LANGUAGE_MODEL,
];

/** OPFS cache key for the language_model external-data sidecar, produced by
 *  the split transform during download. The cache adapter needs this to serve
 *  the sidecar when transformers.js requests it. */
export const JANUS_LANGUAGE_MODEL_SIDECAR_ID =
  (languageModelSplit as Patch).sidecar!.fileId;

/** HF model id passed to transformers.js .from_pretrained(). */
export const JANUS_PRO_1B_MODEL_ID = JANUS_REPO_ID;

/** Number of image tokens generated per image. Janus-Pro produces a 24x24
 *  grid of image tokens that the image_decode head expands to a 384x384
 *  RGB image. This number is fixed by the model architecture; the user
 *  cannot change it via a "steps" slider the way they can with diffusion. */
export const JANUS_NUM_IMAGE_TOKENS = 576;

/** Native output resolution. Janus has no resolution knob - the image
 *  decoder always produces 384x384. */
export const JANUS_NATIVE_RESOLUTION = 384;
