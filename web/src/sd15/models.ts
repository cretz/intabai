// Model bundles for the image-gen tool. Mirrors the ModelFile / ModelSet
// shape used by video-face-swap so the cache layer can stay structurally
// similar.
//
// A "model" from the user's POV is a ModelSet: a named bundle that must be
// fully downloaded before generation is allowed. Multiple ModelSets can
// reference the same ModelFile by id (for example: every SD1.5 finetune
// reuses the same CLIP tokenizer files), and the cache dedupes by id so
// the shared file is only downloaded once.
//
// ModelSet is a discriminated union over the architecture family. Each
// family carries the file pointers it needs (SD1.5 = single CLIP-L; SDXL =
// dual CLIP-L + CLIP-bigG with text_embeds + time_ids on the UNet) plus
// per-family defaults (native resolution, recommended steps/CFG).
//
// RAM policy (enforced by the per-family pipeline, not by this file): every
// network runs strictly sequentially, and each ORT session is created, run,
// then released before the next one is created. Peak GPU memory is
// max(component_size) rather than the sum. Bundle composition here is what
// the user downloads to disk; what is resident in GPU memory at any given
// moment is a separate concern.

import type { ModelFile } from "../shared/model-cache";
import type { OrtModelFile } from "./ort-helpers";
import { ortModelFiles } from "./ort-helpers";
import { patchTransform, type Patch } from "../shared/model-patch";
import sdxlTurboUnetSplit from "../sdxl/patches/sdxl-turbo-unet-q4f16.split.json";
import {
  JANUS_PRO_1B_FILES,
  JANUS_PRO_1B_MODEL_ID,
  JANUS_NATIVE_RESOLUTION,
  JANUS_NUM_IMAGE_TOKENS,
} from "../janus/models";

const HF = "https://huggingface.co";

export type { ModelFile };

/** What a given ModelSet supports. Drives UI feature gating. */
export interface ModelCapabilities {
  txt2img: boolean;
  img2img: boolean;
}

/** Per-model help text for the reference-image control. Different model
 *  families consume the reference image differently. */
export interface Img2ImgHelp {
  /** Short label shown next to the strength slider. */
  strengthLabel: string;
  /** One- or two-sentence explanation of how the reference image will
   *  affect this model's output. Shown in the reference-image fieldset. */
  description: string;
}

/** Per-model defaults + UI hint text. The image-gen UI applies these when
 *  the user picks a model so the steps/CFG/resolution sliders land on
 *  values that actually work for that family. */
export interface ModelDefaults {
  /** Default and native resolution. Used when the user has no saved
   *  setting AND when "reset to defaults" is clicked. */
  width: number;
  height: number;
  /** Mobile fallback resolution (smaller, fits the WebGPU memory budget
   *  on flagship phones). Both axes use this value. */
  mobileResolution: number;
  /** Maximum allowed dimension on either axis. */
  maxResolution: number;
  /** Default denoise step count. */
  steps: number;
  /** Default classifier-free guidance scale. */
  cfg: number;
  /** Help text shown next to the width/height inputs. */
  resolutionHelp: string;
  /** Help text shown next to the steps input. */
  stepsHelp: string;
  /** Help text shown next to the CFG input. */
  cfgHelp: string;
  /** If true, width/height are fixed by the model and inputs are disabled. */
  fixedResolution?: boolean;
  /** If true, step count is fixed by the model and the input is disabled. */
  fixedSteps?: boolean;
}

interface BaseModelInfo {
  id: string;
  name: string;
  description: string;
  capabilities: ModelCapabilities;
  defaults: ModelDefaults;
  /** Reference-image help text. Required when capabilities.img2img is true. */
  img2img?: Img2ImgHelp;
}

/** SD1.5 family: single CLIP-L text encoder, 4-channel latent UNet,
 *  vae scaling factor 0.18215. */
export interface Sd15ModelSet extends BaseModelInfo {
  family: "sd15";
  tokenizer: { vocab: ModelFile; merges: ModelFile };
  textEncoder: OrtModelFile;
  unet: OrtModelFile;
  vaeDecoder: OrtModelFile;
  vaeEncoder?: OrtModelFile;
}

/** SDXL family: dual CLIP-L + OpenCLIP-bigG text encoders producing a
 *  concatenated 2048-dim hidden state plus a 1280-dim pooled "text_embeds"
 *  output, UNet that additionally consumes text_embeds and a 6-element
 *  time_ids tensor, vae scaling factor 0.13025. */
export interface SdxlModelSet extends BaseModelInfo {
  family: "sdxl";
  /** Tokenizer for text_encoder (CLIP-L). */
  tokenizer: { vocab: ModelFile; merges: ModelFile };
  /** Tokenizer for text_encoder_2 (OpenCLIP-bigG). Vocab is byte-identical
   *  to CLIP-L's in every export I have looked at, but it lives in a
   *  separate folder in the source repo so we list it separately. */
  tokenizer2: { vocab: ModelFile; merges: ModelFile };
  textEncoder: OrtModelFile;
  textEncoder2: OrtModelFile;
  unet: OrtModelFile;
  vaeDecoder: OrtModelFile;
  vaeEncoder?: OrtModelFile;
  /** Boundary dtype for UNet / VAE inputs and outputs. Diffusers/optimum
   *  SDXL exports default to fp32 at the boundary (segmind-vega); q4f16
   *  exports like webnn/sdxl-turbo use fp16. Defaults to "float32". */
  boundaryDtype?: "float16" | "float32";
  /** Scheduler type. Standard SDXL uses DDIM; ADD/LCM-distilled models
   *  like SDXL-Turbo use Euler. Defaults to "ddim". */
  schedulerType?: "ddim" | "euler";
}

/** Janus family: autoregressive multimodal LM, NOT diffusion. The pipeline
 *  for this family is implemented entirely on top of @huggingface/transformers
 *  rather than our own ORT plumbing - see web/src/janus/generate.ts and
 *  web/src/janus/cache-adapter.ts for the rationale. The ModelSet entry only
 *  needs to know which HF repo id to hand to from_pretrained() and which
 *  files to pre-cache via OPFS. There is no scheduler / unet / vae split. */
export interface JanusModelSet extends BaseModelInfo {
  family: "janus";
  hfModelId: string;
  /** dtype string passed to from_pretrained(). q4f16 is the only variant we
   *  ship - it produces the smallest viable bundle and the worklog 2026-04-08
   *  smoke confirmed q4f16 weight decompression works in stock ORT-web. */
  dtype: "q4f16";
  /** Native output resolution. Fixed by the image_decode head. */
  nativeResolution: number;
  /** Number of image tokens the LM generates per image. Used as the progress
   *  total. */
  numImageTokens: number;
  /** Flat file list for the model manager. */
  files: ModelFile[];
}

/** Z-Image family: S3-DiT transformer with a Qwen2-class q4f16 text
 *  encoder. 16-channel latents, 5D hidden_states, flow matching scheduler
 *  with shift=3.0. Architecture is fundamentally different from SD/SDXL:
 *  no UNet, no CLIP, no CFG. Text encoder uses transformers.js AutoTokenizer
 *  with a chat template. */
export interface ZImageModelSet extends BaseModelInfo {
  family: "zimage";
  /** Tokenizer files for transformers.js AutoTokenizer. */
  tokenizer: { config: ModelFile; vocab: ModelFile };
  textEncoder: OrtModelFile;
  /** Monolithic transformer (load once, run all steps). */
  transformer: OrtModelFile;
  vaeDecoder: OrtModelFile;
  /** Tiny baked ONNX scheduler step model (~3 KB). Takes noise_pred,
   *  latents, step_info and produces latents_out with the correct flow
   *  matching Euler step. */
  schedulerStep: OrtModelFile;
  /** Tiny baked ONNX VAE pre-process model (~1 KB). Squeezes + scales
   *  the latent before VAE decode. */
  vaePreProcess: OrtModelFile;
  /** Number of DiT inference steps. */
  numInferenceSteps: number;
  /** Text encoder hidden dim (2560 for Z-Image-Turbo). */
  hiddenDim: number;
}

/** Z-Image with sharded transformer. Each shard is a separate ONNX graph
 *  sharing a single external data file. Per step, shards are loaded, run,
 *  and released sequentially - peak GPU memory = max(shard) not sum. */
export interface ZImageShardedModelSet extends Omit<ZImageModelSet, "transformer"> {
  /** Shard graph files (order matters - run sequentially). All reference
   *  the same external data file via dataPath. */
  transformerShards: { graph: ModelFile; data: ModelFile; dataPath: string }[];
}

export type ModelSet = Sd15ModelSet | SdxlModelSet | JanusModelSet | ZImageModelSet | ZImageShardedModelSet;

// =============================================================================
// SD1.5 base, ONNX fp16. From nmkd's repo.
// =============================================================================
//
// Picked over tlwu because the UNet uses external-data layout (a small graph
// file + a separate weights blob) which is the only way to load a 1.7 GB
// model in ORT-web without hitting the wasm 32-bit linear-memory cap (4 GB)
// during session create. tlwu ships a monolithic UNet that triggers
// std::bad_alloc on InferenceSession.create regardless of whether we hand it
// an ArrayBuffer or a streamed blob URL. The smaller networks (text encoder,
// VAEs) are monolithic in nmkd's repo, which is fine because they all fit
// comfortably under the wasm cap on their own.
//
// All weights are fp16 in this bundle. fp16 is the SD1.5 standard for
// in-browser inference - quality loss vs fp32 is negligible and the memory
// savings are essential.

const SD15_BASE_REPO = `${HF}/nmkd/stable-diffusion-1.5-onnx-fp16/resolve/main`;

// CLIP ViT-L/14 tokenizer files. SD1.5's UNet cross-attention is hard-wired
// to this exact text encoder's 768-dim output, so every SD1.5 finetune reuses
// these same two files. nmkd's bytes are identical to tlwu's (same vocab),
// so the committed test fixture under web/test/fixtures/clip/ still matches
// and the tokenizer test continues to pass without regeneration.
const CLIP_VOCAB: ModelFile = {
  id: "sd15_nmkd_vocab",
  name: "CLIP ViT-L/14 vocab.json",
  url: `${SD15_BASE_REPO}/tokenizer/vocab.json`,
  sizeBytes: 1_059_962,
};

const CLIP_MERGES: ModelFile = {
  id: "sd15_nmkd_merges",
  name: "CLIP ViT-L/14 merges.txt",
  url: `${SD15_BASE_REPO}/tokenizer/merges.txt`,
  sizeBytes: 524_619,
};

const SD15_TEXT_ENCODER: ModelFile = {
  id: "sd15_nmkd_text_encoder",
  name: "SD1.5 CLIP text encoder (fp16)",
  url: `${SD15_BASE_REPO}/text_encoder/model.onnx`,
  sizeBytes: 246_476_214,
};

// External-data layout: tiny graph file + large weights sidecar. The graph
// file references the sidecar by the filename "weights.pb" relative to its
// own location.
const SD15_UNET_GRAPH: ModelFile = {
  id: "sd15_nmkd_unet_graph",
  name: "SD1.5 UNet (graph)",
  url: `${SD15_BASE_REPO}/unet/model.onnx`,
  sizeBytes: 1_217_704,
};

const SD15_UNET_DATA: ModelFile = {
  id: "sd15_nmkd_unet_weights",
  name: "SD1.5 UNet (weights, fp16)",
  url: `${SD15_BASE_REPO}/unet/weights.pb`,
  sizeBytes: 1_718_976_000,
};

const SD15_VAE_DECODER: ModelFile = {
  id: "sd15_nmkd_vae_decoder",
  name: "SD1.5 VAE decoder (fp16)",
  url: `${SD15_BASE_REPO}/vae_decoder/model.onnx`,
  sizeBytes: 99_094_195,
};

const SD15_VAE_ENCODER: ModelFile = {
  id: "sd15_nmkd_vae_encoder",
  name: "SD1.5 VAE encoder (fp16)",
  url: `${SD15_BASE_REPO}/vae_encoder/model.onnx`,
  sizeBytes: 68_430_493,
};

export const SD15_BASE_MODEL: Sd15ModelSet = {
  id: "sd15_base",
  name: "Stable Diffusion 1.5 (base, fp16)",
  description:
    "Original SD1.5 weights, ONNX fp16, external-data UNet. Supports " +
    "text-to-image and image-to-image. ~2.13 GB total download.",
  family: "sd15",
  capabilities: { txt2img: true, img2img: true },
  defaults: {
    width: 512,
    height: 512,
    mobileResolution: 384,
    maxResolution: 1024,
    steps: 20,
    cfg: 7.5,
    resolutionHelp:
      "SD1.5 was trained at 512x512. Going above 512 on either axis usually produces duplicated body parts and other glitches. Below 512 loses detail. Multiples of 8.",
    stepsHelp:
      "denoising iterations. 20-50 is the SD1.5 sweet spot. fewer = faster, lower quality.",
    cfgHelp: "prompt adherence strength. 7.5 is the SD1.5 default.",
  },
  img2img: {
    strengthLabel: "strength",
    description:
      "SD1.5 uses the reference as a starting point in latent space, " +
      "then partially re-noises and re-denoises it. The result keeps " +
      "the original layout, composition, and colors and lets the prompt " +
      "vary textures, lighting, and style. Lower strength sticks closer " +
      "to the reference; higher strength drifts further from it. This " +
      "mode does NOT change what is in the image (clothes, pose, " +
      "subject) - it only restyles it. For real edits use inpainting " +
      "(coming later).",
  },
  tokenizer: { vocab: CLIP_VOCAB, merges: CLIP_MERGES },
  textEncoder: SD15_TEXT_ENCODER,
  unet: {
    graph: SD15_UNET_GRAPH,
    data: SD15_UNET_DATA,
    dataPath: "weights.pb",
  },
  vaeDecoder: SD15_VAE_DECODER,
  vaeEncoder: SD15_VAE_ENCODER,
};

// =============================================================================
// Segmind Vega, ONNX fp16. From gfodor/segmind-vega-fp16-onnx.
// =============================================================================
//
// Distilled SDXL (~0.74 B params vs SDXL's 3.5 B) with the standard SDXL
// architecture: dual text encoders (CLIP-L + OpenCLIP-bigG/14), UNet
// consuming concatenated 2048-dim hidden states plus a 1280-dim pooled
// text_embeds and a 6-element time_ids tensor, SDXL VAE with scaling factor
// 0.13025. All four heavy components are external-data with the standard
// diffusers/optimum sidecar filename "model.onnx_data". Bundle ~3.22 GB.
//
// Tokenizer is shared with SD1.5 (CLIP BPE on the same vocab) but the file
// lives at a different URL in this repo, so we declare it as separate
// ModelFile entries to keep cache bookkeeping clean.

const VEGA_REPO = `${HF}/gfodor/segmind-vega-fp16-onnx/resolve/main`;

const VEGA_TOK_VOCAB: ModelFile = {
  id: "vega_tok1_vocab",
  name: "Segmind Vega CLIP-L vocab.json",
  url: `${VEGA_REPO}/tokenizer/vocab.json`,
  sizeBytes: 1_059_962,
};
const VEGA_TOK_MERGES: ModelFile = {
  id: "vega_tok1_merges",
  name: "Segmind Vega CLIP-L merges.txt",
  url: `${VEGA_REPO}/tokenizer/merges.txt`,
  sizeBytes: 524_619,
};
const VEGA_TOK2_VOCAB: ModelFile = {
  id: "vega_tok2_vocab",
  name: "Segmind Vega OpenCLIP-bigG vocab.json",
  url: `${VEGA_REPO}/tokenizer_2/vocab.json`,
  sizeBytes: 1_059_962,
};
const VEGA_TOK2_MERGES: ModelFile = {
  id: "vega_tok2_merges",
  name: "Segmind Vega OpenCLIP-bigG merges.txt",
  url: `${VEGA_REPO}/tokenizer_2/merges.txt`,
  sizeBytes: 524_619,
};

const VEGA_TE1_GRAPH: ModelFile = {
  id: "vega_text_encoder_graph",
  name: "Segmind Vega CLIP-L text encoder (graph)",
  url: `${VEGA_REPO}/text_encoder/model.onnx`,
  sizeBytes: 438_405,
};
const VEGA_TE1_DATA: ModelFile = {
  id: "vega_text_encoder_data",
  name: "Segmind Vega CLIP-L text encoder (weights, fp16)",
  url: `${VEGA_REPO}/text_encoder/model.onnx_data`,
  sizeBytes: 246_120_960,
};

const VEGA_TE2_GRAPH: ModelFile = {
  id: "vega_text_encoder2_graph",
  name: "Segmind Vega OpenCLIP-bigG text encoder (graph)",
  url: `${VEGA_REPO}/text_encoder_2/model.onnx`,
  sizeBytes: 1_179_085,
};
const VEGA_TE2_DATA: ModelFile = {
  id: "vega_text_encoder2_data",
  name: "Segmind Vega OpenCLIP-bigG text encoder (weights, fp16)",
  url: `${VEGA_REPO}/text_encoder_2/model.onnx_data`,
  sizeBytes: 1_389_319_680,
};

const VEGA_UNET_GRAPH: ModelFile = {
  id: "vega_unet_graph",
  name: "Segmind Vega UNet (graph)",
  url: `${VEGA_REPO}/unet/model.onnx`,
  sizeBytes: 428_404,
};
const VEGA_UNET_DATA: ModelFile = {
  id: "vega_unet_data",
  name: "Segmind Vega UNet (weights, fp16)",
  url: `${VEGA_REPO}/unet/model.onnx_data`,
  sizeBytes: 1_490_618_880,
};

const VEGA_VAE_DEC_GRAPH: ModelFile = {
  id: "vega_vae_decoder_graph",
  name: "Segmind Vega VAE decoder (graph)",
  url: `${VEGA_REPO}/vae_decoder/model.onnx`,
  sizeBytes: 157_744,
};
const VEGA_VAE_DEC_DATA: ModelFile = {
  id: "vega_vae_decoder_data",
  name: "Segmind Vega VAE decoder (weights, fp16)",
  url: `${VEGA_REPO}/vae_decoder/model.onnx_data`,
  sizeBytes: 98_965_248,
};

const VEGA_VAE_ENC_GRAPH: ModelFile = {
  id: "vega_vae_encoder_graph",
  name: "Segmind Vega VAE encoder (graph)",
  url: `${VEGA_REPO}/vae_encoder/model.onnx`,
  sizeBytes: 137_934,
};
const VEGA_VAE_ENC_DATA: ModelFile = {
  id: "vega_vae_encoder_data",
  name: "Segmind Vega VAE encoder (weights, fp16)",
  url: `${VEGA_REPO}/vae_encoder/model.onnx_data`,
  sizeBytes: 68_315_904,
};

export const SEGMIND_VEGA_MODEL: SdxlModelSet = {
  id: "segmind_vega",
  name: "Segmind Vega (distilled SDXL, fp16)",
  description:
    "Distilled SDXL with dual CLIP text encoders. Higher quality than SD1.5 " +
    "with sharper details and better prompt adherence, at the cost of a much " +
    "larger ~3.22 GB download. Supports text-to-image and image-to-image.",
  family: "sdxl",
  capabilities: { txt2img: true, img2img: true },
  defaults: {
    // SDXL native resolution is 1024x1024 but the WebGPU memory budget on
    // a typical iGPU does not love that. 768 is a reasonable middle: still
    // gets you the SDXL quality jump over SD1.5 without the activation
    // memory of full 1024. Mobile users get 512.
    width: 768,
    height: 768,
    mobileResolution: 512,
    maxResolution: 1024,
    // Distilled SDXL still wants more steps than SDXL-Turbo (1) but fewer
    // than full SDXL (30+). Vega's repo recommends ~25.
    steps: 25,
    cfg: 7.0,
    resolutionHelp:
      "Segmind Vega is a distilled SDXL trained at 1024x1024. 768x768 is a good speed/quality balance in-browser; 512 is the mobile fallback. Multiples of 8.",
    stepsHelp:
      "denoising iterations. Vega's distillation lets it do well at 20-25 (SDXL-base wants 30+).",
    cfgHelp: "prompt adherence strength. 5-8 is the SDXL range; 7 is a balanced default.",
  },
  img2img: {
    strengthLabel: "strength",
    description:
      "Same strength-based init-image path as SD1.5: encodes the " +
      "reference into latent space, partially re-noises it, and lets the " +
      "prompt restyle the result. SDXL preserves layout and composition " +
      "better than SD1.5 thanks to the dual text encoders, so the " +
      "reference image's structure tends to come through more cleanly. " +
      "Still does NOT change what is in the image - for clothing/pose/" +
      "subject edits you need inpainting (coming later).",
  },
  tokenizer: { vocab: VEGA_TOK_VOCAB, merges: VEGA_TOK_MERGES },
  tokenizer2: { vocab: VEGA_TOK2_VOCAB, merges: VEGA_TOK2_MERGES },
  textEncoder: {
    graph: VEGA_TE1_GRAPH,
    data: VEGA_TE1_DATA,
    dataPath: "model.onnx_data",
  },
  textEncoder2: {
    graph: VEGA_TE2_GRAPH,
    data: VEGA_TE2_DATA,
    dataPath: "model.onnx_data",
  },
  unet: {
    graph: VEGA_UNET_GRAPH,
    data: VEGA_UNET_DATA,
    dataPath: "model.onnx_data",
  },
  vaeDecoder: {
    graph: VEGA_VAE_DEC_GRAPH,
    data: VEGA_VAE_DEC_DATA,
    dataPath: "model.onnx_data",
  },
  vaeEncoder: {
    graph: VEGA_VAE_ENC_GRAPH,
    data: VEGA_VAE_ENC_DATA,
    dataPath: "model.onnx_data",
  },
};

// =============================================================================
// SDXL-Turbo q4f16. From webnn/sdxl-turbo (Microsoft WebNN team export).
// =============================================================================
//
// Adversarial Diffusion Distillation (ADD) of SDXL. 1-step default, no
// classifier-free guidance needed (CFG=1 / disabled). All four ONNX
// components are monolithic q4f16 with no external-data sidecars. Boundary
// dtype is fp16 (unlike Vega's fp32). Total bundle ~2.67 GB. Desktop only
// for now - the 1.97 GB monolithic UNet exceeds mobile GPU buffer limits.
//
// The HF repo ships tokenizer.json (fast-tokenizer format) instead of
// vocab.json + merges.txt. Since the CLIP BPE vocabulary is byte-identical
// across all SDXL exports, we reuse Vega's tokenizer files which our
// ClipTokenizer already consumes.

const TURBO_REPO = `${HF}/webnn/sdxl-turbo/resolve/main`;

const TURBO_TE1: ModelFile = {
  id: "turbo_text_encoder",
  name: "SDXL-Turbo CLIP-L text encoder (q4f16)",
  url: `${TURBO_REPO}/onnx/text_encoder_model_q4f16.onnx`,
  sizeBytes: 124_433_929,
};

const TURBO_TE2: ModelFile = {
  id: "turbo_text_encoder2",
  name: "SDXL-Turbo OpenCLIP-bigG text encoder (q4f16)",
  url: `${TURBO_REPO}/onnx/text_encoder_2_model_q4f16.onnx`,
  sizeBytes: 483_677_630,
};

const TURBO_UNET_GRAPH: ModelFile = {
  id: "turbo_unet",
  name: "SDXL-Turbo UNet (q4f16)",
  url: `${TURBO_REPO}/onnx/unet_model_q4f16.onnx`,
  sizeBytes: 1_965_607_674,
  transform: patchTransform(sdxlTurboUnetSplit as unknown as Patch),
};

const TURBO_UNET_SIDECAR: ModelFile = {
  id: (sdxlTurboUnetSplit as unknown as Patch).sidecar!.fileId,
  name: "SDXL-Turbo UNet (weights)",
  url: "", // created during graph download, never fetched directly
  sizeBytes: 0, // not a separate download; extracted from the graph file
};

const TURBO_VAE_DEC: ModelFile = {
  id: "turbo_vae_decoder",
  name: "SDXL-Turbo VAE decoder (q4f16)",
  url: `${TURBO_REPO}/onnx/vae_decoder_model_q4f16.onnx`,
  sizeBytes: 97_617_257,
};

export const SDXL_TURBO_MODEL: SdxlModelSet = {
  id: "sdxl_turbo",
  name: "SDXL-Turbo (q4f16, 1-step)",
  description:
    "Microsoft WebNN team's q4f16 SDXL-Turbo. 1-step generation with no " +
    "classifier-free guidance needed. ~2.67 GB total download. UNet is split " +
    "into graph + external-data sidecar during download for mobile compatibility.",
  family: "sdxl",
  capabilities: { txt2img: true, img2img: true },
  boundaryDtype: "float16",
  schedulerType: "euler",
  defaults: {
    width: 512,
    height: 512,
    mobileResolution: 512,
    maxResolution: 1024,
    steps: 1,
    cfg: 0,
    resolutionHelp:
      "SDXL-Turbo was trained at 512x512. Higher resolutions may work but are untested with the 1-step distillation.",
    stepsHelp:
      "SDXL-Turbo is distilled for 1-step generation. More steps may improve detail but with diminishing returns.",
    cfgHelp:
      "SDXL-Turbo is trained without classifier-free guidance. Leave at 0 for best results; values above 1 enable CFG (slower, 2 UNet passes per step).",
  },
  img2img: {
    strengthLabel: "strength",
    description:
      "SDXL-Turbo img2img: encodes the reference into latent space and " +
      "partially re-noises it. With 1 step and no CFG the output is fast " +
      "but the prompt has limited steering. Increase steps to 2-4 for " +
      "more prompt influence. Still preserves layout/composition only - " +
      "for real edits use inpainting (coming later).",
  },
  // Reuse Vega's tokenizer files - same CLIP BPE vocab, byte-identical.
  tokenizer: { vocab: VEGA_TOK_VOCAB, merges: VEGA_TOK_MERGES },
  tokenizer2: { vocab: VEGA_TOK2_VOCAB, merges: VEGA_TOK2_MERGES },
  textEncoder: TURBO_TE1,
  textEncoder2: TURBO_TE2,
  unet: {
    graph: TURBO_UNET_GRAPH,
    data: TURBO_UNET_SIDECAR,
    dataPath: "unet_model_q4f16.onnx_data",
  },
  vaeDecoder: TURBO_VAE_DEC,
};

// =============================================================================
// Z-Image-Turbo q4f16. From webnn/Z-Image-Turbo (Microsoft WebNN team).
// =============================================================================
//
// S3-DiT 6B family with a Qwen2-class q4f16 text encoder. Flow matching
// scheduler (shift=3.0), 8 NFEs, no CFG. Highest quality model in the
// lineup but also the heaviest: text encoder ~2.22 GB, transformer ~3.70 GB.
// Desktop only - the transformer alone exceeds any mobile GPU budget.
//
// The text encoder uses a Qwen2 tokenizer with a chat template, handled
// via transformers.js AutoTokenizer (same dynamic-import pattern as Janus).
// The tokenizer files are tokenizer.json (7.3 MB) + tokenizer_config.json.

const ZIMAGE_REPO = `${HF}/webnn/Z-Image-Turbo/resolve/main`;

const ZIMAGE_TOK_VOCAB: ModelFile = {
  id: "zimage_tokenizer_vocab",
  name: "Z-Image tokenizer.json",
  url: `${ZIMAGE_REPO}/tokenizer/tokenizer.json`,
  sizeBytes: 7_335_749,
};
const ZIMAGE_TOK_CONFIG: ModelFile = {
  id: "zimage_tokenizer_config",
  name: "Z-Image tokenizer_config.json",
  url: `${ZIMAGE_REPO}/tokenizer/tokenizer_config.json`,
  sizeBytes: 5_404,
};

const ZIMAGE_TE_GRAPH: ModelFile = {
  id: "zimage_text_encoder_graph",
  name: "Z-Image text encoder (q4f16, graph)",
  url: `${ZIMAGE_REPO}/onnx/text_encoder_model_q4f16.onnx`,
  sizeBytes: 690_765_724,
};
const ZIMAGE_TE_DATA: ModelFile = {
  id: "zimage_text_encoder_data",
  name: "Z-Image text encoder (q4f16, weights)",
  url: `${ZIMAGE_REPO}/onnx/text_encoder_model_q4f16.onnx_data`,
  sizeBytes: 1_526_231_040,
};

const ZIMAGE_XFMR_GRAPH: ModelFile = {
  id: "zimage_transformer_graph",
  name: "Z-Image transformer (q4f16, graph)",
  url: `${ZIMAGE_REPO}/onnx/transformer_model_q4f16.onnx`,
  sizeBytes: 1_675_741_721,
};
const ZIMAGE_XFMR_DATA: ModelFile = {
  id: "zimage_transformer_data",
  name: "Z-Image transformer (q4f16, weights)",
  url: `${ZIMAGE_REPO}/onnx/transformer_model_q4f16.onnx_data`,
  sizeBytes: 2_025_062_400,
};

const ZIMAGE_VAE_DEC: ModelFile = {
  id: "zimage_vae_decoder",
  name: "Z-Image VAE decoder (f16)",
  url: `${ZIMAGE_REPO}/onnx/vae_decoder_model_f16.onnx`,
  sizeBytes: 99_284_482,
};

const ZIMAGE_SCHEDULER_STEP: ModelFile = {
  id: "zimage_scheduler_step",
  name: "Z-Image scheduler step (f16)",
  url: `${ZIMAGE_REPO}/onnx/scheduler_step_model_f16.onnx`,
  sizeBytes: 3_235,
};

const ZIMAGE_VAE_PRE: ModelFile = {
  id: "zimage_vae_pre_process",
  name: "Z-Image VAE pre-process (f16)",
  url: `${ZIMAGE_REPO}/onnx/vae_pre_process_model_f16.onnx`,
  sizeBytes: 905,
};

export const ZIMAGE_TURBO_MODEL: ZImageModelSet = {
  id: "zimage_turbo",
  name: "Z-Image-Turbo (S3-DiT 6B, q4f16)",
  description:
    "Microsoft WebNN team's q4f16 Z-Image-Turbo. S3-DiT 6B architecture " +
    "with a Qwen2-class text encoder. Highest quality model available, " +
    "but slow (~286s on iGPU, much faster on discrete GPU). ~6.02 GB " +
    "total download. Desktop only.",
  family: "zimage",
  capabilities: { txt2img: true, img2img: false },
  numInferenceSteps: 9,
  hiddenDim: 2560,
  defaults: {
    width: 512,
    height: 512,
    mobileResolution: 512,
    maxResolution: 768,
    steps: 9,
    cfg: 0,
    resolutionHelp: "Z-Image supports 512x512 and 768x768.",
    stepsHelp: "DiT inference steps. 9 is the default. 3-9 range.",
    cfgHelp: "Z-Image-Turbo does not use classifier-free guidance.",
    fixedResolution: false,
    fixedSteps: false,
  },
  tokenizer: { config: ZIMAGE_TOK_CONFIG, vocab: ZIMAGE_TOK_VOCAB },
  textEncoder: {
    graph: ZIMAGE_TE_GRAPH,
    data: ZIMAGE_TE_DATA,
    dataPath: "text_encoder_model_q4f16.onnx_data",
  },
  transformer: {
    graph: ZIMAGE_XFMR_GRAPH,
    data: ZIMAGE_XFMR_DATA,
    dataPath: "transformer_model_q4f16.onnx_data",
  },
  vaeDecoder: ZIMAGE_VAE_DEC,
  schedulerStep: ZIMAGE_SCHEDULER_STEP,
  vaePreProcess: ZIMAGE_VAE_PRE,
};

// =============================================================================
// Z-Image-Turbo (reordered) - dev/test variant with node-reordered transformer
// for sharding. Uses local dev server for transformer, HF for everything else.
// =============================================================================

// TODO: switch to HF URL once published: `${HF}/cretz/Z-Image-Turbo/resolve/main`
const ZIMAGE_REORDERED_REPO = "/local-models";

const ZIMAGE_REORDERED_TOK_VOCAB: ModelFile = {
  id: "zimage_reordered_tokenizer_vocab",
  name: "Z-Image tokenizer.json",
  url: `${ZIMAGE_REORDERED_REPO}/tokenizer/tokenizer.json`,
  sizeBytes: 7_335_749,
};
const ZIMAGE_REORDERED_TOK_CONFIG: ModelFile = {
  id: "zimage_reordered_tokenizer_config",
  name: "Z-Image tokenizer_config.json",
  url: `${ZIMAGE_REORDERED_REPO}/tokenizer/tokenizer_config.json`,
  sizeBytes: 5_404,
};
const ZIMAGE_REORDERED_TE_GRAPH: ModelFile = {
  id: "zimage_reordered_text_encoder_graph",
  name: "Z-Image text encoder (q4f16, graph)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/text_encoder_model_q4f16.onnx`,
  sizeBytes: 690_765_724,
};
const ZIMAGE_REORDERED_TE_DATA: ModelFile = {
  id: "zimage_reordered_text_encoder_data",
  name: "Z-Image text encoder (q4f16, weights)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/text_encoder_model_q4f16.onnx_data`,
  sizeBytes: 1_526_231_040,
};
const ZIMAGE_REORDERED_XFMR_GRAPH: ModelFile = {
  id: "zimage_reordered_transformer_graph",
  name: "Z-Image transformer reordered (q4f16, graph)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16.onnx`,
  sizeBytes: 1_675_741_721,
};
const ZIMAGE_REORDERED_XFMR_DATA: ModelFile = {
  id: "zimage_reordered_transformer_data",
  name: "Z-Image transformer reordered (q4f16, weights)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16.onnx_data`,
  sizeBytes: 2_025_062_400,
};
const ZIMAGE_REORDERED_VAE_DEC: ModelFile = {
  id: "zimage_reordered_vae_decoder",
  name: "Z-Image VAE decoder (f16)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/vae_decoder_model_f16.onnx`,
  sizeBytes: 99_284_482,
};
const ZIMAGE_REORDERED_SCHEDULER_STEP: ModelFile = {
  id: "zimage_reordered_scheduler_step",
  name: "Z-Image scheduler step (f16)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/scheduler_step_model_f16.onnx`,
  sizeBytes: 3_235,
};
const ZIMAGE_REORDERED_VAE_PRE: ModelFile = {
  id: "zimage_reordered_vae_pre_process",
  name: "Z-Image VAE pre-process (f16)",
  url: `${ZIMAGE_REORDERED_REPO}/onnx/vae_pre_process_model_f16.onnx`,
  sizeBytes: 905,
};

export const ZIMAGE_TURBO_REORDERED_MODEL: ZImageModelSet = {
  id: "zimage_turbo_reordered",
  name: "Z-Image-Turbo Reordered (dev)",
  description:
    "Dev/test: Z-Image-Turbo with node-reordered transformer graph. " +
    "Same weights and behavior as the original, but with contiguous " +
    "layer nodes for sharding. Served from local dev server.",
  family: "zimage",
  capabilities: { txt2img: true, img2img: false },
  numInferenceSteps: 9,
  hiddenDim: 2560,
  defaults: {
    width: 512,
    height: 512,
    mobileResolution: 512,
    maxResolution: 768,
    steps: 9,
    cfg: 0,
    resolutionHelp: "Z-Image supports 512x512 and 768x768.",
    stepsHelp: "DiT inference steps. 9 is the default. 3-9 range.",
    cfgHelp: "Z-Image-Turbo does not use classifier-free guidance.",
    fixedResolution: false,
    fixedSteps: false,
  },
  tokenizer: { config: ZIMAGE_REORDERED_TOK_CONFIG, vocab: ZIMAGE_REORDERED_TOK_VOCAB },
  textEncoder: {
    graph: ZIMAGE_REORDERED_TE_GRAPH,
    data: ZIMAGE_REORDERED_TE_DATA,
    dataPath: "text_encoder_model_q4f16.onnx_data",
  },
  transformer: {
    graph: ZIMAGE_REORDERED_XFMR_GRAPH,
    data: ZIMAGE_REORDERED_XFMR_DATA,
    dataPath: "transformer_model_q4f16.onnx_data",
  },
  vaeDecoder: ZIMAGE_REORDERED_VAE_DEC,
  schedulerStep: ZIMAGE_REORDERED_SCHEDULER_STEP,
  vaePreProcess: ZIMAGE_REORDERED_VAE_PRE,
};

// =============================================================================
// Z-Image-Turbo Sharded (5x ~800 MB transformer shards for mobile).
// Same reordered graph, topologically split into 5 sub-graphs that share
// a single external data file. Served from local dev server.
// =============================================================================

const ZIMAGE_SHARD_GRAPHS: ModelFile[] = [
  { id: "zimage_sharded_xfmr_s0", name: "Z-Image transformer shard 0", url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16_shard0.onnx`, sizeBytes: 504_645_236 },
  { id: "zimage_sharded_xfmr_s1", name: "Z-Image transformer shard 1", url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16_shard1.onnx`, sizeBytes: 332_079_565 },
  { id: "zimage_sharded_xfmr_s2", name: "Z-Image transformer shard 2", url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16_shard2.onnx`, sizeBytes: 346_799_091 },
  { id: "zimage_sharded_xfmr_s3", name: "Z-Image transformer shard 3", url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16_shard3.onnx`, sizeBytes: 360_370_916 },
  { id: "zimage_sharded_xfmr_s4", name: "Z-Image transformer shard 4", url: `${ZIMAGE_REORDERED_REPO}/onnx/transformer_model_q4f16_shard4.onnx`, sizeBytes: 131_759_709 },
];

export const ZIMAGE_TURBO_SHARDED_MODEL: ZImageShardedModelSet = {
  id: "zimage_turbo_sharded",
  name: "Z-Image-Turbo Sharded (dev)",
  description:
    "Dev/test: Z-Image-Turbo with 5-shard transformer (~800 MB/shard). " +
    "Same output as the monolithic model but loads one shard at a time " +
    "to fit mobile GPU memory budgets.",
  family: "zimage",
  capabilities: { txt2img: true, img2img: false },
  numInferenceSteps: 9,
  hiddenDim: 2560,
  defaults: {
    width: 512,
    height: 512,
    mobileResolution: 512,
    maxResolution: 768,
    steps: 9,
    cfg: 0,
    resolutionHelp: "Z-Image supports 512x512 and 768x768.",
    stepsHelp: "DiT inference steps. 9 is the default. 3-9 range.",
    cfgHelp: "Z-Image-Turbo does not use classifier-free guidance.",
    fixedResolution: false,
    fixedSteps: false,
  },
  tokenizer: { config: ZIMAGE_REORDERED_TOK_CONFIG, vocab: ZIMAGE_REORDERED_TOK_VOCAB },
  textEncoder: {
    graph: ZIMAGE_REORDERED_TE_GRAPH,
    data: ZIMAGE_REORDERED_TE_DATA,
    dataPath: "text_encoder_model_q4f16.onnx_data",
  },
  transformerShards: ZIMAGE_SHARD_GRAPHS.map((graph) => ({
    graph,
    data: ZIMAGE_REORDERED_XFMR_DATA,
    dataPath: "transformer_model_q4f16.onnx_data",
  })),
  vaeDecoder: ZIMAGE_REORDERED_VAE_DEC,
  schedulerStep: ZIMAGE_REORDERED_SCHEDULER_STEP,
  vaePreProcess: ZIMAGE_REORDERED_VAE_PRE,
};

// =============================================================================
// Janus-Pro-1B, ONNX q4f16. From onnx-community/Janus-Pro-1B-ONNX.
// =============================================================================
//
// Autoregressive multimodal LM. Fundamentally different architecture from
// sd15/sdxl - no UNet, no scheduler, generates 576 image tokens sequentially
// then decodes them in a single pass. Quality is research-tier and generally
// below SD1.5, but the bundle is small (~1.34 GB), the resolution is fixed
// at 384x384, and we get to validate a non-diffusion family in our pipeline
// registry. Pipeline implemented on top of @huggingface/transformers rather
// than our own ORT - see web/src/janus/generate.ts.

export const JANUS_PRO_1B_MODEL: JanusModelSet = {
  id: "janus_pro_1b",
  name: "Janus-Pro-1B (autoregressive)",
  description:
    "DeepSeek's Janus-Pro-1B multimodal LM, ONNX q4f16. Autoregressive " +
    "image generation: a 24-layer transformer produces 576 image tokens " +
    "one at a time, then a single decode pass turns them into a 384x384 " +
    "image. ~1.34 GB total download. Quality is research-tier (typically " +
    "below SD1.5); the appeal is the small bundle and the radically " +
    "different architecture. No reference-image / img2img support.",
  family: "janus",
  hfModelId: JANUS_PRO_1B_MODEL_ID,
  dtype: "q4f16",
  nativeResolution: JANUS_NATIVE_RESOLUTION,
  numImageTokens: JANUS_NUM_IMAGE_TOKENS,
  files: JANUS_PRO_1B_FILES,
  capabilities: { txt2img: true, img2img: false },
  defaults: {
    // Janus output resolution is hard-wired to 384x384 by image_decode.
    // The width/height inputs in the UI are still shown but the pipeline
    // ignores them - the result is always 384x384.
    width: JANUS_NATIVE_RESOLUTION,
    height: JANUS_NATIVE_RESOLUTION,
    mobileResolution: JANUS_NATIVE_RESOLUTION,
    maxResolution: JANUS_NATIVE_RESOLUTION,
    // "steps" for an autoregressive model is the number of generated image
    // tokens. The user cannot change this - it is fixed by architecture.
    steps: JANUS_NUM_IMAGE_TOKENS,
    // CFG IS used inside generate_images (transformers.js applies the
    // standard 5.0 default for Janus). We expose the slider but the
    // pipeline currently passes our default through to generation_config.
    cfg: 5.0,
    resolutionHelp: "Janus-Pro is hard-wired to 384x384.",
    stepsHelp: "Janus generates 576 image tokens per image. Fixed by architecture.",
    cfgHelp: "classifier-free guidance scale. 5.0 is the Janus default.",
    fixedResolution: true,
    fixedSteps: true,
  },
};

/** All model sets the image-gen tool offers. Order = display order,
 *  recommended first by quality/speed/compatibility balance. */
export const IMAGE_GEN_MODELS: ModelSet[] = [
  JANUS_PRO_1B_MODEL,
  SDXL_TURBO_MODEL,
  ZIMAGE_TURBO_MODEL,
  ZIMAGE_TURBO_REORDERED_MODEL,
  ZIMAGE_TURBO_SHARDED_MODEL,
  SEGMIND_VEGA_MODEL,
  SD15_BASE_MODEL,
];

/** Flatten a ModelSet to the list of files that must be cached. */
export function modelSetFiles(set: ModelSet): ModelFile[] {
  if (set.family === "janus") {
    return [...set.files];
  }
  if (set.family === "zimage") {
    const files: ModelFile[] = [];
    files.push(set.tokenizer.config, set.tokenizer.vocab);
    files.push(...ortModelFiles(set.textEncoder));
    if ("transformerShards" in set) {
      const seen = new Set<string>();
      for (const shard of set.transformerShards) {
        files.push(shard.graph);
        if (!seen.has(shard.data.id)) {
          seen.add(shard.data.id);
          files.push(shard.data);
        }
      }
    } else {
      files.push(...ortModelFiles(set.transformer));
    }
    files.push(...ortModelFiles(set.schedulerStep));
    files.push(...ortModelFiles(set.vaePreProcess));
    files.push(...ortModelFiles(set.vaeDecoder));
    return files;
  }
  // sd15 / sdxl
  const files: ModelFile[] = [];
  files.push(set.tokenizer.vocab, set.tokenizer.merges);
  if (set.family === "sdxl") {
    files.push(set.tokenizer2.vocab, set.tokenizer2.merges);
    files.push(...ortModelFiles(set.textEncoder));
    files.push(...ortModelFiles(set.textEncoder2));
  } else {
    files.push(...ortModelFiles(set.textEncoder));
  }
  files.push(...ortModelFiles(set.unet));
  files.push(...ortModelFiles(set.vaeDecoder));
  if (set.vaeEncoder) files.push(...ortModelFiles(set.vaeEncoder));
  return files;
}
