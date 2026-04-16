// Temporary smoke-test tool. Pulls a candidate ONNX bundle from HF, attempts
// InferenceSession.create on each component, and (for the UNet) attempts one
// dummy session.run with zero tensors. Everything is appended to a textarea
// for copy/paste back to the dev. Tied to the worklog "Model exploration" plan.
//
// Not part of the shipping product. Lives under tools/model-smoke/. Independent
// OPFS dir so it does not contend with the image-gen cache.

import * as ort from "onnxruntime-web";

import { ModelCache, type ModelFile } from "../shared/model-cache";
import { initThemeSelect } from "../shared/theme";
import { gaussianNoise, mulberry32 } from "../image-gen/generate-utils";
import { f16BitsToF32, f32ToF16Array } from "../sd15/fp16";

{
  const sel = document.getElementById("theme-select");
  if (sel instanceof HTMLSelectElement) initThemeSelect(sel);
}

// Sana DiT (1.2 GB external-data) blew the wasm32 4 GB cap when ORT-web
// requested 4.41 GB initial linear memory. The biggest contributor to wasm
// initial memory is per-thread stack provisioning - dropping numThreads to
// 1 reduces the initial allocation enough for ~1.2 GB models to fit. If
// this still isn't enough we can also try graphOptimizationLevel: "disabled"
// in the session options below.
ort.env.wasm.numThreads = 1;

interface Component {
  /** Display name in the log. */
  name: string;
  /** ONNX graph file. */
  graph: ModelFile;
  /**
   * Optional external-data sidecar. When present, passed to ORT via
   * sessionOptions.externalData with the path the graph references.
   */
  externalData?: { file: ModelFile; pathInGraph: string };
  /**
   * If set, after session.create succeeds, also try one dummy session.run
   * with zero tensors at these input shapes. This is what catches WebGPU
   * op-coverage failures (the same class of bug that hit the CLIP text
   * encoder Attention-mask kernel for image-gen).
   */
  dummyRun?: {
    /** Substring matchers for input names + their shape. Default dtype is
     *  float16 (matches fp16 amuse exports); set `dtype: "float32"` for
     *  stock optimum/diffusers exports that didn't fp16-cast. Timestep is
     *  probed: starts with the requested float dtype, falls back to int64
     *  on a dtype error from session.run.
     *  Per-input `dtype` overrides the component default - use when a single
     *  graph mixes float16 + float32 (e.g. fp32 RoPE frequencies) or needs
     *  int64 scalar inputs (e.g. unpatchify size args).
     *  Empty `shape: []` is supported for scalar inputs. */
    dtype?: "float16" | "float32";
    inputs: Array<{
      match: string[];
      shape: number[];
      dtype?: "float16" | "float32" | "int64" | "int32";
      /** Uniform fill value for every element. Defaults to 0. Use for scalar
       *  size/dim inputs that must be non-zero for downstream ops (e.g. the
       *  patch-space dims passed to an unpatchify reshape). */
      fill?: number;
      /** If true, fill with seeded Gaussian noise instead of zeros. For
       *  fp16/fp32 inputs only. Use when zeros collapse the op semantically
       *  (e.g. VAE decoding zero latent yields a gray frame with no useful
       *  signal about whether the decoder is working). */
      gaussian?: boolean;
    }>;
    /** If set, render one frame of the first matching output to the
     *  #smoke-output-canvas. Shape assumption: [B, F, C, H, W] fp16 (NTCHW),
     *  F and C read from the spec below. Use to visually confirm a VAE is
     *  actually producing a picture vs just returning near-zero. */
    renderOutput?: {
      /** Name substring of the output tensor; first match wins. */
      match: string[];
      /** Frame count (F). */
      numFrames: number;
      /** Channel count - must be 3 for RGB rendering. */
      channels: number;
      /** Height. */
      height: number;
      /** Width. */
      width: number;
      /** Which frame index to render (default 0). */
      frameIndex?: number;
      /** Pixel range the tensor lives in: [-1, 1] (default) or [0, 1]. */
      pixelRange?: "-1to1" | "0to1";
    };
    /**
     * If > 1, run session.run() this many times in a row and log each
     * timing separately. The first call always pays for WebGPU shader
     * compilation, kernel pipeline assembly, and weight uploads to GPU
     * VRAM, so the first number is dramatically inflated vs steady-state.
     * For shippability decisions look at runs 2 and 3.
     */
    repeats?: number;
  };
}

interface Candidate {
  id: string;
  label: string;
  components: Component[];
}

const HF_LCM = "https://huggingface.co/TensorStack/Realistic-LCM-amuse/resolve/main";
const HF_SDXL = "https://huggingface.co/TensorStack/SDXL-Lightning-amuse/resolve/main";
const HF_SDXS = "https://huggingface.co/lemonteaa/sdxs-onnx/resolve/main";
const HF_SANA = "https://huggingface.co/brad-agi/sana-0.6b-onnx-webgpu/resolve/main";
const HF_ZIMAGE = "https://huggingface.co/webnn/Z-Image-Turbo/resolve/main";
const HF_VEGA = "https://huggingface.co/gfodor/segmind-vega-fp16-onnx/resolve/main";
const HF_WEBNN_SDXL = "https://huggingface.co/webnn/sdxl-turbo/resolve/main";
const HF_ORT_SD_TURBO = "https://huggingface.co/onnxruntime/sd-turbo/resolve/main";
const HF_JANUS = "https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX/resolve/main";
const HF_NITRO_E = "https://huggingface.co/TensorStack/Nitro-E-onnx/resolve/main";
const LOCAL_FASTWAN = "/local-models/fastwan";

function f(prefix: string, base: string, rel: string, approxBytes: number): ModelFile {
  return {
    id: `${prefix}__${rel.replace(/\//g, "__")}`,
    name: rel,
    url: `${base}/${rel}`,
    sizeBytes: approxBytes,
  };
}

const CANDIDATES: Candidate[] = [
  {
    id: "ort-sd-turbo",
    label: "onnxruntime/sd-turbo (Microsoft official, SD2.1, monolithic fp16)",
    components: [
      // SD2.1-shape monolithic fp16. UNet is the smallest 1-step UNet
      // that's not q4-quantized, ~870 MB on disk. Highest probability of
      // being mobile-feasible since it's smaller than webnn/sdxl-turbo's
      // 1.25 GB UNet that crashed phone Chrome. Also: lacerbi/web-txt2img
      // uses this exact repo so it's externally proven to work in ORT-web.
      {
        name: "vae_decoder",
        graph: f("ortsd", HF_ORT_SD_TURBO, "vae_decoder/model.onnx", 95 * 1024 * 1024),
      },
      {
        name: "text_encoder (OpenCLIP-H/14, 1024-dim)",
        graph: f("ortsd", HF_ORT_SD_TURBO, "text_encoder/model.onnx", 700 * 1024 * 1024),
      },
      {
        name: "unet (monolithic ~870 MB)",
        graph: f("ortsd", HF_ORT_SD_TURBO, "unet/model.onnx", 870 * 1024 * 1024),
      },
    ],
  },
  {
    id: "janus-pro-1b",
    label: "onnx-community/Janus-Pro-1B-ONNX (q4f16, multimodal VLM)",
    components: [
      // Tiny multimodal model that supports text-to-image as one mode.
      // language_model is the only component with .onnx_data sidecar
      // (suggesting it's the biggest piece). Image gen path components:
      // gen_img_embeds, gen_head, image_decode, plus shared prepare_inputs_embeds.
      // Quality is research-tier (below SD1.5) but it's by far the smallest
      // model and uses transformers.js patterns (Xenova/HF browser ML team).
      {
        name: "image_decode q4f16",
        graph: f("janus", HF_JANUS, "onnx/image_decode_q4f16.onnx", 50 * 1024 * 1024),
      },
      {
        name: "gen_head q4f16",
        graph: f("janus", HF_JANUS, "onnx/gen_head_q4f16.onnx", 30 * 1024 * 1024),
      },
      {
        name: "gen_img_embeds q4f16",
        graph: f("janus", HF_JANUS, "onnx/gen_img_embeds_q4f16.onnx", 30 * 1024 * 1024),
      },
      {
        name: "language_model q4f16 (with external data)",
        graph: f("janus", HF_JANUS, "onnx/language_model_q4f16.onnx", 8 * 1024 * 1024),
        // language_model_q4f16 doesn't have a .onnx_data sibling per the
        // listing - only the unquantized language_model.onnx + data and
        // language_model_fp16.onnx + data have sidecars. The q4f16 is
        // monolithic. Will adjust if smoke errors with "data file missing".
      },
    ],
  },
  {
    id: "tensorstack-nitro-e",
    label: "TensorStack/Nitro-E-onnx (DiT, multi-resolution, non-Amuse)",
    components: [
      // TensorStack repo but no amuse_template.json. Tests whether
      // non-Amuse TensorStack exports also use com.microsoft.* contrib
      // ops (the killer for the *-amuse line). Multi-resolution variants:
      // we test the smallest (transformer/512) first - if even that fails
      // with contrib op errors, the whole TensorStack non-Amuse line is
      // also dead. Single text encoder, size unknown (could be CLIP-L
      // class or LM class - the latter is a budget killer).
      {
        name: "vae_decoder",
        graph: f("nitro", HF_NITRO_E, "vae_decoder/model.onnx", 5 * 1024 * 1024),
        externalData: {
          file: f("nitro", HF_NITRO_E, "vae_decoder/model.onnx.data", 100 * 1024 * 1024),
          pathInGraph: "model.onnx.data",
        },
      },
      {
        name: "text_encoder",
        graph: f("nitro", HF_NITRO_E, "text_encoder/model.onnx", 5 * 1024 * 1024),
        externalData: {
          file: f("nitro", HF_NITRO_E, "text_encoder/model.onnx.data", 500 * 1024 * 1024),
          pathInGraph: "model.onnx.data",
        },
      },
      {
        name: "transformer/512 (smallest variant)",
        graph: f("nitro", HF_NITRO_E, "transformer/512/model.onnx", 5 * 1024 * 1024),
        externalData: {
          file: f("nitro", HF_NITRO_E, "transformer/512/model.onnx.data", 800 * 1024 * 1024),
          pathInGraph: "model.onnx.data",
        },
      },
    ],
  },
  {
    id: "webnn-sdxl-turbo",
    label: "webnn/sdxl-turbo (SDXL-Turbo q4f16, Microsoft WebNN export)",
    components: [
      // Phase 1: just session.create on each component to verify q4f16
      // SDXL loads in stock ORT-web. The HF file list shows no .onnx_data
      // sidecars so we treat all four as monolithic q4-quantized files.
      // After load names are logged we add dummyRun specs in phase 2.
      // Use the non-qdq variants first; webnn also ships *_qdq_q4f16
      // (Quantize-Dequantize-aware) variants which are more accurate but
      // larger - try those only if base q4f16 has run-time issues.
      {
        name: "vae_decoder q4f16",
        graph: f("wsdxl", HF_WEBNN_SDXL, "onnx/vae_decoder_model_q4f16.onnx", 50 * 1024 * 1024),
        dummyRun: {
          // webnn/sdxl-turbo uses fp16 at the API boundary (Z-Image used
          // fp32 - MS isn't consistent across their q4f16 exports). Zero
          // tensors work in either dtype since fp16 zero is also Uint16
          // bit pattern 0x0000.
          dtype: "float16",
          inputs: [
            // SDXL uses 4-channel latents (NOT Z-Image's 16). 64x64 = 512px.
            { match: ["latent_sample", "latent"], shape: [1, 4, 64, 64] },
          ],
        },
      },
      {
        name: "text_encoder q4f16 (CLIP-L)",
        graph: f("wsdxl", HF_WEBNN_SDXL, "onnx/text_encoder_model_q4f16.onnx", 75 * 1024 * 1024),
        dummyRun: {
          dtype: "float16",
          inputs: [{ match: ["input_ids"], shape: [1, 77] }],
        },
      },
      {
        name: "text_encoder_2 q4f16 (CLIP-bigG)",
        graph: f("wsdxl", HF_WEBNN_SDXL, "onnx/text_encoder_2_model_q4f16.onnx", 350 * 1024 * 1024),
        dummyRun: {
          dtype: "float16",
          inputs: [{ match: ["input_ids"], shape: [1, 77] }],
        },
      },
      {
        name: "unet q4f16 (SDXL-Turbo, monolithic ~1.25 GB)",
        graph: f("wsdxl", HF_WEBNN_SDXL, "onnx/unet_model_q4f16.onnx", 1.25 * 1024 * 1024 * 1024),
        dummyRun: {
          dtype: "float16",
          repeats: 3,
          inputs: [
            // Standard SDXL UNet input convention. SDXL-Turbo native is
            // 512x512 -> 64x64 latent. 4-channel latents (vs Z-Image's 16).
            { match: ["sample"], shape: [1, 4, 64, 64] },
            { match: ["timestep"], shape: [1] },
            // SDXL concatenates CLIP-L (768) + bigG (1280) = 2048 hidden dim.
            { match: ["encoder_hidden_states", "encoder_hidden_state"], shape: [1, 77, 2048] },
            // Pooled bigG output, used in added_cond_kwargs.
            { match: ["text_embeds"], shape: [1, 1280] },
            // SDXL "added time ids": [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
            { match: ["time_ids"], shape: [1, 6] },
          ],
        },
      },
    ],
  },
  {
    id: "webnn-z-image-turbo",
    label: "webnn/Z-Image-Turbo (q4f16, Microsoft WebNN export)",
    components: [
      {
        // The headline test: does ORT-web's WebGPU EP support the q4
        // weight decompression kernels webnn/ uses? Confirmed yes
        // 2026-04-08 - session.create OK in 4.7s. Now dummyRun.
        // Microsoft's reference (microsoft/webnn-developer-preview demo
        // demos/z-image-turbo) uses fp32 inputs even though weights are
        // q4f16; the q4 part is internal storage, the API boundary is
        // fp32. sequenceLength is hardcoded 113 in MS's reference.
        name: "text_encoder q4f16",
        graph: f("zimage", HF_ZIMAGE, "onnx/text_encoder_model_q4f16.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f(
            "zimage",
            HF_ZIMAGE,
            "onnx/text_encoder_model_q4f16.onnx_data",
            1.2 * 1024 * 1024 * 1024,
          ),
          // Note the underscore: webnn uses ".onnx_data" not ".onnx.data"
          pathInGraph: "text_encoder_model_q4f16.onnx_data",
        },
        dummyRun: {
          dtype: "float32",
          inputs: [
            { match: ["input_ids"], shape: [1, 113] },
            { match: ["attention_mask"], shape: [1, 113] },
          ],
        },
      },
      {
        name: "vae_decoder f16",
        graph: f("zimage", HF_ZIMAGE, "onnx/vae_decoder_model_f16.onnx", 100 * 1024 * 1024),
        dummyRun: {
          dtype: "float32",
          inputs: [
            // Z-Image uses 16-channel latents (vs SD1.5's 4-channel),
            // closer to SD3/Flux family. 64x64 latent = 512x512 image.
            { match: ["latent_sample", "latent"], shape: [1, 16, 64, 64] },
          ],
        },
      },
      {
        name: "transformer q4f16 (DiT)",
        graph: f("zimage", HF_ZIMAGE, "onnx/transformer_model_q4f16.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f(
            "zimage",
            HF_ZIMAGE,
            "onnx/transformer_model_q4f16.onnx_data",
            1.5 * 1024 * 1024 * 1024,
          ),
          pathInGraph: "transformer_model_q4f16.onnx_data",
        },
        dummyRun: {
          dtype: "float32",
          repeats: 3,
          inputs: [
            // 5D! [batch, latent_channels=16, 1, H/8, W/8]. The extra "1"
            // dim comes from MS's reference and is unusual; if shape
            // mismatch, drop the extra 1 to make it 4D and retry.
            { match: ["hidden_states"], shape: [1, 16, 1, 64, 64] },
            { match: ["timestep"], shape: [1] },
            // Text encoder hidden dim is 2560 (Qwen2-2.5B class).
            { match: ["encoder_hidden_states", "encoder_hidden_state"], shape: [1, 113, 2560] },
          ],
        },
      },
    ],
  },
  {
    id: "gfodor-segmind-vega-fp16",
    label: "gfodor/segmind-vega-fp16-onnx (distilled SDXL, 2023)",
    components: [
      {
        name: "vae_decoder",
        graph: f("vega", HF_VEGA, "vae_decoder/model.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f("vega", HF_VEGA, "vae_decoder/model.onnx_data", 100 * 1024 * 1024),
          pathInGraph: "model.onnx_data",
        },
      },
      {
        name: "text_encoder (CLIP-L)",
        graph: f("vega", HF_VEGA, "text_encoder/model.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f("vega", HF_VEGA, "text_encoder/model.onnx_data", 250 * 1024 * 1024),
          pathInGraph: "model.onnx_data",
        },
      },
      {
        name: "text_encoder_2 (CLIP-bigG)",
        graph: f("vega", HF_VEGA, "text_encoder_2/model.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f("vega", HF_VEGA, "text_encoder_2/model.onnx_data", 1.4 * 1024 * 1024 * 1024),
          pathInGraph: "model.onnx_data",
        },
      },
      {
        name: "unet (distilled SDXL ~0.74B params)",
        graph: f("vega", HF_VEGA, "unet/model.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f("vega", HF_VEGA, "unet/model.onnx_data", 1.5 * 1024 * 1024 * 1024),
          pathInGraph: "model.onnx_data",
        },
      },
    ],
  },
  {
    id: "brad-agi-sana-0.6b",
    label: "brad-agi/sana-0.6b-onnx-webgpu (Sana 0.6B DiT, 1024 native)",
    components: [
      {
        // Monolithic int8-quantized Gemma-2 2B. ~2 GB and at risk of the
        // 1.72 GB std::bad_alloc cliff we hit for tlwu's monolithic SD1.5
        // UNet. Worth checking the int8 path first since that's the explicit
        // browser fix in this repo; if it bad_allocs we try the fp16 external
        // data version below.
        name: "text_encoder int8 (Gemma-2 2B, monolithic ~2 GB)",
        graph: f("sana", HF_SANA, "sana_text_encoder_int8.onnx", 2 * 1024 * 1024 * 1024),
      },
      {
        name: "text_encoder fp16 (Gemma-2 2B, external-data)",
        graph: f("sana", HF_SANA, "sana_text_encoder.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f("sana", HF_SANA, "sana_text_encoder.onnx.data", 4 * 1024 * 1024 * 1024),
          pathInGraph: "sana_text_encoder.onnx.data",
        },
      },
      {
        name: "vae 1024 (DC-AE 32x)",
        graph: f("sana", HF_SANA, "1024/sana_vae_1024.onnx", 5 * 1024 * 1024),
        externalData: {
          file: f("sana", HF_SANA, "1024/sana_vae_1024.onnx.data", 120 * 1024 * 1024),
          pathInGraph: "sana_vae_1024.onnx.data",
        },
      },
      {
        name: "dit 1024 (linear-attention DiT)",
        graph: f("sana", HF_SANA, "1024/sana_dit_1024.onnx", 5 * 1024 * 1024),
        externalData: {
          file: f("sana", HF_SANA, "1024/sana_dit_1024.onnx.data", 1.2 * 1024 * 1024 * 1024),
          pathInGraph: "sana_dit_1024.onnx.data",
        },
        // Phase 1: just session.create. We don't know exact Sana DiT input
        // shapes yet (Gemma-2 embed dim, max token length, latent channels
        // for DC-AE 32x). After session.create succeeds we read inputNames /
        // outputNames out of the log and add a dummyRun spec for phase 2.
      },
    ],
  },
  {
    id: "lemonteaa-sdxs-onnx",
    label: "lemonteaa/sdxs-onnx (original 2024 XAIR SDXS, stock diffusers)",
    components: [
      {
        name: "text_encoder (OpenCLIP-H/14, 1024-dim)",
        graph: f("sdxs", HF_SDXS, "text_encoder/model.onnx", 250 * 1024 * 1024),
      },
      {
        name: "vae_decoder",
        graph: f("sdxs", HF_SDXS, "vae_decoder/model.onnx", 95 * 1024 * 1024),
      },
      {
        name: "unet (monolithic, fp32, SD2.1-shape)",
        graph: f("sdxs", HF_SDXS, "unet/model.onnx", 350 * 1024 * 1024),
        dummyRun: {
          // SDXS-512 distilled from SD2.1: OpenCLIP-H/14 -> 1024-dim
          // text embeddings (vs CLIP-L's 768). Latent stays 4-channel /8.
          dtype: "float32",
          inputs: [
            { match: ["sample"], shape: [1, 4, 64, 64] },
            { match: ["timestep"], shape: [1] },
            { match: ["encoder_hidden_state"], shape: [1, 77, 1024] },
          ],
        },
      },
    ],
  },
  {
    id: "realistic-lcm-amuse",
    label: "TensorStack/Realistic-LCM-amuse",
    components: [
      {
        name: "text_encoder (CLIP-L)",
        graph: f("lcm", HF_LCM, "text_encoder/model.onnx", 250 * 1024 * 1024),
      },
      {
        name: "vae_decoder",
        graph: f("lcm", HF_LCM, "vae_decoder/model.onnx", 95 * 1024 * 1024),
      },
      {
        name: "unet (monolithic, expected to bad_alloc)",
        graph: f("lcm", HF_LCM, "unet/model.onnx", 1.72 * 1024 * 1024 * 1024),
        dummyRun: {
          inputs: [
            { match: ["sample"], shape: [1, 4, 64, 64] },
            { match: ["timestep"], shape: [1] },
            { match: ["encoder_hidden_state"], shape: [1, 77, 768] },
          ],
        },
      },
    ],
  },
  {
    id: "sdxl-lightning-amuse",
    label: "TensorStack/SDXL-Lightning-amuse",
    components: [
      {
        name: "text_encoder (CLIP-L)",
        graph: f("sdxl", HF_SDXL, "text_encoder/model.onnx", 250 * 1024 * 1024),
      },
      {
        name: "text_encoder_2 (CLIP-bigG)",
        graph: f("sdxl", HF_SDXL, "text_encoder_2/model.onnx", 1.4 * 1024 * 1024 * 1024),
      },
      {
        name: "vae_decoder",
        graph: f("sdxl", HF_SDXL, "vae_decoder/model.onnx", 95 * 1024 * 1024),
      },
      {
        name: "unet (external-data, ~5.13 GB)",
        graph: f("sdxl", HF_SDXL, "unet/model.onnx", 8 * 1024 * 1024),
        externalData: {
          file: f("sdxl", HF_SDXL, "unet/model.onnx.data", 5.13 * 1024 * 1024 * 1024),
          pathInGraph: "model.onnx.data",
        },
        dummyRun: {
          inputs: [
            { match: ["sample"], shape: [1, 4, 128, 128] },
            { match: ["timestep"], shape: [1] },
            { match: ["encoder_hidden_state"], shape: [1, 77, 2048] },
            { match: ["text_embeds"], shape: [1, 1280] },
            { match: ["time_ids"], shape: [1, 6] },
          ],
        },
      },
    ],
  },
  {
    id: "fastwan-vae",
    label: "FastWan 2.2 LightTAE VAE decoder (local, 36.6 MB)",
    components: [
      {
        name: "vae_decoder (LightTAE)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae_decoder.onnx", 36.6 * 1024 * 1024),
        dummyRun: {
          dtype: "float16",
          inputs: [{ match: ["latents"], shape: [1, 21, 48, 30, 52] }],
        },
      },
    ],
  },
  {
    id: "fastwan-vae-noise",
    label:
      "FastWan 2.2 LightTAE VAE decoder: pure Gaussian-noise input, renders frame 0",
    components: [
      {
        name: "vae_decoder + noise (diagnoses gray-output complaint from full pipeline)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae_decoder.onnx", 36.6 * 1024 * 1024),
        dummyRun: {
          dtype: "float16",
          inputs: [
            {
              match: ["latents"],
              shape: [1, 21, 48, 30, 52],
              gaussian: true,
            },
          ],
          renderOutput: {
            match: ["frames"],
            numFrames: 81,
            channels: 3,
            height: 480,
            width: 832,
            frameIndex: 0,
            pixelRange: "-1to1",
          },
        },
      },
    ],
  },
  {
    id: "fastwan-transformer",
    label: "FastWan 2.2 transformer per-block (local, shell_pre + block_00 + shell_post)",
    components: [
      // Smoke uses trace-tiny shapes (1 latent frame, 4x4 spatial = 4 tokens,
      // 64 text tokens) to confirm op coverage without allocating the full
      // 8190-token attention. Full-scale attention memory is a separate risk
      // tracked in the worklog.
      {
        name: "shell_pre (patch embed + RoPE + condition embed, 179.7 MB)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/shell_pre.onnx", 179.7 * 1024 * 1024),
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 48, 1, 4, 4], dtype: "float16" },
            { match: ["timestep"], shape: [1, 4], dtype: "int64" },
            { match: ["encoder_hidden_states"], shape: [1, 64, 4096], dtype: "float16" },
          ],
        },
      },
      {
        name: "block_00 (one of 30 transformer blocks, 327.5 MB)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/block_00.onnx", 327.5 * 1024 * 1024),
        dummyRun: {
          // freqs_cos/sin are fp32 (RoPE computes in fp32 for precision);
          // everything else fp16 matches the diffusers trace dtype.
          inputs: [
            { match: ["hidden_states"], shape: [1, 4, 3072], dtype: "float16" },
            { match: ["encoder_hidden_states"], shape: [1, 64, 3072], dtype: "float16" },
            { match: ["timestep_proj"], shape: [1, 4, 6, 3072], dtype: "float16" },
            { match: ["freqs_cos"], shape: [1, 4, 1, 128], dtype: "float32" },
            { match: ["freqs_sin"], shape: [1, 4, 1, 128], dtype: "float32" },
          ],
        },
      },
      {
        name: "shell_post (final norm + proj + unpatchify, 1.2 MB)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/shell_post.onnx", 1.2 * 1024 * 1024),
        dummyRun: {
          // ppf/pph/ppw are scalar int64 size args passed through to the
          // unpatchify reshape (they correspond to the runtime's latent dims).
          inputs: [
            { match: ["hidden_states"], shape: [1, 4, 3072], dtype: "float16" },
            { match: ["temb"], shape: [1, 4, 3072], dtype: "float16" },
            // ppf*pph*ppw must equal seq_len (4) and match the upstream
            // hidden_states token count. Trace-tiny values: 1*2*2 = 4.
            { match: ["ppf"], shape: [], dtype: "int64", fill: 1 },
            { match: ["pph"], shape: [], dtype: "int64", fill: 2 },
            { match: ["ppw"], shape: [], dtype: "int64", fill: 2 },
          ],
        },
      },
    ],
  },
  {
    id: "fastwan-transformer-q4",
    label: "FastWan 2.2 transformer q4f16 (full-shape, weight-only 4-bit)",
    components: [
      // Same components as fastwan-transformer-full but q4f16 weights.
      // Block ~92 MB each (28% of fp16), shell_pre 52 MB, shell_post 1.2 MB
      // fp16 (unquantized). Tests whether MatMulNBits kernels work in
      // ORT-web WebGPU + whether q4 gives a meaningful speedup.
      {
        name: "shell_pre q4f16 (full-shape)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer-q4f16/shell_pre.onnx", 52 * 1024 * 1024),
        externalData: {
          file: f(
            "fastwan",
            LOCAL_FASTWAN,
            "onnx/transformer-q4f16/shell_pre.onnx.data",
            52 * 1024 * 1024,
          ),
          pathInGraph: "shell_pre.onnx.data",
        },
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 48, 21, 30, 52], dtype: "float16" },
            { match: ["timestep"], shape: [1, 8190], dtype: "int64" },
            { match: ["encoder_hidden_states"], shape: [1, 512, 4096], dtype: "float16" },
          ],
          repeats: 2,
        },
      },
      {
        name: "block_00 q4f16 (full-shape, 8190 tokens)",
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          "onnx/transformer-q4f16/block_00.onnx",
          92.4 * 1024 * 1024,
        ),
        externalData: {
          file: f(
            "fastwan",
            LOCAL_FASTWAN,
            "onnx/transformer-q4f16/block_00.onnx.data",
            92 * 1024 * 1024,
          ),
          pathInGraph: "block_00.onnx.data",
        },
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["encoder_hidden_states"], shape: [1, 512, 3072], dtype: "float16" },
            { match: ["timestep_proj"], shape: [1, 8190, 6, 3072], dtype: "float16" },
            { match: ["freqs_cos"], shape: [1, 8190, 1, 128], dtype: "float32" },
            { match: ["freqs_sin"], shape: [1, 8190, 1, 128], dtype: "float32" },
          ],
          repeats: 2,
        },
      },
      {
        name: "shell_post (fp16, unquantized)",
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          "onnx/transformer-q4f16/shell_post.onnx",
          1.2 * 1024 * 1024,
        ),
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["temb"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["ppf"], shape: [], dtype: "int64", fill: 21 },
            { match: ["pph"], shape: [], dtype: "int64", fill: 15 },
            { match: ["ppw"], shape: [], dtype: "int64", fill: 26 },
          ],
          repeats: 2,
        },
      },
    ],
  },
  {
    id: "fastwan-transformer-full",
    label: "FastWan 2.2 transformer full-shape (8190 tokens - tests attention memory)",
    components: [
      // Real inference shape. Latent: 21 frames x 30x52 spatial, patch 1x2x2
      // -> seq_len = 21 * 15 * 26 = 8190. Text seq 512.
      // This is the existential test: ORT-web's WebGPU attention must not
      // materialize the full 24-head x 8190x8190 score matrix (~3.1 GB in
      // fp16) or we'll hit maxBufferSize (2 GB). Run #2 (warm) is the real
      // number - #1 includes shader compile.
      {
        name: "shell_pre full-shape (21f x 30x52 latent, 512 text tokens)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/shell_pre.onnx", 179.7 * 1024 * 1024),
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 48, 21, 30, 52], dtype: "float16" },
            { match: ["timestep"], shape: [1, 8190], dtype: "int64" },
            { match: ["encoder_hidden_states"], shape: [1, 512, 4096], dtype: "float16" },
          ],
          repeats: 2,
        },
      },
      {
        name: "block_00 full-shape (8190 tokens - attention memory test)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/block_00.onnx", 327.5 * 1024 * 1024),
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["encoder_hidden_states"], shape: [1, 512, 3072], dtype: "float16" },
            { match: ["timestep_proj"], shape: [1, 8190, 6, 3072], dtype: "float16" },
            { match: ["freqs_cos"], shape: [1, 8190, 1, 128], dtype: "float32" },
            { match: ["freqs_sin"], shape: [1, 8190, 1, 128], dtype: "float32" },
          ],
          repeats: 2,
        },
      },
      {
        name: "shell_post full-shape (ppf=21, pph=15, ppw=26)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/shell_post.onnx", 1.2 * 1024 * 1024),
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["temb"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["ppf"], shape: [], dtype: "int64", fill: 21 },
            { match: ["pph"], shape: [], dtype: "int64", fill: 15 },
            { match: ["ppw"], shape: [], dtype: "int64", fill: 26 },
          ],
          repeats: 2,
        },
      },
    ],
  },
  {
    id: "fastwan-text-encoder-q4",
    label: "FastWan 2.2 text encoder q4f16 (UMT5-XXL layer_00 + shell_post)",
    components: [
      // Per-layer UMT5 export: each layer takes pre-embedded hidden states
      // (JS does the token embedding lookup from embedding.bin) and a 4D
      // extended attention mask. 512 seq_len, d_model 4096. 24 identical
      // layers; smoke runs layer_00 + shell_post.
      {
        name: "layer_00 q4f16 (UMT5 block, 108.6 MB)",
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          "onnx/text-encoder-q4f16/layer_00.onnx",
          108.6 * 1024 * 1024,
        ),
        externalData: {
          file: f(
            "fastwan",
            LOCAL_FASTWAN,
            "onnx/text-encoder-q4f16/layer_00.onnx.data",
            108 * 1024 * 1024,
          ),
          pathInGraph: "layer_00.onnx.data",
        },
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 512, 4096], dtype: "float16" },
            { match: ["attention_mask"], shape: [1, 1, 1, 512], dtype: "float16" },
          ],
          repeats: 2,
        },
      },
      {
        name: "shell_post (final UMT5LayerNorm, fp16 unquantized)",
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          "onnx/text-encoder-q4f16/shell_post.onnx",
          0.02 * 1024 * 1024,
        ),
        dummyRun: {
          inputs: [{ match: ["hidden_states"], shape: [1, 512, 4096], dtype: "float16" }],
          repeats: 2,
        },
      },
    ],
  },
];

const cache = new ModelCache({ opfsDirName: "intabai-model-smoke" });

const $log = document.getElementById("log") as HTMLTextAreaElement;
const $status = document.getElementById("status-line")!;
const $progress = document.getElementById("progress") as HTMLProgressElement;
const $select = document.getElementById("model-select") as HTMLSelectElement;
const $component = document.getElementById("component-select") as HTMLSelectElement;
const $run = document.getElementById("run-btn") as HTMLButtonElement;
const $clear = document.getElementById("clear-btn") as HTMLButtonElement;

function log(line = "") {
  $log.value += line + "\n";
  $log.scrollTop = $log.scrollHeight;
}

function setStatus(s: string) {
  $status.textContent = s;
}

function renderOutputFrame(
  spec: NonNullable<Component["dummyRun"]>["renderOutput"],
  results: ort.InferenceSession.OnnxValueMapType,
): void {
  if (!spec) return;
  const keys = Object.keys(results);
  const outName = keys.find((k) => spec.match.some((m) => k.includes(m))) ?? keys[0];
  const out = results[outName];
  if (!out) {
    log(`renderOutput: no output matched ${spec.match.join("/")}`);
    return;
  }
  const raw = out.data;
  const canvas = document.getElementById("smoke-output-canvas") as HTMLCanvasElement | null;
  if (!canvas) return;
  const { numFrames, channels, height, width } = spec;
  const frameIndex = spec.frameIndex ?? 0;
  if (channels !== 3) {
    log(`renderOutput: only channels=3 supported (got ${channels})`);
    return;
  }
  const plane = height * width;
  const expectedLen = numFrames * channels * plane;
  if (raw.length !== expectedLen) {
    log(
      `renderOutput: length ${raw.length} != expected ${expectedLen} ` +
        `(numFrames=${numFrames}, C=${channels}, H=${height}, W=${width})`,
    );
    return;
  }
  // Convert fp16 bits (Uint16Array) or fp32 (Float32Array) to a single
  // float value per pixel per channel.
  const readF = (idx: number): number => {
    if (raw instanceof Uint16Array) return f16BitsToF32(raw[idx]);
    if (raw instanceof Float32Array) return raw[idx];
    return Number(raw[idx as keyof typeof raw]);
  };
  const range = spec.pixelRange ?? "-1to1";
  const mapPixel = range === "0to1" ? (v: number) => v * 255 : (v: number) => (v * 0.5 + 0.5) * 255;
  const base = frameIndex * channels * plane;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    const r = readF(base + i);
    const g = readF(base + plane + i);
    const b = readF(base + 2 * plane + i);
    rgba[i * 4 + 0] = Math.max(0, Math.min(255, Math.round(mapPixel(r))));
    rgba[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(mapPixel(g))));
    rgba[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(mapPixel(b))));
    rgba[i * 4 + 3] = 255;
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  log(`renderOutput: drew frame ${frameIndex} of "${outName}" to canvas`);
}

async function dumpEnvironment() {
  log("=== environment ===");
  log(`time: ${new Date().toISOString()}`);
  log(`ua: ${navigator.userAgent}`);
  log(
    `platform: ${(navigator as unknown as { userAgentData?: { platform?: string } }).userAgentData?.platform ?? "n/a"}`,
  );
  log(`hardwareConcurrency: ${navigator.hardwareConcurrency}`);
  const dm = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  log(`deviceMemory: ${dm ?? "n/a"} GB`);
  log(`webgpu present: ${"gpu" in navigator}`);
  if ("gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.info as unknown as Record<string, string> | undefined;
        log(`gpu vendor: ${info?.vendor ?? "?"}`);
        log(`gpu architecture: ${info?.architecture ?? "?"}`);
        log(`gpu device: ${info?.device ?? "?"}`);
        log(`gpu description: ${info?.description ?? "?"}`);
        const limits = adapter.limits;
        log(
          `maxBufferSize: ${limits.maxBufferSize} (${(limits.maxBufferSize / 1024 / 1024 / 1024).toFixed(2)} GB)`,
        );
        log(`maxStorageBufferBindingSize: ${limits.maxStorageBufferBindingSize}`);
        log(`features: ${[...adapter.features].join(", ")}`);
      } else {
        log("gpu adapter: null (no compatible adapter)");
      }
    } catch (err) {
      log(`gpu adapter error: ${(err as Error).message}`);
    }
  }
  log(
    `ort version: ${(ort as unknown as { env: { versions: Record<string, string> } }).env.versions?.common ?? "?"}`,
  );
  log("");
}

async function downloadAll(comps: Component[]) {
  const files: ModelFile[] = [];
  for (const c of comps) {
    files.push(c.graph);
    if (c.externalData) files.push(c.externalData.file);
  }
  const cached = await cache.getCachedStatus(files);
  const missing = files.filter((file) => !cached.get(file.id));
  log(`=== download ===`);
  log(
    `files needed: ${files.length}, already cached: ${files.length - missing.length}, missing: ${missing.length}`,
  );
  for (const m of missing) {
    log(`  to fetch: ${m.name} (~${(m.sizeBytes / 1024 / 1024).toFixed(1)} MB est)`);
  }
  if (missing.length === 0) {
    log("nothing to download");
    log("");
    return;
  }
  const t0 = performance.now();
  const totalSize = missing.reduce((s, f) => s + f.sizeBytes, 0);
  // Downloads run in parallel, so events interleave. Track per-fileId latest
  // bytesLoaded and sum for the global bar.
  const bytesByFile = new Map<string, number>();
  await cache.downloadFiles(files, (p) => {
    bytesByFile.set(p.fileId, p.bytesLoaded);
    let totalLoaded = 0;
    for (const v of bytesByFile.values()) totalLoaded += v;
    setStatus(
      `downloading ${(totalLoaded / 1024 / 1024).toFixed(1)} / ${(totalSize / 1024 / 1024).toFixed(1)} MB`,
    );
    $progress.value = (totalLoaded / Math.max(1, totalSize)) * 100;
  });
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  log(`download done in ${dt}s`);
  log("");
}

function providers(): string[] {
  const out: string[] = [];
  if ("gpu" in navigator) out.push("webgpu");
  out.push("wasm");
  return out;
}

function findName(names: readonly string[], matchers: string[]): string | null {
  for (const m of matchers) {
    for (const n of names) {
      if (n.toLowerCase().includes(m.toLowerCase())) return n;
    }
  }
  return null;
}

async function tryComponent(c: Component) {
  log(`--- ${c.name} ---`);
  // Log actual on-disk sizes (rather than the rough estimates in
  // CANDIDATES). Cheap stat via FileSystemFileHandle.getFile().
  const graphSize = await cache.getFileSize(c.graph).catch(() => -1);
  log(
    `graph: ${c.graph.name}${graphSize >= 0 ? ` (${(graphSize / 1024 / 1024).toFixed(1)} MB)` : ""}`,
  );
  if (c.externalData) {
    const dataSize = await cache.getFileSize(c.externalData.file).catch(() => -1);
    log(
      `externalData: ${c.externalData.file.name}${dataSize >= 0 ? ` (${(dataSize / 1024 / 1024).toFixed(1)} MB)` : ""} (as "${c.externalData.pathInGraph}")`,
    );
  }

  setStatus(`loading ${c.name}`);

  // Use blob URLs (avoids copying multi-GB files into wasm heap as ArrayBuffer).
  const { url: graphUrl, revoke: revokeGraph } = await cache.loadFileAsBlobUrl(c.graph);
  let revokeData = () => {};
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: providers(),
    graphOptimizationLevel: "all",
  };
  if (c.externalData) {
    const { url, revoke } = await cache.loadFileAsBlobUrl(c.externalData.file);
    revokeData = revoke;
    (
      sessionOptions as unknown as { externalData: Array<{ path: string; data: string }> }
    ).externalData = [{ path: c.externalData.pathInGraph, data: url }];
  }

  let session: ort.InferenceSession | null = null;
  const tCreate0 = performance.now();
  try {
    session = await ort.InferenceSession.create(graphUrl, sessionOptions);
    const dt = (performance.now() - tCreate0).toFixed(0);
    log(`session.create OK in ${dt}ms`);
    log(`  inputNames: ${session.inputNames.join(", ")}`);
    log(`  outputNames: ${session.outputNames.join(", ")}`);
  } catch (err) {
    const dt = (performance.now() - tCreate0).toFixed(0);
    log(`session.create FAILED in ${dt}ms: ${(err as Error).message}`);
  } finally {
    revokeGraph();
    revokeData();
  }

  if (session && c.dummyRun) {
    setStatus(`dummy run ${c.name}`);
    const dummyDtype = c.dummyRun.dtype ?? "float16";
    type AnyDtype = "float16" | "float32" | "int64" | "int32";
    const buildTensor = (dtype: AnyDtype, len: number, shape: number[], fill = 0) => {
      if (dtype === "float32") {
        const a = new Float32Array(len);
        if (fill !== 0) a.fill(fill);
        return new ort.Tensor("float32", a, shape);
      }
      if (dtype === "int64") {
        const a = new BigInt64Array(len);
        if (fill !== 0) a.fill(BigInt(fill));
        return new ort.Tensor("int64", a, shape);
      }
      if (dtype === "int32") {
        const a = new Int32Array(len);
        if (fill !== 0) a.fill(fill);
        return new ort.Tensor("int32", a, shape);
      }
      // float16 zero is Uint16 0x0000; nonzero fp16 fill would need a cast
      // helper - skip for now since fill is only used for int scalars.
      return new ort.Tensor("float16", new Uint16Array(len), shape);
    };
    try {
      const feeds: Record<string, ort.Tensor> = {};
      for (const spec of c.dummyRun.inputs) {
        const inputName = findName(session.inputNames, spec.match);
        if (!inputName) {
          log(`  dummyRun: no input matched ${spec.match.join("/")} - skipping`);
          continue;
        }
        const len = spec.shape.reduce((a, b) => a * b, 1);
        if (spec.gaussian) {
          const rng = mulberry32(12345);
          const noise = gaussianNoise(len, rng);
          const d = spec.dtype ?? dummyDtype;
          if (d === "float16") {
            feeds[inputName] = new ort.Tensor("float16", f32ToF16Array(noise), spec.shape);
          } else if (d === "float32") {
            feeds[inputName] = new ort.Tensor("float32", noise, spec.shape);
          } else {
            log(`  dummyRun: gaussian unsupported for dtype ${d}, falling back to zeros`);
            feeds[inputName] = buildTensor(d, len, spec.shape);
          }
          continue;
        }
        if (spec.dtype) {
          feeds[inputName] = buildTensor(spec.dtype, len, spec.shape, spec.fill);
          continue;
        }
        // Heuristic fallback: transformer-style int64 inputs (input_ids,
        // attention_mask, token_type_ids) get a BigInt64Array of zeros.
        const isIntInput = spec.match.some((m) => /input_ids|attention_mask|token_type/i.test(m));
        if (isIntInput) {
          feeds[inputName] = new ort.Tensor("int64", new BigInt64Array(len), spec.shape);
        } else {
          feeds[inputName] = buildTensor(dummyDtype, len, spec.shape);
        }
      }
      const repeats = c.dummyRun.repeats ?? 1;
      const tRun0 = performance.now();
      let lastResults: ort.InferenceSession.OnnxValueMapType | null = null;
      try {
        lastResults = await session.run(feeds);
        const dt0 = performance.now() - tRun0;
        if (repeats > 1) {
          log(`session.run #1 (cold, includes shader compile) OK in ${dt0.toFixed(0)}ms`);
          for (let i = 2; i <= repeats; i++) {
            const t = performance.now();
            lastResults = await session.run(feeds);
            log(`session.run #${i} (warm) OK in ${(performance.now() - t).toFixed(0)}ms`);
          }
        } else {
          log(`session.run (dummy zeros) OK in ${dt0.toFixed(0)}ms`);
        }
        if (c.dummyRun.renderOutput && lastResults) {
          renderOutputFrame(c.dummyRun.renderOutput, lastResults);
        }
      } catch (err) {
        const msg = (err as Error).message;
        // Detect dtype mismatches and rebuild offending inputs.
        // Two common cases:
        //  (a) timestep wants int64 but we sent float - swap timestep to int64
        //  (b) input_ids wants int32 but the int-input heuristic sent int64 -
        //      swap all int-shaped feeds to int32 (Int32Array)
        const wantsInt64 = /expected:\s*\(tensor\(int64\)\)/i.test(msg);
        const wantsInt32 = /expected:\s*\(tensor\(int32\)\)/i.test(msg);
        if (wantsInt64 || wantsInt32) {
          const fix = wantsInt32 ? "int32" : "int64";
          log(`  dummyRun: dtype rejected (${msg.slice(0, 120)}), retry with ${fix}`);
          for (const spec of c.dummyRun.inputs) {
            const inputName = findName(session.inputNames, spec.match);
            if (!inputName) continue;
            const len = spec.shape.reduce((a, b) => a * b, 1);
            const isTimestep = spec.match.some((m) => m.toLowerCase().includes("timestep"));
            const isIntInput = spec.match.some((m) =>
              /input_ids|attention_mask|token_type/i.test(m),
            );
            // For wantsInt64: only swap timestep (the "current" wrong input)
            // For wantsInt32: only swap int-shaped inputs to int32
            if (wantsInt64 && isTimestep) {
              feeds[inputName] = new ort.Tensor("int64", new BigInt64Array(len), spec.shape);
            } else if (wantsInt32 && isIntInput) {
              feeds[inputName] = new ort.Tensor("int32", new Int32Array(len), spec.shape);
            }
          }
          const tRun1 = performance.now();
          try {
            await session.run(feeds);
            log(
              `session.run (dummy zeros, ${fix} fallback) OK in ${(performance.now() - tRun1).toFixed(0)}ms`,
            );
          } catch (err2) {
            log(`session.run FAILED: ${(err2 as Error).message}`);
          }
        } else {
          log(`session.run FAILED in ${(performance.now() - tRun0).toFixed(0)}ms: ${msg}`);
        }
      }
    } catch (err) {
      log(`dummyRun setup error: ${(err as Error).message}`);
    }
  }

  if (session) {
    try {
      await session.release();
    } catch {
      // ignore
    }
  }
  log("");
}

async function runSmoke() {
  $run.disabled = true;
  $clear.disabled = true;
  $log.value = "";
  $progress.value = 0;
  try {
    const cand = CANDIDATES.find((c) => c.id === $select.value)!;
    const sel = $component.value;
    const comps =
      sel === "__all__" ? cand.components : cand.components.filter((c) => c.name === sel);
    log(`### smoke test: ${cand.label}`);
    if (sel !== "__all__") log(`(component filter: ${sel})`);
    log("");
    await dumpEnvironment();
    await downloadAll(comps);
    log("=== component checks ===");
    for (const c of comps) {
      try {
        await tryComponent(c);
      } catch (err) {
        log(`unexpected error on ${c.name}: ${(err as Error).message}`);
        log("");
      }
    }
    log("### done");
    setStatus("done");
  } catch (err) {
    log(`fatal: ${(err as Error).stack ?? (err as Error).message}`);
    setStatus("error");
  } finally {
    $run.disabled = false;
    $clear.disabled = false;
    $progress.value = 100;
  }
}

async function clearCache() {
  $clear.disabled = true;
  setStatus("clearing cache");
  try {
    await cache.clearAll();
    setStatus("cache cleared");
    log("(smoke cache cleared)");
  } catch (err) {
    setStatus(`clear failed: ${(err as Error).message}`);
  } finally {
    $clear.disabled = false;
  }
}

function repopulateComponents() {
  const cand = CANDIDATES.find((c) => c.id === $select.value)!;
  $component.innerHTML = "";
  const all = document.createElement("option");
  all.value = "__all__";
  all.textContent = "(all components)";
  $component.appendChild(all);
  for (const c of cand.components) {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent = c.name;
    $component.appendChild(opt);
  }
}

function populateModels() {
  $select.innerHTML = "";
  for (const c of CANDIDATES) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label;
    $select.appendChild(opt);
  }
}

populateModels();
$select.addEventListener("change", repopulateComponents);
repopulateComponents();

$run.addEventListener("click", () => void runSmoke());
$clear.addEventListener("click", () => void clearCache());

setStatus("idle");
