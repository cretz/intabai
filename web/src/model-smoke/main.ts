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
import { copyF16Bits, f16BitsToF32, f16ToF32Array, f32ToF16Array } from "../sd15/fp16";

/** One-line stats for diagnosing whether a captured/output fp16 tensor is
 *  finite, distinguishing NaN from +/-Inf (the LTX bisect on
 *  dec_block_00_res_0 produced bothNan=243816 at conv3d but other-AI's
 *  CPU-ORT replay showed conv3d max=+30.7, so we need to know which
 *  category the "non-finite" actually is). */
function statsFp16Brief(log: (s: string) => void, name: string, bits: Uint16Array): void {
  const n = bits.length;
  const f32 = f16ToF32Array(bits);
  let min = Infinity, max = -Infinity, sum = 0, sumSq = 0, nan = 0, posInf = 0, negInf = 0;
  for (let i = 0; i < n; i++) {
    const v = f32[i];
    if (Number.isNaN(v)) { nan++; continue; }
    if (v === Infinity) { posInf++; continue; }
    if (v === -Infinity) { negInf++; continue; }
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
  }
  const finite = n - nan - posInf - negInf;
  const mean = finite > 0 ? sum / finite : 0;
  const variance = finite > 0 ? Math.max(0, sumSq / finite - mean * mean) : 0;
  const std = Math.sqrt(variance);
  log(
    `stats[${name}] n=${n} finite=${finite} nan=${nan} +inf=${posInf} -inf=${negInf} ` +
      `min=${min.toFixed(4)} max=${max.toFixed(4)} mean=${mean.toFixed(4)} std=${std.toFixed(4)}`,
  );
}

function dumpFp16Tensor(log: (s: string) => void, name: string, bits: Uint16Array): void {
  const n = bits.length;
  // Stats in f32 space for sanity.
  let min = Infinity,
    max = -Infinity,
    sum = 0,
    nan = 0,
    zeros = 0;
  const f32 = f16ToF32Array(bits);
  for (let i = 0; i < n; i++) {
    const v = f32[i];
    if (Number.isNaN(v)) {
      nan++;
      continue;
    }
    if (v === 0) zeros++;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / Math.max(1, n - nan);
  log(
    `stats[${name}] n=${n} min=${min.toFixed(4)} max=${max.toFixed(4)} mean=${mean.toFixed(6)} nan=${nan} zeros=${zeros}`,
  );
  const hex = Array.from(bits.subarray(0, 32), (b) => b.toString(16).padStart(4, "0")).join(",");
  log(`${name}[0:32] hex=${hex}`);
  const f32slice = Array.from(f32.subarray(0, 32), (v) => v.toFixed(4)).join(",");
  log(`${name}[0:32] f32=[${f32slice}]`);
}

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

// ?ortverbose=1 raises ORT-web log level so op-level dispatch traces appear in
// the console. Useful for locating the last op before a WebGPU device-lost /
// TDR: the final "[V:...]" line identifies which kernel was running.
if (new URLSearchParams(location.search).get("ortverbose") === "1") {
  ort.env.logLevel = "verbose";
}

interface Component {
  /** Display name in the log. */
  name: string;
  /** ONNX graph file. */
  graph: ModelFile;
  /**
   * Optional external-data sidecar. When present, passed to ORT via
   * sessionOptions.externalData with the path the graph references.
   * Use the array form for multi-shard exports (e.g. LTX transformer's
   * 7 shard.bin files).
   */
  externalData?:
    | { file: ModelFile; pathInGraph: string }
    | Array<{ file: ModelFile; pathInGraph: string }>;
  /**
   * Override the default "all" graph optimization level. Use "disabled" when
   * exposing intermediate graph outputs so ORT-web's memory planner doesn't
   * promote every live MatMul output to a persistent buffer and OOM at
   * session create (Fastwan block_00_debug.onnx symptom).
   */
  graphOptLevel?: "disabled" | "basic" | "extended" | "all";
  /**
   * Override execution provider preference for this component. Default is
   * the global `providers()` (webgpu first, wasm fallback). Force a single
   * EP for wasm-vs-webgpu A/B diffs. Pair two components with the same
   * `diffPair` value to auto-diff outputs across EPs.
   */
  executionProviders?: ("webgpu" | "wasm")[];
  /**
   * Tag pairing two components whose captured outputs should be diffed
   * after both run. When the second component with the same diffPair tag
   * finishes, every output named in both is compared in fp32 space and
   * the first one with mean-relative-diff above ~5% is logged loudly. Use
   * with `executionProviders: ["wasm"]` on one and `["webgpu"]` on the
   * other to localise EP-specific kernel bugs to a single op.
   */
  diffPair?: string;
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
      /** Deterministic fp16 fill: val[i] = sin(i * freq + offset) * amplitude,
       *  cast fp32 -> fp16. Matches the fill in
       *  web/scripts/fastwan/probe-attn-matmul.py so browser and Python see the
       *  same bytes. Only meaningful for dtype: "float16".
       *  Default freq/amp if only `sinOffset` provided: freq=0.0017 amp=8.0. */
      sinFill?: { offset: number; freq?: number; amplitude?: number };
      /** If true, dump stats + first-32-hex of this input after building
       *  it (for parity diff with Python). */
      dumpInput?: boolean;
      /** If set, reuse a tensor produced by a previous component in the
       *  same run (output name matched by this substring). Lets us chain
       *  e.g. decoder_init's cache_out_NN into decoder_step's cache_in_NN
       *  so the step runs against real signal rather than zeros. */
      fromPrevOutput?: string;
      /** If set, fetch the LTXBLK26-format bundle at `url` once per run
       *  and use the tensor named `tensorName` from it as this input.
       *  Bundle format is produced by LTX_DUMP_BLOCK26_INPUTS in
       *  web/src/ltx/generate.ts. Lets a model-smoke diff feed a block
       *  with REAL captured production-shape activations instead of
       *  synthetic sinFill, which is the only way to localise a kernel
       *  bug without NaN-overflow contamination. dtype is always float16. */
      fromCapture?: { url: string; tensorName: string };
    }>;
    /** If true, after dummy run, iterate outputs and emit stats + first-32-hex
     *  per output so we can diff against a CPU reference log. */
    dumpOutputs?: boolean;
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
    id: "fastwan-vae-kl",
    label: "FastWan 2.2 full AutoencoderKLWan decoder (streaming, local, 2.2 GB)",
    components: (() => {
      // The full 3D-causal-conv VAE. Exported as two ONNX graphs (init,
      // step) that share a 32-tensor cache I/O contract. Smoke runs each
      // with the exact cache shapes from the export probe (see
      // notes/export-vae-kl-streaming-init.log, slot 00..31).
      const CACHE_SHAPES: number[][] = [
        [1, 48, 2, 30, 52],
        ...Array(11).fill([1, 1024, 2, 30, 52]),
        ...Array(7).fill([1, 1024, 2, 60, 104]),
        [1, 1024, 2, 120, 208],
        ...Array(5).fill([1, 512, 2, 120, 208]),
        [1, 512, 2, 240, 416],
        ...Array(6).fill([1, 256, 2, 240, 416]),
      ];
      const cacheInputs = CACHE_SHAPES.map((shape, i) => {
        const nn = i.toString().padStart(2, "0");
        return {
          match: [`cache_in_${nn}`],
          shape,
          dtype: "float16" as const,
          // Chain init's cache_out_NN into step's cache_in_NN so the step
          // runs against real signal rather than zeros. Falls back to zeros
          // if init wasn't run (e.g. component filter = step only).
          fromPrevOutput: `cache_out_${nn}`,
        };
      });
      return [
        {
          name: "decoder_init (frame 0: latent -> 1 frame + 32 caches, 1.1 GB, Conv2D-decomposed)",
          graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae/decoder_init.onnx", 1117366418),
          dummyRun: {
            dtype: "float16" as const,
            inputs: [{ match: ["latent"], shape: [1, 48, 1, 30, 52], gaussian: true }],
            dumpOutputs: true,
          },
        },
        {
          name: "decoder_step (frame 1+: latent + 32 caches -> 4 frames + 32 caches, 1.1 GB, Conv2D-decomposed)",
          graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae/decoder_step.onnx", 1110666229),
          dummyRun: {
            dtype: "float16" as const,
            inputs: [
              { match: ["latent"], shape: [1, 48, 1, 30, 52], gaussian: true },
              ...cacheInputs,
            ],
            dumpOutputs: true,
          },
        },
      ];
    })(),
  },
  {
    id: "fastwan-vae-kl-split-init",
    label:
      "FastWan 2.2 full AutoencoderKLWan decoder_init, 34-way halved-resnet split (TDR test, local, 1.1 GB)",
    components: (() => {
      // 34-way sub-graphs produced by split-fastwan-vae-decoder.py --preset fine40.
      // Each resnet is cut at its conv1/Conv output, so one session.run covers
      // only one Conv3D + norm/nonlinearity (half a resnet). The 20-way split
      // still TDR'd on mid_block resnets (5.5s each at the TDR edge); halving
      // drops each resnet half to ~2-3s, comfortably under the watchdog.
      type Part = {
        name: string;
        sizeBytes: number;
        input: string;
        output: string;
        extraInput?: string;
      };
      const parts: Part[] = [
        {
          name: "part_00_pre",
          sizeBytes: 2669148,
          input: "latent",
          output: "/decoder/conv_in/Conv_output_0",
        },
        {
          name: "part_01_mid_r0_a",
          sizeBytes: 56638207,
          input: "/decoder/conv_in/Conv_output_0",
          output: "/decoder/mid_block/resnets.0/conv1/Conv_output_0",
        },
        {
          name: "part_01_mid_r0_b",
          sizeBytes: 56638737,
          input: "/decoder/mid_block/resnets.0/conv1/Conv_output_0",
          output: "/decoder/mid_block/resnets.0/Add_output_0",
          extraInput: "/decoder/conv_in/Conv_output_0",
        },
        {
          name: "part_03_mid_attn",
          sizeBytes: 8430181,
          input: "/decoder/mid_block/resnets.0/Add_output_0",
          output: "/decoder/mid_block/attentions.0/Add_1_output_0",
        },
        {
          name: "part_04_mid_r1_a",
          sizeBytes: 56638326,
          input: "/decoder/mid_block/attentions.0/Add_1_output_0",
          output: "/decoder/mid_block/resnets.1/conv1/Conv_output_0",
        },
        {
          name: "part_04_mid_r1_b",
          sizeBytes: 56638865,
          input: "/decoder/mid_block/resnets.1/conv1/Conv_output_0",
          output: "/decoder/mid_block/resnets.1/Add_output_0",
          extraInput: "/decoder/mid_block/attentions.0/Add_1_output_0",
        },
        {
          name: "part_06_up0_r0_a",
          sizeBytes: 56638630,
          input: "/decoder/mid_block/resnets.1/Add_output_0",
          output: "/decoder/up_blocks.0/resnets.0/conv1/Conv_output_0",
        },
        {
          name: "part_06_up0_r0_b",
          sizeBytes: 56639182,
          input: "/decoder/up_blocks.0/resnets.0/conv1/Conv_output_0",
          output: "/decoder/up_blocks.0/resnets.0/Add_output_0",
          extraInput: "/decoder/mid_block/resnets.1/Add_output_0",
        },
        {
          name: "part_08_up0_r1_a",
          sizeBytes: 56638636,
          input: "/decoder/up_blocks.0/resnets.0/Add_output_0",
          output: "/decoder/up_blocks.0/resnets.1/conv1/Conv_output_0",
        },
        {
          name: "part_08_up0_r1_b",
          sizeBytes: 56639188,
          input: "/decoder/up_blocks.0/resnets.1/conv1/Conv_output_0",
          output: "/decoder/up_blocks.0/resnets.1/Add_output_0",
          extraInput: "/decoder/up_blocks.0/resnets.0/Add_output_0",
        },
        {
          name: "part_10_up0_r2_a",
          sizeBytes: 56638636,
          input: "/decoder/up_blocks.0/resnets.1/Add_output_0",
          output: "/decoder/up_blocks.0/resnets.2/conv1/Conv_output_0",
        },
        {
          name: "part_10_up0_r2_b",
          sizeBytes: 56639188,
          input: "/decoder/up_blocks.0/resnets.2/conv1/Conv_output_0",
          output: "/decoder/up_blocks.0/resnets.2/Add_output_0",
          extraInput: "/decoder/up_blocks.0/resnets.1/Add_output_0",
        },
        {
          name: "part_12_up0_upsample",
          sizeBytes: 18909499,
          input: "/decoder/up_blocks.0/resnets.2/Add_output_0",
          output: "/decoder/up_blocks.0/Add_output_0",
          extraInput: "/decoder/mid_block/resnets.1/Add_output_0",
        },
        {
          name: "part_13_up1_r0_a",
          sizeBytes: 56638648,
          input: "/decoder/up_blocks.0/Add_output_0",
          output: "/decoder/up_blocks.1/resnets.0/conv1/Conv_output_0",
        },
        {
          name: "part_13_up1_r0_b",
          sizeBytes: 56639172,
          input: "/decoder/up_blocks.1/resnets.0/conv1/Conv_output_0",
          output: "/decoder/up_blocks.1/resnets.0/Add_output_0",
          extraInput: "/decoder/up_blocks.0/Add_output_0",
        },
        {
          name: "part_15_up1_r1_a",
          sizeBytes: 56638636,
          input: "/decoder/up_blocks.1/resnets.0/Add_output_0",
          output: "/decoder/up_blocks.1/resnets.1/conv1/Conv_output_0",
        },
        {
          name: "part_15_up1_r1_b",
          sizeBytes: 56639188,
          input: "/decoder/up_blocks.1/resnets.1/conv1/Conv_output_0",
          output: "/decoder/up_blocks.1/resnets.1/Add_output_0",
          extraInput: "/decoder/up_blocks.1/resnets.0/Add_output_0",
        },
        {
          name: "part_17_up1_r2_a",
          sizeBytes: 56638636,
          input: "/decoder/up_blocks.1/resnets.1/Add_output_0",
          output: "/decoder/up_blocks.1/resnets.2/conv1/Conv_output_0",
        },
        {
          name: "part_17_up1_r2_b",
          sizeBytes: 56639188,
          input: "/decoder/up_blocks.1/resnets.2/conv1/Conv_output_0",
          output: "/decoder/up_blocks.1/resnets.2/Add_output_0",
          extraInput: "/decoder/up_blocks.1/resnets.1/Add_output_0",
        },
        {
          name: "part_19_up1_upsample",
          sizeBytes: 18909554,
          input: "/decoder/up_blocks.1/resnets.2/Add_output_0",
          output: "/decoder/up_blocks.1/Add_output_0",
          extraInput: "/decoder/up_blocks.0/Add_output_0",
        },
        {
          name: "part_20_up2_r0_a",
          sizeBytes: 28326072,
          input: "/decoder/up_blocks.1/Add_output_0",
          output: "/decoder/up_blocks.2/resnets.0/conv1/Conv_output_0",
        },
        {
          name: "part_20_up2_r0_b",
          sizeBytes: 15224930,
          input: "/decoder/up_blocks.2/resnets.0/conv1/Conv_output_0",
          output: "/decoder/up_blocks.2/resnets.0/Add_output_0",
          extraInput: "/decoder/up_blocks.1/Add_output_0",
        },
        {
          name: "part_22_up2_r1_a",
          sizeBytes: 14169260,
          input: "/decoder/up_blocks.2/resnets.0/Add_output_0",
          output: "/decoder/up_blocks.2/resnets.1/conv1/Conv_output_0",
        },
        {
          name: "part_22_up2_r1_b",
          sizeBytes: 14169812,
          input: "/decoder/up_blocks.2/resnets.1/conv1/Conv_output_0",
          output: "/decoder/up_blocks.2/resnets.1/Add_output_0",
          extraInput: "/decoder/up_blocks.2/resnets.0/Add_output_0",
        },
        {
          name: "part_24_up2_r2_a",
          sizeBytes: 14169260,
          input: "/decoder/up_blocks.2/resnets.1/Add_output_0",
          output: "/decoder/up_blocks.2/resnets.2/conv1/Conv_output_0",
        },
        {
          name: "part_24_up2_r2_b",
          sizeBytes: 14169812,
          input: "/decoder/up_blocks.2/resnets.2/conv1/Conv_output_0",
          output: "/decoder/up_blocks.2/resnets.2/Add_output_0",
          extraInput: "/decoder/up_blocks.2/resnets.1/Add_output_0",
        },
        {
          name: "part_26_up2_upsample",
          sizeBytes: 4750922,
          input: "/decoder/up_blocks.2/resnets.2/Add_output_0",
          output: "/decoder/up_blocks.2/Add_output_0",
          extraInput: "/decoder/up_blocks.1/Add_output_0",
        },
        {
          name: "part_27_up3_r0_a",
          sizeBytes: 7090872,
          input: "/decoder/up_blocks.2/Add_output_0",
          output: "/decoder/up_blocks.3/resnets.0/conv1/Conv_output_0",
        },
        {
          name: "part_27_up3_r0_b",
          sizeBytes: 3820130,
          input: "/decoder/up_blocks.3/resnets.0/conv1/Conv_output_0",
          output: "/decoder/up_blocks.3/resnets.0/Add_output_0",
          extraInput: "/decoder/up_blocks.2/Add_output_0",
        },
        {
          name: "part_29_up3_r1_a",
          sizeBytes: 3551404,
          input: "/decoder/up_blocks.3/resnets.0/Add_output_0",
          output: "/decoder/up_blocks.3/resnets.1/conv1/Conv_output_0",
        },
        {
          name: "part_29_up3_r1_b",
          sizeBytes: 3551956,
          input: "/decoder/up_blocks.3/resnets.1/conv1/Conv_output_0",
          output: "/decoder/up_blocks.3/resnets.1/Add_output_0",
          extraInput: "/decoder/up_blocks.3/resnets.0/Add_output_0",
        },
        {
          name: "part_31_up3_r2_a",
          sizeBytes: 3551404,
          input: "/decoder/up_blocks.3/resnets.1/Add_output_0",
          output: "/decoder/up_blocks.3/resnets.2/conv1/Conv_output_0",
        },
        {
          name: "part_31_up3_r2_b",
          sizeBytes: 3551956,
          input: "/decoder/up_blocks.3/resnets.2/conv1/Conv_output_0",
          output: "/decoder/up_blocks.3/resnets.2/Add_output_0",
          extraInput: "/decoder/up_blocks.3/resnets.1/Add_output_0",
        },
        {
          name: "part_33_tail",
          sizeBytes: 32130432,
          input: "/decoder/up_blocks.3/resnets.2/Add_output_0",
          output: "frames",
        },
      ];
      return parts.map((p, i) => {
        const inputs: Array<{
          match: string[];
          shape: number[];
          gaussian?: boolean;
          fromPrevOutput?: string;
        }> = [
          i === 0
            ? { match: [p.input], shape: [1, 48, 1, 30, 52], gaussian: true }
            : { match: [p.input], shape: [], fromPrevOutput: p.input },
        ];
        if (p.extraInput) {
          inputs.push({ match: [p.extraInput], shape: [], fromPrevOutput: p.extraInput });
        }
        return {
          name: `decoder_init ${p.name}`,
          graph: f("fastwan", LOCAL_FASTWAN, `onnx/vae/decoder_init_${p.name}.onnx`, p.sizeBytes),
          dummyRun: {
            dtype: "float16" as const,
            inputs,
          },
        };
      });
    })(),
  },
  {
    id: "fastwan-part-13a-pre-conv-tiled",
    label: "FastWan part_13a: pre + 6 Conv tiles across sessions (TDR fix test)",
    components: (() => {
      const tiles = Array.from({ length: 6 }, (_, i) => ({
        name: `part_13a_conv tile ${i + 1}/6 (H=10 slice)`,
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          `onnx/vae/decoder_init_part_13_up1_r0_a_conv_tile${i}of6.onnx`,
          56625890,
        ),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [
            {
              match: ["/decoder/up_blocks.1/resnets.0/conv1/Pad_output_0"],
              shape: [],
              fromPrevOutput: "/decoder/up_blocks.1/resnets.0/conv1/Pad_output_0",
            },
          ],
        },
      }));
      return [
        {
          name: "part_13a_pre (RMSNorm+SiLU+Pad)",
          graph: f(
            "fastwan",
            LOCAL_FASTWAN,
            "onnx/vae/decoder_init_part_13_up1_r0_a_pre.onnx",
            9506,
          ),
          dummyRun: {
            dtype: "float16" as const,
            inputs: [
              {
                match: ["/decoder/up_blocks.0/Add_output_0"],
                shape: [1, 1024, 1, 60, 104],
                gaussian: true,
              },
            ],
          },
        },
        ...tiles,
      ];
    })(),
  },
  {
    id: "fastwan-part-13a-pre-conv-split",
    label: "FastWan part_13a split at Pad (_pre then _conv, separate sessions, TDR fix test)",
    // Part_13a TDRs standalone because ~18 pre-Conv ops + one big Conv3D all
    // run inside one session.run, so cumulative GPU time exceeds Windows D3D12
    // TDR (~2-3s). Splitting at the Pad output into two session.runs forces a
    // GPU idle between them (mapAsync drains the queue at session boundary),
    // resetting the TDR clock. Probe A proved the Conv alone is under budget
    // at this shape; the pre side is ~18 cheap element-wise ops. If both pass,
    // the fix generalizes to every half-resnet up1 component (60x104). Up2 and
    // up3 will still need per-Conv tile-across-sessions on top.
    components: [
      {
        name: "part_13a_pre (RMSNorm+SiLU+Pad)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae/decoder_init_part_13_up1_r0_a_pre.onnx", 9506),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [
            {
              match: ["/decoder/up_blocks.0/Add_output_0"],
              shape: [1, 1024, 1, 60, 104],
              gaussian: true,
            },
          ],
        },
      },
      {
        name: "part_13a_conv (Conv3D 1024->1024)",
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          "onnx/vae/decoder_init_part_13_up1_r0_a_conv.onnx",
          56625941,
        ),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [
            {
              match: ["/decoder/up_blocks.1/resnets.0/conv1/Pad_output_0"],
              shape: [],
              fromPrevOutput: "/decoder/up_blocks.1/resnets.0/conv1/Pad_output_0",
            },
          ],
        },
      },
    ],
  },
  {
    id: "fastwan-part-13a-standalone",
    label:
      "FastWan part_13a standalone (real op chain + weights, random input, cumulative-TDR test, 54 MB)",
    // Probe A (just Conv3D, 1024->1024 60x104) ran clean standalone. Part_13a
    // in the 34-way split (same Conv + RMS-norm + SiLU + Pad + Slice) TDRs when
    // chained. Running part_13a alone with random input isolates: if this TDRs,
    // the delta vs probe A is purely the non-Conv ops adding cumulative GPU
    // time within one session.run -> fix is splitting to one Conv per session.
    // If clean, something about prior-part activations or cross-session state
    // is implicated instead.
    components: [
      {
        name: "part_13a standalone (random input)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae/decoder_init_part_13_up1_r0_a.onnx", 56638648),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [
            {
              match: ["/decoder/up_blocks.0/Add_output_0"],
              shape: [1, 1024, 1, 60, 104],
              gaussian: true,
            },
          ],
        },
      },
    ],
  },
  {
    id: "fastwan-conv3d-chain-3x",
    label: "FastWan Conv3D chain 3x n=5 sequential (cumulative-submit TDR test)",
    // Three separate sessions of chain-n=5. Each session.run is ~9s wall on
    // its own; between them JS awaits, letting the GPU queue drain. If the
    // real VAE TDR is cumulative-within-one-submit, 3 sequential runs should
    // be clean (27s of compute but split across 3 drain boundaries). If they
    // still TDR, the theory is wrong and something else is in play.
    components: Array.from({ length: 3 }, (_, i) => ({
      name: `Conv3D chain n=5 run ${i + 1}/3 (88 Gop x 5, separate session)`,
      graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae/conv3d_chain_n5.onnx", 70.8 * 1024 * 1024),
      dummyRun: {
        dtype: "float16" as const,
        inputs: [{ match: ["x"], shape: [1, 512, 2, 60, 104], gaussian: true }],
      },
    })),
  },
  {
    id: "fastwan-conv3d-chain",
    label: "FastWan Conv3D chain TDR probe (1/3/5 Convs back-to-back, 127 MB)",
    components: (() => {
      // Tests cumulative-submit TDR: is it per-dispatch or per-session.run?
      // Each Conv is 512->512 k3p1 on [1,512,2,60,104] — ~88 Gop, which ran
      // clean as a single-tile probe. If N=1 clean, N=3 TDR, then ORT-web
      // batches everything into one command buffer and we need to split.
      const sizes = [
        { n: 1, bytes: 14.2 * 1024 * 1024 },
        { n: 3, bytes: 42.5 * 1024 * 1024 },
        { n: 5, bytes: 70.8 * 1024 * 1024 },
      ];
      return sizes.map((p) => ({
        name: `Conv3D chain n=${p.n} (${p.n} x 88 Gop 512->512 k3p1 [1,512,2,60,104])`,
        graph: f("fastwan", LOCAL_FASTWAN, `onnx/vae/conv3d_chain_n${p.n}.onnx`, p.bytes),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [{ match: ["x"], shape: [1, 512, 2, 60, 104], gaussian: true }],
        },
      }));
    })(),
  },
  {
    id: "fastwan-conv3d-probe",
    label: "FastWan Conv3D TDR probe (5 shapes covering heaviest decoder convs, ~110 MB)",
    components: (() => {
      // Five single-Conv3D probes at T_in=4 (matches step graph T). Each is
      // pad=0 kernel=3x3x3 matching real-VAE Conv3Ds. Running all in sequence
      // tells us which shape TDRs in isolation. Prior run proved E
      // ([1,256,2,240,416] 256->256) runs in 5.6s; not tested here.
      const probes: {
        tag: string;
        inC: number;
        outC: number;
        H: number;
        W: number;
        bytes: number;
      }[] = [
        {
          tag: "Btile_1024to512_15x208",
          inC: 1024,
          outC: 512,
          H: 17,
          W: 210,
          bytes: 28.3 * 1024 * 1024,
        },
        {
          tag: "G_1024to1024_30x52",
          inC: 1024,
          outC: 1024,
          H: 32,
          W: 54,
          bytes: 56.6 * 1024 * 1024,
        },
        {
          tag: "A_1024to1024_60x104",
          inC: 1024,
          outC: 1024,
          H: 62,
          W: 106,
          bytes: 56.6 * 1024 * 1024,
        },
        {
          tag: "B_1024to512_120x208",
          inC: 1024,
          outC: 512,
          H: 122,
          W: 210,
          bytes: 28.3 * 1024 * 1024,
        },
        {
          tag: "C_512to512_120x208",
          inC: 512,
          outC: 512,
          H: 122,
          W: 210,
          bytes: 14.2 * 1024 * 1024,
        },
        {
          tag: "D_512to256_240x416",
          inC: 512,
          outC: 256,
          H: 242,
          W: 418,
          bytes: 7.1 * 1024 * 1024,
        },
        { tag: "F_256to3_240x416", inC: 256, outC: 3, H: 242, W: 418, bytes: 0.1 * 1024 * 1024 },
      ];
      return probes.map((p) => ({
        name: `Conv3D ${p.tag} [1,${p.inC},4,${p.H},${p.W}] -> [1,${p.outC},2,${p.H - 2},${p.W - 2}]`,
        graph: f("fastwan", LOCAL_FASTWAN, `onnx/vae/conv3d_probe_${p.tag}.onnx`, p.bytes),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [{ match: ["x"], shape: [1, p.inC, 4, p.H, p.W], gaussian: true }],
        },
      }));
    })(),
  },
  {
    id: "fastwan-vae-final-block",
    label: "FastWan 2.2 AutoencoderKLWan final up_block probe (worst-case Conv3D, 25 MB)",
    components: [
      // Isolates up_blocks[-1] at full 240x416 spatial to answer: does per-block
      // chunking dodge the TDR, or is a single Conv3D dispatch on 240x416x256ch
      // the hang? Captured in frame-0 mode (T=1); real step-mode T=4 would be
      // ~4x heavier, so if this TDRs chunking won't save us.
      {
        name: "final up_block (x[1,512,1,240,416] + 6 caches -> y + caches)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/vae/probe_final_block.onnx", 25.1 * 1024 * 1024),
        dummyRun: {
          dtype: "float16" as const,
          inputs: [
            { match: ["x"], shape: [1, 512, 1, 240, 416], gaussian: true },
            { match: ["cache_in_25"], shape: [1, 512, 1, 240, 416] },
            { match: ["cache_in_26"], shape: [1, 256, 1, 240, 416] },
            { match: ["cache_in_27"], shape: [1, 256, 1, 240, 416] },
            { match: ["cache_in_28"], shape: [1, 256, 1, 240, 416] },
            { match: ["cache_in_29"], shape: [1, 256, 1, 240, 416] },
            { match: ["cache_in_30"], shape: [1, 256, 1, 240, 416] },
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
    label: "FastWan 2.2 LightTAE VAE decoder: pure Gaussian-noise input, renders frame 0",
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
        graph: f(
          "fastwan",
          LOCAL_FASTWAN,
          "onnx/transformer-q4f16/shell_pre.onnx",
          52 * 1024 * 1024,
        ),
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
    id: "fastwan-block-00-debug",
    label: "FastWan block_00_debug (10 exposed taps; graphOpt disabled to dodge session OOM)",
    components: [
      {
        name: "block_00_debug full-shape (8190 tokens, graphOpt=disabled)",
        graph: f("fastwan", LOCAL_FASTWAN, "onnx/transformer/block_00_debug.onnx", 183_706),
        externalData: {
          file: f(
            "fastwan",
            LOCAL_FASTWAN,
            "onnx/transformer/block_00_debug.onnx.data",
            327_362_560,
          ),
          pathInGraph: "block_00_debug.onnx.data",
        },
        graphOptLevel: "disabled",
        dummyRun: {
          inputs: [
            { match: ["hidden_states"], shape: [1, 8190, 3072], dtype: "float16" },
            { match: ["encoder_hidden_states"], shape: [1, 512, 3072], dtype: "float16" },
            { match: ["timestep_proj"], shape: [1, 8190, 6, 3072], dtype: "float16" },
            { match: ["freqs_cos"], shape: [1, 8190, 1, 128], dtype: "float32" },
            { match: ["freqs_sin"], shape: [1, 8190, 1, 128], dtype: "float32" },
          ],
          repeats: 1,
        },
      },
    ],
  },
  {
    id: "fastwan-attn-probe",
    label: "FastWan attn probe (isolated softmax(Q.Kt/sqrt)·V at heads=24, d=128, varying seq)",
    components: [
      // Minimal ONNX: Transpose(K) -> MatMul -> Mul(scale) -> Softmax -> MatMul(.V).
      // Inputs filled deterministically via sinFill so browser + Python
      // see identical fp16 bytes. Diff first-32 hex of output against the
      // hex printed by web/scripts/fastwan/probe-attn-matmul.py.
      // Intermediate scores buffer: heads*seq^2*2 bytes.
      //   seq=1024:  48 MB  (well within maxBufferSize)
      //   seq=2048: 192 MB
      //   seq=4096: 768 MB  (may OOM on mobile)
      //   seq=8190: 3.22 GB (exceeds typical maxBufferSize; skip until others prove bug)
      ...(
        [
          { seq: 1024, heads: 24, tag: "" },
          { seq: 2048, heads: 24, tag: "" },
          { seq: 4096, heads: 24, tag: "" },
          { seq: 8190, heads: 1, tag: "-h1" },
          { seq: 8190, heads: 24, tag: "-h24" },
        ] as const
      ).map(({ seq, heads, tag }) => ({
        name: `probe seq=${seq} heads=${heads} (intermediate scores ~${((heads * seq * seq * 2) / 1024 / 1024).toFixed(0)} MB)`,
        graph: f("fastwan", LOCAL_FASTWAN, `onnx/probe/probe-${seq}${tag}.onnx`, 2048),
        graphOptLevel: "disabled" as const,
        dummyRun: {
          inputs: [
            {
              match: ["q"],
              shape: [1, heads, seq, 128],
              dtype: "float16" as const,
              sinFill: { offset: 0.0 },
            },
            {
              match: ["k"],
              shape: [1, heads, seq, 128],
              dtype: "float16" as const,
              sinFill: { offset: 1.1 },
            },
            {
              match: ["v"],
              shape: [1, heads, seq, 128],
              dtype: "float16" as const,
              sinFill: { offset: 2.2 },
            },
          ],
          dumpOutputs: true,
          repeats: 1,
        },
      })),
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
  // LTX 2B distilled transformer block_00, tapped to expose every intermediate
  // node output. Runs the same tapped graph twice (wasm + webgpu) with
  // identical sinFill inputs, then diffs every output and reports the
  // first node whose mean-rel-diff exceeds 5%. Used to localise the
  // "webgpu green-tile garbage vs wasm coherent output" regression
  // documented in worklog.md "Active bug" to a single ORT-web kernel.
  //
  // Build the tapped graph with:
  //   uv run --with onnx python web/scripts/ltx/expose_block_intermediates.py \
  //       notes/models/ltx/hf-repo/onnx/transformer-fp16/block_00.onnx
  //
  // n_tokens=128, n_text=64 match the python script's shape-inference dims
  // so tap shapes line up.
  ...(() => {
    const LTX_LOCAL = "/local-models/ltx";
    const N_TOK = 128;
    const N_TXT = 64;
    const ltxBlockTappedFiles = () => ({
      graph: f("ltx", LTX_LOCAL, "onnx/transformer-fp16/block_00_tapped.onnx", 240 * 1024),
      data: f(
        "ltx",
        LTX_LOCAL,
        "onnx/transformer-fp16/block_00.onnx.data",
        128 * 1024 * 1024,
      ),
    });
    // Realistic input magnitudes - the previous run with default sinFill
    // amp=8 overflowed fp16 in layer-norm variance + adaln modulation
    // and produced NaN floods on BOTH EPs, masking real kernel divergence.
    // Real-pipeline ranges:
    //   hidden_states post-patchify: std~0.5-1
    //   freqs_cos/sin: literally cos/sin in [-1, 1]
    //   timestep_mod = 1 + small adaln scale, so the sin part is small
    //   encoder_proj (T5 hidden, projected): magnitudes ~1-3
    //   encoder_attn_bias: 0 for real tokens (we have no padding here)
    const sharedDummyRun = {
      inputs: [
        { match: ["hidden_states"], shape: [1, N_TOK, 2048], sinFill: { offset: 0.0, amplitude: 1.0 } },
        { match: ["freqs_cos"], shape: [1, N_TOK, 2048], sinFill: { offset: 1.1, amplitude: 1.0 } },
        { match: ["freqs_sin"], shape: [1, N_TOK, 2048], sinFill: { offset: 2.2, amplitude: 1.0 } },
        { match: ["timestep_mod"], shape: [1, 1, 12288], sinFill: { offset: 3.3, amplitude: 0.3 } },
        { match: ["encoder_proj"], shape: [1, N_TXT, 2048], sinFill: { offset: 4.4, amplitude: 1.5 } },
        { match: ["encoder_attn_bias"], shape: [1, 1, N_TXT], dtype: "float16" as const },
      ],
      dumpOutputs: false,
      repeats: 1,
    };
    const files = ltxBlockTappedFiles();
    return [
      {
        id: "ltx-block-00-diff",
        label: "LTX block_00 tapped (wasm vs webgpu per-op diff)",
        components: [
          {
            name: "block_00_tapped (wasm)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "block_00.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["wasm" as const],
            diffPair: "ltx-block-00",
            dummyRun: sharedDummyRun,
          },
          {
            name: "block_00_tapped (webgpu)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "block_00.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["webgpu" as const],
            diffPair: "ltx-block-00",
            dummyRun: sharedDummyRun,
          },
        ],
      } satisfies Candidate,
      // REAL captured inputs at production shape (N_TOK=196). hidden_states
      // = shell_pre_hidden from the LTXALL28 all-blocks bundle (the tensor
      // fed into block_00 in pass1 step 1). aux tensors (freqs_cos/sin,
      // timestep_mod, encoder_proj, encoder_attn_bias) come from the
      // LTXBLK26 block_26-input bundle - those are shell_pre outputs and
      // are identical for every block in the same step. Both bundles
      // captured from the WebGPU run, so this isolates "what does block_00
      // do differently between EPs given the EXACT input wgpu produced?".
      // First op with mean-rel-diff > 5% AND non-trivial maxAbs is the
      // first kernel to localize. shell_pre_hidden is essentially
      // identical between wgpu and wasm (max|d|=0.008) so this is the
      // cleanest place to find single-kernel fp16 accumulation bugs.
      ((): Candidate => {
        const allUrl = `${LTX_LOCAL}/captures/ltx_all_blocks_pass1_step1_webgpu.bin`;
        const auxUrl = `${LTX_LOCAL}/captures/block_26_pass1_step1_inputs_webgpu.bin`;
        const inputs = [
          { match: ["hidden_states"], shape: [], fromCapture: { url: allUrl, tensorName: "shell_pre_hidden" } },
          { match: ["freqs_cos"], shape: [], fromCapture: { url: auxUrl, tensorName: "freqs_cos" } },
          { match: ["freqs_sin"], shape: [], fromCapture: { url: auxUrl, tensorName: "freqs_sin" } },
          { match: ["timestep_mod"], shape: [], fromCapture: { url: auxUrl, tensorName: "timestep_mod" } },
          { match: ["encoder_proj"], shape: [], fromCapture: { url: auxUrl, tensorName: "encoder_proj" } },
          { match: ["encoder_attn_bias"], shape: [], fromCapture: { url: auxUrl, tensorName: "encoder_attn_bias" } },
        ];
        const realDummy = { inputs, dumpOutputs: false, repeats: 1 };
        return {
          id: "ltx-block-00-real-diff",
          label: "LTX block_00 tapped, REAL captured inputs (wasm vs webgpu per-op diff)",
          components: [
            {
              name: "block_00_tapped real (wasm)",
              graph: files.graph,
              externalData: { file: files.data, pathInGraph: "block_00.onnx.data" },
              graphOptLevel: "disabled" as const,
              executionProviders: ["wasm" as const],
              diffPair: "ltx-block-00-real",
              dummyRun: realDummy,
            },
            {
              name: "block_00_tapped real (webgpu)",
              graph: files.graph,
              externalData: { file: files.data, pathInGraph: "block_00.onnx.data" },
              graphOptLevel: "disabled" as const,
              executionProviders: ["webgpu" as const],
              diffPair: "ltx-block-00-real",
              dummyRun: realDummy,
            },
          ],
        };
      })(),
    ];
  })(),
  // LTX block_26 tapped (wasm vs webgpu). In a real pass1 step 1 with
  // n_tokens=196 we observe block_26 producing a webgpu-only outlier
  // (max=+141 vs wasm max=+33) on top of compounded drift through
  // blocks 00-25. This candidate runs block_26 with the same synthetic
  // sinFill inputs that block_00 was clean against. If block_26 also
  // comes back clean at synthetic inputs, the production divergence is
  // amplification of upstream drift, not a per-block kernel bug, and
  // the next test is a real-captured-input replay.
  //
  // Build with:
  //   uv run --with onnx python web/scripts/ltx/expose_block_intermediates.py \
  //       notes/models/ltx/hf-repo/onnx/transformer-fp16/block_26.onnx
  ...(() => {
    const LTX_LOCAL = "/local-models/ltx";
    const N_TOK = 128;
    const N_TXT = 64;
    const ltxBlockTappedFiles = () => ({
      graph: f("ltx", LTX_LOCAL, "onnx/transformer-fp16/block_26_tapped.onnx", 240 * 1024),
      data: f(
        "ltx",
        LTX_LOCAL,
        "onnx/transformer-fp16/block_26.onnx.data",
        128 * 1024 * 1024,
      ),
    });
    // hidden_states amplitude lowered from 1.0 -> 0.3: at amp=1.0 the FF
    // up-projection in block_26 overflows fp16 and floods both EPs with
    // NaN, masking real kernel divergence. block_00 stayed in range at
    // amp=1.0 because its weights are smaller. Production never hits the
    // overflow regime because real post-26-block activations have
    // correlation structure that the FF weights handle gracefully;
    // uncorrelated sinFill at the same magnitude is far harsher on the
    // matmul accumulator.
    const sharedDummyRun = {
      inputs: [
        { match: ["hidden_states"], shape: [1, N_TOK, 2048], sinFill: { offset: 0.0, amplitude: 0.3 } },
        { match: ["freqs_cos"], shape: [1, N_TOK, 2048], sinFill: { offset: 1.1, amplitude: 1.0 } },
        { match: ["freqs_sin"], shape: [1, N_TOK, 2048], sinFill: { offset: 2.2, amplitude: 1.0 } },
        { match: ["timestep_mod"], shape: [1, 1, 12288], sinFill: { offset: 3.3, amplitude: 0.3 } },
        { match: ["encoder_proj"], shape: [1, N_TXT, 2048], sinFill: { offset: 4.4, amplitude: 1.5 } },
        { match: ["encoder_attn_bias"], shape: [1, 1, N_TXT], dtype: "float16" as const },
      ],
      dumpOutputs: false,
      repeats: 1,
    };
    const files = ltxBlockTappedFiles();
    return [
      {
        id: "ltx-block-26-diff",
        label: "LTX block_26 tapped (wasm vs webgpu per-op diff)",
        components: [
          {
            name: "block_26_tapped (wasm)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "block_26.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["wasm" as const],
            diffPair: "ltx-block-26",
            dummyRun: sharedDummyRun,
          },
          {
            name: "block_26_tapped (webgpu)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "block_26.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["webgpu" as const],
            diffPair: "ltx-block-26",
            dummyRun: sharedDummyRun,
          },
        ],
      } satisfies Candidate,
      // Same block_26_tapped graph as above, but fed with REAL captured
      // pass1-step-1 production-shape inputs from a generate run with
      // LTX_DUMP_BLOCK26_INPUTS=true (see web/src/ltx/generate.ts). This
      // is the only clean per-op diff: synthetic sinFill at any
      // amplitude floods the cross-attention K-norm path with NaN/Inf
      // overflow because uncorrelated inputs break RMSNorm's variance
      // estimate, contaminating the per-element comparison.
      //
      // Capture file lives at notes/models/ltx/hf-repo/captures/
      // (served by the local-model-proxy at runtime).
      // 4-cell input/exec swap experiment to localize the +141 spike.
      // Two candidates, two pairs:
      //   pair "wgpu-cap" (fixed input = webgpu capture): A=wgpu/wgpu vs B=wasm/wgpu
      //   pair "wasm-cap" (fixed input = wasm   capture): C=wgpu/wasm vs D=wasm/wasm
      // Reading:
      //   - small diff in BOTH pairs => kernels innocent on either input;
      //     bug is upstream cumulative input drift amplified at block_26.
      //   - large diff in BOTH pairs => some op in block_26 genuinely
      //     diverges between EPs regardless of input. Localize via per-op
      //     dumpOutputs.
      //   - one pair small, one large => the "input" that triggers
      //     divergence is itself an artifact of the producing EP; bug is
      //     in an op that's input-sensitive in a way only one EP's
      //     intermediate values exercise.
      // Capture files are produced with LTX_DUMP_BLOCK26_INPUTS=true in
      // generate.ts, named block_26_pass1_step1_inputs_{webgpu,wasm}.bin.
      ...((): Candidate[] => {
        const capUrl = (ep: "webgpu" | "wasm") =>
          `${LTX_LOCAL}/captures/block_26_pass1_step1_inputs_${ep}.bin`;
        const inputsFromCap = (ep: "webgpu" | "wasm") => [
          { match: ["hidden_states"], shape: [], fromCapture: { url: capUrl(ep), tensorName: "hidden_states" } },
          { match: ["freqs_cos"], shape: [], fromCapture: { url: capUrl(ep), tensorName: "freqs_cos" } },
          { match: ["freqs_sin"], shape: [], fromCapture: { url: capUrl(ep), tensorName: "freqs_sin" } },
          { match: ["timestep_mod"], shape: [], fromCapture: { url: capUrl(ep), tensorName: "timestep_mod" } },
          { match: ["encoder_proj"], shape: [], fromCapture: { url: capUrl(ep), tensorName: "encoder_proj" } },
          { match: ["encoder_attn_bias"], shape: [], fromCapture: { url: capUrl(ep), tensorName: "encoder_attn_bias" } },
        ];
        const cell = (
          name: string,
          ep: "webgpu" | "wasm",
          capEp: "webgpu" | "wasm",
          diffPair: string,
        ) => ({
          name,
          graph: files.graph,
          externalData: { file: files.data, pathInGraph: "block_26.onnx.data" },
          graphOptLevel: "disabled" as const,
          executionProviders: [ep],
          diffPair,
          dummyRun: {
            inputs: inputsFromCap(capEp),
            dumpOutputs: false,
            repeats: 1,
          },
        });
        return [
          {
            id: "ltx-block-26-real-diff-wgpu-cap",
            label: "LTX block_26 tapped, REAL inputs from WEBGPU capture (wasm vs webgpu)",
            components: [
              cell("block_26 wgpu-cap (webgpu)", "webgpu", "webgpu", "ltx-block-26-real-wgpu-cap"),
              cell("block_26 wgpu-cap (wasm)",   "wasm",   "webgpu", "ltx-block-26-real-wgpu-cap"),
            ],
          } satisfies Candidate,
          {
            id: "ltx-block-26-real-diff-wasm-cap",
            label: "LTX block_26 tapped, REAL inputs from WASM capture (wasm vs webgpu)",
            components: [
              cell("block_26 wasm-cap (webgpu)", "webgpu", "wasm", "ltx-block-26-real-wasm-cap"),
              cell("block_26 wasm-cap (wasm)",   "wasm",   "wasm", "ltx-block-26-real-wasm-cap"),
            ],
          } satisfies Candidate,
        ];
      })(),
    ];
  })(),
  // LTX shell_pre tapped (wasm vs webgpu). shell_pre runs once per
  // denoising step and produces 5 of the 6 inputs to every transformer
  // block (timestep_mod, freqs_cos/sin, encoder_proj, encoder_attn_bias).
  // If shell_pre's webgpu output drifts from wasm, all 28 blocks
  // downstream see corrupt conditioning every step. block_00_diff
  // showed kernels are byte-equivalent there, so shell_pre is the
  // primary suspect for the production webgpu-vs-wasm divergence.
  //
  // Build with:
  //   uv run --with onnx python web/scripts/ltx/expose_block_intermediates.py \
  //       notes/models/ltx/hf-repo/onnx/transformer/shell_pre.onnx
  ...(() => {
    const LTX_LOCAL = "/local-models/ltx";
    const N_TOK = 128;
    const N_TXT = 64;
    const filesShellPre = () => ({
      graph: f("ltx", LTX_LOCAL, "onnx/transformer/shell_pre_tapped.onnx", 152 * 1024),
      data: f(
        "ltx",
        LTX_LOCAL,
        "onnx/transformer/shell_pre.onnx.data",
        82 * 1024 * 1024,
      ),
    });
    const sharedDummyRun = {
      inputs: [
        { match: ["hidden_states"], shape: [1, N_TOK, 128], sinFill: { offset: 0.0, amplitude: 0.5 } },
        // indices_grid is fp32 RoPE indices (frame, h, w). Real values are
        // small non-negative ints; sinFill bounded works as a stand-in.
        { match: ["indices_grid"], shape: [1, 3, N_TOK], dtype: "float32" as const },
        { match: ["encoder_hidden_states"], shape: [1, N_TXT, 4096], sinFill: { offset: 1.5, amplitude: 1.0 } },
        // attention mask: 1 = real token. fill=1 keeps everything live.
        { match: ["encoder_attention_mask"], shape: [1, N_TXT], dtype: "int64" as const, fill: 1 },
        // timestep: ~0.9 sigma in the distilled sampler's pass1 range.
        { match: ["timestep"], shape: [1], sinFill: { offset: 0.9, amplitude: 0.0 } },
      ],
      dumpOutputs: false,
      repeats: 1,
    };
    const files = filesShellPre();
    return [
      {
        id: "ltx-shell-pre-diff",
        label: "LTX shell_pre tapped (wasm vs webgpu per-op diff)",
        components: [
          {
            name: "shell_pre_tapped (wasm)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "shell_pre.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["wasm" as const],
            diffPair: "ltx-shell-pre",
            dummyRun: sharedDummyRun,
          },
          {
            name: "shell_pre_tapped (webgpu)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "shell_pre.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["webgpu" as const],
            diffPair: "ltx-shell-pre",
            dummyRun: sharedDummyRun,
          },
        ],
      } satisfies Candidate,
    ];
  })(),
  // LTX VAE dec_block_00_res_0 tapped (wasm vs webgpu per-op diff). The
  // VAE-shard bundle (LTXVAE13) localised divergence to res_0: input
  // (dec_shell_pre.hidden) matches wasm within fp16 noise, output
  // explodes (maxAbs=7.1, std 5.7x wasm). After the fp16-misc + Pow
  // patches landed without effect, we tap every node output to find
  // the first diverging op. Inputs replayed from the captured wasm
  // VAE-shard bundle (hidden_in = dec_shell_pre.hidden, ts_embed =
  // dec_block_00_pre.ts_embed). Same fixed input on both EPs => any
  // diff above fp16 ULP at a tap is a kernel/export bug.
  //
  // Build with:
  //   uv run --with onnx python notes/scripts/tap_res0.py
  ...(() => {
    const LTX_LOCAL = "/local-models/ltx";
    const files = {
      graph: f("ltx", LTX_LOCAL, "onnx/vae/dec_block_00_res_0_tapped.onnx", 64 * 1024),
      data: f("ltx", LTX_LOCAL, "onnx/vae/dec_block_00_res_0.onnx.data", 110 * 1024 * 1024),
    };
    const capUrl = `${LTX_LOCAL}/captures/ltx_vae_decode_wasm.bin`;
    const realDummy = {
      inputs: [
        { match: ["hidden_in"], shape: [], fromCapture: { url: capUrl, tensorName: "dec_shell_pre.hidden" } },
        { match: ["ts_embed"], shape: [], fromCapture: { url: capUrl, tensorName: "dec_block_00_pre.ts_embed" } },
      ],
      dumpOutputs: false,
      repeats: 1,
    };
    return [
      {
        id: "ltx-vae-res0-real-diff",
        label: "LTX VAE dec_block_00_res_0 tapped, REAL captured inputs (wasm vs webgpu per-op diff)",
        components: [
          {
            name: "dec_block_00_res_0_tapped real (wasm)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "dec_block_00_res_0.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["wasm" as const],
            diffPair: "ltx-vae-res0-real",
            dummyRun: realDummy,
          },
          {
            name: "dec_block_00_res_0_tapped real (webgpu)",
            graph: files.graph,
            externalData: { file: files.data, pathInGraph: "dec_block_00_res_0.onnx.data" },
            graphOptLevel: "disabled" as const,
            executionProviders: ["webgpu" as const],
            diffPair: "ltx-vae-res0-real",
            dummyRun: realDummy,
          },
        ],
      } satisfies Candidate,
    ];
  })(),
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

function externalDataList(
  c: Component,
): Array<{ file: ModelFile; pathInGraph: string }> {
  if (!c.externalData) return [];
  return Array.isArray(c.externalData) ? c.externalData : [c.externalData];
}

async function downloadAll(comps: Component[]) {
  // Dedupe by file id - components in a diffPair share the same graph/data
  // files, and concurrent downloadFiles for the same OPFS id triggers
  // NoModificationAllowedError on createWritable.
  const seen = new Set<string>();
  const files: ModelFile[] = [];
  const push = (f: ModelFile) => {
    if (!seen.has(f.id)) {
      seen.add(f.id);
      files.push(f);
    }
  };
  for (const c of comps) {
    push(c.graph);
    for (const e of externalDataList(c)) push(e.file);
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

async function tryComponent(
  c: Component,
  prevOutputs: Record<string, ort.Tensor> = {},
): Promise<Record<string, ort.Tensor>> {
  log(`--- ${c.name} ---`);
  // Log actual on-disk sizes (rather than the rough estimates in
  // CANDIDATES). Cheap stat via FileSystemFileHandle.getFile().
  const graphSize = await cache.getFileSize(c.graph).catch(() => -1);
  log(
    `graph: ${c.graph.name}${graphSize >= 0 ? ` (${(graphSize / 1024 / 1024).toFixed(1)} MB)` : ""}`,
  );
  for (const e of externalDataList(c)) {
    const dataSize = await cache.getFileSize(e.file).catch(() => -1);
    log(
      `externalData: ${e.file.name}${dataSize >= 0 ? ` (${(dataSize / 1024 / 1024).toFixed(1)} MB)` : ""} (as "${e.pathInGraph}")`,
    );
  }

  setStatus(`loading ${c.name}`);

  // Use blob URLs (avoids copying multi-GB files into wasm heap as ArrayBuffer).
  const { url: graphUrl, revoke: revokeGraph } = await cache.loadFileAsBlobUrl(c.graph);
  const revokeFns: Array<() => void> = [];
  const revokeData = () => {
    for (const r of revokeFns) r();
  };
  const ortVerbose = new URLSearchParams(location.search).get("ortverbose") === "1";
  const eps = c.executionProviders ?? providers();
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: eps,
    graphOptimizationLevel: c.graphOptLevel ?? "all",
    ...(ortVerbose ? { logSeverityLevel: 0, logVerbosityLevel: 1 } : {}),
  };
  if (c.executionProviders) {
    log(`executionProviders: ${eps.join(", ")} (forced)`);
  }
  if (c.graphOptLevel) {
    log(`graphOptimizationLevel: ${c.graphOptLevel}`);
  }
  const extDataList = externalDataList(c);
  if (extDataList.length > 0) {
    const wired: Array<{ path: string; data: string }> = [];
    for (const e of extDataList) {
      const { url, revoke } = await cache.loadFileAsBlobUrl(e.file);
      revokeFns.push(revoke);
      wired.push({ path: e.pathInGraph, data: url });
    }
    (
      sessionOptions as unknown as { externalData: Array<{ path: string; data: string }> }
    ).externalData = wired;
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

  let lastResults: ort.InferenceSession.OnnxValueMapType | null = null;
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
      const captureCache = new Map<string, { manifest: Record<string, { shape: number[]; byteOffset: number; byteLength: number }>; payloadBase: number; buf: ArrayBuffer }>();
      for (const spec of c.dummyRun.inputs) {
        const inputName = findName(session.inputNames, spec.match);
        if (!inputName) {
          log(`  dummyRun: no input matched ${spec.match.join("/")} - skipping`);
          continue;
        }
        const len = spec.shape.reduce((a, b) => a * b, 1);
        if (spec.fromCapture) {
          const { url, tensorName } = spec.fromCapture;
          let entry = captureCache.get(url);
          if (!entry) {
            const resp = await fetch(url);
            if (!resp.ok) {
              log(`  dummyRun: fromCapture fetch ${url} -> ${resp.status}, falling back to zeros`);
              feeds[inputName] = buildTensor(spec.dtype ?? "float16", len, spec.shape);
              continue;
            }
            const buf = await resp.arrayBuffer();
            const dv = new DataView(buf);
            const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
            if (magic !== "LTXBLK26" && magic !== "LTXALL28" && magic !== "LTXVAE13") {
              log(`  dummyRun: fromCapture ${url} bad magic ${JSON.stringify(magic)}, falling back to zeros`);
              feeds[inputName] = buildTensor(spec.dtype ?? "float16", len, spec.shape);
              continue;
            }
            const headerLen = dv.getUint32(8, true);
            const headerJson = new TextDecoder().decode(new Uint8Array(buf, 12, headerLen));
            const header = JSON.parse(headerJson) as { tensors: Record<string, { shape: number[]; byteOffset: number; byteLength: number }> };
            entry = { manifest: header.tensors, payloadBase: 12 + headerLen, buf };
            captureCache.set(url, entry);
            log(`  dummyRun: fromCapture loaded ${url} (${(buf.byteLength / 1024 / 1024).toFixed(2)} MB, ${Object.keys(header.tensors).length} tensors)`);
          }
          const meta = entry.manifest[tensorName];
          if (!meta) {
            log(`  dummyRun: fromCapture ${url} has no tensor ${JSON.stringify(tensorName)}, falling back to zeros`);
            feeds[inputName] = buildTensor(spec.dtype ?? "float16", len, spec.shape);
            continue;
          }
          const bytes = new Uint8Array(entry.buf, entry.payloadBase + meta.byteOffset, meta.byteLength);
          const u16 = new Uint16Array(bytes.byteLength / 2);
          new Uint8Array(u16.buffer).set(bytes);
          feeds[inputName] = new ort.Tensor("float16", u16, meta.shape);
          statsFp16Brief(log, `input ${inputName} (from ${tensorName})`, u16);
          if (spec.dumpInput) dumpFp16Tensor(log, inputName, u16);
          continue;
        }
        if (spec.fromPrevOutput) {
          const prevName = findName(Object.keys(prevOutputs), [spec.fromPrevOutput]);
          const prevT = prevName ? prevOutputs[prevName] : undefined;
          if (!prevT) {
            log(
              `  dummyRun: fromPrevOutput ${spec.fromPrevOutput} not found - falling back to zeros`,
            );
            feeds[inputName] = buildTensor(spec.dtype ?? dummyDtype, len, spec.shape);
          } else {
            feeds[inputName] = prevT;
          }
          continue;
        }
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
        if (spec.sinFill) {
          const freq = spec.sinFill.freq ?? 0.0017;
          const amp = spec.sinFill.amplitude ?? 8.0;
          const off = spec.sinFill.offset;
          const f32 = new Float32Array(len);
          for (let i = 0; i < len; i++) f32[i] = Math.fround(Math.sin(i * freq + off) * amp);
          feeds[inputName] = new ort.Tensor("float16", f32ToF16Array(f32), spec.shape);
          if (spec.dumpInput) dumpFp16Tensor(log, inputName, feeds[inputName].data as Uint16Array);
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
        if (lastResults) {
          for (const name of Object.keys(lastResults)) {
            if (!/^conv3d/.test(name)) continue;
            const t = lastResults[name] as ort.Tensor;
            if (t.type !== "float16") continue;
            const bits = copyF16Bits(t.data as ArrayBufferView);
            statsFp16Brief(log, `output ${name}`, bits);
          }
        }
        if (c.dummyRun.renderOutput && lastResults) {
          renderOutputFrame(c.dummyRun.renderOutput, lastResults);
        }
        if (c.dummyRun.dumpOutputs && lastResults) {
          for (const name of Object.keys(lastResults)) {
            const t = lastResults[name] as ort.Tensor;
            if (t.type === "float16") {
              const bits = copyF16Bits(t.data as ArrayBufferView);
              dumpFp16Tensor(log, name, bits);
            } else {
              log(`  dumpOutputs: ${name} type=${t.type} (fp16 dump only)`);
            }
          }
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

  // Capture output tensors by copying their fp16 bits into standalone tensors
  // so they survive session.release() and can feed the next component.
  const captured: Record<string, ort.Tensor> = {};
  if (session && lastResults) {
    for (const name of Object.keys(lastResults)) {
      const t = lastResults[name] as ort.Tensor;
      if (t.type === "float16") {
        const bits = copyF16Bits(t.data as ArrayBufferView);
        captured[name] = new ort.Tensor("float16", bits, t.dims);
      }
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
  return captured;
}

/**
 * Diff two captured fp16 output maps in fp32 space. Walks `b` in
 * insertion order (graph output order from session.run, which mirrors
 * onnx.GraphProto.output) so the first divergent op is the upstream-most
 * one. Logs every output's diff plus a single "FIRST DIVERGENCE" line
 * for the first one over `threshold` mean-relative-diff.
 */
function diffCapturedFp16(
  pairId: string,
  aLabel: string,
  a: Record<string, ort.Tensor>,
  bLabel: string,
  b: Record<string, ort.Tensor>,
  threshold: number,
): void {
  log(`=== diff ${pairId}: ${aLabel} vs ${bLabel} (threshold mean-rel-diff > ${(threshold * 100).toFixed(1)}%) ===`);
  let firstBad: string | null = null;
  let compared = 0;
  for (const name of Object.keys(b)) {
    const tb = b[name];
    const ta = a[name];
    if (!ta) continue;
    if (tb.type !== "float16" || ta.type !== "float16") continue;
    // Use copyF16Bits (not `data as Uint16Array`): Chrome 147+ returns
    // tensor.data as Float16Array, which silently corrupts when iterated
    // as if it were Uint16Array (already-converted f32 numbers get
    // reinterpreted as fp16 bit patterns -> spurious NaN/Inf flooding).
    // Diagnosed 2026-05-06 on the LTX VAE res_0 bisect: the broken diff
    // reported bothNan=243816 at conv3d while actual conv3d output was
    // 100% finite, range [-680, +30.7].
    const fa = f16ToF32Array(copyF16Bits(ta.data as ArrayBufferView));
    const fb = f16ToF32Array(copyF16Bits(tb.data as ArrayBufferView));
    if (fa.length !== fb.length) {
      log(`  ${name}: SHAPE MISMATCH ${fa.length} vs ${fb.length}`);
      continue;
    }
    let maxAbs = 0;
    let sumAbs = 0;
    let sumRefAbs = 0;
    let bothNan = 0;
    let aNanOnly = 0;
    let bNanOnly = 0;
    for (let i = 0; i < fa.length; i++) {
      const va = fa[i];
      const vb = fb[i];
      const aN = Number.isNaN(va) || !Number.isFinite(va);
      const bN = Number.isNaN(vb) || !Number.isFinite(vb);
      if (aN && bN) {
        bothNan++;
        continue;
      }
      if (aN) {
        aNanOnly++;
        continue;
      }
      if (bN) {
        bNanOnly++;
        continue;
      }
      const d = Math.abs(va - vb);
      if (d > maxAbs) maxAbs = d;
      sumAbs += d;
      sumRefAbs += Math.abs(va);
    }
    const finiteN = Math.max(1, fa.length - bothNan - aNanOnly - bNanOnly);
    const meanAbs = sumAbs / finiteN;
    const meanRel = sumAbs / Math.max(1e-12, sumRefAbs);
    const asymN = aNanOnly + bNanOnly;
    const asymRatio = asymN / fa.length;
    // EP-specific overflow (one side NaN, other finite) is ALSO a kernel
    // divergence - flag it as such.
    const overThresh = meanRel > threshold || asymRatio > threshold;
    const flag = overThresh ? "  ** OVER THRESHOLD **" : "";
    log(
      `  ${name}: maxAbs=${maxAbs.toExponential(2)} meanAbs=${meanAbs.toExponential(2)} meanRel=${(meanRel * 100).toFixed(2)}% bothNan=${bothNan} ${aLabel}-only-NaN=${aNanOnly} ${bLabel}-only-NaN=${bNanOnly}${flag}`,
    );
    compared++;
    if (firstBad === null && overThresh) firstBad = name;
  }
  log(`compared ${compared} fp16 outputs`);
  if (firstBad) {
    log(`>>> FIRST DIVERGENCE (graph order): ${firstBad}`);
  } else {
    log(`>>> no output exceeded threshold`);
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
    // Accumulate across components — fromPrevOutput can reference any tensor
    // produced by any earlier component in the chain, not just N-1. Needed for
    // VAE-split parts whose side-input (e.g. block_input for an up_block's
    // avg_shortcut) comes from several components upstream.
    let prevOutputs: Record<string, ort.Tensor> = {};
    const diffPairBaselines = new Map<
      string,
      { label: string; outputs: Record<string, ort.Tensor> }
    >();
    for (const c of comps) {
      try {
        const out = await tryComponent(c, prevOutputs);
        prevOutputs = { ...prevOutputs, ...out };
        if (c.diffPair) {
          const prior = diffPairBaselines.get(c.diffPair);
          if (!prior) {
            diffPairBaselines.set(c.diffPair, { label: c.name, outputs: out });
          } else {
            diffCapturedFp16(c.diffPair, prior.label, prior.outputs, c.name, out, 0.05);
            diffPairBaselines.delete(c.diffPair);
          }
        }
      } catch (err) {
        log(`unexpected error on ${c.name}: ${(err as Error).message}`);
        log("");
        prevOutputs = {};
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
