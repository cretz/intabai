// FastWan 2.2 TI2V-5B ONNX file manifest.
//
// Files are served in dev via the vite proxy at `/local-models/fastwan/`
// (see web/vite.config.ts), which maps to
// notes/models/fastwan/hf-repo/ on disk. Swap FASTWAN_BASE to the HF repo
// URL once we upload (target name `cretz/FastWan2.2-TI2V-5B-ONNX`, not yet
// created - do not invent the URL, verify the repo before swapping).
//
// Sizes are approximate and match the shipping layout documented in
// notes/worklog.md (section "Text encoder shipping payload" and
// "q4f16 quantization: DONE"). They drive the download progress bar only,
// so a few-KB discrepancy does not matter; we don't hash-verify yet.

import type { ModelFile } from "../shared/model-cache";
import type { OrtModelFile } from "../sd15/ort-helpers";

/** Base URL for FastWan asset fetches. Dev uses the vite proxy; flip to
 *  the HF raw URL once the repo is published. */
const FASTWAN_BASE = "/local-models/fastwan/onnx";

// ---- Raw ModelFile entries -------------------------------------------------
//
// IDs are globally unique within the OPFS cache (the cache dedupes by id),
// so we prefix with `fastwan_` to avoid collisions with image-gen files.

function mf(id: string, name: string, rel: string, sizeBytes: number): ModelFile {
  return {
    id: `fastwan_${id}`,
    name,
    url: `${FASTWAN_BASE}/${rel}`,
    sizeBytes,
  };
}

// VAE decoder (LightTAE) - single fp16 file, no external data. Used for
// previews always, and for final decode by default.
export const FASTWAN_VAE_FILE: ModelFile = mf(
  "vae_decoder",
  "vae_decoder.onnx",
  "vae_decoder.onnx",
  36_600_000,
);

// Full AutoencoderKLWan streaming decoder. Two monolithic ONNX graphs
// (Conv3D-decomposed to Conv2D-sum in export, see export-fastwan-vae-kl-
// streaming.py --decompose-conv3d). Gated on ?vaekl=1 for final decode;
// previews always stay on LightTAE. Each graph ~1.1 GB, no external data.
export const FASTWAN_VAE_KL_INIT_FILE: ModelFile = mf(
  "vae_kl_init_decomp",
  "vae/decoder_init.onnx",
  "vae/decoder_init.onnx",
  1_117_366_418,
);

export const FASTWAN_VAE_KL_STEP_FILE: ModelFile = mf(
  "vae_kl_step_decomp",
  "vae/decoder_step.onnx",
  "vae/decoder_step.onnx",
  1_110_666_229,
);

/** `?vaekl=1` on the tool URL swaps the final VAE decode from LightTAE to
 *  the full AutoencoderKLWan streaming decoder. Previews remain on
 *  LightTAE (Wan VAE is too slow for in-loop preview). Adds ~2.2 GB to
 *  downloads. */
export const FASTWAN_USE_VAE_KL =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("vaekl") === "1";

// ---- Text encoder q4f16 ----------------------------------------------------
// Embedding table lives outside the ONNX graph; see fastwan/embedding.ts for
// the rationale. Two plain binary files.

export const FASTWAN_EMBEDDING_Q8_FILE: ModelFile = mf(
  "text_embedding_q8",
  "embedding_q8.bin",
  "text-encoder-q4f16/embedding_q8.bin",
  1_050_000_000,
);

export const FASTWAN_EMBEDDING_SCALES_FILE: ModelFile = mf(
  "text_embedding_scales",
  "embedding_scales.bin",
  "text-encoder-q4f16/embedding_scales.bin",
  512_768,
);

/** `?textfp16=1` on the tool URL swaps the text encoder from q4f16 to
 *  pre-quant fp16 (desktop-only, ~9 GB extra). Diagnostic for suspected
 *  WebGPU q4 kernel bugs in the text encoder path. */
const TEXT_FP16 =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("textfp16") === "1";

/** `?textwasm=1` on the tool URL forces the text encoder onto the wasm
 *  CPU execution provider instead of WebGPU. Diagnostic for suspected
 *  WebGPU kernel divergence vs CPU ONNX (which we've verified matches
 *  PyTorch to cosine 0.9998). Very slow (~5-10 min for 24 layers) but
 *  decisive. Pair with the "stop after text encoder" checkbox. */
export const FASTWAN_TEXT_ENCODER_WASM =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("textwasm") === "1";

/** `?textmaskmag=N` on the tool URL overrides the text encoder's fp16
 *  additive attention-mask magnitude for padding positions. Default is
 *  65504 (fp16 min finite). Hypothesis: WebGPU SDPA mishandles fp16
 *  scores near the min finite value when the mask is added pre-softmax,
 *  producing the 10-15% per-value divergence we see vs wasm. Try
 *  `?textmaskmag=10000` (-1e4) or `?textmaskmag=1000` (-1e3). Wasm
 *  matches PyTorch with -65504 so only WebGPU needs this. */
export const FASTWAN_TEXT_MASK_MAG =
  (() => {
    if (typeof window === "undefined") return 65504;
    const raw = new URLSearchParams(window.location.search).get("textmaskmag");
    if (!raw) return 65504;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : 65504;
  })();

/** `?transformerwasm=1` on the tool URL forces the transformer
 *  (shell_pre + 30 blocks + shell_post) onto the wasm CPU execution
 *  provider. Diagnostic for suspected WebGPU block kernel divergence
 *  vs CPU ONNX. Extremely slow (hours for a full run); intended to be
 *  paired with the "stop after block_00" checkbox for a one-block
 *  byte-diff against scripts/dump-reference-block-00.py. */
export const FASTWAN_TRANSFORMER_WASM =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("transformerwasm") === "1";

/** `?debugblock00=1` on the tool URL swaps block_00 for block_00_debug.onnx,
 *  which exposes every intermediate tensor as a graph output. Paired with
 *  the "stop after block_00" checkbox and transformer.ts's per-tap hex
 *  dump, this produces a line-per-tap log that diffs directly against
 *  `dump-reference-block-00.py --debug`. Only meaningful in fp16 precision
 *  (the debug graph is monolithic fp16; q4 has external data and wasn't
 *  regenerated). */
export const FASTWAN_DEBUG_BLOCK00 =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("debugblock00") === "1";

/** `?numsteps=N` overrides the denoising step count. Defaults to 4 to
 *  match the HF Space (KingNish/wan2-2-fast), which uses
 *  UniPCMultistepScheduler at num_inference_steps=4. */
export const FASTWAN_NUM_STEPS_OVERRIDE: number | null = (() => {
  if (typeof window === "undefined") return null;
  const raw = new URLSearchParams(window.location.search).get("numsteps");
  if (!raw) return null;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 1 && n <= 20 ? n : null;
})();

/** `?flowshift=X` overrides the flow-matching shift applied to the UniPC
 *  sigma schedule. Default 8.0 matches the HF Space; the scheduler config
 *  file ships 5.0 but the Space overrides to 8.0 in its pipeline setup. */
export const FASTWAN_FLOW_SHIFT: number = (() => {
  if (typeof window === "undefined") return 8.0;
  const raw = new URLSearchParams(window.location.search).get("flowshift");
  if (!raw) return 8.0;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 8.0;
})();

/** `?txnoopt=1` disables graph-level optimization for transformer block
 *  sessions (shell_pre, 30 blocks, shell_post). Diagnostic: if block_00
 *  output changes significantly, ORT-web's WebGPU EP is applying a
 *  fusion/rewrite that produces different numerics than the CPU EP. If
 *  output is unchanged, the bug is in a kernel, not in graph optimisation.
 *  Precedent: `sdxl/text-encoders.ts` uses "disabled" to sidestep an
 *  ORT-web optimizer bug. */
export const FASTWAN_TX_NOOPT =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("txnoopt") === "1";

function textEncoderLayerFile(i: number): OrtModelFile {
  const idx = String(i).padStart(2, "0");
  const subdir = TEXT_FP16 ? "text-encoder" : "text-encoder-q4f16";
  const idSuffix = TEXT_FP16 ? "_fp16" : "";
  const graph = mf(
    `text_layer_${idx}${idSuffix}`,
    `${subdir}/layer_${idx}.onnx`,
    `${subdir}/layer_${idx}.onnx`,
    TEXT_FP16 ? 500_000 : 110_000,
  );
  const data = mf(
    `text_layer_${idx}_data${idSuffix}`,
    `${subdir}/layer_${idx}.onnx.data`,
    `${subdir}/layer_${idx}.onnx.data`,
    TEXT_FP16 ? 385_900_000 : 108_600_000,
  );
  return { graph, data, dataPath: `layer_${idx}.onnx.data` };
}

export const FASTWAN_TEXT_ENCODER_LAYERS: OrtModelFile[] = Array.from(
  { length: 24 },
  (_, i) => textEncoderLayerFile(i),
);

export const FASTWAN_TEXT_ENCODER_SHELL_POST: OrtModelFile = mf(
  `text_shell_post${TEXT_FP16 ? "_fp16" : ""}`,
  TEXT_FP16 ? "text-encoder/shell_post.onnx" : "text-encoder-q4f16/shell_post.onnx",
  TEXT_FP16 ? "text-encoder/shell_post.onnx" : "text-encoder-q4f16/shell_post.onnx",
  20_000,
);

// ---- Transformer ----------------------------------------------------------
// Two precision variants ship: q4f16 (default, ~2.8 GB transformer, works on
// mobile) and fp16 (desktop-only, ~9.4 GB, ~25% faster per forward because
// ORT-web's MatMulNBits kernel dequants on every pass so q4 isn't free).

export type FastwanTransformerPrecision = "q4f16" | "fp16";

/** shell_pre is always served as the fp16 single-file graph (179 MB). The
 *  q4f16 shell_pre (52 MB) produces NaN in timestep_proj under ORT-web's
 *  WebGPU MatMulNBits kernel at accuracy_level=4; see worklog 2026-04-19.
 *  A re-quantize at accuracy_level=1 exists but has never been browser-
 *  tested - if that turns out clean we can reclaim ~128 MB on the q4f16
 *  bundle by switching the q4 branch to it. Until then fp16 is correct
 *  for both precision variants, and the cost is the same 179 MB file in
 *  both bundles (no duplicate download since cache id is shared). */
function transformerShellPre(_precision: FastwanTransformerPrecision): OrtModelFile {
  return mf(
    "tx_shell_pre_fp16",
    "transformer/shell_pre.onnx (fp16)",
    "transformer/shell_pre.onnx",
    179_722_412,
  );
}

function transformerBlockFile(
  i: number,
  precision: FastwanTransformerPrecision,
): OrtModelFile {
  const idx = String(i).padStart(2, "0");
  if (precision === "fp16") {
    // Debug swap: ?debugblock00=1 replaces block_00 with the graph that
    // exposes every intermediate as an output, for per-op diffing against
    // dump-reference-block-00.py --debug. Distinct cache id so OPFS doesn't
    // serve the stale normal block. Only block 0; other blocks stay normal.
    if (i === 0 && FASTWAN_DEBUG_BLOCK00) {
      // v4: switched to external-data layout. Monolithic 327 MB session
      // create OOM'd with shell_pre (180 MB) already resident because
      // ORT-web's wasm loader held the full proto in linear memory at
      // init. With external data, graph proto is tiny and weights stream
      // to GPU via the sidecar, same as regular q4 blocks.
      // v7: adds to_k/MatMul and to_v/MatMul to the v6 tap set so we can
      // diff K and V against CPU ORT. Still skips any tap between
      // attn{1,2}/MatMul and attn{1,2}/MatMul_1 (score matrix is
      // seq²·heads·fp16 ≈ 3.2 GB per tap, OOMs session create).
      const graph = mf(
        "tx_block_00_fp16_debug_v7",
        "transformer/block_00_debug.onnx (fp16)",
        "transformer/block_00_debug.onnx",
        183_808,
      );
      const data = mf(
        "tx_block_00_fp16_debug_v7_data",
        "transformer/block_00_debug.onnx.data",
        "transformer/block_00_debug.onnx.data",
        327_362_560,
      );
      return { graph, data, dataPath: "block_00_debug.onnx.data" };
    }
    // Blocks have attn1 seq-chunked (N=3) to skirt the 2 GiB WebGPU
    // maxBufferSize cliff at seq=8190 x heads=24. Chunked file is ~1.1 KB
    // larger; cache id bumped so OPFS re-downloads.
    return mf(
      `tx_block_${idx}_fp16_chunked`,
      `transformer/block_${idx}.onnx (fp16)`,
      `transformer/block_${idx}.onnx`,
      327_507_628,
    );
  }
  // Cache ids bumped to _chunked after re-quantizing from attn1-chunked fp16
  // blocks. See notes/ort-fp16-bugs.md section 5.
  const graph = mf(
    `tx_block_${idx}_chunked`,
    `transformer/block_${idx}.onnx`,
    `transformer-q4f16/block_${idx}.onnx`,
    170_000,
  );
  const data = mf(
    `tx_block_${idx}_data_chunked`,
    `transformer/block_${idx}.onnx.data`,
    `transformer-q4f16/block_${idx}.onnx.data`,
    92_000_000,
  );
  return { graph, data, dataPath: `block_${idx}.onnx.data` };
}

function transformerBlocks(precision: FastwanTransformerPrecision): OrtModelFile[] {
  return Array.from({ length: 30 }, (_, i) => transformerBlockFile(i, precision));
}

// shell_post was below the quantize threshold so q4f16/ and transformer/ hold
// bit-identical fp16 graphs. Separate cache ids so OPFS entries match each
// variant's download group cleanly.
function transformerShellPost(precision: FastwanTransformerPrecision): OrtModelFile {
  if (precision === "fp16") {
    return mf(
      "tx_shell_post_fp16",
      "transformer/shell_post.onnx (fp16)",
      "transformer/shell_post.onnx",
      1_200_000,
    );
  }
  return mf(
    "tx_shell_post",
    "transformer/shell_post.onnx",
    "transformer-q4f16/shell_post.onnx",
    1_200_000,
  );
}

export interface FastwanTransformerFiles {
  shellPre: OrtModelFile;
  blocks: OrtModelFile[];
  shellPost: OrtModelFile;
}

export function fastwanTransformerFiles(
  precision: FastwanTransformerPrecision,
): FastwanTransformerFiles {
  return {
    shellPre: transformerShellPre(precision),
    blocks: transformerBlocks(precision),
    shellPost: transformerShellPost(precision),
  };
}

// ---- Tokenizer -------------------------------------------------------------
// UMT5 tokenizer.json bundled with the source pipeline. Served from the same
// vite proxy root for dev.

export const FASTWAN_TOKENIZER_FILE: ModelFile = {
  id: "fastwan_tokenizer_json",
  name: "tokenizer/tokenizer.json",
  url: "/local-models/fastwan/tokenizer/tokenizer.json",
  sizeBytes: 16_500_000,
};

// ---- Aggregate -------------------------------------------------------------

import { ortModelFiles } from "../sd15/ort-helpers";

/** Every ModelFile needed to run FastWan 2.2 at the given transformer
 *  precision. The cache consumes this list to drive downloads and the
 *  "all cached?" check. The text encoder and VAE are the same across
 *  precision variants; only the transformer files differ. */
export function fastwanAllFiles(precision: FastwanTransformerPrecision): ModelFile[] {
  const tx = fastwanTransformerFiles(precision);
  const files: ModelFile[] = [
    FASTWAN_VAE_FILE,
    FASTWAN_EMBEDDING_Q8_FILE,
    FASTWAN_EMBEDDING_SCALES_FILE,
    FASTWAN_TOKENIZER_FILE,
    ...ortModelFiles(FASTWAN_TEXT_ENCODER_SHELL_POST),
    ...ortModelFiles(tx.shellPre),
    ...ortModelFiles(tx.shellPost),
  ];
  if (FASTWAN_USE_VAE_KL) {
    files.push(FASTWAN_VAE_KL_INIT_FILE, FASTWAN_VAE_KL_STEP_FILE);
  }
  for (const layer of FASTWAN_TEXT_ENCODER_LAYERS) files.push(...ortModelFiles(layer));
  for (const block of tx.blocks) files.push(...ortModelFiles(block));
  return files;
}
