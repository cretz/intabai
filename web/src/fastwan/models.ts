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

// VAE decoder (LightTAE) - single fp16 file, no external data.
export const FASTWAN_VAE_FILE: ModelFile = mf(
  "vae_decoder",
  "vae_decoder.onnx",
  "vae_decoder.onnx",
  36_600_000,
);

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
    return mf(
      `tx_block_${idx}_fp16`,
      `transformer/block_${idx}.onnx (fp16)`,
      `transformer/block_${idx}.onnx`,
      327_506_456,
    );
  }
  const graph = mf(
    `tx_block_${idx}`,
    `transformer/block_${idx}.onnx`,
    `transformer-q4f16/block_${idx}.onnx`,
    170_000,
  );
  const data = mf(
    `tx_block_${idx}_data`,
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
  for (const layer of FASTWAN_TEXT_ENCODER_LAYERS) files.push(...ortModelFiles(layer));
  for (const block of tx.blocks) files.push(...ortModelFiles(block));
  return files;
}
