// FastWan 2.2 TI2V-5B ONNX file manifest.
//
// Files are served from the published HF repo
// (cretz/FastWan2.2-TI2V-5B-ONNX-sharded). To re-test against a local
// export, flip FASTWAN_BASE back to `/local-models/fastwan/onnx` (served
// by the vite dev proxy in web/vite.config.ts).

import type { ModelFile } from "../shared/model-cache";
import type { OrtModelFile } from "../sd15/ort-helpers";
import type { FastwanResolution } from "./transformer";

/** Base URL for FastWan asset fetches. */
const FASTWAN_BASE =
  "https://huggingface.co/cretz/FastWan2.2-TI2V-5B-ONNX-sharded/resolve/main/onnx";

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

// ---- VAE decoder (LightTAE) -----------------------------------------------
// Single fp16 file, no external data. Per-resolution: each resolution ships
// its own LightTAE export at the latent geometry the transformer produces.

const LIGHTTAE_FILE_BYTES: Record<FastwanResolution, number> = {
  480: 29_549_754,
  576: 33_807_564,
};

export function fastwanVaeFile(resolution: FastwanResolution): ModelFile {
  return mf(
    `vae_decoder_${resolution}`,
    `vae_decoder-${resolution}.onnx`,
    `vae_decoder-${resolution}.onnx`,
    LIGHTTAE_FILE_BYTES[resolution],
  );
}

// ---- Denoising loop constants ---------------------------------------------

/** Active denoising step count. Matches KingNish/wan2-2-fast
 *  (UniPCMultistepScheduler at num_inference_steps=4). */
export const FASTWAN_NUM_STEPS = 4;

/** Flow-matching shift applied to the UniPC sigma schedule. The model's
 *  scheduler config ships 5.0 but the HF Space overrides to 8.0 in its
 *  pipeline setup; we mirror that. */
export const FASTWAN_FLOW_SHIFT = 8.0;

// ---- Text encoder q4f16 ---------------------------------------------------
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

// Text encoder ships in two precisions keyed to the transformer precision
// selection. q4f16 has ~47% drift vs PyTorch on CPU ORT (MatMulNBits is too
// lossy for UMT5's 50k-magnitude intermediates even at accuracy_level=1);
// fp16 matches PyTorch to cosine=1.0. Mobile pays the drift; desktop users
// running the fp16 transformer also get the fp16 text encoder.
function textEncoderLayerFile(
  i: number,
  precision: FastwanTransformerPrecision,
): OrtModelFile {
  const idx = String(i).padStart(2, "0");
  if (precision === "fp16") {
    return mf(
      `text_layer_${idx}_fp16`,
      `text-encoder/layer_${idx}.onnx (fp16)`,
      `text-encoder/layer_${idx}.onnx`,
      385_924_507,
    );
  }
  const graph = mf(
    `text_layer_${idx}`,
    `text-encoder-q4f16/layer_${idx}.onnx`,
    `text-encoder-q4f16/layer_${idx}.onnx`,
    110_000,
  );
  const data = mf(
    `text_layer_${idx}_data`,
    `text-encoder-q4f16/layer_${idx}.onnx.data`,
    `text-encoder-q4f16/layer_${idx}.onnx.data`,
    108_600_000,
  );
  return { graph, data, dataPath: `layer_${idx}.onnx.data` };
}

export function fastwanTextEncoderLayers(
  precision: FastwanTransformerPrecision,
): OrtModelFile[] {
  return Array.from({ length: 24 }, (_, i) => textEncoderLayerFile(i, precision));
}

export function fastwanTextEncoderShellPost(
  precision: FastwanTransformerPrecision,
): OrtModelFile {
  if (precision === "fp16") {
    return mf(
      "text_shell_post_fp16",
      "text-encoder/shell_post.onnx (fp16)",
      "text-encoder/shell_post.onnx",
      20_000,
    );
  }
  return mf(
    "text_shell_post",
    "text-encoder-q4f16/shell_post.onnx",
    "text-encoder-q4f16/shell_post.onnx",
    20_000,
  );
}

// ---- Transformer ----------------------------------------------------------
// Two precision variants ship: q4f16 (default, ~2.8 GB transformer, works on
// mobile) and fp16 (desktop-only, ~9.4 GB, ~25% faster per forward because
// ORT-web's MatMulNBits kernel dequants on every pass so q4 isn't free).

export type FastwanTransformerPrecision = "q4f16" | "fp16";

/** shell_pre is always served as the fp16 single-file graph (180 MB) for
 *  both precision variants. The q4f16 shell_pre (52 MB) produces NaN in
 *  timestep_proj under ORT-web's WebGPU MatMulNBits kernel at
 *  accuracy_level=4 (worklog 2026-04-19). Cache id is shared so neither
 *  bundle pays a duplicate download. */
function transformerShellPre(
  _precision: FastwanTransformerPrecision,
  resolution: FastwanResolution,
): OrtModelFile {
  return mf(
    `tx_shell_pre_fp16_${resolution}`,
    `transformer-${resolution}/shell_pre.onnx (fp16)`,
    `transformer-${resolution}/shell_pre.onnx`,
    179_722_412,
  );
}

function transformerBlockFile(
  i: number,
  precision: FastwanTransformerPrecision,
  resolution: FastwanResolution,
): OrtModelFile {
  const idx = String(i).padStart(2, "0");
  if (precision === "fp16") {
    return mf(
      `tx_block_${idx}_fp16_${resolution}`,
      `transformer-${resolution}/block_${idx}.onnx (fp16)`,
      `transformer-${resolution}/block_${idx}.onnx`,
      327_507_628,
    );
  }
  const graph = mf(
    `tx_block_${idx}_${resolution}`,
    `transformer-${resolution}/block_${idx}.onnx`,
    `transformer-q4f16-${resolution}/block_${idx}.onnx`,
    170_000,
  );
  const data = mf(
    `tx_block_${idx}_data_${resolution}`,
    `transformer-${resolution}/block_${idx}.onnx.data`,
    `transformer-q4f16-${resolution}/block_${idx}.onnx.data`,
    92_000_000,
  );
  return { graph, data, dataPath: `block_${idx}.onnx.data` };
}

function transformerBlocks(
  precision: FastwanTransformerPrecision,
  resolution: FastwanResolution,
): OrtModelFile[] {
  return Array.from({ length: 30 }, (_, i) =>
    transformerBlockFile(i, precision, resolution),
  );
}

function transformerShellPost(
  precision: FastwanTransformerPrecision,
  resolution: FastwanResolution,
): OrtModelFile {
  if (precision === "fp16") {
    return mf(
      `tx_shell_post_fp16_${resolution}`,
      `transformer-${resolution}/shell_post.onnx (fp16)`,
      `transformer-${resolution}/shell_post.onnx`,
      1_200_000,
    );
  }
  return mf(
    `tx_shell_post_${resolution}`,
    `transformer-${resolution}/shell_post.onnx`,
    `transformer-q4f16-${resolution}/shell_post.onnx`,
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
  resolution: FastwanResolution,
): FastwanTransformerFiles {
  return {
    shellPre: transformerShellPre(precision, resolution),
    blocks: transformerBlocks(precision, resolution),
    shellPost: transformerShellPost(precision, resolution),
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
export function fastwanAllFiles(
  precision: FastwanTransformerPrecision,
  resolution: FastwanResolution,
): ModelFile[] {
  const tx = fastwanTransformerFiles(precision, resolution);
  const textEncoderLayers = fastwanTextEncoderLayers(precision);
  const textEncoderShellPost = fastwanTextEncoderShellPost(precision);
  const files: ModelFile[] = [
    fastwanVaeFile(resolution),
    FASTWAN_EMBEDDING_Q8_FILE,
    FASTWAN_EMBEDDING_SCALES_FILE,
    FASTWAN_TOKENIZER_FILE,
    ...ortModelFiles(textEncoderShellPost),
    ...ortModelFiles(tx.shellPre),
    ...ortModelFiles(tx.shellPost),
  ];
  for (const layer of textEncoderLayers) files.push(...ortModelFiles(layer));
  for (const block of tx.blocks) files.push(...ortModelFiles(block));
  return files;
}
