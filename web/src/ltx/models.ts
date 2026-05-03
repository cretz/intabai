// LTX-Video 2B 0.9.8 distilled file manifest.
//
// Currently served from the local vite proxy at /local-models/ltx/onnx
// (which maps to notes/models/ltx/hf-repo/onnx/, populated by the python
// scripts under web/scripts/ltx/). Will switch to a public HF repo once
// the export is validated end-to-end.
//
// T5 layout: per-block layer-stream (mirrors fastwan UMT5). 24 q4f16
// blocks + tiny shell_post + JS-side embedding lookup. Block 0 owns the
// relative-attention bias and outputs a position_bias tensor that is
// threaded through blocks 1..23.
//
// Transformer layout: monolithic dynamo export, sharded into 7 ~600 MB
// external-data files. Will be redone per-block to fit mobile maxBufferSize.

import type { ModelFile } from "../shared/model-cache";
import type { OrtModelFile } from "../sd15/ort-helpers";

const LTX_BASE = "/local-models/ltx/onnx";

function mf(id: string, name: string, rel: string, sizeBytes: number): ModelFile {
  return {
    id: `ltx_${id}`,
    name,
    url: `${LTX_BASE}/${rel}`,
    sizeBytes,
  };
}

// ---- Transformer (per-block layer-stream, q4f16) ---------------------------
// shell_pre + 28 blocks + shell_post. shell_pre is fp16 (67 MB); blocks are
// q4f16 (~38 MB each, ~1.07 GB total); shell_post is too small to quantize
// and ships as fp16 (~0.09 MB).

export const LTX_TX_NUM_BLOCKS = 28;

function ltxTxShellPreFile(): OrtModelFile {
  const graph = mf(
    "tx_shell_pre",
    "transformer/shell_pre.onnx",
    "transformer/shell_pre.onnx",
    151_832,
  );
  const data = mf(
    "tx_shell_pre_data",
    "transformer/shell_pre.onnx.data",
    "transformer/shell_pre.onnx.data",
    67_040_595,
  );
  return { graph, data, dataPath: "shell_pre.onnx.data" };
}

function ltxTxShellPostFile(): OrtModelFile {
  const graph = mf(
    "tx_shell_post",
    "transformer/shell_post.onnx",
    "transformer/shell_post.onnx",
    86_250,
  );
  const data = mf(
    "tx_shell_post_data",
    "transformer/shell_post.onnx.data",
    "transformer/shell_post.onnx.data",
    536_576,
  );
  return { graph, data, dataPath: "shell_post.onnx.data" };
}

function ltxTxBlockFile(i: number): OrtModelFile {
  const idx = String(i).padStart(2, "0");
  const graph = mf(
    `tx_block_${idx}_fp16`,
    `transformer-fp16/block_${idx}.onnx`,
    `transformer-fp16/block_${idx}.onnx`,
    233_481,
  );
  const data = mf(
    `tx_block_${idx}_data_fp16`,
    `transformer-fp16/block_${idx}.onnx.data`,
    `transformer-fp16/block_${idx}.onnx.data`,
    117_440_512,
  );
  return { graph, data, dataPath: `block_${idx}.onnx.data` };
}

export interface LtxTxFiles {
  shellPre: OrtModelFile;
  blocks: OrtModelFile[];
  shellPost: OrtModelFile;
}

export function ltxTransformerFiles(): LtxTxFiles {
  return {
    shellPre: ltxTxShellPreFile(),
    blocks: Array.from({ length: LTX_TX_NUM_BLOCKS }, (_, i) => ltxTxBlockFile(i)),
    shellPost: ltxTxShellPostFile(),
  };
}

// ---- VAE (per-shard layer-stream, fp16) -----------------------------------
// CausalVideoAutoencoder split per encoder.down_blocks / decoder.up_blocks.
// Encoder: enc_shell_pre + 9 down_blocks (block_08 split into 2 per-resnet
// sub-files) + enc_shell_post. Decoder: dec_shell_pre + 7 up_blocks
// (block_00 split into pre + 5 per-resnet sub-files) + dec_shell_post.
// Conv-heavy, so q4f16 quantization is a no-op -- ships fp16.
//
// File sizes verified 2026-05-03 from notes/models/ltx/hf-repo/onnx/vae/.
// Largest single shard is enc_block_08_res_0/1 at ~453 MB on disk; well
// under the wasm32 4 GB cap and the mobile maxBufferSize ceiling we're
// targeting (~2 GB for headroom).

interface VaeShardSpec {
  /** Stem under hf-repo/onnx/vae/, e.g. "enc_block_00". */
  stem: string;
  /** .onnx graph size in bytes. */
  graphBytes: number;
  /** .onnx.data sidecar size in bytes. */
  dataBytes: number;
  /** If set, load the `<stem>_pixnorm_fp32.onnx` variant produced by
   *  patch_vae_pixnorm_fp32.py. The patched graph routes the PixelNorm
   *  chain (Pow->ReduceMean->Sqrt->Div) through fp32 to dodge the fp16
   *  saturation -> WebGPU NaN cascade documented in attic/ort-fp16-bugs.md.
   *  The .onnx.data sidecar is shared with the unpatched stem. */
  pixnormFp32GraphBytes?: number;
}

function vaeShard(spec: VaeShardSpec): OrtModelFile {
  const graphStem = spec.pixnormFp32GraphBytes !== undefined
    ? `${spec.stem}_pixnorm_fp32`
    : spec.stem;
  const graphBytes = spec.pixnormFp32GraphBytes ?? spec.graphBytes;
  const graph = mf(
    `vae_${graphStem}`,
    `vae/${graphStem}.onnx`,
    `vae/${graphStem}.onnx`,
    graphBytes,
  );
  const data = mf(
    `vae_${spec.stem}_data`,
    `vae/${spec.stem}.onnx.data`,
    `vae/${spec.stem}.onnx.data`,
    spec.dataBytes,
  );
  return { graph, data, dataPath: `${spec.stem}.onnx.data` };
}

// Encoder: ordered as the runtime executes them. pixnormFp32GraphBytes is
// set on every shard that contains a Pow->ReduceMean->Sqrt->Div PixelNorm
// chain (verified by web/scripts/ltx/patch_vae_pixnorm_fp32.py); the patched
// graph reuses the unpatched .onnx.data sidecar.
const VAE_ENC_SHARDS: VaeShardSpec[] = [
  { stem: "enc_shell_pre",     graphBytes:    16_928, dataBytes:       331_776 },
  { stem: "enc_block_00",      graphBytes:   168_057, dataBytes:     7_077_888, pixnormFp32GraphBytes: 169_897 },
  { stem: "enc_block_01",      graphBytes:    30_550, dataBytes:       442_368 },
  { stem: "enc_block_02",      graphBytes:   249_539, dataBytes:    42_532_864, pixnormFp32GraphBytes: 252_318 },
  { stem: "enc_block_03",      graphBytes:    28_612, dataBytes:     3_604_480 },
  { stem: "enc_block_04",      graphBytes:   249_572, dataBytes:   169_934_848, pixnormFp32GraphBytes: 252_351 },
  { stem: "enc_block_05",      graphBytes:    35_743, dataBytes:     3_538_944 },
  { stem: "enc_block_06",      graphBytes:    83_771, dataBytes:   226_557_952, pixnormFp32GraphBytes:  84_691 },
  { stem: "enc_block_07",      graphBytes:    35_559, dataBytes:    14_221_312 },
  { stem: "enc_block_08_res_0",graphBytes:    34_035, dataBytes:   453_050_368, pixnormFp32GraphBytes:  34_495 },
  { stem: "enc_block_08_res_1",graphBytes:    34_035, dataBytes:   453_050_368, pixnormFp32GraphBytes:  34_495 },
  { stem: "enc_shell_post",    graphBytes:    14_675, dataBytes:    14_331_904, pixnormFp32GraphBytes:  14_902 },
];

// Decoder: dec_block_00 split as pre + 5 res sub-files.
const VAE_DEC_SHARDS: VaeShardSpec[] = [
  { stem: "dec_shell_pre",     graphBytes:    12_121, dataBytes:     7_143_424 },
  { stem: "dec_block_00_pre",  graphBytes:    22_418, dataBytes:    35_717_120 },
  { stem: "dec_block_00_res_0",graphBytes:    59_894, dataBytes:   113_311_744, pixnormFp32GraphBytes:  60_354 },
  { stem: "dec_block_00_res_1",graphBytes:    59_894, dataBytes:   113_311_744, pixnormFp32GraphBytes:  60_354 },
  { stem: "dec_block_00_res_2",graphBytes:    59_894, dataBytes:   113_311_744, pixnormFp32GraphBytes:  60_354 },
  { stem: "dec_block_00_res_3",graphBytes:    59_894, dataBytes:   113_311_744, pixnormFp32GraphBytes:  60_354 },
  { stem: "dec_block_00_res_4",graphBytes:    59_894, dataBytes:   113_311_744, pixnormFp32GraphBytes:  60_354 },
  { stem: "dec_block_01",      graphBytes:    44_633, dataBytes:   226_557_952 },
  { stem: "dec_block_02",      graphBytes:   403_271, dataBytes:   151_060_480, pixnormFp32GraphBytes: 405_596 },
  { stem: "dec_block_03",      graphBytes:    44_621, dataBytes:    56_688_640 },
  { stem: "dec_block_04",      graphBytes:   403_241, dataBytes:    38_076_416, pixnormFp32GraphBytes: 405_566 },
  { stem: "dec_block_05",      graphBytes:    44_621, dataBytes:    14_221_312 },
  { stem: "dec_block_06",      graphBytes:   405_115, dataBytes:     9_641_472, pixnormFp32GraphBytes: 407_440 },
  { stem: "dec_shell_post",    graphBytes:    52_458, dataBytes:       595_968, pixnormFp32GraphBytes:  52_685 },
];

export interface LtxVaeFiles {
  encShellPre: OrtModelFile;
  encBlocks: OrtModelFile[];           // includes block_08_res_0/1 in order
  encShellPost: OrtModelFile;
  decShellPre: OrtModelFile;
  /** dec_block_00_pre, dec_block_00_res_{0..4}, dec_block_{01..06}. */
  decBlocks: OrtModelFile[];
  decShellPost: OrtModelFile;
}

// ---- Spatial upscaler (monolithic fp16) ----------------------------------
// LatentUpsampler 482 MB safetensors -> 252 MB fp16 ONNX. Doubles spatial,
// keeps temporal. Small enough to ship as one file.

export function ltxUpscalerFile(): OrtModelFile {
  const graph = mf("upscaler", "upscaler.onnx", "upscaler.onnx", 284_141);
  const data = mf(
    "upscaler_data",
    "upscaler.onnx.data",
    "upscaler.onnx.data",
    252_510_208,
  );
  return { graph, data, dataPath: "upscaler.onnx.data" };
}

/** Per-channel latent statistics (`mean_of_means` then `std_of_means`,
 *  each 128 fp32 values = 1024 bytes total). Used by the sampler to map
 *  between the pipeline-normalized latent space (transformer denoises in
 *  this space) and the un-normalized space the decoder takes. Not used
 *  inside vae.ts itself. */
export const LTX_VAE_PER_CHANNEL_STATS_FILE: ModelFile = mf(
  "vae_per_channel_stats",
  "vae/per_channel_stats.f32",
  "vae/per_channel_stats.f32",
  1024,
);

export function ltxVaeFiles(): LtxVaeFiles {
  const enc = VAE_ENC_SHARDS.map(vaeShard);
  const dec = VAE_DEC_SHARDS.map(vaeShard);
  return {
    encShellPre: enc[0],
    encBlocks: enc.slice(1, enc.length - 1),
    encShellPost: enc[enc.length - 1],
    decShellPre: dec[0],
    decBlocks: dec.slice(1, dec.length - 1),
    decShellPost: dec[dec.length - 1],
  };
}

// ---- T5-XXL text encoder (per-block layer-stream) -------------------------
// PixArt-XL T5 encoder, 4.76B params. 24 q4f16 blocks (~108.6 MB each,
// ~2.61 GB total) + tiny shell_post + JS-side embedding lookup.
//
// Block I/O signature:
//   block_00: (hidden_states, attention_mask) -> (hidden_states_out, position_bias)
//   block_NN (1..23): (hidden_states, attention_mask, position_bias) -> hidden_states_out
//   shell_post: hidden_states -> last_hidden_state

export const LTX_T5_NUM_BLOCKS = 24;

const T5_BLOCK_GRAPH_BYTES_FIRST = 30_251;
const T5_BLOCK_GRAPH_BYTES_OTHER = 19_435;
const T5_BLOCK_DATA_BYTES_FIRST = 108_548_096;
const T5_BLOCK_DATA_BYTES_OTHER = 108_544_000;
const T5_SHELL_POST_BYTES = 18_223;

function ltxT5BlockFile(i: number): OrtModelFile {
  const idx = String(i).padStart(2, "0");
  const isFirst = i === 0;
  const graph = mf(
    `t5_block_${idx}`,
    `t5/block_${idx}.onnx`,
    `t5/block_${idx}.onnx`,
    isFirst ? T5_BLOCK_GRAPH_BYTES_FIRST : T5_BLOCK_GRAPH_BYTES_OTHER,
  );
  const data = mf(
    `t5_block_${idx}_data`,
    `t5/block_${idx}.onnx.data`,
    `t5/block_${idx}.onnx.data`,
    isFirst ? T5_BLOCK_DATA_BYTES_FIRST : T5_BLOCK_DATA_BYTES_OTHER,
  );
  return { graph, data, dataPath: `block_${idx}.onnx.data` };
}

export function ltxT5BlockFiles(): OrtModelFile[] {
  return Array.from({ length: LTX_T5_NUM_BLOCKS }, (_, i) => ltxT5BlockFile(i));
}

export function ltxT5ShellPostFile(): OrtModelFile {
  return mf("t5_shell_post", "t5/shell_post.onnx", "t5/shell_post.onnx", T5_SHELL_POST_BYTES);
}

// ---- T5 embedding (JS-side int8 quantized lookup) -------------------------
// 32128 x 4096 fp16 table = 263 MB raw. Per-row symmetric int8 quant
// halves that to 131 MB body + 64 KB fp16 scales. JS lookup dequants on
// demand; no GPU footprint.

export const LTX_T5_EMBEDDING_Q8_FILE: ModelFile = mf(
  "t5_embedding_q8",
  "embedding_q8.bin",
  "t5/embedding_q8.bin",
  131_596_288,
);

export const LTX_T5_EMBEDDING_SCALES_FILE: ModelFile = mf(
  "t5_embedding_scales",
  "embedding_scales.bin",
  "t5/embedding_scales.bin",
  64_256,
);

// ---- Tokenizer ------------------------------------------------------------

export const LTX_T5_TOKENIZER_FILE: ModelFile = mf(
  "t5_tokenizer",
  "tokenizer.json",
  "t5/tokenizer.json",
  2_423_929,
);

// ---- Aggregate ------------------------------------------------------------

import { ortModelFiles } from "../sd15/ort-helpers";

export function ltxAllFiles(): ModelFile[] {
  const tx = ltxTransformerFiles();
  const vae = ltxVaeFiles();
  const files: ModelFile[] = [
    LTX_T5_EMBEDDING_Q8_FILE,
    LTX_T5_EMBEDDING_SCALES_FILE,
    LTX_T5_TOKENIZER_FILE,
    LTX_VAE_PER_CHANNEL_STATS_FILE,
    ...ortModelFiles(ltxUpscalerFile()),
    ...ortModelFiles(ltxT5ShellPostFile()),
    ...ortModelFiles(tx.shellPre),
    ...ortModelFiles(tx.shellPost),
    ...ortModelFiles(vae.encShellPre),
    ...ortModelFiles(vae.encShellPost),
    ...ortModelFiles(vae.decShellPre),
    ...ortModelFiles(vae.decShellPost),
  ];
  for (const block of ltxT5BlockFiles()) files.push(...ortModelFiles(block));
  for (const block of tx.blocks) files.push(...ortModelFiles(block));
  for (const block of vae.encBlocks) files.push(...ortModelFiles(block));
  for (const block of vae.decBlocks) files.push(...ortModelFiles(block));
  return files;
}
