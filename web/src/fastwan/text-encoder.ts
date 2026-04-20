// UMT5-XXL text encoder, run layer-by-layer.
//
// The encoder was exported as 24 independent q4f16 layer ONNX files plus a
// tiny `shell_post.onnx` (final RMSNorm). We load one layer at a time, run
// it, and dispose the session before the next one, so peak GPU memory is
// one layer (~108 MB) rather than 24 × 108 MB. The layers are shape-
// identical, so the run loop is trivial.
//
// This pattern matches the transformer block loop we do later; text
// encoding runs once per generation (24 × ~190 ms ≈ 4.5 s on desktop
// WebGPU), so the load/dispose cost is negligible against the ~10-minute
// transformer compute.
//
// Input to layer_00 comes from TokenEmbedding.embed() (int8 table lookup
// in JS; see embedding.ts for why the table is not inside the graph).
// Attention mask is fp16 additive: 0 for attended positions, -65504 (fp16
// min finite) for padding. UMT5 layers add this mask onto the pre-softmax
// scores, so any finite "large negative" value works; -inf is not safe
// because fp16 -inf through SDPA can produce NaNs on some EPs.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits, f32ToF16Bits } from "../sd15/fp16";
import { TokenEmbedding, UMT5_HIDDEN_SIZE } from "./embedding";
import { FASTWAN_TEXT_MASK_MAG } from "./models";

/** UMT5-XXL layer count (encoder-only; it's an encoder model). */
export const UMT5_NUM_LAYERS = 24;

/** Max sequence length the FastWan pipeline feeds to the transformer. */
export const UMT5_MAX_SEQ_LEN = 512;

/** fp16 bit pattern for 0.0. */
const F16_ZERO = 0x0000;
/** fp16 bit pattern for the mask-fill magnitude. Negated so attended=0,
 *  padded=-|mag|. Default -65504 (fp16 min finite); overridable via
 *  `?textmaskmag=N` for the WebGPU SDPA fp16-near-min-finite hypothesis. */
const F16_NEG_LARGE = f32ToF16Bits(-FASTWAN_TEXT_MASK_MAG);

export interface TextEncoderFiles {
  /** 24 layer files, index 0..23. Each is a (graph, data) pair. */
  layers: OrtModelFile[];
  /** Tiny final RMSNorm. */
  shellPost: OrtModelFile;
}

export type LayerProgress = (layerIndex: number, totalLayers: number) => void;

export interface TextEncodeResult {
  /** [1, seqLen, 4096] fp16 bit patterns, row-major. */
  hiddenStates: Uint16Array;
  /** [1, 1, 1, seqLen] fp16 bit patterns. Needed by the transformer. */
  attentionMaskF16: Uint16Array;
  /** Number of non-padding tokens at the start of the sequence. */
  validLength: number;
  /** Always UMT5_MAX_SEQ_LEN. */
  seqLen: number;
}

export class TextEncoder {
  constructor(
    private readonly cache: ModelCache,
    private readonly files: TextEncoderFiles,
    private readonly embedding: TokenEmbedding,
    private readonly providers: ("webgpu" | "wasm")[] = ["webgpu", "wasm"],
  ) {
    if (files.layers.length !== UMT5_NUM_LAYERS) {
      throw new Error(
        `expected ${UMT5_NUM_LAYERS} UMT5 layer files, got ${files.layers.length}`,
      );
    }
  }

  /** Encode a pre-tokenized, pre-padded id sequence of length UMT5_MAX_SEQ_LEN.
   *  `validLength` is the number of leading non-padding tokens (pad ids at
   *  validLength..seqLen-1 are masked out of attention). */
  async encode(
    tokenIds: ArrayLike<number>,
    validLength: number,
    onLayer?: LayerProgress,
    onStatsFp16?: (name: string, bits: Uint16Array) => void,
  ): Promise<TextEncodeResult> {
    const seqLen = UMT5_MAX_SEQ_LEN;
    if (tokenIds.length !== seqLen) {
      throw new Error(`expected ${seqLen} token ids, got ${tokenIds.length}`);
    }
    if (validLength < 0 || validLength > seqLen) {
      throw new Error(`validLength ${validLength} out of range [0, ${seqLen}]`);
    }

    // Embedding lookup (JS).
    let hidden = this.embedding.embed(tokenIds);
    onStatsFp16?.("embed.out[all]", hidden);
    onStatsFp16?.(
      `embed.out[valid:0..${validLength}]`,
      hidden.subarray(0, validLength * UMT5_HIDDEN_SIZE),
    );

    // Attention mask [1, 1, 1, seqLen] fp16.
    const mask = new Uint16Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      mask[i] = i < validLength ? F16_ZERO : F16_NEG_LARGE;
    }
    onStatsFp16?.("mask", mask);
    const maskDims = [1, 1, 1, seqLen];
    const hiddenDims = [1, seqLen, UMT5_HIDDEN_SIZE];

    // Run the 24 encoder layers, disposing each session before loading
    // the next to keep peak GPU memory at one layer.
    for (let i = 0; i < UMT5_NUM_LAYERS; i++) {
      const session = await createSession(
        this.cache,
        this.files.layers[i],
        this.providers,
      );
      try {
        const feeds: Record<string, ort.Tensor> = {
          hidden_states: new ort.Tensor("float16", hidden, hiddenDims),
          attention_mask: new ort.Tensor("float16", mask, maskDims),
        };
        const results = await session.run(feeds);
        const out = pickOutput(results);
        // Copy out before releasing the session - ORT-web's WebGPU tensor
        // data can be backed by GPU memory that goes away on release().
        hidden = copyF16Bits(out.data as ArrayBufferView);
      } finally {
        await session.release();
      }
      if (i === 0 || i === 11 || i === UMT5_NUM_LAYERS - 1) {
        onStatsFp16?.(
          `text_enc.layer_${i.toString().padStart(2, "0")}.out[valid:0..${validLength}]`,
          hidden.subarray(0, validLength * UMT5_HIDDEN_SIZE),
        );
      }
      onLayer?.(i + 1, UMT5_NUM_LAYERS + 1);
    }

    // Final RMSNorm.
    const post = await createSession(
      this.cache,
      this.files.shellPost,
      this.providers,
    );
    try {
      const results = await post.run({
        hidden_states: new ort.Tensor("float16", hidden, hiddenDims),
      });
      const out = pickOutput(results);
      hidden = copyF16Bits(out.data as ArrayBufferView);
    } finally {
      await post.release();
    }
    // Diffusers WanPipeline.encode_prompt zero-fills positions beyond the
    // real token count before handing embeddings to the transformer. Our
    // transformer blocks have no encoder_attention_mask input, so without
    // this step cross-attention treats 504 padded positions as real signal
    // and the prompt gets drowned in the padding. See worklog 2026-04-19
    // "prompt-agnostic fabric texture" diagnosis.
    for (let i = validLength * UMT5_HIDDEN_SIZE; i < hidden.length; i++) {
      hidden[i] = F16_ZERO;
    }
    onLayer?.(UMT5_NUM_LAYERS + 1, UMT5_NUM_LAYERS + 1);

    return {
      hiddenStates: hidden,
      attentionMaskF16: mask,
      validLength,
      seqLen,
    };
  }
}

function pickOutput(results: ort.InferenceSession.OnnxValueMapType): ort.Tensor {
  // Export uses `hidden_states_out` for layers and `last_hidden_state`
  // for shell_post. Accept either, fall back to the first output.
  const keys = Object.keys(results);
  const name =
    keys.find((k) => k === "hidden_states_out" || k === "last_hidden_state") ??
    keys[0];
  const t = results[name];
  if (!t) throw new Error("text encoder session produced no output");
  return t as ort.Tensor;
}
