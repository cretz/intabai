// PixArt-XL T5-XXL text encoder, run layer-by-layer.
//
// Mirrors fastwan/text-encoder.ts. 24 q4f16 blocks plus a tiny shell_post
// (final T5LayerNorm). Each block is loaded, run, and released before
// the next is loaded so peak GPU memory is one block (~108 MB) rather
// than 24 * 108 MB.
//
// T5 differs from UMT5 in that only block 0 owns the relative-attention
// bias. block_00.onnx returns (hidden_states_out, position_bias);
// block_NN for N >= 1 takes position_bias as input and returns
// hidden_states_out. We thread that one tensor through all 23 later
// blocks. Sequence length is dynamic (export used dynamic_axes), so the
// first run at a new length pays a few seconds of WebGPU shader compile,
// then caches.
//
// Embedding lookup happens in JS via LtxT5Embedding (see embedding.ts).
// Attention mask is fp16 additive: 0 attended, -65504 masked (fp16 -inf
// can NaN through some EPs).

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits, f32ToF16Bits } from "../sd15/fp16";
import { LtxT5Embedding, LTX_T5_HIDDEN_SIZE } from "./embedding";
import { LTX_T5_NUM_BLOCKS } from "./models";
import { assertF16View, assertOrtOutput } from "./validate";

const F16_ZERO = 0x0000;
const F16_NEG_LARGE = f32ToF16Bits(-65504);

export interface LtxT5Files {
  blocks: OrtModelFile[];
  shellPost: OrtModelFile;
}

export interface BlockTiming {
  blockIndex: number;
  totalBlocks: number;
  loadMs: number;
  runMs: number;
  releaseMs: number;
}

export type BlockProgress = (info: BlockTiming) => void;

export interface LtxT5EncodeResult {
  /** [1, seqLen, 4096] fp16 bits, row-major. */
  hiddenStates: Uint16Array;
  /** [1, 1, 1, seqLen] fp16 bits. */
  attentionMaskF16: Uint16Array;
  /** Number of leading non-padding tokens. */
  validLength: number;
  seqLen: number;
}

export class LtxT5Encoder {
  constructor(
    private readonly cache: ModelCache,
    private readonly files: LtxT5Files,
    private readonly embedding: LtxT5Embedding,
    private readonly providers: ("webgpu" | "wasm")[] = ["webgpu", "wasm"],
  ) {
    if (files.blocks.length !== LTX_T5_NUM_BLOCKS) {
      throw new Error(
        `expected ${LTX_T5_NUM_BLOCKS} T5 block files, got ${files.blocks.length}`,
      );
    }
  }

  /** Encode a tokenized prompt. `tokenIds.length` is treated as the full
   *  sequence length (no padding done here); `validLength` is the number
   *  of leading real tokens (positions [validLength, seqLen) are masked
   *  out of attention and zeroed in the final hidden states). */
  async encode(
    tokenIds: ArrayLike<number> | ArrayLike<bigint>,
    validLength: number,
    onBlock?: BlockProgress,
  ): Promise<LtxT5EncodeResult> {
    const seqLen = tokenIds.length;
    if (validLength < 0 || validLength > seqLen) {
      throw new Error(`validLength ${validLength} out of range [0, ${seqLen}]`);
    }

    let hidden = this.embedding.embed(tokenIds);

    const mask = new Uint16Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      mask[i] = i < validLength ? F16_ZERO : F16_NEG_LARGE;
    }
    const maskDims = [1, 1, 1, seqLen];
    const hiddenDims = [1, seqLen, LTX_T5_HIDDEN_SIZE];

    // position_bias is produced by block_00 and threaded through 1..23.
    let positionBias: Uint16Array | null = null;
    let positionBiasDims: number[] | null = null;

    for (let i = 0; i < LTX_T5_NUM_BLOCKS; i++) {
      const tLoad = performance.now();
      const session = await createSession(this.cache, this.files.blocks[i], this.providers);
      const loadMs = performance.now() - tLoad;

      // Block 0 owns relative_attention_bias and combines it with the
      // additive attention_mask into position_bias. Blocks 1..23 receive
      // that combined tensor and never reference attention_mask -- the
      // legacy ONNX tracer drops unused inputs, so feeding attention_mask
      // there fails with "invalid input 'attention_mask'".
      const feeds: Record<string, ort.Tensor> = {
        hidden_states: new ort.Tensor("float16", hidden, hiddenDims),
      };
      if (i === 0) {
        feeds.attention_mask = new ort.Tensor("float16", mask, maskDims);
      } else {
        if (!positionBias || !positionBiasDims) {
          throw new Error("block_00 did not produce position_bias");
        }
        feeds.position_bias = new ort.Tensor("float16", positionBias, positionBiasDims);
      }

      let runMs = 0;
      let releaseMs = 0;
      try {
        const tRun = performance.now();
        const results = await session.run(feeds);
        runMs = performance.now() - tRun;

        const hiddenOut = pickOutput(results, ["hidden_states_out", "last_hidden_state"]);
        assertOrtOutput(`t5.block_${i}.hidden_states_out`, hiddenOut, {
          type: "float16",
          dims: [1, seqLen, LTX_T5_HIDDEN_SIZE],
        });
        const hiddenView = assertF16View(`t5.block_${i}.hidden_states_out.data`, hiddenOut.data, seqLen * LTX_T5_HIDDEN_SIZE);
        hidden = copyF16Bits(hiddenView);
        if (i === 0) {
          const pb = pickOutput(results, ["position_bias"]);
          assertOrtOutput("t5.block_00.position_bias", pb, { type: "float16" });
          const pbDims = pb.dims as number[];
          let pbCount = 1;
          for (const d of pbDims) pbCount *= d;
          const pbView = assertF16View("t5.block_00.position_bias.data", pb.data, pbCount);
          positionBias = copyF16Bits(pbView);
          positionBiasDims = pbDims.slice();
        }
        for (const k in results) (results[k] as ort.Tensor).dispose?.();
      } finally {
        for (const k in feeds) feeds[k].dispose?.();
        const tRel = performance.now();
        await session.release();
        releaseMs = performance.now() - tRel;
      }
      onBlock?.({
        blockIndex: i,
        totalBlocks: LTX_T5_NUM_BLOCKS,
        loadMs,
        runMs,
        releaseMs,
      });
    }

    // Final T5LayerNorm.
    const post = await createSession(this.cache, this.files.shellPost, this.providers);
    const postFeed = new ort.Tensor("float16", hidden, hiddenDims);
    try {
      const results = await post.run({ hidden_states: postFeed });
      const out = pickOutput(results, ["last_hidden_state", "hidden_states_out"]);
      assertOrtOutput("t5.shell_post", out, {
        type: "float16",
        dims: [1, seqLen, LTX_T5_HIDDEN_SIZE],
      });
      const view = assertF16View("t5.shell_post.data", out.data, seqLen * LTX_T5_HIDDEN_SIZE);
      hidden = copyF16Bits(view);
      for (const k in results) (results[k] as ort.Tensor).dispose?.();
    } finally {
      postFeed.dispose?.();
      await post.release();
    }

    // Zero out positions beyond validLength so cross-attention in the
    // transformer doesn't pick up signal from padded slots. The LTX
    // pipeline does this via the prompt_attention_mask passed to
    // PixArt-style cross-attention; our transformer export uses an
    // explicit mask too (encoder_attention_mask), but zeroing the hidden
    // states is a belt-and-braces guard that matches the FastWan fix.
    for (let i = validLength * LTX_T5_HIDDEN_SIZE; i < hidden.length; i++) {
      hidden[i] = F16_ZERO;
    }

    return {
      hiddenStates: hidden,
      attentionMaskF16: mask,
      validLength,
      seqLen,
    };
  }
}

function pickOutput(
  results: ort.InferenceSession.OnnxValueMapType,
  preferred: string[],
): ort.Tensor {
  for (const name of preferred) {
    if (results[name]) return results[name] as ort.Tensor;
  }
  const keys = Object.keys(results);
  const t = keys[0] ? results[keys[0]] : undefined;
  if (!t) throw new Error("session produced no output");
  return t as ort.Tensor;
}
