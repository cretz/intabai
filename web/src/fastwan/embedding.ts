// UMT5 token embedding lookup, done in JS rather than in the ONNX graph.
//
// The UMT5-XXL embedding table is [vocab=256384, hidden=4096] fp16, which
// is 2.10 GB as a single tensor. WebGPU enforces a per-buffer size limit
// (`maxBufferSize`, ~256 MB on mobile, ~2 GB on many desktops), so keeping
// the table inside the ONNX graph would refuse to load on most devices.
// We extracted it at export time to two flat files and do the lookup here;
// the op is a pure indexed copy with no arithmetic, so GPU offers no
// speedup over a memcpy, and it runs once per generation anyway.
//
// Quantization: per-row symmetric int8 with fp16 scales.
//   row[i][j]  (fp16)  ~=  int8_q[i][j] * scale_fp16[i]
// Files written by web/scripts/quantize-embedding.py and shipped in
// `text-encoder-q4f16/` alongside the transformer layers:
//   embedding_q8.bin      [vocab, hidden] int8, row-major, 1.05 GB
//   embedding_scales.bin  [vocab]         fp16 scales,     0.51 MB
//
// Accuracy per the quantize script: mean abs err ~7e-2 against original
// fp16. If end-to-end output quality drops, tighten to per-block int8/int4
// rather than moving the lookup back into ONNX.

import { f16BitsToF32, f32ToF16Bits } from "../sd15/fp16";

/** UMT5-XXL hidden size. Hardcoded by the model. */
export const UMT5_HIDDEN_SIZE = 4096;

/** UMT5-XXL vocab size. Used for bounds checking only. */
export const UMT5_VOCAB_SIZE = 256384;

export class TokenEmbedding {
  /** Signed int8 view of the quantized embedding body, [vocab, hidden]. */
  private readonly q8: Int8Array;
  /** fp16 bit patterns of the per-row scales, one per vocab id. */
  private readonly scales: Uint16Array;
  private readonly hidden: number;
  private readonly vocab: number;

  constructor(
    embeddingQ8: ArrayBuffer,
    embeddingScales: ArrayBuffer,
    hidden: number = UMT5_HIDDEN_SIZE,
  ) {
    this.q8 = new Int8Array(embeddingQ8);
    this.scales = new Uint16Array(embeddingScales);
    this.hidden = hidden;
    this.vocab = this.scales.length;
    const expectedBody = this.vocab * hidden;
    if (this.q8.length !== expectedBody) {
      throw new Error(
        `embedding_q8 length ${this.q8.length} != vocab*hidden ${expectedBody} ` +
          `(vocab=${this.vocab}, hidden=${hidden})`,
      );
    }
  }

  /** Look up a batch of token ids and return fp16 bits of shape
   *  `[tokenIds.length, hidden]`, row-major. */
  embed(tokenIds: ArrayLike<number>): Uint16Array {
    const n = tokenIds.length;
    const out = new Uint16Array(n * this.hidden);
    for (let t = 0; t < n; t++) {
      this.embedRow(tokenIds[t], out, t * this.hidden);
    }
    return out;
  }

  /** Dequant one row into `out[outOffset..outOffset+hidden]` as fp16 bits. */
  embedRow(tokenId: number, out: Uint16Array, outOffset: number): void {
    if (tokenId < 0 || tokenId >= this.vocab) {
      throw new Error(`token id ${tokenId} out of range [0, ${this.vocab})`);
    }
    const scale = f16BitsToF32(this.scales[tokenId]);
    const base = tokenId * this.hidden;
    for (let j = 0; j < this.hidden; j++) {
      out[outOffset + j] = f32ToF16Bits(this.q8[base + j] * scale);
    }
  }
}
