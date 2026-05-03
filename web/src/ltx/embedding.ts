// PixArt-XL T5 token embedding lookup, JS-side.
//
// Same pattern as fastwan/embedding.ts: keep the embedding table out of
// the ONNX graph (no GPU buffer, no maxBufferSize concerns) and dequant
// per row at lookup time. T5 vocab is 32128 (vs 256384 for UMT5) so the
// raw fp16 table is only 263 MB; per-row symmetric int8 halves to 131 MB.

import { f16BitsToF32, f32ToF16Bits } from "../sd15/fp16";

/** PixArt T5-XXL hidden size. */
export const LTX_T5_HIDDEN_SIZE = 4096;

/** PixArt T5-XXL vocab size. */
export const LTX_T5_VOCAB_SIZE = 32128;

export class LtxT5Embedding {
  private readonly q8: Int8Array;
  private readonly scales: Uint16Array;
  private readonly hidden: number;
  private readonly vocab: number;

  constructor(
    embeddingQ8: ArrayBuffer,
    embeddingScales: ArrayBuffer,
    hidden: number = LTX_T5_HIDDEN_SIZE,
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

  /** Look up `tokenIds` and return fp16 bits, shape `[tokenIds.length, hidden]`,
   *  row-major. */
  embed(tokenIds: ArrayLike<number> | ArrayLike<bigint>): Uint16Array {
    const n = tokenIds.length;
    const out = new Uint16Array(n * this.hidden);
    for (let t = 0; t < n; t++) {
      const raw = tokenIds[t];
      const id = typeof raw === "bigint" ? Number(raw) : raw;
      this.embedRow(id, out, t * this.hidden);
    }
    return out;
  }

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
