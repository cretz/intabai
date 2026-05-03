// PixArt-XL T5 tokenizer for LTX-Video prompts.
//
// Same direct-construction pattern as fastwan/tokenizer.ts: load
// tokenizer.json from cache, instantiate T5Tokenizer ourselves. Avoids
// AutoTokenizer.from_pretrained (which captures fetch globally).
//
// LTX-2B doesn't bake a max_length the way FastWan does (UMT5_MAX_SEQ_LEN
// = 512). The pipeline truncates per call; we expose `validLength` and
// let the caller decide the encode-time padding. T5 EOS (</s>) is
// appended by the tokenizer; we leave the rest unpadded and let the
// transformer attend over `validLength` tokens.

import type { ModelCache } from "../shared/model-cache";
import { LTX_T5_TOKENIZER_FILE } from "./models";

/** T5 pad token id (matches sentencepiece pad). */
export const LTX_T5_PAD_ID = 0;

export interface TokenizedPrompt {
  /** Variable-length token sequence including trailing </s>. */
  ids: BigInt64Array;
  /** Number of real tokens (== ids.length; kept for parity with fastwan). */
  validLength: number;
}

let cachedTokenizer: unknown = null;

export async function loadTokenizer(cache: ModelCache): Promise<unknown> {
  if (cachedTokenizer) return cachedTokenizer;
  const tokenizerJson = await cache.loadFileText(LTX_T5_TOKENIZER_FILE);
  let tokJson: unknown;
  try {
    tokJson = JSON.parse(tokenizerJson);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`tokenizer.json parse failed: ${msg}`);
  }
  const transformers = await import("@huggingface/transformers");
  const anyTransformers = transformers as Record<string, unknown>;
  const TokClass = anyTransformers["T5Tokenizer"] ?? anyTransformers["PreTrainedTokenizer"];
  if (!TokClass || typeof TokClass !== "function") {
    throw new Error("transformers.js does not export T5Tokenizer/PreTrainedTokenizer");
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  cachedTokenizer = new (TokClass as any)(tokJson, {});
  return cachedTokenizer;
}

/** Tokenize and pad/truncate to `maxLength`. T5 input_ids are int64 in
 *  the ONNX graph, so we return BigInt64Array directly. */
export function tokenize(
  tokenizer: unknown,
  prompt: string,
  maxLength: number,
): TokenizedPrompt {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tok = tokenizer as any;
  const result = tok([prompt], {
    padding: false,
    truncation: true,
    max_length: maxLength,
    return_tensor: false,
  });
  const raw: number[] = result.input_ids[0];
  const validLength = Math.min(raw.length, maxLength);
  const ids = new BigInt64Array(maxLength);
  for (let i = 0; i < validLength; i++) ids[i] = BigInt(raw[i]);
  for (let i = validLength; i < maxLength; i++) ids[i] = BigInt(LTX_T5_PAD_ID);
  return { ids, validLength };
}
