// UMT5 tokenizer for FastWan 2.2 prompts.
//
// Loads transformers.js lazily and constructs a T5Tokenizer directly from
// the cached tokenizer.json. UMT5 uses the same tokenizer class as T5
// (SentencePiece-derived); the distinction is at the model level, not the
// tokenizer. Same direct-construction pattern as zimage/generate.ts (we
// avoid AutoTokenizer.from_pretrained because it captures fetch at module
// load and is fiddly to redirect at OPFS; the explicit new T5Tokenizer()
// call with cached JSON is reliable).
//
// Output contract matches what text-encoder.ts expects:
//   - ids: length UMT5_MAX_SEQ_LEN (512), padded with pad_token_id=0
//   - validLength: number of leading real tokens (prompt + </s>) before padding

import type { ModelCache } from "../shared/model-cache";
import { FASTWAN_TOKENIZER_FILE } from "./models";
import { UMT5_MAX_SEQ_LEN } from "./text-encoder";

/** UMT5 pad token id; matches sentencepiece pad token. */
const UMT5_PAD_ID = 0;

export interface TokenizedPrompt {
  /** Length UMT5_MAX_SEQ_LEN, pad with UMT5_PAD_ID. */
  ids: Int32Array;
  /** Number of non-pad leading tokens. */
  validLength: number;
}

let cachedTokenizer: unknown = null;

export async function loadTokenizer(cache: ModelCache): Promise<unknown> {
  if (cachedTokenizer) return cachedTokenizer;
  const tokenizerJson = await cache.loadFileText(FASTWAN_TOKENIZER_FILE);
  const tokJson = JSON.parse(tokenizerJson);
  const transformers = await import("@huggingface/transformers");
  const anyTransformers = transformers as Record<string, unknown>;
  const TokClass =
    anyTransformers["T5Tokenizer"] ?? anyTransformers["PreTrainedTokenizer"];
  if (!TokClass || typeof TokClass !== "function") {
    throw new Error("transformers.js does not export T5Tokenizer/PreTrainedTokenizer");
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  cachedTokenizer = new (TokClass as any)(tokJson, {});
  return cachedTokenizer;
}

/** Tokenize a prompt to a UMT5_MAX_SEQ_LEN-long id sequence with trailing
 *  pad. Appends EOS (</s>) via transformers.js default T5 behavior, then
 *  pads with UMT5_PAD_ID. Truncates to UMT5_MAX_SEQ_LEN. */
export function tokenize(tokenizer: unknown, prompt: string): TokenizedPrompt {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tok = tokenizer as any;
  const result = tok([prompt], {
    padding: false,
    truncation: true,
    max_length: UMT5_MAX_SEQ_LEN,
    return_tensor: false,
  });
  const raw: number[] = result.input_ids[0];
  const validLength = Math.min(raw.length, UMT5_MAX_SEQ_LEN);
  const ids = new Int32Array(UMT5_MAX_SEQ_LEN);
  for (let i = 0; i < validLength; i++) ids[i] = raw[i];
  for (let i = validLength; i < UMT5_MAX_SEQ_LEN; i++) ids[i] = UMT5_PAD_ID;
  return { ids, validLength };
}
