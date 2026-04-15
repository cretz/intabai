// CLIP ViT-L/14 BPE tokenizer, hand-ported from openai/CLIP's
// `simple_tokenizer.py`. SD1.5's text encoder requires this exact tokenizer:
// the same byte-level pre-encoding, the same </w> end-of-word marker, the
// same merge ordering, the same special token ids. Any deviation produces
// embeddings the UNet was not trained on and degrades output silently.
//
// Sources:
//   https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
//   https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz
//
// Vocabulary and merges are loaded at runtime from the model bundle (see
// models.ts). We do NOT bundle them into the JS - they live in the same OPFS
// cache as the ONNX weights and are downloaded as part of the model set.
//
// Notes on intentional simplifications vs the reference:
// - We do not run ftfy. ftfy fixes mojibake (e.g. text that was UTF-8 decoded
//   as latin-1). This is rare for prompts typed into a textarea in a modern
//   browser, where input is already valid UTF-16. If a user pastes mojibake
//   they get mojibake embeddings; that is fine.
// - basic_clean in the reference unescapes HTML entities. We do the same with
//   a tiny inline unescape because users sometimes paste &amp; from web text.
// - whitespace_clean collapses runs of whitespace and trims, matching the
//   reference exactly.

const BOS_TOKEN = "<|startoftext|>";
const EOS_TOKEN = "<|endoftext|>";

/** SD1.5's CLIP text encoder always sees a fixed-length 77-token sequence. */
export const CLIP_CONTEXT_LENGTH = 77;

/**
 * Result of tokenizing one prompt. ids and attentionMask are both length
 * CLIP_CONTEXT_LENGTH (77). attentionMask is 1 for real tokens (including
 * BOS/EOS) and 0 for padding. ids uses EOS as the pad token, matching the
 * reference SD1.5 / diffusers behavior.
 */
export interface TokenizedPrompt {
  ids: Int32Array;
  attentionMask: Int32Array;
  /** Number of non-padding tokens, including BOS and EOS. */
  length: number;
  /** True if the input was longer than CLIP_CONTEXT_LENGTH-2 BPE tokens
   *  and had to be truncated before EOS. */
  truncated: boolean;
}

/**
 * Build the GPT-2 / CLIP "bytes_to_unicode" mapping. This produces a
 * reversible bijection from the 256 byte values to a set of 256 visible
 * unicode characters, so that BPE - which operates on strings - can
 * losslessly represent arbitrary UTF-8 bytes. Identical to
 * `bytes_to_unicode` in the reference.
 */
function buildByteEncoder(): string[] {
  const bs: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b); // '!' .. '~'
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
  for (let b = 0xae; b <= 0xff; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const out = Array.from<string>({ length: 256 });
  for (let i = 0; i < bs.length; i++) {
    out[bs[i]] = String.fromCodePoint(cs[i]);
  }
  return out;
}

/** CLIP's pre-tokenization regex, ported verbatim. Matches contractions,
 *  letter runs, single digits, and runs of non-letter/non-digit/non-space
 *  punctuation. Unicode property escapes require ES2018+, which we have. */
const PRETOKEN_PATTERN =
  /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;

function basicClean(text: string): string {
  // Minimal HTML entity unescape. The reference uses Python's html.unescape;
  // we cover the five XML entities plus &nbsp;, which is what real-world
  // pasted text actually contains. Numeric entities are rare in prompts.
  return text
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ")
    .trim();
}

function whitespaceClean(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** All adjacent symbol pairs in a word. Used by the BPE merge loop. */
function getPairs(word: string[]): Set<string> {
  const pairs = new Set<string>();
  for (let i = 0; i < word.length - 1; i++) {
    pairs.add(word[i] + "\u0000" + word[i + 1]);
  }
  return pairs;
}

export class ClipTokenizer {
  private readonly encoder: Map<string, number>;
  private readonly bpeRanks: Map<string, number>;
  private readonly byteEncoder: string[];
  private readonly cache = new Map<string, string>();
  readonly bosId: number;
  readonly eosId: number;

  /**
   * @param vocab Parsed contents of vocab.json (token string -> id).
   * @param merges Contents of merges.txt as a single string. The first line
   *               is a "#version" comment per the GPT-2/CLIP convention and
   *               is skipped.
   */
  constructor(vocab: Record<string, number>, merges: string) {
    this.encoder = new Map(Object.entries(vocab));
    this.byteEncoder = buildByteEncoder();

    const lines = merges.split("\n");
    // Skip the version header (line 0) and any trailing blank line.
    const merged: [string, string][] = [];
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];
      if (!line) continue;
      const sp = line.indexOf(" ");
      if (sp < 0) continue;
      merged.push([line.slice(0, sp), line.slice(sp + 1)]);
    }
    this.bpeRanks = new Map();
    for (let i = 0; i < merged.length; i++) {
      this.bpeRanks.set(merged[i][0] + "\u0000" + merged[i][1], i);
    }

    const bos = this.encoder.get(BOS_TOKEN);
    const eos = this.encoder.get(EOS_TOKEN);
    if (bos === undefined || eos === undefined) {
      throw new Error("CLIP vocab.json is missing <|startoftext|> or <|endoftext|>");
    }
    this.bosId = bos;
    this.eosId = eos;
  }

  /** Run the BPE merge loop on a single byte-encoded pre-token. Returns the
   *  merged subwords joined with spaces, matching the reference's `bpe()`
   *  output shape. The trailing </w> marker is applied to the LAST char of
   *  the input before merging, per CLIP's convention. */
  private bpe(token: string): string {
    const cached = this.cache.get(token);
    if (cached !== undefined) return cached;

    const chars = Array.from(token);
    if (chars.length === 0) return "";
    // Append the end-of-word marker to the last character.
    const word: string[] = chars.slice(0, -1);
    word.push(chars[chars.length - 1] + "</w>");

    let pairs = getPairs(word);
    if (pairs.size === 0) {
      const result = word[0];
      this.cache.set(token, result);
      return result;
    }

    // Repeatedly find the highest-priority adjacent pair and merge it.
    while (true) {
      let bestRank = Infinity;
      let bestPair: string | null = null;
      let bestFirst = "";
      let bestSecond = "";
      for (const pair of pairs) {
        const rank = this.bpeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
          const sep = pair.indexOf("\u0000");
          bestFirst = pair.slice(0, sep);
          bestSecond = pair.slice(sep + 1);
        }
      }
      if (bestPair === null) break;

      const newWord: string[] = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(bestFirst, i);
        if (j === -1) {
          for (let k = i; k < word.length; k++) newWord.push(word[k]);
          break;
        }
        for (let k = i; k < j; k++) newWord.push(word[k]);
        if (j < word.length - 1 && word[j] === bestFirst && word[j + 1] === bestSecond) {
          newWord.push(bestFirst + bestSecond);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }
      word.length = 0;
      for (const w of newWord) word.push(w);
      if (word.length === 1) break;
      pairs = getPairs(word);
    }

    const result = word.join(" ");
    this.cache.set(token, result);
    return result;
  }

  /** Encode a prompt to the fixed-length id sequence the CLIP text encoder
   *  expects. Always returns CLIP_CONTEXT_LENGTH ids. */
  encode(text: string): TokenizedPrompt {
    const cleaned = whitespaceClean(basicClean(text)).toLowerCase();

    const bpeIds: number[] = [];
    const matches = cleaned.matchAll(PRETOKEN_PATTERN);
    for (const m of matches) {
      const pretoken = m[0];
      // UTF-8 encode then map each byte through byte_encoder to a visible
      // unicode char. The result is the input to BPE.
      const bytes = new TextEncoder().encode(pretoken);
      let mapped = "";
      for (let i = 0; i < bytes.length; i++) {
        mapped += this.byteEncoder[bytes[i]];
      }
      const merged = this.bpe(mapped);
      for (const sub of merged.split(" ")) {
        const id = this.encoder.get(sub);
        if (id === undefined) {
          // Should be impossible if vocab and merges agree, but bail loudly
          // rather than silently producing wrong embeddings.
          throw new Error(`CLIP tokenizer: unknown subword "${sub}"`);
        }
        bpeIds.push(id);
      }
    }

    // Reserve two slots for BOS and EOS. Anything beyond that gets truncated.
    const maxBody = CLIP_CONTEXT_LENGTH - 2;
    const truncated = bpeIds.length > maxBody;
    const body = truncated ? bpeIds.slice(0, maxBody) : bpeIds;

    const ids = new Int32Array(CLIP_CONTEXT_LENGTH);
    const mask = new Int32Array(CLIP_CONTEXT_LENGTH);
    ids[0] = this.bosId;
    mask[0] = 1;
    for (let i = 0; i < body.length; i++) {
      ids[i + 1] = body[i];
      mask[i + 1] = 1;
    }
    const eosIdx = body.length + 1;
    ids[eosIdx] = this.eosId;
    mask[eosIdx] = 1;
    // Pad with EOS (matches diffusers / SD1.5 reference).
    for (let i = eosIdx + 1; i < CLIP_CONTEXT_LENGTH; i++) {
      ids[i] = this.eosId;
    }

    return {
      ids,
      attentionMask: mask,
      length: eosIdx + 1,
      truncated,
    };
  }
}
