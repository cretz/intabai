// Verifies our hand-ported CLIP BPE tokenizer (sd15/tokenizer.ts) against
// a Python reference produced by transformers.CLIPTokenizer. The reference
// output is committed at fixtures/clip/expected-ids.json - see
// scripts/clip_reference_test_gen.py for the regenerator.
//
// This test is intentionally JS-only and reads only committed fixture files
// so it can run in CI without Python, transformers, or network access.

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, it, expect } from "vitest";

import {
  ClipTokenizer,
  CLIP_CONTEXT_LENGTH,
} from "../src/sd15/tokenizer";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "fixtures", "clip");

interface ExpectedPrompt {
  text: string;
  ids: number[];
}

interface ExpectedFile {
  context_length: number;
  pad_token: string;
  prompts: ExpectedPrompt[];
}

const vocab = JSON.parse(
  readFileSync(join(FIXTURES, "vocab.json"), "utf8"),
) as Record<string, number>;
const merges = readFileSync(join(FIXTURES, "merges.txt"), "utf8");
const expected = JSON.parse(
  readFileSync(join(FIXTURES, "expected-ids.json"), "utf8"),
) as ExpectedFile;

const tokenizer = new ClipTokenizer(vocab, merges);

describe("ClipTokenizer", () => {
  it("matches the reference context length", () => {
    expect(expected.context_length).toBe(CLIP_CONTEXT_LENGTH);
  });

  for (const prompt of expected.prompts) {
    const label =
      prompt.text.length > 60
        ? `${prompt.text.slice(0, 57)}...`
        : prompt.text || "(empty)";
    it(`encodes: ${label}`, () => {
      const got = tokenizer.encode(prompt.text);
      expect(Array.from(got.ids)).toEqual(prompt.ids);
    });
  }
});
