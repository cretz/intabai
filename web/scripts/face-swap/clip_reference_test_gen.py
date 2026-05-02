#!/usr/bin/env python3
"""Regenerate the CLIP tokenizer reference output used by tokenizer.test.ts.

This is a *dev-time* tool. CI does not run it. The committed
expected-ids.json file is the reference our TypeScript ClipTokenizer is
checked against; this script is what produced it.

Run when you want to add prompts to the test set or after changing the
fixture vocab/merges files. Requires `transformers` (pip install
transformers).

    cd web
    python scripts/clip_reference_test_gen.py

Output: web/test/fixtures/clip/expected-ids.json

The expected-ids.json shape is:
    {
      "prompts": [
        {
          "text": "<input prompt>",
          "ids":  [<int>, <int>, ...]   # length 77, padded with EOS
        },
        ...
      ]
    }
"""

import json
import os
import sys

# Quiet the noisy startup warnings: we don't need PyTorch for tokenization,
# and the symlink warning is a Windows-only quirk of huggingface_hub. Set
# these BEFORE importing transformers so they're respected.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from transformers import CLIPTokenizer
except ImportError:
    sys.stderr.write(
        "transformers is required: pip install transformers\n"
    )
    sys.exit(1)

# Prompts cover empty input, single token, normal sentences, punctuation,
# contractions, mixed case (we lowercase), unicode, repeated words, and a
# long prompt that should hit the 77-token truncation path.
PROMPTS = [
    "",
    "cat",
    "a cat",
    "a photo of an astronaut riding a horse",
    "don't worry, it's fine!",
    "RED cube on a BLUE sphere",
    "café münchen",
    "the the the the the",
    "highly detailed, 8k, cinematic lighting, masterpiece, "
    "best quality, intricate details, volumetric lighting, "
    "trending on artstation, photorealistic, ultra realistic, "
    "award winning photography, depth of field, bokeh, "
    "studio lighting, professional, sharp focus, hdr, "
    "vibrant colors, dramatic atmosphere, epic composition, "
    "concept art, digital painting, hyperrealistic, "
    "octane render, unreal engine 5",
]

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.dirname(HERE)
FIXTURE_DIR = os.path.join(WEB_ROOT, "test", "fixtures", "clip")
VOCAB = os.path.join(FIXTURE_DIR, "vocab.json")
MERGES = os.path.join(FIXTURE_DIR, "merges.txt")
OUT = os.path.join(FIXTURE_DIR, "expected-ids.json")


def main() -> None:
    if not os.path.isfile(VOCAB) or not os.path.isfile(MERGES):
        sys.stderr.write(
            f"Missing fixture files in {FIXTURE_DIR}.\n"
            f"  vocab.json: {os.path.isfile(VOCAB)}\n"
            f"  merges.txt: {os.path.isfile(MERGES)}\n"
        )
        sys.exit(1)

    # transformers v5 broke direct CLIPTokenizer(vocab_file=..., merges_file=...)
    # construction: it silently no-ops and reports vocab_size=2 with bos=0/eos=2,
    # producing nonsense ids. The only path that actually works is
    # from_pretrained, which downloads the tokenizer files (no model weights)
    # to ~/.cache/huggingface/hub/ on first call, then reuses them forever.
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Sanity check: the JS test loads the committed fixture vocab/merges from
    # tlwu/stable-diffusion-v1-5-onnxruntime, while transformers downloads from
    # openai/clip-vit-large-patch14. They are supposed to be byte-identical
    # (the tlwu repo just copies them), but if HF ever ships a divergent
    # version we want to know loudly here, not as a mysterious test failure.
    # Drift check: HF's openai/clip-vit-large-patch14 vocab.json is compact
    # JSON, while tlwu's copy is pretty-printed with 2-space indent. They are
    # semantically identical (same 49408 tokens, same id mapping) but not
    # byte-equal, so we compare parsed contents, not bytes.
    hf_dir = os.path.dirname(tok.init_kwargs.get("vocab_file", "")) or None
    if hf_dir:
        hf_vocab = os.path.join(hf_dir, "vocab.json")
        hf_merges = os.path.join(hf_dir, "merges.txt")
        if os.path.isfile(hf_vocab):
            with open(hf_vocab, "r", encoding="utf-8") as f:
                hf_v = json.load(f)
            with open(VOCAB, "r", encoding="utf-8") as f:
                fix_v = json.load(f)
            if hf_v != fix_v:
                sys.stderr.write(
                    "vocab.json id mapping from HF differs from committed "
                    "fixture - investigate before regenerating.\n"
                )
                sys.exit(1)
        if os.path.isfile(hf_merges):
            # merges.txt is line-oriented; compare as normalized line lists
            # to ignore trailing newlines and CRLF differences.
            with open(hf_merges, "r", encoding="utf-8") as f:
                hf_m = [ln.rstrip() for ln in f if ln.strip()]
            with open(MERGES, "r", encoding="utf-8") as f:
                fix_m = [ln.rstrip() for ln in f if ln.strip()]
            if hf_m != fix_m:
                sys.stderr.write(
                    "merges.txt from HF differs from committed fixture - "
                    "investigate before regenerating.\n"
                )
                sys.exit(1)

    out_prompts = []
    for text in PROMPTS:
        # SD1.5 / diffusers convention: pad to 77 with the EOS token, return
        # plain int ids. Truncate anything longer than 77.
        encoded = tok(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors=None,
        )
        out_prompts.append({"text": text, "ids": encoded["input_ids"]})

    payload = {
        "tokenizer": "openai/clip-vit-large-patch14 (via tlwu sd1.5 onnx)",
        "context_length": 77,
        "pad_token": "<|endoftext|>",
        "prompts": out_prompts,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(out_prompts)} prompts to {OUT}")


if __name__ == "__main__":
    main()
