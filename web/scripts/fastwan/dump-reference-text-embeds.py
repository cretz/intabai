"""Dump reference UMT5 text embeds for a prompt to .npy so we can compare
against what our JS pipeline produces in the browser.

Produces:
  ref_tokens.npy     int32 [1, 512]  tokenized + padded ids
  ref_embeds.npy     fp16  [1, 512, 4096]  post-final-LayerNorm, post-zero-pad
  ref_meta.json      {"prompt", "valid_length", "token_ids"}
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path,
                    help="FastWan source checkout root (has text_encoder/, tokenizer/)")
    ap.add_argument("--prompt", default="a calm zen garden at dawn")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--output-dir", type=Path, default=Path("./ref-out"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, UMT5EncoderModel

    print(f"Loading tokenizer from {args.source_dir / 'tokenizer'}...")
    tok = AutoTokenizer.from_pretrained(str(args.source_dir / "tokenizer"))

    # Match diffusers WanPipeline._get_t5_prompt_embeds exactly.
    inputs = tok(
        [args.prompt],
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = inputs.input_ids
    mask = inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    valid_length = int(seq_lens[0].item())
    print(f"  tokenized: {valid_length} valid tokens (padded to {args.max_length})")
    print(f"  first {valid_length} ids: {ids[0, :valid_length].tolist()}")

    print(f"Loading UMT5EncoderModel from {args.source_dir / 'text_encoder'}...")
    model = UMT5EncoderModel.from_pretrained(
        str(args.source_dir / "text_encoder"), torch_dtype=torch.float16
    )
    model.eval()

    print("Running forward...")
    with torch.inference_mode():
        out = model(ids, mask).last_hidden_state  # fp16 [1, seq, 4096]

    # Match diffusers: slice to valid tokens, zero-pad rest.
    out = out[0, :valid_length]  # [valid, 4096]
    padded = torch.cat(
        [out, out.new_zeros(args.max_length - valid_length, out.size(1))], dim=0
    )
    padded = padded.unsqueeze(0).cpu().numpy()  # [1, 512, 4096] fp16

    print(f"  embeds: shape={padded.shape} dtype={padded.dtype}")
    arr_f32 = padded.astype(np.float32)
    print(f"  stats (all):   min={arr_f32.min():.4f} max={arr_f32.max():.4f} "
          f"mean={arr_f32.mean():.6f} nan={int(np.isnan(arr_f32).sum())}")
    valid_slice = arr_f32[0, :valid_length]
    print(f"  stats (valid): min={valid_slice.min():.4f} max={valid_slice.max():.4f} "
          f"mean={valid_slice.mean():.6f}")

    np.save(args.output_dir / "ref_tokens.npy", ids.cpu().numpy().astype(np.int32))
    np.save(args.output_dir / "ref_embeds.npy", padded)
    (args.output_dir / "ref_meta.json").write_text(json.dumps({
        "prompt": args.prompt,
        "valid_length": valid_length,
        "token_ids": ids[0, :valid_length].tolist(),
        "stats_all": {"min": float(arr_f32.min()), "max": float(arr_f32.max()),
                      "mean": float(arr_f32.mean())},
        "stats_valid": {"min": float(valid_slice.min()), "max": float(valid_slice.max()),
                        "mean": float(valid_slice.mean())},
    }, indent=2))
    print(f"Wrote {args.output_dir}/ref_tokens.npy, ref_embeds.npy, ref_meta.json")


if __name__ == "__main__":
    main()
