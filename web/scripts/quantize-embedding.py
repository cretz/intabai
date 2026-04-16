#!/usr/bin/env python3
"""Quantize a raw fp16 embedding table to per-row symmetric int8.

Input format (same as export-fastwan-text-encoder.py writes):
  embedding.bin: raw fp16 little-endian, shape [vocab, d_model]

Output format:
  embedding_q8.bin:     int8 body, shape [vocab, d_model], row-major
  embedding_scales.bin: fp16 scales, shape [vocab]

Per-row symmetric quantization:
    scale[i] = max(abs(row[i])) / 127
    q[i][j] = round(row[i][j] / scale[i]).clip(-127, 127)
    dequant: row[i][j] = q[i][j] * scale[i]

JS-side lookup for a token id:
  1. Read 4096 int8 bytes at offset `id * 4096` from embedding_q8.bin
  2. Read 2 bytes (fp16 scale) at offset `id * 2` from embedding_scales.bin
  3. Dequant: out[j] = int8_body[j] * scale

Per-row scale (rather than per-block) is appropriate for embedding lookup:
we don't do any dot products that would amplify quantization error, just
token->row fetch. Int8 is safe margin; int4 would halve again but risks
visible damage on rare tokens.

Usage:
    uv run quantize-embedding.py <embedding.bin> [--vocab N] [--dim D]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to embedding.bin (raw fp16)")
    parser.add_argument("--vocab", type=int, default=256384,
                        help="Vocabulary size (default: 256384 for UMT5)")
    parser.add_argument("--dim", type=int, default=4096,
                        help="Embedding dimension (default: 4096 for UMT5)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: alongside input)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    q8_path = output_dir / "embedding_q8.bin"
    scales_path = output_dir / "embedding_scales.bin"

    vocab, dim = args.vocab, args.dim
    expected_bytes = vocab * dim * 2
    actual_bytes = args.input.stat().st_size
    if actual_bytes != expected_bytes:
        print(f"Error: size mismatch. Expected {expected_bytes} "
              f"({vocab}x{dim}x2), got {actual_bytes}", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {args.input} ({actual_bytes / 1e9:.2f} GB, {vocab} x {dim} fp16)")
    print(f"Output: {q8_path} + {scales_path}")

    t0 = time.time()
    # Memory-map the input to avoid loading 2.1 GB into RAM at once.
    embed_fp16 = np.memmap(args.input, dtype=np.float16, mode="r", shape=(vocab, dim))

    # Per-row max-abs scale. Compute in fp32 for accuracy, store fp16.
    print("Computing per-row scales...")
    row_max = np.zeros(vocab, dtype=np.float32)
    chunk = 4096
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        row_max[start:end] = np.abs(embed_fp16[start:end].astype(np.float32)).max(axis=1)
    # Avoid div-by-zero for zero rows (pad tokens, unused entries).
    scales = np.where(row_max > 0, row_max / 127.0, 1.0).astype(np.float32)
    print(f"  scale range: [{scales.min():.2e}, {scales.max():.2e}], "
          f"median {np.median(scales):.2e}")

    # Quantize and write int8 body in streaming chunks.
    print("Quantizing and writing int8 body...")
    with open(q8_path, "wb") as f_out:
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            rows = embed_fp16[start:end].astype(np.float32)
            chunk_scales = scales[start:end, None]
            q = np.round(rows / chunk_scales).clip(-127, 127).astype(np.int8)
            f_out.write(q.tobytes())

    # Scales as fp16 (enough precision, half the size).
    scales_fp16 = scales.astype(np.float16)
    with open(scales_path, "wb") as f_out:
        f_out.write(scales_fp16.tobytes())

    q8_size = q8_path.stat().st_size
    scales_size = scales_path.stat().st_size
    total_out = q8_size + scales_size
    dt = time.time() - t0

    print(f"\nDone in {dt:.1f}s:")
    print(f"  embedding_q8.bin:     {q8_size / 1e9:.2f} GB")
    print(f"  embedding_scales.bin: {scales_size / 1e6:.2f} MB")
    print(f"  total: {total_out / 1e9:.2f} GB "
          f"({100 * total_out / actual_bytes:.0f}% of fp16)")

    # Quick accuracy check: dequant a random subset, compare to original.
    print("\nAccuracy check (100 random rows):")
    rng = np.random.default_rng(42)
    sample = rng.choice(vocab, size=100, replace=False)
    q_body = np.memmap(q8_path, dtype=np.int8, mode="r", shape=(vocab, dim))
    orig = embed_fp16[sample].astype(np.float32)
    dequant = q_body[sample].astype(np.float32) * scales[sample, None]
    abs_err = np.abs(dequant - orig)
    rel_err = abs_err / (np.abs(orig) + 1e-8)
    print(f"  abs err: mean {abs_err.mean():.2e}, max {abs_err.max():.2e}")
    print(f"  rel err: mean {rel_err[np.abs(orig) > 1e-3].mean():.2e} "
          f"(excluding near-zero entries)")


if __name__ == "__main__":
    main()
