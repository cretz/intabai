#!/usr/bin/env python3
"""Standalone driver: rewrite a FastWan transformer block ONNX to chunk attn1.

See lib/chunk_attn1.py for the rewrite itself. This script is used for
one-off reprocessing of existing block_NN.onnx files without re-running the
full transformer export. The same rewrite is folded into
export-fastwan-transformer.py so fresh exports are chunked automatically.

Usage:
    uv run chunk-attn1-onnx.py <input.onnx> <output.onnx> [--n 3]
"""

import argparse
import sys
from pathlib import Path

import onnx

from lib.chunk_attn1 import chunk_attn1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--n", type=int, default=3, help="number of seq chunks")
    ap.add_argument("--seq-len", type=int, required=True,
                    help="exact attn1 seq_len (e.g. 4725 for 480x480, 8190 for 480x832)")
    args = ap.parse_args()

    print(f"loading {args.input}")
    model = onnx.load(str(args.input))
    print(f"rewriting attn1 with N={args.n} chunks (size {args.seq_len // args.n})")
    model = chunk_attn1(model, args.n, seq_len=args.seq_len)

    print("checking model")
    onnx.checker.check_model(model, full_check=False)

    print(f"saving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
