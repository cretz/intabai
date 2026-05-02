#!/usr/bin/env python3
"""Quantize FastWan 2.2 transformer per-block ONNX files to q4f16.

Uses onnxruntime's MatMulNBitsQuantizer (weight-only RTN quantization on
MatMul weights) to shrink each ~327 MB fp16 block down to ~80 MB q4f16.
Shell_pre (~180 MB) is also quantized. Shell_post is left as fp16 (1.2 MB
isn't worth the quality tradeoff).

Writes outputs to <input_dir>-q4f16/ by default (e.g. `transformer/` ->
`transformer-q4f16/`, `text-encoder/` -> `text-encoder-q4f16/`).

Usage:
    uv run quantize-fastwan-blocks.py <input-dir>

Example:
    uv run quantize-fastwan-blocks.py \\
        ../../notes/models/fastwan/hf-repo/onnx/transformer
"""

import argparse
import gc
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil
from onnxruntime.quantization import matmul_nbits_quantizer

# Silence the "skip to quantize /block/Cast_N ..." INFO spam (one line per
# non-MatMul node, ~800 per block). WARNING level still surfaces real issues
# like "tensor too small to quantize".
logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)


def start_ram_watchdog(max_gb: float, log_fn):
    proc = psutil.Process(os.getpid())
    peak = [0.0]

    def tick():
        while True:
            rss_gb = proc.memory_info().rss / 1e9
            if rss_gb > peak[0]:
                peak[0] = rss_gb
            if rss_gb > max_gb:
                log_fn(f"!!! RAM {rss_gb:.2f} GB > {max_gb:.1f} GB budget, KILLING")
                os._exit(2)
            time.sleep(0.5)

    threading.Thread(target=tick, daemon=True).start()
    return peak


def quantize_one(
    input_path: Path,
    output_path: Path,
    block_size: int,
    accuracy_level: int,
    log_fn,
):
    t0 = time.time()
    in_size = input_path.stat().st_size
    log_fn(f"  loading {input_path.name} ({in_size / 1e6:.1f} MB)")

    # Pass the path directly so the quantizer owns the load/save lifecycle.
    # accuracy_level: 0=unset (uses input dtype), 1=fp32, 2=fp16, 3=bf16,
    # 4=int8. Level 4 is aggressive and loses precision for extreme-magnitude
    # inputs (UMT5-XXL embeds reach ±187 and overflow int8 scales), so the
    # text-encoder quantize passes accuracy_level=1 (fp32 accumulator).
    # Transformer blocks use level 4 with no observed accuracy issues.
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        str(input_path),
        bits=4,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=accuracy_level,
    )
    quant.process()

    # onnx.save_model opens the external-data sidecar in append mode, so a
    # pre-existing .data file from an earlier run would silently double in
    # size. Delete both before writing.
    data_path = output_path.parent / (output_path.name + ".data")
    if output_path.exists():
        output_path.unlink()
    if data_path.exists():
        data_path.unlink()

    # quant.model is an ORT ONNXModel wrapper; save_model_to_file writes the
    # graph + an adjacent ".data" sidecar when use_external_data_format=True.
    quant.model.save_model_to_file(str(output_path), use_external_data_format=True)

    del quant
    gc.collect()

    out_size = output_path.stat().st_size
    data_file = output_path.parent / (output_path.name + ".data")
    out_data_size = data_file.stat().st_size if data_file.exists() else 0
    total_out = out_size + out_data_size
    dt = time.time() - t0
    log_fn(
        f"  -> {output_path.name} ({total_out / 1e6:.1f} MB graph+data, "
        f"{100 * total_out / in_size:.0f}% of fp16, {dt:.1f}s)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing shell_pre.onnx + block_*.onnx + shell_post.onnx",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Quantization block size (32 = higher quality, 128 = smaller)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>-q4f16 sibling)",
    )
    parser.add_argument(
        "--ram-budget-gb",
        type=float,
        default=10.0,
        help="Process-total RAM budget; kill self if exceeded",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only quantize files matching this prefix (e.g. 'block_00' for smoke)",
    )
    parser.add_argument(
        "--copy-below-mb",
        type=float,
        default=5.0,
        help="Files smaller than this (MB) are copied as-is instead of quantized",
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=4,
        help="MatMulNBits accuracy_level: 0=unset, 1=fp32, 2=fp16, 3=bf16, 4=int8. "
        "Pass 1 for text-encoder (extreme-magnitude inputs overflow int8).",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (args.input_dir.parent / f"{args.input_dir.name}-q4f16")
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)

    log(f"Input:  {args.input_dir}")
    log(f"Output: {output_dir}")
    log(f"Block size: {args.block_size}")
    log(f"Accuracy level: {args.accuracy_level}")
    log(f"RAM budget: {args.ram_budget_gb:.1f} GB (self-kill if exceeded)")
    ram_peak = start_ram_watchdog(args.ram_budget_gb, log)

    # Auto-discover all .onnx files. Files smaller than --copy-below-mb are
    # copied as-is (too small to benefit from MatMulNBits quantization).
    all_onnx = sorted(args.input_dir.glob("*.onnx"))
    if args.only:
        all_onnx = [p for p in all_onnx if p.name.startswith(args.only)]

    for in_path in all_onnx:
        name = in_path.name
        size_mb = in_path.stat().st_size / 1e6
        out_path = output_dir / name
        if size_mb < args.copy_below_mb:
            shutil.copy2(in_path, out_path)
            log(f"  copied {name} ({size_mb:.2f} MB, fp16, too small to quantize)")
            continue
        quantize_one(in_path, out_path, args.block_size, args.accuracy_level, log)

    total = sum(p.stat().st_size for p in output_dir.iterdir())
    log(f"Total output size: {total / 1e9:.2f} GB")
    log(f"Peak RAM observed: {ram_peak[0]:.2f} GB")
    log("Done.")


if __name__ == "__main__":
    main()
