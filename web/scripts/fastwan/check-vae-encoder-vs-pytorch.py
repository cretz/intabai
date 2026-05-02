"""Numerical diff: vae_encoder-{H}.onnx vs PyTorch LightTAEEncoder.

Loads a fresh LightTAEEncoder with the same weights the export script
uses, runs it on a random fp16 image batch, then runs the same input
through onnxruntime CPU on the exported ONNX file. Reports max abs diff.

This validates the ONNX export step (no traced bugs, no shape
mismatches). It does NOT validate that LightTAEEncoder matches the
upstream LightX2V TAEHV reference - that requires installing the
lightx2v package; we get that confidence by copying tae.py verbatim.
For semantic validation, also run the round-trip: encode an image,
decode via vae_decoder-{H}.onnx, eyeball the reconstruction.

Usage:
    uv run check-vae-encoder-vs-pytorch.py \\
        <weights_path> <onnx_path> --pixel-height 576 --pixel-width 576
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Re-import the encoder class from the export script so we test
# exactly what was exported. Use spec_from_file_location since the
# script name has dashes.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "export_fastwan_vae_encoder",
    Path(__file__).parent / "export-fastwan-vae-encoder.py",
)
_export_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_export_mod)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights_path", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--pixel-height", type=int, required=True)
    parser.add_argument("--pixel-width", type=int, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ---- Load PyTorch model ----
    print(f"[{time.strftime('%H:%M:%S')}] loading PyTorch encoder from {args.weights_path}")
    from safetensors.torch import load_file
    state_dict = load_file(str(args.weights_path), device="cpu")
    model = _export_mod.LightTAEEncoder()
    encoder_sd = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.load_state_dict(encoder_sd, strict=True)
    model.eval()
    model = model.half()

    # ---- Build identical input ----
    B = 1
    T = args.frames
    H = args.pixel_height
    W = args.pixel_width
    # Use [0,1] range as our assumed input convention (decoder clamps to [0,1]
    # at output, so encoder input is presumed symmetric).
    frames = torch.rand(B, T, 3, H, W, dtype=torch.float16)

    # ---- PyTorch reference ----
    print(f"[{time.strftime('%H:%M:%S')}] running PyTorch forward")
    t0 = time.time()
    with torch.no_grad():
        torch_out = model(frames).cpu().numpy()
    print(f"  pytorch: {torch_out.shape} {torch_out.dtype}, took {time.time() - t0:.2f}s")

    # ---- ONNX inference ----
    print(f"[{time.strftime('%H:%M:%S')}] running onnxruntime on {args.onnx_path}")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])
    feeds = {sess.get_inputs()[0].name: frames.cpu().numpy()}
    t0 = time.time()
    onnx_out = sess.run(None, feeds)[0]
    print(f"  onnx:    {onnx_out.shape} {onnx_out.dtype}, took {time.time() - t0:.2f}s")

    # ---- Diff ----
    diff = np.abs(torch_out.astype(np.float32) - onnx_out.astype(np.float32))
    print()
    print(f"  max abs diff: {diff.max():.6e}")
    print(f"  mean abs diff: {diff.mean():.6e}")
    print(f"  pytorch range: [{torch_out.min():.4f}, {torch_out.max():.4f}]")
    print(f"  onnx range:    [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")

    # Tolerances: fp16 round-trip noise is typically <1e-2; anything
    # larger usually indicates a real divergence.
    if diff.max() > 5e-2:
        print("\nWARNING: max diff > 5e-2; export likely has a structural bug")
        sys.exit(1)
    if diff.max() > 5e-3:
        print("\nNOTE: max diff > 5e-3; tolerable for fp16 but worth investigating")
    print("\nOK")


if __name__ == "__main__":
    main()
