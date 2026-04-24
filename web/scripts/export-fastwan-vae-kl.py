#!/usr/bin/env python3
"""Export the full Wan 2.2 VAE decoder (AutoencoderKLWan) to ONNX.

This is the 3D-causal-conv VAE from the Wan 2.2 stack. Unlike LightTAE
(Conv2D per-frame), it maintains temporal context via a per-conv feature
cache and is the suspect fix for the motion-quality gap we see with
LightTAE in the browser pipeline.

The diffusers `_decode` implementation does a Python `for i in range(T)`
loop that calls `self.decoder(z[:, :, i:i+1])` and maintains the cache
between iterations. Tracing this unrolls the loop into a single large
graph at the fixed input T we specify.

  Inputs:
    latents   [B, 48, T, H, W]   float16/float32   (NCTHW, Wan latent space)

  Output:
    frames    [B, 3, T*4-3, H*16, W*16]   same dtype as input   (NCTHW, [-1, 1])

For the default Wan 2.2 480x832 81-frame output:
  Input:  [1, 48, 21, 30, 52]
  Output: [1, 3, 81, 480, 832]

Usage:
    uv run export-fastwan-vae-kl.py <source_dir> <output_dir> [--dtype fp16]

    source_dir must contain vae/ subdir with config.json and
    diffusion_pytorch_model.safetensors (the diffusers layout).

Example:
    uv run export-fastwan-vae-kl.py \\
      ../../../notes/models/fastwan/source \\
      ../../../notes/models/fastwan/hf-repo/onnx
"""

import argparse
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="FastWan diffusers clone root (contains vae/ subdir)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory; writes vae_decoder_kl.onnx (+ .data)",
    )
    parser.add_argument("--latent-frames", type=int, default=21)
    parser.add_argument("--latent-height", type=int, default=30)
    parser.add_argument("--latent-width", type=int, default=52)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Export dtype. fp16 keeps output size manageable (~1.3 GB) and "
        "matches the LightTAE path. fp32 is useful for numerical reference.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--output-name",
        default="vae_decoder_kl.onnx",
        help="ONNX file name within output_dir",
    )
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    vae_dir = args.source_dir / "vae"
    if not (vae_dir / "config.json").exists():
        print(f"Error: {vae_dir / 'config.json'} not found", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    log_path = args.log or (args.output_dir / "export-vae-kl.log")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Export log: {log_path}")
    log(f"Target: {output_path}")

    # ---- Load model ----
    log(f"Loading AutoencoderKLWan from {vae_dir}...")
    t0 = time.time()
    from diffusers import AutoencoderKLWan

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype)
    vae.eval()
    vae.use_tiling = False
    vae.use_slicing = False
    log(f"  loaded in {time.time() - t0:.1f}s, dtype={dtype}")

    # `WanUpsample` uses `mode="nearest-exact"` which maps to
    # aten::_upsample_nearest_exact2d - unsupported in opset 18 and awkward
    # in later opsets. For an integer scale factor of 2, "nearest-exact" and
    # "nearest" produce identical output (tie-breaking only matters when
    # scale_factor is non-integer). Swap the mode in place.
    import torch.nn as nn
    patched = 0
    for mod in vae.modules():
        if isinstance(mod, nn.Upsample) and mod.mode == "nearest-exact":
            mod.mode = "nearest"
            patched += 1
    log(f"  patched {patched} Upsample modules nearest-exact -> nearest")

    param_bytes = sum(p.numel() * p.element_size() for p in vae.parameters())
    log(f"  param memory: {param_bytes / 1e9:.2f} GB")

    # Decoder-only export wrapper. AutoencoderKLWan.decode() returns a
    # DecoderOutput; we want a raw tensor for ONNX.
    class DecoderWrap(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents, return_dict=False)[0]

    wrap = DecoderWrap(vae).eval()

    # ---- Build dummy input ----
    B = 1
    C = 48
    T = args.latent_frames
    H = args.latent_height
    W = args.latent_width
    log(f"  dummy input: latents [{B},{C},{T},{H},{W}] {dtype}")
    dummy = torch.randn(B, C, T, H, W, dtype=dtype)

    # ---- Verify forward pass ----
    log("  verifying forward pass (can take a minute on CPU)...")
    t0 = time.time()
    with torch.no_grad():
        test_out = wrap(dummy)
    log(
        f"  forward pass OK: output shape {list(test_out.shape)} dtype {test_out.dtype}, "
        f"took {time.time() - t0:.1f}s"
    )

    # ---- Export to ONNX ----
    # Weights are ~2.6 GB so we need external-data format. torch.onnx with
    # dynamo=False writes the external data alongside automatically when the
    # model is large; we point it at a specific name via the .data sibling.
    log(f"Exporting to ONNX (opset {args.opset})...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrap,
            (dummy,),
            str(output_path),
            opset_version=args.opset,
            input_names=["latents"],
            output_names=["frames"],
            dynamo=False,
        )
    log(f"  export completed in {time.time() - t0:.1f}s")

    onnx_size = output_path.stat().st_size
    log(f"  {output_path.name}: {onnx_size / 1e6:.1f} MB")
    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        log(f"  {data_path.name}: {data_path.stat().st_size / 1e9:.2f} GB")

    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
