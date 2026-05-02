"""Round-trip sanity test: encode a real image then decode it.

Validates the input-normalization assumption for the LightTAE encoder.
The encoder's expected RGB range is not documented anywhere we copied
from; we assumed [0,1] because the decoder clamps its output to [0,1].
This test confirms by encoding-then-decoding a real image and
comparing the reconstruction against the input.

Uses PyTorch end-to-end (not the exported ONNX) because:
  - The exported vae_decoder-576.onnx is shape-locked to T_lat=21,
    while a single-image encode produces T_lat=1. PyTorch is dynamic.
  - We're testing the normalization convention, not the ONNX export.
    ONNX export was already validated by check-vae-encoder-vs-pytorch.py.

Saves the input and reconstructed image side-by-side to <output_dir>
so you can eyeball them. Reports PSNR.

Usage:
    uv run check-vae-roundtrip.py <weights_path> <image_path> <output_dir> \\
        [--pixel-size 576] [--input-range 0_1|m1_1]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights_path", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pixel-size", type=int, default=576,
                        help="Square H=W size, must be multiple of 16")
    parser.add_argument("--input-range", choices=["0_1", "m1_1"], default="0_1",
                        help="RGB normalization range: [0,1] or [-1,1]")
    args = parser.parse_args()

    if args.pixel_size % 16 != 0:
        sys.exit(f"--pixel-size must be multiple of 16, got {args.pixel_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    enc_mod = _load("export_fastwan_vae_encoder", "export-fastwan-vae-encoder.py")
    dec_mod = _load("export_fastwan_vae", "export-fastwan-vae.py")

    # ---- Load image ----
    from PIL import Image
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.image_path}")
    img = Image.open(args.image_path).convert("RGB").resize((args.pixel_size, args.pixel_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    chw = arr.transpose(2, 0, 1)  # CHW

    if args.input_range == "0_1":
        norm = chw  # [0,1]
    else:  # m1_1
        norm = chw * 2.0 - 1.0  # [-1,1]
    print(f"  input range mode: {args.input_range}; tensor range "
          f"[{norm.min():.3f}, {norm.max():.3f}]")

    # T=1 padded to T=4 by repeating (matches reference encode_video pad)
    frames = torch.from_numpy(norm).half().unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
    frames = frames.repeat(1, 4, 1, 1, 1)  # [1,4,3,H,W]

    # ---- Load encoder + decoder weights ----
    from safetensors.torch import load_file
    sd = load_file(str(args.weights_path), device="cpu")

    enc = enc_mod.LightTAEEncoder()
    enc.load_state_dict({k: v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    enc.eval().half()

    dec = dec_mod.LightTAEDecoder()
    dec_sd = {k: v for k, v in sd.items() if k.startswith("decoder.")}
    dec.load_state_dict(dec.patch_tgrow_layers(dec_sd))
    dec.eval().half()

    # ---- Encode ----
    print(f"[{time.strftime('%H:%M:%S')}] encoding [1,4,3,{args.pixel_size},{args.pixel_size}]")
    t0 = time.time()
    with torch.no_grad():
        latents = enc(frames)
    print(f"  latents: {tuple(latents.shape)} range "
          f"[{latents.min().item():.3f}, {latents.max().item():.3f}], "
          f"{time.time() - t0:.2f}s")

    # ---- Decode ----
    # Decoder expects [B, T_lat, 48, H_lat, W_lat] (NTCHW).
    print(f"[{time.strftime('%H:%M:%S')}] decoding")
    t0 = time.time()
    with torch.no_grad():
        recon = dec(latents)
    # recon: [1, T_out, 3, H, W] in [0,1]
    print(f"  recon: {tuple(recon.shape)}, {time.time() - t0:.2f}s")

    # ---- Compare ----
    # Take first reconstructed frame, undo input normalization to land
    # back in [0,1] for visual + PSNR comparison.
    recon_first = recon[0, 0].float().cpu().numpy()  # [3, H, W] in [0,1]
    target = chw  # [3, H, W] in [0,1]

    diff = recon_first - target
    mse = float((diff ** 2).mean())
    psnr = float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
    print()
    print(f"  MSE:  {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB  (>20 dB ~ recognizable, >25 dB ~ good)")

    # ---- Save side-by-side ----
    side = np.concatenate([target, recon_first], axis=2)  # [3, H, 2W]
    side = np.clip(side, 0, 1)
    side_img = (side.transpose(1, 2, 0) * 255).astype(np.uint8)
    out_path = args.output_dir / f"roundtrip_{args.input_range}_{args.pixel_size}.png"
    Image.fromarray(side_img).save(out_path)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
