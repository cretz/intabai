#!/usr/bin/env python3
"""Export LightTAE (TAEHV) Wan 2.2 video VAE encoder to ONNX.

Companion to export-fastwan-vae.py (decoder export). The encoder is
needed for the I2V path: an input image is VAE-encoded to a single
latent frame which the denoise loop pins as frame 0 of the latent
buffer (Wan 2.2 TI2V uses expand_timesteps with a frame-0 mask, no
channel concat, no CLIP).

Source: github.com/ModelTC/LightX2V, lightx2v/models/video_encoders/hf/tae.py
Weights: huggingface.co/lightx2v/Autoencoders (lighttaew2_2.safetensors)

  Inputs:
    frames     [B, T, 3, H, W]    float16   (NTCHW, RGB [0,1])

  Output:
    latents    [B, T_lat, 48, H_lat, W_lat]   float16

    T_lat   = T / 4         (4x temporal pool; T is padded to mult of 4 by repeating last frame)
    H_lat   = H / 16
    W_lat   = W / 16

For I2V single-image conditioning at 576x576:
  Input:  [1, 1, 3, 576, 576]    -> internally padded to T=4 by repeating
  Output: [1, 1, 48, 36, 36]

Usage:
    uv run export-fastwan-vae-encoder.py <weights_path> <output_dir>
      --pixel-height 576 --pixel-width 576

Output file: <output_dir>/vae_encoder.onnx (rename to vae_encoder-576.onnx etc.)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- LightTAE encoder (inlined from lightx2v tae.py to avoid dep) ---

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out), act_func,
            conv(n_out, n_out), act_func,
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class LightTAEEncoder(nn.Module):
    """Standalone LightTAE Wan 2.2 encoder for ONNX export.

    Wraps the encode_video parallel path into a single forward() call.
    Input: NTCHW RGB [0,1], Output: NTCHW 48-channel latents.

    Caller must pre-pad T so that T_in % 4 == 0 (we still also pad
    inside forward defensively, but ONNX trace bakes in the static
    T_in shape so callers should pass T_in directly).
    """

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.image_channels = 3
        self.latent_channels = 48

        act_func = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size ** 2, 64),
            act_func,
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            conv(64, self.latent_channels),
        )

    def forward(self, frames):
        # frames: [1, T, 3, H, W] - batch is always 1 for browser inference.
        # T is expected to be a multiple of 4 already (caller pads by
        # repeating last frame). For single-image I2V conditioning, the
        # caller passes T=4 with the same image repeated.
        N, T, _C_in, H_in, W_in = frames.shape

        # ONNX pixel_shuffle/unshuffle only supports 4D input, so flatten
        # batch*time -> [N*T, 3, H, W] -> pixel_unshuffle -> [N*T, 12, H/2, W/2]
        x = frames.reshape(N * T, _C_in, H_in, W_in)
        x = F.pixel_unshuffle(x, self.patch_size)

        for b in self.encoder:
            if isinstance(b, MemBlock):
                NT, Cc, Hh, Ww = x.shape
                Tt = NT // N
                _x = x.reshape(N, Tt, Cc, Hh, Ww)
                # mem = shift-by-one along T, frame 0 sees zeros
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :Tt].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)

        NT, Cc, Hh, Ww = x.shape
        Tt = NT // N
        return x.view(N, Tt, Cc, Hh, Ww)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("weights_path", type=Path, help="Path to lighttaew2_2.safetensors")
    parser.add_argument("output_dir", type=Path, help="Output directory for ONNX file")
    parser.add_argument("--pixel-height", type=int, required=True,
                        help="Pixel height of input image (e.g. 480, 576). Must be multiple of 16.")
    parser.add_argument("--pixel-width", type=int, required=True,
                        help="Pixel width of input image. Must be multiple of 16.")
    parser.add_argument("--frames", type=int, default=4,
                        help="Number of input frames (must be multiple of 4). "
                             "Default 4 = single image padded by caller.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    if not args.weights_path.exists():
        print(f"Error: weights not found at {args.weights_path}", file=sys.stderr)
        sys.exit(1)

    if args.frames % 4 != 0:
        print(f"Error: --frames must be a multiple of 4, got {args.frames}", file=sys.stderr)
        sys.exit(1)
    if args.pixel_height % 16 != 0 or args.pixel_width % 16 != 0:
        print("Error: pixel-height and pixel-width must be multiples of 16", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "vae_encoder.onnx"

    log_file = open(args.log, "w", buffering=1) if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_file is not None:
            log_file.write(line + "\n")

    if log_file is not None:
        log(f"Export log: {args.log}")

    # ---- Load model ----
    log(f"Loading LightTAE encoder from {args.weights_path}...")
    t0 = time.time()

    from safetensors.torch import load_file
    state_dict = load_file(str(args.weights_path), device="cpu")

    model = LightTAEEncoder()
    encoder_sd = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    missing = set(model.state_dict().keys()) - set(encoder_sd.keys())
    unexpected = set(encoder_sd.keys()) - set(model.state_dict().keys())
    if missing:
        log(f"  WARNING missing keys ({len(missing)}): {sorted(missing)[:5]}...")
    if unexpected:
        log(f"  WARNING unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...")
    model.load_state_dict(encoder_sd, strict=True)
    model.eval()
    model = model.half()
    log(f"  loaded in {time.time() - t0:.1f}s")

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    log(f"  model size: {param_bytes / 1e6:.1f} MB")

    # ---- Build dummy inputs ----
    B = 1
    T = args.frames
    C = 3
    H = args.pixel_height
    W = args.pixel_width
    expected_T_lat = T // 4
    expected_H_lat = H // 16
    expected_W_lat = W // 16

    log(f"  dummy input: frames [{B},{T},{C},{H},{W}]")
    log(f"  expected output: [{B},{expected_T_lat},48,{expected_H_lat},{expected_W_lat}]")

    # Use [0,1] range to match expected RGB normalization
    dummy_frames = torch.rand(B, T, C, H, W, dtype=torch.float16)

    # ---- Verify forward pass ----
    log("  verifying forward pass...")
    t0 = time.time()
    with torch.no_grad():
        test_out = model(dummy_frames)
    log(f"  forward pass OK: output shape {list(test_out.shape)}, took {time.time() - t0:.1f}s")
    out_T, out_C, out_H, out_W = test_out.shape[1:]
    if (out_T, out_C, out_H, out_W) != (expected_T_lat, 48, expected_H_lat, expected_W_lat):
        log(f"  WARNING shape mismatch vs expected")

    # ---- Export to ONNX ----
    log(f"Exporting to ONNX (opset {args.opset})...")
    log(f"  output: {output_path}")
    t0 = time.time()

    torch.onnx.export(
        model,
        (dummy_frames,),
        str(output_path),
        opset_version=args.opset,
        input_names=["frames"],
        output_names=["latents"],
        dynamo=False,
    )
    export_time = time.time() - t0
    log(f"  export completed in {export_time:.1f}s")

    onnx_size = output_path.stat().st_size
    log(f"  ONNX file size: {onnx_size / 1e6:.1f} MB")

    log("Done.")
    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
