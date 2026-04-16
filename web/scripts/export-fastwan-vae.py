#!/usr/bin/env python3
"""Export LightTAE (TAEHV) Wan 2.2 video VAE decoder to ONNX.

LightTAE is a tiny (~46 MB) Conv2D-only VAE decoder that replaces the full
Wan-VAE (2.63 GB, 3D causal conv). It decodes Wan latents to video frames
using temporal reshaping tricks (MemBlock, TPool, TGrow) instead of 3D conv.

Source: github.com/ModelTC/LightX2V, lightx2v/models/video_encoders/hf/tae.py
Weights: huggingface.co/lightx2v/Autoencoders (lighttaew2_2.safetensors)

The model's decode_video() uses apply_model_with_memblocks() which has a
parallel mode that flattens batch*time and iterates blocks sequentially.
We wrap this for ONNX export.

  Inputs:
    latents    [B, T, 48, H, W]    float16   (NTCHW, Wan 2.2 latent space)

  Output:
    frames     [B, T_out, 3, H_out, W_out]   float16   (NTCHW, RGB [0,1])

    T_out = T * 4 - 3   (4x temporal upsample, trim first 3 frames)
    H_out = H * 16      (3x spatial upsample(2x) + pixel_shuffle(2))
    W_out = W * 16

For standard Wan 480x832 81-frame output:
  Input:  [1, 21, 48, 30, 52]   (81/4=~21 latent frames, 480/16=30, 832/16=52)
  Output: [1, 81, 3, 480, 832]

Usage:
    uv run export-fastwan-vae.py <weights_path> <output_dir>

Example:
    uv run export-fastwan-vae.py \
      ../../notes/models/fastwan/lightx2v-weights/lighttaew2_2.safetensors \
      ../../notes/models/fastwan/onnx
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- LightTAE model (inlined from lightx2v tae.py to avoid dep) ---

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
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


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class LightTAEDecoder(nn.Module):
    """Standalone LightTAE Wan 2.2 decoder for ONNX export.

    Wraps the decode_video parallel path into a single forward() call.
    Input: NTCHW latents, Output: NTCHW RGB frames.
    """

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.latent_channels = 48
        self.frames_to_trim = 3  # 2^2 - 1

        act_func = nn.ReLU(inplace=True)
        n_f = [256, 128, 64, 64]

        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[1], 2),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[2], 2),
            conv(n_f[2], n_f[3], bias=False),
            act_func,
            conv(n_f[3], 3 * self.patch_size ** 2),
        )

    def patch_tgrow_layers(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def forward(self, latents):
        # latents: [1, T, 48, H, W] - batch is always 1 for browser inference
        x = latents.squeeze(0)  # [T, C, H, W]

        for b in self.decoder:
            if isinstance(b, MemBlock):
                # Shift-by-one memory: previous frame for each position
                mem = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)
                x = b(x, mem)
            else:
                x = b(x)

        x = x.clamp(0, 1)
        # pixel_shuffle on 4D (ONNX only supports 4D pixel_shuffle)
        T_out, C_out, H_out, W_out = x.shape
        x = F.pixel_shuffle(x, self.patch_size)  # [T_out, 3, H*2, W*2]
        x = x.unsqueeze(0)  # [1, T_out, 3, H*2, W*2]
        return x[:, self.frames_to_trim:]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "weights_path",
        type=Path,
        help="Path to lighttaew2_2.safetensors",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for ONNX file",
    )
    parser.add_argument("--latent-frames", type=int, default=21,
                        help="Number of latent frames (default: 21, producing 81 output frames)")
    parser.add_argument("--latent-height", type=int, default=30,
                        help="Latent height (default: 30, for 480px output)")
    parser.add_argument("--latent-width", type=int, default=52,
                        help="Latent width (default: 52, for 832px output)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    if not args.weights_path.exists():
        print(f"Error: weights not found at {args.weights_path}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "vae_decoder.onnx"

    log_path = args.log or (args.output_dir / "export-vae.log")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Export log: {log_path}")

    # ---- Load model ----
    log(f"Loading LightTAE decoder from {args.weights_path}...")
    t0 = time.time()

    from safetensors.torch import load_file
    state_dict = load_file(str(args.weights_path), device="cpu")

    model = LightTAEDecoder()
    decoder_sd = {k: v for k, v in state_dict.items() if k.startswith("decoder.")}
    model.load_state_dict(model.patch_tgrow_layers(decoder_sd))
    model.eval()
    model = model.half()
    log(f"  loaded in {time.time() - t0:.1f}s")

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    log(f"  model size: {param_bytes / 1e6:.1f} MB")

    # ---- Build dummy inputs ----
    B = 1
    T = args.latent_frames
    C = 48
    H = args.latent_height
    W = args.latent_width
    expected_out_frames = T * 4 - 3
    expected_out_h = H * 16
    expected_out_w = W * 16

    log(f"  dummy input: latents [{B},{T},{C},{H},{W}]")
    log(f"  expected output: [{B},{expected_out_frames},3,{expected_out_h},{expected_out_w}]")

    dummy_latents = torch.randn(B, T, C, H, W, dtype=torch.float16)

    # ---- Verify forward pass ----
    log("  verifying forward pass...")
    t0 = time.time()
    with torch.no_grad():
        test_out = model(dummy_latents)
    log(f"  forward pass OK: output shape {list(test_out.shape)}, took {time.time() - t0:.1f}s")

    # ---- Export to ONNX ----
    log(f"Exporting to ONNX (opset {args.opset})...")
    log(f"  output: {output_path}")
    t0 = time.time()

    torch.onnx.export(
        model,
        (dummy_latents,),
        str(output_path),
        opset_version=args.opset,
        input_names=["latents"],
        output_names=["frames"],
        dynamo=False,
    )
    export_time = time.time() - t0
    log(f"  export completed in {export_time:.1f}s")

    onnx_size = output_path.stat().st_size
    log(f"  ONNX file size: {onnx_size / 1e6:.1f} MB")

    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
