"""Reference PyTorch decode of N(0,1) latent at the same shape JS uses.

Runs decoder.forward end-to-end on a fixed-seed Gaussian latent at shape
(1, 128, 4, 10, 10) - matches our SKIP_TRANSFORMER bypass test - and reports
pixel mean/std/min/max plus a center-frame PNG.

If pixel stats here ~match JS (mean ~= -0.7, std ~= 0.67), the export is
faithful and "noise in -> garbage" is just OOD model behavior.

If they diverge significantly, we have a real export bug somewhere between
PyTorch and ONNX/ORT.

Run from intabai/web/scripts:
  uv run python ltx/reference_decode.py \
    --space-path C:/work/personal/intabai/notes/ltx-video-distilled-space \
    --ckpt C:/work/personal/intabai/notes/models/ltx/source/ltxv-2b-0.9.8-distilled.safetensors \
    --stats C:/work/personal/intabai/notes/models/ltx/hf-repo/onnx/vae/per_channel_stats.f32 \
    --out-dir C:/work/personal/intabai/notes/ref-decode
"""
import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import psutil
import torch


def start_ram_watchdog(max_gb: float):
    proc = psutil.Process(os.getpid())
    peak = [0.0]

    def tick():
        while True:
            rss = proc.memory_info().rss / 1e9
            if rss > peak[0]:
                peak[0] = rss
            if rss > max_gb:
                print(f"!!! RAM {rss:.2f} GB > {max_gb:.1f} GB, KILLING", flush=True)
                os._exit(2)
            time.sleep(0.5)

    threading.Thread(target=tick, daemon=True).start()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--latent-t", type=int, default=4)
    ap.add_argument("--latent-hw", type=int, default=10)
    ap.add_argument("--decode-timestep", type=float, default=0.05)
    ap.add_argument("--decode-noise-scale", type=float, default=0.025)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--ram-budget-gb", type=float, default=14.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ram_peak = start_ram_watchdog(args.ram_budget_gb)
    print(f"RAM budget: {args.ram_budget_gb:.1f} GB")

    sys.path.insert(0, str(args.space_path))
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )

    print(f"Loading VAE from {args.ckpt}")
    vae = CausalVideoAutoencoder.from_pretrained(str(args.ckpt))
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    vae = vae.to(dtype).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    decoder = vae.decoder
    spatial_ds = vae.spatial_downscale_factor
    temporal_ds = vae.temporal_downscale_factor
    print(f"  spatial_downscale={spatial_ds}")
    print(f"  temporal_downscale={temporal_ds}")
    print(f"  timestep_conditioning={decoder.timestep_conditioning}")

    # Drop encoder to free ~half the VAE weight RAM. We only need decoder.
    vae.encoder = None
    gc.collect()

    # Load per-channel stats (128 fp32 mean, 128 fp32 std).
    raw = np.fromfile(args.stats, dtype=np.float32)
    assert raw.shape == (256,), raw.shape
    mean = torch.from_numpy(raw[:128].copy()).to(dtype)
    std = torch.from_numpy(raw[128:].copy()).to(dtype)
    print(f"  stats: mean[0..3]={mean[:4].tolist()} std[0..3]={std[:4].tolist()}")

    # Generate N(0,1) latent with deterministic seed (Box-Muller via torch).
    g = torch.Generator().manual_seed(args.seed)
    latent_norm = torch.randn(
        1, 128, args.latent_t, args.latent_hw, args.latent_hw,
        generator=g, dtype=torch.float32,
    ).to(dtype)
    print(f"  latent (normalized) shape={tuple(latent_norm.shape)}")
    print(
        f"  latent_norm: mean={latent_norm.float().mean():.4f} "
        f"std={latent_norm.float().std():.4f}"
    )

    # un_normalize: per-channel  x * std + mean.
    latent = latent_norm * std.view(1, 128, 1, 1, 1) + mean.view(1, 128, 1, 1, 1)
    print(
        f"  latent (un_normalized): mean={latent.float().mean():.4f} "
        f"std={latent.float().std():.4f}"
    )

    # Mix in noise per pipeline_ltx_video: latent = (1-s)*latent + s*noise.
    s = args.decode_noise_scale
    if s > 0:
        noise = torch.randn(latent.shape, generator=g, dtype=torch.float32).to(dtype)
        latent = latent * (1 - s) + noise * s
    print(
        f"  latent (post noise-mix s={s}): mean={latent.float().mean():.4f} "
        f"std={latent.float().std():.4f}"
    )

    # Run decoder.
    target_shape = (
        1, 3,
        (args.latent_t - 1) * temporal_ds + 1,
        args.latent_hw * spatial_ds,
        args.latent_hw * spatial_ds,
    )
    print(f"  decoder target_shape={target_shape}")

    timestep = torch.tensor([args.decode_timestep], dtype=dtype)
    with torch.no_grad():
        pixels = decoder(latent, target_shape=target_shape, timestep=timestep)
    pixels = pixels.float()
    print(f"  pixels shape={tuple(pixels.shape)}")
    print(
        f"  pixels: mean={pixels.mean().item():.4f} "
        f"std={pixels.std().item():.4f} "
        f"min={pixels.min().item():.4f} "
        f"max={pixels.max().item():.4f}"
    )

    # Per-channel pixel stats (R,G,B).
    for c in range(3):
        p = pixels[0, c]
        print(
            f"  ch{c}: mean={p.mean().item():.4f} std={p.std().item():.4f} "
            f"min={p.min().item():.4f} max={p.max().item():.4f}"
        )

    # Save center frame as PNG (clamped to [-1, 1] -> [0, 255]).
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow not installed, skipping PNG save")
        return
    T = pixels.shape[2]
    cf = T // 2
    img = pixels[0, :, cf].clamp(-1, 1).add(1).mul(0.5).mul(255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    Image.fromarray(img).save(args.out_dir / "ref_center_frame.png")
    print(f"  saved ref_center_frame.png ({img.shape})")

    # Also save raw float pixel tensor for fine-grained comparison.
    np.save(args.out_dir / "ref_pixels.npy", pixels.cpu().numpy())
    print(f"  saved ref_pixels.npy")
    print(f"Peak RAM: {ram_peak[0]:.2f} GB")


if __name__ == "__main__":
    main()
