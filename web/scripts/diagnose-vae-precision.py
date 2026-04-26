#!/usr/bin/env python3
"""Numerical diagnostic: is VAE precision/architecture the FastWan blockiness bug?

Loads a cached "final latent" (transformer output, normalized space) and
decodes it four ways:
  A: Wan VAE (AutoencoderKLWan) fp32 -- reference
  B: Wan VAE fp16
  C: LightTAE fp32
  D: LightTAE fp16

Frames are normalized to [0,1] for fair pairwise comparison. Pairwise
max_abs / mean_abs / RMSE reported. Frame tensors cached to .npy under
notes/diagnose-vae/ so we can re-analyze without re-decoding.

Each model is loaded, used, then explicitly freed before the next one
loads -- keeps PyTorch CPU RAM under ~10 GB even with fp32 Wan VAE.

Wan VAE expects DENORMALIZED latent (x * std[c] + mean[c]). LightTAE
takes the raw normalized latent (matches what we ship in browser).

Default latent: notes/fastwan-final_latent_normalized-1x48x21x30x52-fp32.bin
(480x832 geometry, T=21). Blockiness reproduces at this shape per
worklog so it's a fair diagnostic input.

Usage:
    uv run diagnose-vae-precision.py [--latent PATH] [--out-dir PATH]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -- LightTAE decoder (copied from export-fastwan-vae.py to avoid import) --

def _conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), act_func, _conv(n_out, n_out), act_func, _conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class LightTAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.latent_channels = 48
        self.frames_to_trim = 3

        act_func = nn.ReLU(inplace=True)
        n_f = [256, 128, 64, 64]

        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(self.latent_channels, n_f[0]),
            act_func,
            _MemBlock(n_f[0], n_f[0], act_func),
            _MemBlock(n_f[0], n_f[0], act_func),
            _MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2),
            _TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),
            _MemBlock(n_f[1], n_f[1], act_func),
            _MemBlock(n_f[1], n_f[1], act_func),
            _MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2),
            _TGrow(n_f[1], 2),
            _conv(n_f[1], n_f[2], bias=False),
            _MemBlock(n_f[2], n_f[2], act_func),
            _MemBlock(n_f[2], n_f[2], act_func),
            _MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2),
            _TGrow(n_f[2], 2),
            _conv(n_f[2], n_f[3], bias=False),
            act_func,
            _conv(n_f[3], 3 * self.patch_size ** 2),
        )

    def patch_tgrow_layers(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, _TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def forward(self, latents):
        # latents: [1, T, 48, H, W]
        x = latents.squeeze(0)
        for b in self.decoder:
            if isinstance(b, _MemBlock):
                mem = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)
                x = b(x, mem)
            else:
                x = b(x)
        x = x.clamp(0, 1)
        x = F.pixel_shuffle(x, self.patch_size)
        x = x.unsqueeze(0)
        return x[:, self.frames_to_trim:]


# -- Helpers --

def free():
    gc.collect()


def to_canonical_uint01(frames_thwc_or_tchw, layout):
    """Bring frames to canonical [T, H, W, 3] float32 in [0, 1]."""
    arr = frames_thwc_or_tchw
    if layout == "NTCHW":
        # [1, T, 3, H, W] -> [T, H, W, 3]
        arr = arr.squeeze(0).permute(0, 2, 3, 1).contiguous()
    elif layout == "NCTHW":
        # [1, 3, T, H, W] -> [T, H, W, 3]
        arr = arr.squeeze(0).permute(1, 2, 3, 0).contiguous()
    else:
        raise ValueError(layout)
    arr = arr.float()
    # If looks like [-1, 1] (Wan VAE native range), shift to [0, 1].
    if arr.min().item() < -0.05:
        arr = (arr + 1.0) / 2.0
    arr = arr.clamp(0.0, 1.0)
    return arr.cpu().numpy()


def pairwise_stats(label_a, a, label_b, b):
    diff = (a.astype(np.float32) - b.astype(np.float32))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return f"{label_a} vs {label_b}: max_abs={max_abs:.4f}  mean_abs={mean_abs:.4f}  rmse={rmse:.4f}"


# -- Main --

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=Path,
                   default=Path("../../../notes/fastwan-final_latent_normalized-1x48x21x30x52-fp32.bin"))
    p.add_argument("--latent-shape", nargs=5, type=int, default=[1, 48, 21, 30, 52],
                   metavar=("B", "C", "T", "H", "W"))
    p.add_argument("--vae-source", type=Path,
                   default=Path("../../../notes/models/fastwan/source/vae"))
    p.add_argument("--lighttae-weights", type=Path,
                   default=Path("../../../notes/models/fastwan/lightx2v-weights/lighttaew2_2.safetensors"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("../../../notes/diagnose-vae"))
    p.add_argument("--log", type=Path,
                   default=Path("../../../notes/diagnose-vae/diagnose-vae-precision.log"))
    p.add_argument("--skip-fp32-wan", action="store_true",
                   help="Skip Wan VAE fp32 decode (RAM escape hatch)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log, "w", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Log: {args.log}")
    log(f"Output dir: {args.out_dir}")
    log(f"Latent: {args.latent}  shape={args.latent_shape}")

    # -- Load latent --
    if not args.latent.exists():
        log(f"FATAL: latent not found at {args.latent}")
        sys.exit(1)
    raw = np.fromfile(args.latent, dtype=np.float32)
    expected = int(np.prod(args.latent_shape))
    if raw.size != expected:
        log(f"FATAL: latent file has {raw.size} floats, expected {expected}")
        sys.exit(1)
    latent_norm_np = raw.reshape(args.latent_shape)
    log(f"  latent stats: min={latent_norm_np.min():.3f} max={latent_norm_np.max():.3f} "
        f"mean={latent_norm_np.mean():.3f} std={latent_norm_np.std():.3f}")

    # -- Load Wan VAE config to get latents_mean / latents_std --
    cfg_path = args.vae_source / "config.json"
    cfg = json.loads(cfg_path.read_text())
    mean_arr = np.asarray(cfg["latents_mean"], dtype=np.float32)
    std_arr = np.asarray(cfg["latents_std"], dtype=np.float32)
    assert mean_arr.shape == (48,) and std_arr.shape == (48,), (mean_arr.shape, std_arr.shape)
    log(f"  latents_mean[0..4] = {mean_arr[:4]}")
    log(f"  latents_std[0..4]  = {std_arr[:4]}")

    # Wan-VAE-input (denormalized): x * std + mean per channel.
    # Latent is [B, C, T, H, W]; broadcast over C.
    latent_denorm_np = (latent_norm_np * std_arr.reshape(1, 48, 1, 1, 1)
                        + mean_arr.reshape(1, 48, 1, 1, 1)).astype(np.float32)
    log(f"  denormalized stats: min={latent_denorm_np.min():.3f} max={latent_denorm_np.max():.3f} "
        f"mean={latent_denorm_np.mean():.3f} std={latent_denorm_np.std():.3f}")

    T = args.latent_shape[2]
    H = args.latent_shape[3]
    W = args.latent_shape[4]
    expected_frames = T * 4 - 3
    expected_h = H * 16
    expected_w = W * 16
    log(f"  expected output frames: T_out={expected_frames} H={expected_h} W={expected_w}")

    out_paths = {}

    # ---- A: Wan VAE fp32 ----
    if not args.skip_fp32_wan:
        log("")
        log("=== A: Wan VAE fp32 ===")
        from diffusers import AutoencoderKLWan
        t0 = time.time()
        vae = AutoencoderKLWan.from_pretrained(str(args.vae_source), torch_dtype=torch.float32)
        vae.eval()
        try:
            vae.enable_slicing()
        except Exception as e:
            log(f"  enable_slicing unavailable: {e}")
        try:
            vae.enable_tiling()
        except Exception as e:
            log(f"  enable_tiling unavailable: {e}")
        log(f"  loaded in {time.time()-t0:.1f}s")

        latent_t = torch.from_numpy(latent_denorm_np).to(torch.float32)
        t0 = time.time()
        with torch.no_grad():
            out = vae.decode(latent_t, return_dict=False)[0]  # [1, 3, T_out, H, W]
        log(f"  decoded in {time.time()-t0:.1f}s -> shape {list(out.shape)} "
            f"min={out.min().item():.3f} max={out.max().item():.3f}")
        frames_a = to_canonical_uint01(out, "NCTHW")
        del out, vae, latent_t
        free()
        path_a = args.out_dir / "frames_A_wan_fp32.npy"
        np.save(path_a, frames_a)
        out_paths["A_wan_fp32"] = (path_a, frames_a.shape)
        log(f"  saved {path_a}  shape={frames_a.shape}")
        del frames_a
        free()
    else:
        log("(skipping A: Wan VAE fp32)")

    # ---- B: Wan VAE fp16 ----
    log("")
    log("=== B: Wan VAE fp16 ===")
    from diffusers import AutoencoderKLWan
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(str(args.vae_source), torch_dtype=torch.float16)
    vae.eval()
    try:
        vae.enable_slicing()
    except Exception:
        pass
    try:
        vae.enable_tiling()
    except Exception:
        pass
    log(f"  loaded in {time.time()-t0:.1f}s")

    latent_t = torch.from_numpy(latent_denorm_np).to(torch.float16)
    t0 = time.time()
    with torch.no_grad():
        out = vae.decode(latent_t, return_dict=False)[0]
    log(f"  decoded in {time.time()-t0:.1f}s -> shape {list(out.shape)} "
        f"min={out.float().min().item():.3f} max={out.float().max().item():.3f}")
    frames_b = to_canonical_uint01(out, "NCTHW")
    del out, vae, latent_t
    free()
    path_b = args.out_dir / "frames_B_wan_fp16.npy"
    np.save(path_b, frames_b)
    out_paths["B_wan_fp16"] = (path_b, frames_b.shape)
    log(f"  saved {path_b}  shape={frames_b.shape}")
    del frames_b
    free()

    # ---- C: LightTAE fp32 ----
    log("")
    log("=== C: LightTAE fp32 ===")
    from safetensors.torch import load_file
    t0 = time.time()
    sd = load_file(str(args.lighttae_weights), device="cpu")
    decoder_sd = {k: v.float() for k, v in sd.items() if k.startswith("decoder.")}
    del sd
    model = LightTAEDecoder()
    model.load_state_dict(model.patch_tgrow_layers(decoder_sd))
    del decoder_sd
    model.eval()
    log(f"  loaded fp32 in {time.time()-t0:.1f}s")

    # LightTAE wants [B, T, C, H, W]; latent_norm_np is [B, C, T, H, W]
    latent_lt = torch.from_numpy(latent_norm_np).permute(0, 2, 1, 3, 4).contiguous().to(torch.float32)
    t0 = time.time()
    with torch.no_grad():
        out = model(latent_lt)  # [1, T_out, 3, H, W]
    log(f"  decoded in {time.time()-t0:.1f}s -> shape {list(out.shape)} "
        f"min={out.min().item():.3f} max={out.max().item():.3f}")
    frames_c = to_canonical_uint01(out, "NTCHW")
    del out, model, latent_lt
    free()
    path_c = args.out_dir / "frames_C_lighttae_fp32.npy"
    np.save(path_c, frames_c)
    out_paths["C_lighttae_fp32"] = (path_c, frames_c.shape)
    log(f"  saved {path_c}  shape={frames_c.shape}")
    del frames_c
    free()

    # ---- D: LightTAE fp16 ----
    log("")
    log("=== D: LightTAE fp16 ===")
    t0 = time.time()
    sd = load_file(str(args.lighttae_weights), device="cpu")
    decoder_sd = {k: v.float() for k, v in sd.items() if k.startswith("decoder.")}
    del sd
    model = LightTAEDecoder()
    model.load_state_dict(model.patch_tgrow_layers(decoder_sd))
    del decoder_sd
    model.eval()
    model = model.half()
    log(f"  loaded fp16 in {time.time()-t0:.1f}s")

    latent_lt = torch.from_numpy(latent_norm_np).permute(0, 2, 1, 3, 4).contiguous().to(torch.float16)
    t0 = time.time()
    with torch.no_grad():
        out = model(latent_lt)
    log(f"  decoded in {time.time()-t0:.1f}s -> shape {list(out.shape)} "
        f"min={out.float().min().item():.3f} max={out.float().max().item():.3f}")
    frames_d = to_canonical_uint01(out, "NTCHW")
    del out, model, latent_lt
    free()
    path_d = args.out_dir / "frames_D_lighttae_fp16.npy"
    np.save(path_d, frames_d)
    out_paths["D_lighttae_fp16"] = (path_d, frames_d.shape)
    log(f"  saved {path_d}  shape={frames_d.shape}")
    del frames_d
    free()

    # ---- Pairwise diffs ----
    log("")
    log("=== Pairwise diffs (frames in [0,1]) ===")
    loaded = {k: np.load(v[0]) for k, v in out_paths.items()}
    keys = list(loaded.keys())
    for k, arr in loaded.items():
        log(f"  {k}: shape={arr.shape} min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
    log("")

    # Shape check
    shapes = {k: a.shape for k, a in loaded.items()}
    if len(set(shapes.values())) != 1:
        log(f"WARNING: output shapes differ: {shapes}")

    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            a, b = loaded[ka], loaded[kb]
            if a.shape != b.shape:
                log(f"  {ka} vs {kb}: SHAPE MISMATCH {a.shape} vs {b.shape}")
                continue
            log("  " + pairwise_stats(ka, a, kb, b))

    log("")
    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
