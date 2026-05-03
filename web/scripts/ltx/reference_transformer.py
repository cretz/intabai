"""Reference PyTorch transformer forward on canonical deterministic input.

Mirrors the JS first-step transformer call so we can compare noise_pred
sample values across the two implementations without needing matched RNGs.

Canonical input (must match what JS emits):
  hidden_states         (1, 128, 128) all 1.0  fp16  [N_tokens=128, C=128]
  encoder_hidden_states (1, 256, 4096) all 0.0 fp16
  encoder_attention_mask (1, 256) all 1 int64
  indices_grid          (1, 3, 128) computed from latent shape (4, 8, 4)
                        with FPS=30, scale=(8, 32, 32), causal_fix
  timestep              (1,) = 1.0 fp16

Outputs (logged):
  noise_pred[0, 0, :8]       # token 0, channels 0..7
  noise_pred[0, 1, :8]       # token 1
  noise_pred[0, 64, :8]      # mid token
  noise_pred[0, 127, :8]     # last token
  per-channel sums over all tokens (to detect gross-mass divergence)

Run from intabai/web/scripts:
  uv run python ltx/reference_transformer.py \
    --space-path C:/work/personal/intabai/notes/ltx-video-distilled-space \
    --ckpt C:/work/personal/intabai/notes/models/ltx/source/ltxv-2b-0.9.8-distilled.safetensors
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


def build_indices_grid(latent_t, latent_h, latent_w, fps=30):
    """Match JS patchifier.buildIndicesGrid: pixel-space coordinates,
    scaled by (8, 32, 32), with causal-fix on temporal and /fps."""
    n = latent_t * latent_h * latent_w
    grid = torch.zeros(1, 3, n, dtype=torch.float32)
    idx = 0
    for t in range(latent_t):
        for h in range(latent_h):
            for w in range(latent_w):
                # Temporal: pixel_t = max(latent_t * 8 - 7, 0), then /fps.
                pix_t = max(t * 8 - 7, 0)
                grid[0, 0, idx] = pix_t / fps
                grid[0, 1, idx] = h * 32
                grid[0, 2, idx] = w * 32
                idx += 1
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--latent-t", type=int, default=4)
    ap.add_argument("--latent-h", type=int, default=8)
    ap.add_argument("--latent-w", type=int, default=4)
    ap.add_argument("--n-text", type=int, default=256)
    ap.add_argument("--ram-budget-gb", type=float, default=14.0)
    args = ap.parse_args()

    ram_peak = start_ram_watchdog(args.ram_budget_gb)
    print(f"RAM budget: {args.ram_budget_gb:.1f} GB")

    sys.path.insert(0, str(args.space_path))
    from ltx_video.models.transformers.transformer3d import Transformer3DModel

    print(f"Loading transformer from {args.ckpt}")
    t0 = time.time()
    tx = Transformer3DModel.from_pretrained(str(args.ckpt))
    tx = tx.to(torch.float16).eval()
    for p in tx.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  num blocks: {len(tx.transformer_blocks)}")
    print(f"  hidden size: {tx.inner_dim}")
    print(f"  use_rope: {getattr(tx, 'use_rope', None)}")
    print(f"  positional_embedding_max_pos: {getattr(tx, 'positional_embedding_max_pos', None)}")
    print(f"  timestep_scale_multiplier: {getattr(tx, 'timestep_scale_multiplier', None)}")

    n_tokens = args.latent_t * args.latent_h * args.latent_w
    print(f"  n_tokens={n_tokens}")

    hidden = torch.ones(1, n_tokens, 128, dtype=torch.float16)
    enc_hidden = torch.zeros(1, args.n_text, 4096, dtype=torch.float16)
    enc_mask = torch.ones(1, args.n_text, dtype=torch.int64)
    timestep = torch.ones(1, dtype=torch.float16)
    indices_grid = build_indices_grid(args.latent_t, args.latent_h, args.latent_w)

    print(f"  hidden: shape={tuple(hidden.shape)} mean={hidden.float().mean():.4f}")
    print(f"  indices_grid[0, :, :4]={indices_grid[0, :, :4].tolist()}")
    print(f"  indices_grid[0, :, -4:]={indices_grid[0, :, -4:].tolist()}")

    print("Running transformer.forward (canonical input)...")
    t0 = time.time()
    with torch.no_grad():
        out = tx(
            hidden_states=hidden,
            indices_grid=indices_grid,
            encoder_hidden_states=enc_hidden,
            timestep=timestep,
            encoder_attention_mask=enc_mask,
            return_dict=True,
        )
    np_pred = out.sample.float()
    print(f"  forward done in {time.time() - t0:.1f}s")
    print(f"  noise_pred shape={tuple(np_pred.shape)}")
    print(
        f"  np overall: mean={np_pred.mean():.5f} std={np_pred.std():.5f} "
        f"min={np_pred.min():.5f} max={np_pred.max():.5f}"
    )

    # Sample tokens.
    for tok_idx in [0, 1, 64, n_tokens // 2, n_tokens - 1]:
        if tok_idx >= n_tokens:
            continue
        vals = np_pred[0, tok_idx, :8].tolist()
        s = sum(np_pred[0, tok_idx, :8].tolist())
        print(f"  token{tok_idx:>4d} ch0..7=[{', '.join(f'{v:+.5f}' for v in vals)}] sum8={s:+.5f}")

    # Per-channel sums over all tokens (first 8 channels).
    sums = np_pred[0].sum(dim=0)[:8].tolist()
    print(f"  per-channel sums (over {n_tokens} tokens) ch0..7=[{', '.join(f'{v:+.4f}' for v in sums)}]")

    print(f"Peak RAM: {ram_peak[0]:.2f} GB")


if __name__ == "__main__":
    main()
