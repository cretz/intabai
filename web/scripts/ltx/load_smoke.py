#!/usr/bin/env python3
"""Load LTX 2B distilled transformer + VAE on CPU and run a tiny forward
to confirm weights wire up.

Run via the shared web/scripts pyproject:

    cd intabai/web/scripts && uv run python ltx/load_smoke.py \
        --space-path /abs/path/to/ltx-video-distilled-space \
        --ckpt /abs/path/to/ltxv-2b-0.9.8-distilled.safetensors
"""
import argparse
import sys
from pathlib import Path

import torch


def smoke_transformer(space_path: Path, ckpt: Path) -> None:
    sys.path.insert(0, str(space_path))
    from ltx_video.models.transformers.transformer3d import Transformer3DModel

    print(f"Loading transformer (bf16) from {ckpt} ...")
    transformer = Transformer3DModel.from_pretrained(str(ckpt), torch_dtype=torch.bfloat16)
    transformer.eval()
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"  transformer: {n_params/1e9:.2f}B params, dtype={next(transformer.parameters()).dtype}")
    print(f"  num_layers={len(transformer.transformer_blocks)}")
    print(f"  inner_dim={transformer.inner_dim}, heads={transformer.num_attention_heads}")
    print(f"  positional_embedding_type={transformer.positional_embedding_type}")
    print(f"  positional_embedding_max_pos={transformer.positional_embedding_max_pos}")
    print(f"  positional_embedding_theta={transformer.positional_embedding_theta}")

    print("Transformer smoke forward (1x128x1x4x4, caption 8 tokens) ...")
    with torch.no_grad():
        latent = torch.randn(1, 128, 1, 4, 4, dtype=torch.bfloat16)
        encoder_hidden_states = torch.randn(1, 8, 4096, dtype=torch.bfloat16)
        encoder_attention_mask = torch.ones(1, 8, dtype=torch.long)
        timestep = torch.tensor([500.0], dtype=torch.bfloat16)
        out = transformer(
            hidden_states=latent,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            return_dict=False,
        )[0]
        print(f"  output: {tuple(out.shape)} dtype={out.dtype}")


def smoke_vae(space_path: Path, ckpt: Path) -> None:
    sys.path.insert(0, str(space_path))
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )

    print(f"Loading VAE (bf16) from {ckpt} ...")
    vae = CausalVideoAutoencoder.from_pretrained(str(ckpt), torch_dtype=torch.bfloat16)
    vae.eval()
    n_vae = sum(p.numel() for p in vae.parameters())
    print(f"  vae: {n_vae/1e6:.1f}M params")
    print(f"  spatial_downsample={vae.spatial_downscale_factor}")
    print(f"  temporal_downsample={vae.temporal_downscale_factor}")

    print("VAE encode 1x3x1x64x64, decode round-trip ...")
    with torch.no_grad():
        x = torch.randn(1, 3, 1, 64, 64, dtype=torch.bfloat16)
        latents = vae.encode(x).latent_dist.sample()
        print(f"  encoded latents: {tuple(latents.shape)}")
        decoded = vae.decode(latents, timestep=torch.tensor([0.05], dtype=torch.bfloat16)).sample
        print(f"  decoded: {tuple(decoded.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--what", choices=["transformer", "vae", "both"], default="transformer")
    args = ap.parse_args()

    if args.what in ("transformer", "both"):
        smoke_transformer(args.space_path, args.ckpt)
    if args.what in ("vae", "both"):
        smoke_vae(args.space_path, args.ckpt)
    print("OK")


if __name__ == "__main__":
    main()
