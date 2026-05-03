#!/usr/bin/env python3
"""Verify the per-resnet split in convert_vae_layers.py is sound for this
checkpoint. The split iterates `blk.res_blocks` only; if attention_blocks
is set on any UNetMidBlock3D we'd silently drop the resnet<->attention
interleave and produce wrong output.

Run:
  cd intabai/web/scripts && uv run python ltx/check_vae_invariants.py \\
    --space-path C:/work/personal/intabai/notes/ltx-video-distilled-space \\
    --ckpt C:/work/personal/intabai/notes/models/ltx/source/ltxv-2b-0.9.8-distilled.safetensors
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.space_path))
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
        UNetMidBlock3D,
    )

    print(f"Loading VAE from {args.ckpt}")
    vae = CausalVideoAutoencoder.from_pretrained(str(args.ckpt))
    vae.eval()

    enc = vae.encoder
    dec = vae.decoder

    fails = []

    def check(label, cond, detail=""):
        ok = "OK" if cond else "FAIL"
        line = f"  [{ok}] {label}"
        if detail:
            line += f"  -- {detail}"
        print(line)
        if not cond:
            fails.append(label)

    print("Top-level config:")
    check("use_quant_conv == False", vae.use_quant_conv is False, str(vae.use_quant_conv))
    check(
        "normalize_latent_channels == False",
        vae.normalize_latent_channels is False,
        str(vae.normalize_latent_channels),
    )
    check(
        "encoder.latent_log_var == 'uniform'",
        enc.latent_log_var == "uniform",
        enc.latent_log_var,
    )
    check("encoder.patch_size == 4", enc.patch_size == 4, str(enc.patch_size))
    check("decoder.patch_size == 4", dec.patch_size == 4, str(dec.patch_size))
    check(
        "decoder.timestep_conditioning == True",
        dec.timestep_conditioning is True,
        str(dec.timestep_conditioning),
    )

    print("UNetMidBlock3D attention_blocks must be None for the per-resnet split:")
    for i, blk in enumerate(enc.down_blocks):
        if isinstance(blk, UNetMidBlock3D):
            attn = getattr(blk, "attention_blocks", None)
            check(
                f"enc.down_blocks[{i}].attention_blocks is None",
                attn is None,
                f"len={len(attn) if attn is not None else 0}",
            )
    for i, blk in enumerate(dec.up_blocks):
        if isinstance(blk, UNetMidBlock3D):
            attn = getattr(blk, "attention_blocks", None)
            check(
                f"dec.up_blocks[{i}].attention_blocks is None",
                attn is None,
                f"len={len(attn) if attn is not None else 0}",
            )

    print("Counts (must match what the converter assumes):")
    e8 = enc.down_blocks[8]
    check(
        "enc.down_blocks[8] is UNetMidBlock3D",
        isinstance(e8, UNetMidBlock3D),
        type(e8).__name__,
    )
    check(
        "enc.down_blocks[8].res_blocks == 2",
        len(e8.res_blocks) == 2,
        str(len(e8.res_blocks)),
    )
    d0 = dec.up_blocks[0]
    check(
        "dec.up_blocks[0] is UNetMidBlock3D",
        isinstance(d0, UNetMidBlock3D),
        type(d0).__name__,
    )
    check(
        "dec.up_blocks[0].res_blocks == 5",
        len(d0.res_blocks) == 5,
        str(len(d0.res_blocks)),
    )

    print("Pipeline-level normalization tensors present on vae:")
    check(
        "vae.mean_of_means present",
        hasattr(vae, "mean_of_means"),
        "shape="
        + str(tuple(vae.mean_of_means.shape))
        if hasattr(vae, "mean_of_means")
        else "absent",
    )
    check(
        "vae.std_of_means present",
        hasattr(vae, "std_of_means"),
        "shape="
        + str(tuple(vae.std_of_means.shape))
        if hasattr(vae, "std_of_means")
        else "absent",
    )

    if fails:
        print(f"\n{len(fails)} FAIL(s):")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print("\nAll invariants hold. Existing exports are sound.")


if __name__ == "__main__":
    main()
