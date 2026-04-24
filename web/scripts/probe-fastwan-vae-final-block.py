#!/usr/bin/env python3
"""Export just the FINAL up_block of the AutoencoderKLWan decoder, at full
spatial (240x416). This is the worst-case Conv3D workload in the VAE.

Purpose: if the full decoder_init/step TDRs on WebGPU even with
graphOptLevel=disabled, we need to know whether per-block chunking
(splitting the decoder into separate ONNX graphs) would help, or whether
a single Conv3D dispatch on 240x416x256ch is the real problem. This probe
isolates up_blocks[-1] so the smoke harness can answer that directly.

RAM discipline: one forward pass on CPU, with a hook that aborts right
after up_blocks[-1] runs. After the probe we drop everything except the
final up_block before tracing.

Usage:
    uv run probe-fastwan-vae-final-block.py <source_dir> <output_dir>
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


class _AbortForward(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="diffusers VAE root (contains vae/)")
    parser.add_argument("output_dir", type=Path, help="output dir (vae/probe_final_block.onnx)")
    parser.add_argument("--latent-height", type=int, default=30)
    parser.add_argument("--latent-width", type=int, default=52)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    vae_dir = args.source_dir / "vae"
    if not (vae_dir / "config.json").exists():
        print(f"Error: {vae_dir / 'config.json'} not found", file=sys.stderr)
        sys.exit(1)

    vae_out = args.output_dir / "vae"
    vae_out.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (args.output_dir / "probe-final-block.log")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Log: {log_path}")

    from diffusers import AutoencoderKLWan

    dtype = torch.float16
    log(f"Loading VAE {vae_dir} dtype={dtype}...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype)
    vae.eval()
    vae.use_tiling = False
    vae.use_slicing = False
    for mod in vae.modules():
        if isinstance(mod, nn.Upsample) and mod.mode == "nearest-exact":
            mod.mode = "nearest"
    log(f"  loaded in {time.time() - t0:.1f}s")

    del vae.encoder
    del vae.quant_conv
    gc.collect()

    B, C, H, W = 1, 48, args.latent_height, args.latent_width

    # Hook: capture (x, feat_idx snapshot) right before up_blocks[-1] runs,
    # and (feat_idx snapshot) right after. Then raise to abort the forward.
    probe: dict = {}
    final_block = vae.decoder.up_blocks[-1]

    def pre_hook(_mod, inputs, kwargs):
        x = inputs[0] if inputs else kwargs["x"]
        feat_idx = kwargs.get("feat_idx")
        probe["input_shape"] = tuple(x.shape)
        probe["input_dtype"] = x.dtype
        probe["feat_idx_before"] = int(feat_idx[0])

    def post_hook(_mod, inputs, kwargs, output):
        feat_idx = kwargs.get("feat_idx")
        probe["feat_idx_after"] = int(feat_idx[0])
        feat_cache = kwargs["feat_cache"]
        # Snapshot the cache slots this block just wrote.
        slot_shapes = []
        for i in range(probe["feat_idx_before"], probe["feat_idx_after"]):
            slot = feat_cache[i]
            if isinstance(slot, torch.Tensor):
                slot_shapes.append(tuple(slot.shape))
            else:
                slot_shapes.append(None)
        probe["slot_shapes"] = slot_shapes
        probe["output_shape"] = tuple(output.shape)
        raise _AbortForward()

    h1 = final_block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = final_block.register_forward_hook(post_hook, with_kwargs=True)

    log("Running one CPU forward pass (aborts at final up_block)...")
    t0 = time.time()
    probe_latent = torch.zeros(B, C, 1, H, W, dtype=dtype)
    vae.clear_cache()
    try:
        with torch.no_grad():
            x = vae.post_quant_conv(probe_latent)
            _ = vae.decoder(
                x,
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=True,
            )
    except _AbortForward:
        pass
    finally:
        h1.remove()
        h2.remove()
    log(f"  done in {time.time() - t0:.1f}s")

    log(f"  final up_block input shape: {probe['input_shape']} dtype={probe['input_dtype']}")
    log(f"  feat_idx range: {probe['feat_idx_before']} .. {probe['feat_idx_after']}")
    log(f"  slot count: {len(probe['slot_shapes'])}")
    for i, s in enumerate(probe["slot_shapes"]):
        log(f"    slot {probe['feat_idx_before'] + i:02d}: {s}")
    log(f"  final up_block output shape: {probe['output_shape']}")

    # Free everything except the final block + its cache metadata.
    del probe_latent, x, vae.post_quant_conv
    del vae.decoder.conv_in, vae.decoder.mid_block
    del vae.decoder.norm_out, vae.decoder.conv_out
    # Drop up_blocks[0..-2].
    keep = vae.decoder.up_blocks[-1]
    vae.decoder.up_blocks = nn.ModuleList([keep])
    gc.collect()

    input_shape = probe["input_shape"]
    slot_shapes = probe["slot_shapes"]
    for s in slot_shapes:
        if s is None:
            raise RuntimeError("final up_block had a non-tensor cache slot; unexpected")
    N = len(slot_shapes)

    class FinalBlockWrapper(nn.Module):
        """Wraps up_blocks[-1] with explicit cache-tensor I/O. Step-mode
        (pre-populated cache tensors, not 'Rep'), so the trace captures
        the worst-case Conv3D workload in isolation."""

        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x, *caches):
            feat_map: list = list(caches)
            conv_idx = [0]
            out = self.block(x, feat_cache=feat_map, feat_idx=conv_idx, first_chunk=False)
            return tuple([out, *feat_map])

    wrap = FinalBlockWrapper(keep).eval()
    dummy_x = torch.zeros(input_shape, dtype=dtype)
    dummy_caches = tuple(torch.zeros(s, dtype=dtype) for s in slot_shapes)

    log("Forward smoke-check on wrapper...")
    t0 = time.time()
    with torch.no_grad():
        test_out = wrap(dummy_x, *dummy_caches)
    log(
        f"  forward OK: frames_like {list(test_out[0].shape)} + {N} caches, "
        f"took {time.time() - t0:.1f}s"
    )

    cache_in_names = [f"cache_in_{probe['feat_idx_before'] + i:02d}" for i in range(N)]
    cache_out_names = [f"cache_out_{probe['feat_idx_before'] + i:02d}" for i in range(N)]

    out_path = vae_out / "probe_final_block.onnx"
    log(f"Exporting to {out_path}...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrap,
            (dummy_x, *dummy_caches),
            str(out_path),
            opset_version=args.opset,
            input_names=["x", *cache_in_names],
            output_names=["y", *cache_out_names],
            dynamo=False,
        )
    log(f"  export: {time.time() - t0:.1f}s, {out_path.stat().st_size / 1e6:.1f} MB")
    data_path = out_path.with_suffix(".onnx.data")
    if data_path.exists():
        log(f"  data: {data_path.stat().st_size / 1e6:.1f} MB")

    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
