#!/usr/bin/env python3
"""Export AutoencoderKLWan decoder as two single-frame ONNX graphs that share
a cache-tensor I/O contract. Trace RAM per export stays at single-frame scale.

The diffusers implementation runs a Python `for i in range(num_frame)` loop,
calling `self.decoder(x[:, :, i:i+1])` with a `feat_cache` list shared across
iterations. Each causal conv reads + writes its slot in that list to emulate
temporal context. Tracing the whole loop unrolls it and blows up peak RAM
(~8 GB for T=3, linear in T → ~56 GB for T=21 which is what we need).

Instead we thread the cache as ONNX I/O tensors and export two graphs:

  vae/decoder_init.onnx   in: latent[1,48,1,H,W]
                          out: frames[1,3,1,H*16,W*16] + N cache tensors

  vae/decoder_step.onnx   in: latent[1,48,1,H,W] + N cache tensors
                          out: frames[1,3,4,H*16,W*16] + N updated cache tensors

Frame 0 produces 1 output frame (no temporal upsample happens yet because
caches are zero-context). Frame 1..20 each produce 4 output frames. Total
output: 1 + 20*4 = 81 frames. Matches the original `_decode`'s 21 → 81.

Weight sharing: both ONNX graphs are exported with the same weights baked in
(or via external-data). For now we accept 2x weight disk cost (~2.8 GB
total). Deduplication to a shared .data sidecar is a post-export cleanup if
we ever need it.

Usage:
    uv run export-fastwan-vae-kl-streaming.py <source_dir> <output_dir>
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class WanCausalConv2dDecomposed(nn.Module):
    """Drop-in replacement for WanCausalConv3d. Preserves the
    `forward(x, cache_x=None)` signature and the cache-concat + causal-pad
    behavior exactly. Replaces the inner Conv3D with:
        Conv3d(X, W)[:,:,t] = sum_{kt=0..kT-1} Conv2d(X[:,:,t+kt], W[:,:,kt])

    After F.pad applies both spatial and causal-time padding, the time axis
    of x is already kT-aligned per output frame, so we just slice + Conv2D.

    ONNX cost per call (T_out=1 typical):
        kT=3:  3 Slice + 3 Conv2D + 3 Add + 1 Unsqueeze
        kT=1:  1 Slice + 1 Conv2D + 1 Add (bias) + 1 Unsqueeze
    """

    def __init__(self, causal_conv3d: nn.Conv3d):
        super().__init__()
        kT, _, _ = causal_conv3d.kernel_size
        self.kT = kT
        # 6-tuple (W_l, W_r, H_l, H_r, T_l, T_r) in F.pad order. Time is
        # causal: T_l = 2*padding_time, T_r = 0. Populated by WanCausalConv3d
        # __init__ and overwritten there each instance; store a fresh tuple.
        self._padding = tuple(causal_conv3d._padding)
        self.stride = causal_conv3d.stride[1:]
        self.dilation = causal_conv3d.dilation[1:]
        self.groups = causal_conv3d.groups
        # Slice the kT axis. `contiguous()` copies each slice once. Net RAM
        # after the caller's setattr releases `causal_conv3d`: identical to
        # the original 5D weight (kT * [Cout,Cin,kH,kW] == [Cout,Cin,kT,kH,kW]).
        self.weight_slices = nn.ParameterList([
            nn.Parameter(causal_conv3d.weight.data[:, :, t].contiguous(),
                         requires_grad=False)
            for t in range(kT)
        ])
        if causal_conv3d.bias is not None:
            self.bias = nn.Parameter(causal_conv3d.bias.data.clone(),
                                     requires_grad=False)
        else:
            self.bias = None

    def forward(self, x, cache_x=None):
        # Match WanCausalConv3d.forward exactly for cache handling.
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        # x now has spatial+time padding already applied. Conv2D uses (0,0).
        T_in = x.shape[2]
        T_out = T_in - self.kT + 1
        outs = []
        for t_out in range(T_out):
            acc = None
            for kt in range(self.kT):
                s = x[:, :, t_out + kt]
                y = F.conv2d(s, self.weight_slices[kt], None,
                             self.stride, (0, 0), self.dilation, self.groups)
                acc = y if acc is None else acc + y
            if self.bias is not None:
                acc = acc + self.bias.view(1, -1, 1, 1)
            outs.append(acc.unsqueeze(2))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)


def decompose_conv3d(module: nn.Module) -> int:
    """Recursively replace every WanCausalConv3d with WanCausalConv2dDecomposed.
    Returns count of replaced modules. Uses late import to avoid a hard
    diffusers version dep at module load."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, WanCausalConv3d):
            setattr(module, name, WanCausalConv2dDecomposed(child))
            count += 1
        else:
            count += decompose_conv3d(child)
    return count


def _verify_decomp_parity():
    """Standalone parity: real WanCausalConv3d vs WanCausalConv2dDecomposed on
    random small input. Covers cache_x=None and cache_x=tensor paths. ~KB RAM.
    Raises on mismatch."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    torch.manual_seed(0)
    # (kT, padding_time, bias, use_cache)
    # (kT, padding_time, bias, use_cache, T_in)
    # For pT=0 kT=3 we need T_in >= 3 since no time-pad is applied.
    cases = [
        (3, 1, True,  False, 2),
        (3, 1, True,  True,  2),
        (3, 1, False, False, 2),
        (1, 0, True,  False, 2),
        (3, 0, True,  True,  3),
    ]
    for kT, pT, bias, use_cache, T_in in cases:
        conv = WanCausalConv3d(4, 6, kernel_size=(kT, 3, 3),
                               padding=(pT, 1, 1)).eval()
        if not bias:
            conv.bias = None
        wrap = WanCausalConv2dDecomposed(conv).eval()
        x = torch.randn(1, 4, T_in, 5, 7)
        cache_x = torch.randn(1, 4, 1, 5, 7) if use_cache else None
        with torch.no_grad():
            a = conv(x if cache_x is None else x.clone(),
                     cache_x.clone() if cache_x is not None else None)
            b = wrap(x if cache_x is None else x.clone(),
                     cache_x.clone() if cache_x is not None else None)
        assert a.shape == b.shape, (
            f"shape mismatch kT={kT} pT={pT} bias={bias} cache={use_cache}: "
            f"{a.shape} vs {b.shape}"
        )
        diff = (a - b).abs().max().item()
        assert diff < 1e-5, (
            f"parity fail kT={kT} pT={pT} bias={bias} cache={use_cache}: "
            f"max abs diff {diff}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source_dir", type=Path, help="diffusers VAE root (contains vae/)")
    parser.add_argument("output_dir", type=Path, help="output dir for .onnx files")
    parser.add_argument("--latent-height", type=int, default=30)
    parser.add_argument("--latent-width", type=int, default=52)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--dtype", choices=["fp16", "fp32"], default="fp16",
    )
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--skip-init", action="store_true", help="skip init graph export")
    parser.add_argument("--skip-step", action="store_true", help="skip step graph export")
    parser.add_argument(
        "--decompose-conv3d", action="store_true",
        help="Replace every nn.Conv3d with kT * Conv2d + time-sum before trace. "
             "Emits ONNX with zero Conv3D nodes — trades Conv3DNaive (slow on "
             "ORT-web WebGPU) for the mature Conv2D kernel.",
    )
    args = parser.parse_args()

    vae_dir = args.source_dir / "vae"
    if not (vae_dir / "config.json").exists():
        print(f"Error: {vae_dir / 'config.json'} not found", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae_out = args.output_dir / "vae"
    vae_out.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (args.output_dir / "export-vae-kl-streaming.log")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Export log: {log_path}")

    # ---- Load model ----
    from diffusers import AutoencoderKLWan

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    log(f"Loading VAE {vae_dir} dtype={dtype}...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype)
    vae.eval()
    vae.use_tiling = False
    vae.use_slicing = False
    # Patch nearest-exact -> nearest (unsupported in opset 18; equivalent for
    # integer scale=2).
    patched = 0
    for mod in vae.modules():
        if isinstance(mod, nn.Upsample) and mod.mode == "nearest-exact":
            mod.mode = "nearest"
            patched += 1
    log(f"  loaded in {time.time() - t0:.1f}s, patched {patched} Upsample modes")

    # We only need the decoder (and post_quant_conv). Drop the encoder to free
    # ~half the params.
    del vae.encoder
    del vae.quant_conv
    gc.collect()
    param_bytes = sum(p.numel() * p.element_size() for p in vae.parameters())
    log(f"  decoder-only param memory: {param_bytes / 1e9:.2f} GB")

    if args.decompose_conv3d:
        log("Verifying WanCausalConv3d -> Conv2D-sum parity on synthetic input...")
        _verify_decomp_parity()
        log("  parity OK (max abs diff < 1e-5 across kT=1/3, cache None/tensor, bias on/off)")
        log("Walking VAE, replacing WanCausalConv3d -> WanCausalConv2dDecomposed...")
        replaced = decompose_conv3d(vae)
        gc.collect()
        post_bytes = sum(p.numel() * p.element_size() for p in vae.parameters())
        log(f"  replaced {replaced} causal convs; param memory now {post_bytes / 1e9:.2f} GB")

    B, C, H, W = 1, 48, args.latent_height, args.latent_width

    # ---- PROBE: run 2 frames with original logic to discover cache shapes ----
    # The feat_cache slot count and per-slot tensor shape depends on the decoder
    # architecture + the input resolution. After frame 1 every slot is a real
    # tensor (frame-0 "Rep" sentinels have been overwritten). We snapshot those
    # shapes to build the ONNX I/O contract.
    log("Probing cache shapes (runs 2 frames on CPU, ~3 min)...")
    probe_latent = torch.zeros(B, C, 2, H, W, dtype=dtype)
    vae.clear_cache()
    with torch.no_grad():
        x = vae.post_quant_conv(probe_latent)
        for i in range(2):
            vae._conv_idx = [0]
            if i == 0:
                _ = vae.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=vae._feat_map,
                    feat_idx=vae._conv_idx,
                    first_chunk=True,
                )
            else:
                _ = vae.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=vae._feat_map,
                    feat_idx=vae._conv_idx,
                )

    # `_conv_num` counts every WanCausalConv3d module (incl. conv_shortcut
    # which doesn't use feat_idx), so trailing slots stay None. Truncate to
    # the actual feat_idx reached after the second frame.
    used_count = vae._conv_idx[0]
    cache_shapes: list[tuple[int, ...]] = []
    for i in range(used_count):
        slot = vae._feat_map[i]
        if not isinstance(slot, torch.Tensor):
            raise RuntimeError(
                f"cache slot {i} after frame 1 is not a tensor: {slot!r} "
                f"(type {type(slot).__name__})"
            )
        cache_shapes.append(tuple(slot.shape))
    log(f"  feat_idx reached {used_count} of {len(vae._feat_map)} allocated slots")
    log(f"  cache entries: {len(cache_shapes)}")
    total_cache_bytes = sum(
        int(torch.prod(torch.tensor(s)).item()) for s in cache_shapes
    ) * (2 if dtype == torch.float16 else 4)
    log(f"  cache total bytes: {total_cache_bytes / 1e6:.1f} MB")
    for i, s in enumerate(cache_shapes):
        log(f"    slot {i:02d}: {s}")

    vae.clear_cache()
    del probe_latent, x
    gc.collect()

    # ---- Build the two wrapper modules ----
    # Both share the same underlying `vae` — we trace them separately. The
    # wrappers convert the feat_cache Python list into explicit tensor I/O so
    # the ONNX graph surfaces cache state on its inputs/outputs.

    N = len(cache_shapes)

    class DecoderInit(nn.Module):
        """Traced with feat_cache=[None]*N. Captures frame-0 semantics.
        After this forward each of the N slots is EITHER a tensor (most
        convs) OR the string "Rep" (temporal-upsample time_convs). We
        replace "Rep" slots with zero tensors of the expected shape so
        every output is a tensor — the tracer emits those zeros as
        graph Constants."""

        def __init__(self, vae, cache_shapes):
            super().__init__()
            self.vae = vae
            self.cache_shapes = cache_shapes

        def forward(self, latent):
            x = self.vae.post_quant_conv(latent)
            feat_map: list = [None] * len(self.cache_shapes)
            conv_idx = [0]
            out = self.vae.decoder(
                x, feat_cache=feat_map, feat_idx=conv_idx, first_chunk=True
            )
            if self.vae.config.patch_size is not None:
                from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

                out = unpatchify(out, patch_size=self.vae.config.patch_size)
            out = torch.clamp(out, -1.0, 1.0)

            # Normalize cache outputs to the step graph's expected shapes.
            # - "Rep"/None slots -> zeros of cache_shapes[i].
            # - Real tensor slots with T < target_T (frame-0 emits T=1 for
            #   slots that will settle at T=2 after frame 1): zero-left-pad
            #   along time. This is numerically equivalent to what the step
            #   graph's internal causal-time-pad would have produced had it
            #   been fed a T<target cache, since WanCausalConv3d's pad adds
            #   zeros to the front of the time axis.
            cache_tensors: list[torch.Tensor] = []
            for i, slot in enumerate(feat_map):
                target_shape = self.cache_shapes[i]
                target_T = target_shape[2]
                if isinstance(slot, torch.Tensor):
                    if slot.shape[2] < target_T:
                        pad_T = target_T - slot.shape[2]
                        pad = torch.zeros(
                            slot.shape[0], slot.shape[1], pad_T,
                            slot.shape[3], slot.shape[4],
                            dtype=latent.dtype,
                        )
                        slot = torch.cat([pad, slot], dim=2)
                    cache_tensors.append(slot)
                else:
                    cache_tensors.append(
                        torch.zeros(target_shape, dtype=latent.dtype)
                    )
            return tuple([out, *cache_tensors])

    class DecoderStep(nn.Module):
        """Traced with feat_cache pre-populated with tensors. Captures
        frame-1+ semantics (uniform across all subsequent frames)."""

        def __init__(self, vae, N):
            super().__init__()
            self.vae = vae
            self.N = N

        def forward(self, latent, *caches):
            x = self.vae.post_quant_conv(latent)
            feat_map: list = list(caches)
            conv_idx = [0]
            out = self.vae.decoder(
                x, feat_cache=feat_map, feat_idx=conv_idx, first_chunk=False
            )
            if self.vae.config.patch_size is not None:
                from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

                out = unpatchify(out, patch_size=self.vae.config.patch_size)
            out = torch.clamp(out, -1.0, 1.0)
            return tuple([out, *feat_map])

    # Shared dummy inputs.
    dummy_latent_1f = torch.zeros(B, C, 1, H, W, dtype=dtype)
    cache_names = [f"cache_in_{i:02d}" for i in range(N)]
    cache_out_names = [f"cache_out_{i:02d}" for i in range(N)]

    # ---- Export INIT graph ----
    if not args.skip_init:
        log("Exporting init graph...")
        init_wrap = DecoderInit(vae, cache_shapes).eval()
        # Shape check on CPU to catch errors before the slow trace.
        t0 = time.time()
        with torch.no_grad():
            test_out = init_wrap(dummy_latent_1f)
        log(
            f"  init forward OK: frames {list(test_out[0].shape)}, "
            f"+ {N} cache tensors, took {time.time() - t0:.1f}s"
        )
        init_path = vae_out / "decoder_init.onnx"
        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(
                init_wrap,
                (dummy_latent_1f,),
                str(init_path),
                opset_version=args.opset,
                input_names=["latent"],
                output_names=["frames", *cache_out_names],
                dynamo=False,
            )
        log(f"  init export: {time.time() - t0:.1f}s, {init_path.stat().st_size / 1e6:.1f} MB")
        data_path = init_path.with_suffix(".onnx.data")
        if data_path.exists():
            log(f"  init data: {data_path.stat().st_size / 1e6:.1f} MB")
        del init_wrap, test_out
        gc.collect()

    # ---- Export STEP graph ----
    if not args.skip_step:
        log("Exporting step graph...")
        step_wrap = DecoderStep(vae, N).eval()
        # Pre-populated caches for the trace. Use random (not zeros) so any
        # dtype/shape mismatch fails loudly rather than getting masked by
        # zeros propagation.
        dummy_caches = tuple(
            torch.zeros(s, dtype=dtype) for s in cache_shapes
        )
        t0 = time.time()
        with torch.no_grad():
            test_out = step_wrap(dummy_latent_1f, *dummy_caches)
        log(
            f"  step forward OK: frames {list(test_out[0].shape)}, "
            f"+ {N} cache tensors, took {time.time() - t0:.1f}s"
        )
        step_path = vae_out / "decoder_step.onnx"
        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(
                step_wrap,
                (dummy_latent_1f, *dummy_caches),
                str(step_path),
                opset_version=args.opset,
                input_names=["latent", *cache_names],
                output_names=["frames", *cache_out_names],
                dynamo=False,
            )
        log(f"  step export: {time.time() - t0:.1f}s, {step_path.stat().st_size / 1e6:.1f} MB")
        data_path = step_path.with_suffix(".onnx.data")
        if data_path.exists():
            log(f"  step data: {data_path.stat().st_size / 1e6:.1f} MB")

    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
