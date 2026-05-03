#!/usr/bin/env python3
"""Export LTX 2B distilled VAE per-block for browser layer-streaming.

Encoder and decoder are split as:

  enc_shell_pre   : patchify(patch_size=4) + conv_in   (3 -> 128 ch)
  enc_block_NN    : one entry of vae.encoder.down_blocks (10 total)
  enc_shell_post  : conv_norm_out (PixelNorm) + SiLU + conv_out + uniform-logvar
                    repeat (concat last channel 126x to make 256-ch output)

  dec_shell_pre   : conv_in (128 -> 512 ch)
  dec_block_NN    : one entry of vae.decoder.up_blocks (10 total)
  dec_shell_post  : conv_norm_out + SiLU + conv_out + unpatchify(patch_size=4)

I/O is fp16. Spatial/temporal axes (T/H/W) are dynamic via torch.export.Dim
so a single set of weights serves any user-selected duration/resolution.

The browser side runs encoder one block at a time, then JS splits the
256-ch moments into mean+logvar (logvar is the same channel repeated 128x
per the uniform setting), samples mean + exp(0.5*logvar) * randn, and
streams the result through the decoder per-block.

Run:
  cd intabai/web/scripts && uv run python ltx/convert_vae_layers.py \\
    --space-path C:/work/personal/intabai/notes/ltx-video-distilled-space \\
    --ckpt C:/work/personal/intabai/notes/models/ltx/source/ltxv-2b-0.9.8-distilled.safetensors \\
    --out-dir C:/work/personal/intabai/notes/models/ltx/staging/vae

The 2B distilled config has timestep_conditioning=False, so neither encoder
nor decoder takes a timestep input. The pipeline-level decode_noise_scale
(0.025) is applied JS-side to the latent before decode.
"""
import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

# torch.onnx dynamo prints unicode (✅/❌); cp1252 on Windows will crash mid-export.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import psutil
import torch
import torch.nn as nn
from einops import rearrange
from torch.export import Dim


PATCH_SIZE_HW = 4  # OURS_VAE_CONFIG.patch_size
LATENT_CHANNELS = 128
ENC_OUT_CHANNELS = 256  # 128 mean + 128 logvar (uniform-repeated)

# Down/up_blocks that exceed the mobile WebGPU maxBufferSize as monolithic
# files. Split per-ResnetBlock3D to keep each artifact under ~250 MB fp16.
SPLIT_ENC_BLOCKS = {8}   # 4-resnet UNetMid @ 2048ch (~906 MB intact)
SPLIT_DEC_BLOCKS = {0}   # 4-resnet UNetMid @ 1024ch (~602 MB intact)

# Decoder uses timestep-conditioned adaln in UNetMidBlock3Ds and in shell_post
# (last_time_embedder + last_scale_shift_table). decode_timestep=0.05 from the
# pipeline yaml is the default at inference; the JS side feeds it in.


def start_ram_watchdog(max_gb: float, log_fn):
    proc = psutil.Process(os.getpid())
    peak = [0.0]

    def tick():
        while True:
            rss = proc.memory_info().rss / 1e9
            if rss > peak[0]:
                peak[0] = rss
            if rss > max_gb:
                log_fn(f"!!! RAM {rss:.2f} GB > {max_gb:.1f} GB, KILLING")
                os._exit(2)
            time.sleep(0.5)

    threading.Thread(target=tick, daemon=True).start()
    return peak


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------


class EncShellPre(nn.Module):
    """patchify(patch=4) + conv_in. Input (1,3,T,H,W), output (1,128,T,H/4,W/4)."""

    def __init__(self, encoder):
        super().__init__()
        self.conv_in = encoder.conv_in

    def forward(self, sample):
        # rearrange: b c (f 1) (h 4) (w 4) -> b (c*16) f h w
        sample = rearrange(
            sample,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=1, q=PATCH_SIZE_HW, r=PATCH_SIZE_HW,
        )
        return self.conv_in(sample)


class EncDownWrapper(nn.Module):
    """One encoder.down_blocks[i]. Pass-through; conv layers default causal=True."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, sample):
        return self.block(sample)


class EncResnetWrapper(nn.Module):
    """One ResnetBlock3D out of an encoder UNetMidBlock3D. Encoder has no
    timestep_conditioning so we pass timestep=None."""

    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, hidden):
        return self.resnet(hidden, causal=True, timestep=None)


class EncShellPost(nn.Module):
    """conv_norm_out (PixelNorm) + SiLU + conv_out -> 129ch -> uniform repeat to 256ch."""

    def __init__(self, encoder):
        super().__init__()
        self.norm = encoder.conv_norm_out
        self.act = encoder.conv_act
        self.conv_out = encoder.conv_out

    def forward(self, sample):
        sample = self.norm(sample)
        sample = self.act(sample)
        sample = self.conv_out(sample)
        # sample is (B, 129, F, H, W). Last channel = log-variance (broadcast).
        # Repeat last channel to total 256 channels: 128 mean + 128 logvar.
        last = sample[:, -1:, :, :, :]
        # repeat count is fixed at export (sample.shape[1] - 2 = 127, plus the
        # original 129 -> 256). Use 127 explicitly to keep the graph static.
        repeated = last.expand(-1, 127, -1, -1, -1)
        return torch.cat([sample, repeated], dim=1)


# ---------------------------------------------------------------------------
# Decoder wrappers
# ---------------------------------------------------------------------------


class DecShellPre(nn.Module):
    """decoder.conv_in + scale the timestep by timestep_scale_multiplier.

    Inputs:  latent (1,128,T,H,W) fp16, timestep (1,) fp16 in [0,1]
    Outputs: hidden (1, base*4, T, H, W) fp16, scaled_timestep (1,) fp32
    """

    def __init__(self, decoder):
        super().__init__()
        self.conv_in = decoder.conv_in
        self.causal = decoder.causal
        # timestep_scale_multiplier is a fp32 scalar Parameter (init 1000.0).
        self.timestep_scale_multiplier = decoder.timestep_scale_multiplier

    def forward(self, latent, timestep):
        hidden = self.conv_in(latent, causal=self.causal)
        scaled_timestep = timestep.float() * self.timestep_scale_multiplier
        return hidden, scaled_timestep


class DecMidBlockWrapper(nn.Module):
    """A UNetMidBlock3D up_block. Takes (hidden, scaled_timestep)."""

    def __init__(self, block, causal):
        super().__init__()
        self.block = block
        self.causal = causal

    def forward(self, hidden, scaled_timestep):
        return self.block(hidden, causal=self.causal, timestep=scaled_timestep)


class DecPlainBlockWrapper(nn.Module):
    """A non-mid up_block (ResnetBlock3D channel halver, DepthToSpaceUpsample)."""

    def __init__(self, block, causal):
        super().__init__()
        self.block = block
        self.causal = causal

    def forward(self, hidden):
        return self.block(hidden, causal=self.causal)


class DecMidPre(nn.Module):
    """Run a decoder UNetMidBlock3D's time_embedder so its res_blocks can
    each be exported as their own sub-file. Outputs the embedded timestep
    (shape (B, in_channels*4, 1, 1, 1)) for each per-resnet sub-block to
    consume."""

    def __init__(self, mid_block):
        super().__init__()
        self.time_embedder = mid_block.time_embedder

    def forward(self, hidden, scaled_timestep):
        batch_size = hidden.shape[0]
        ts_embed = self.time_embedder(
            timestep=scaled_timestep.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=batch_size,
            hidden_dtype=hidden.dtype,
        )
        ts_embed = ts_embed.view(batch_size, ts_embed.shape[-1], 1, 1, 1)
        return ts_embed


class DecResnetWrapper(nn.Module):
    """One ResnetBlock3D out of a decoder UNetMidBlock3D, with the time
    embedding precomputed by DecMidPre."""

    def __init__(self, resnet, causal):
        super().__init__()
        self.resnet = resnet
        self.causal = causal

    def forward(self, hidden, ts_embed):
        return self.resnet(hidden, causal=self.causal, timestep=ts_embed)


class DecShellPost(nn.Module):
    """norm_out + adaln (last_time_embedder + last_scale_shift_table) + SiLU
    + conv_out + unpatchify.

    Inputs:  hidden (1, C, T, H, W) fp16, scaled_timestep (1,) fp32
    Output:  pixels (1, 3, T*1, H*4, W*4) fp16
    """

    def __init__(self, decoder):
        super().__init__()
        self.norm = decoder.conv_norm_out
        self.act = decoder.conv_act
        self.conv_out = decoder.conv_out
        self.causal = decoder.causal
        self.last_time_embedder = decoder.last_time_embedder
        self.last_scale_shift_table = decoder.last_scale_shift_table

    def forward(self, sample, scaled_timestep):
        batch_size = sample.shape[0]
        sample = self.norm(sample)

        embedded = self.last_time_embedder(
            timestep=scaled_timestep.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=batch_size,
            hidden_dtype=sample.dtype,
        )
        embedded = embedded.view(batch_size, embedded.shape[-1], 1, 1, 1)
        ada_values = self.last_scale_shift_table[None, ..., None, None, None] + embedded.reshape(
            batch_size, 2, -1, embedded.shape[-3], embedded.shape[-2], embedded.shape[-1]
        )
        shift, scale = ada_values.unbind(dim=1)
        sample = sample * (1 + scale) + shift

        sample = self.act(sample)
        sample = self.conv_out(sample, causal=self.causal)
        sample = rearrange(
            sample,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=1, q=PATCH_SIZE_HW, r=PATCH_SIZE_HW,
        )
        return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def export_module(
    module: nn.Module,
    sample_inputs: tuple,
    out_path: Path,
    input_names: list,
    output_names: list,
    dynamic_shapes: tuple,
    opset: int,
    log,
):
    """dynamic_shapes is a tuple/list aligned 1:1 with sample_inputs (positional
    form). Each entry is either {} (all static) or a dict mapping axis index
    to a torch.export.Dim."""
    t0 = time.time()
    torch.onnx.export(
        module,
        sample_inputs,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=True,
    )
    graph_size = out_path.stat().st_size
    data_path = out_path.with_suffix(out_path.suffix + ".data")
    data_size = data_path.stat().st_size if data_path.exists() else 0
    total_mb = (graph_size + data_size) / 1e6
    log(f"  {out_path.name} -> {total_mb:.1f} MB ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ram-budget-gb", type=float, default=14.0)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument(
        "--smoke-frames", type=int, default=9,
        help="Pixel-space frame count for tracing (must be 8k+1)",
    )
    ap.add_argument("--smoke-hw", type=int, default=256, help="Pixel H=W for tracing")
    ap.add_argument(
        "--only", type=str, default="all",
        choices=["all", "encoder", "decoder", "shells"],
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"RAM budget: {args.ram_budget_gb:.1f} GB")
    ram_peak = start_ram_watchdog(args.ram_budget_gb, log)

    sys.path.insert(0, str(args.space_path))
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
        UNetMidBlock3D,
    )

    log(f"Loading VAE from {args.ckpt}")
    vae = CausalVideoAutoencoder.from_pretrained(str(args.ckpt))
    vae = vae.to(torch.float16).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    encoder = vae.encoder
    decoder = vae.decoder
    log(
        f"  spatial_downsample={vae.spatial_downscale_factor} "
        f"temporal_downsample={vae.temporal_downscale_factor}"
    )
    log(f"  down_blocks={len(encoder.down_blocks)} up_blocks={len(decoder.up_blocks)}")

    # ----- capture intermediate shapes by running encoder + decoder eagerly
    F_px, HW_px = args.smoke_frames, args.smoke_hw
    pixel_shape = (1, 3, F_px, HW_px, HW_px)
    log(f"Tracing eager forward at pixel shape {pixel_shape}")

    enc_block_inputs: list[torch.Tensor] = []
    with torch.no_grad():
        x = torch.randn(*pixel_shape, dtype=torch.float16)
        # patchify + conv_in
        x_post_patch = rearrange(
            x, "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=1, q=PATCH_SIZE_HW, r=PATCH_SIZE_HW,
        )
        h_enc = encoder.conv_in(x_post_patch)
        log(f"  after enc_shell_pre: {tuple(h_enc.shape)}")
        for i, blk in enumerate(encoder.down_blocks):
            enc_block_inputs.append(h_enc.clone())
            h_enc = blk(h_enc)
            log(f"  after enc_block_{i:02d}: {tuple(h_enc.shape)}")
        enc_post_input_shape = tuple(h_enc.shape)
        # encoder shell_post: norm + act + conv_out + uniform repeat -> 256ch
        h_enc = encoder.conv_norm_out(h_enc)
        h_enc = encoder.conv_act(h_enc)
        h_enc = encoder.conv_out(h_enc)
        last = h_enc[:, -1:, :, :, :].expand(-1, 127, -1, -1, -1)
        h_enc = torch.cat([h_enc, last], dim=1)
        moments_shape = tuple(h_enc.shape)
        log(f"  encoder moments: {moments_shape} (expect C=256)")
        assert moments_shape[1] == ENC_OUT_CHANNELS, moments_shape

        # Build a representative latent (128-ch slice of moments).
        latent = torch.randn(
            1, LATENT_CHANNELS,
            moments_shape[2], moments_shape[3], moments_shape[4],
            dtype=torch.float16,
        )
        log(f"Decoder input (latent): {tuple(latent.shape)}")

        # Trace decoder with timestep_conditioning. decode_timestep yaml
        # default = 0.05.
        ts_raw = torch.tensor([0.05], dtype=torch.float16)
        ts_scaled = ts_raw.float() * decoder.timestep_scale_multiplier
        h_dec = decoder.conv_in(latent, causal=decoder.causal)
        dec_block_inputs: list[torch.Tensor] = []
        dec_block_is_mid: list[bool] = []
        log(f"  after dec_shell_pre: {tuple(h_dec.shape)} (ts_cond={decoder.timestep_conditioning})")
        for i, blk in enumerate(decoder.up_blocks):
            dec_block_inputs.append(h_dec.clone())
            is_mid = isinstance(blk, UNetMidBlock3D) and decoder.timestep_conditioning
            dec_block_is_mid.append(is_mid)
            if is_mid:
                h_dec = blk(h_dec, causal=decoder.causal, timestep=ts_scaled)
            else:
                h_dec = blk(h_dec, causal=decoder.causal)
            kind = type(blk).__name__
            log(f"  after dec_block_{i:02d} ({kind}, ts={is_mid}): {tuple(h_dec.shape)}")
        dec_post_input_shape = tuple(h_dec.shape)
        log(f"  dec_shell_post input: {dec_post_input_shape}")

    gc.collect()

    # Dim instances. We let T, H, W be free; ORT-web shape-specializes per
    # call, so the same ONNX serves any 8k+1 / 32-divisible request.
    # Bounds cover BOTH the pixel-space tensors (shell_pre input up to
    # 1280x1280, ~256 frames) and the deeper-block intermediates (post-patch
    # up to 1280/4=320, halved at each compress_all). Keep wide.
    DT = Dim("t", min=1, max=512)
    DH = Dim("h", min=1, max=2048)
    DW = Dim("w", min=1, max=2048)

    do_enc = args.only in ("all", "encoder", "shells")
    do_dec = args.only in ("all", "decoder", "shells")
    blocks_only = args.only == "shells"

    chw_dyn = {2: DT, 3: DH, 4: DW}

    # ---- Encoder exports ----------------------------------------------------
    if do_enc:
        log("=== ENCODER ===")

        log("Exporting enc_shell_pre")
        export_module(
            EncShellPre(encoder).eval(),
            (torch.randn(*pixel_shape, dtype=torch.float16),),
            args.out_dir / "enc_shell_pre.onnx",
            ["pixels"], ["hidden"],
            (chw_dyn,),
            args.opset, log,
        )
        gc.collect()

        if not blocks_only:
            for i, blk in enumerate(encoder.down_blocks):
                sample = enc_block_inputs[i]
                if i in SPLIT_ENC_BLOCKS:
                    # ResnetBlock3Ds preserve shape, so the same `sample`
                    # serves as the representative for every sub-file.
                    n_res = len(blk.res_blocks)
                    log(f"Exporting enc_block_{i:02d} as {n_res} per-resnet sub-files")
                    for j, resnet in enumerate(blk.res_blocks):
                        export_module(
                            EncResnetWrapper(resnet).eval(),
                            (sample,),
                            args.out_dir / f"enc_block_{i:02d}_res_{j}.onnx",
                            ["hidden_in"], ["hidden_out"],
                            (chw_dyn,),
                            args.opset, log,
                        )
                        gc.collect()
                else:
                    log(f"Exporting enc_block_{i:02d}")
                    export_module(
                        EncDownWrapper(blk).eval(),
                        (sample,),
                        args.out_dir / f"enc_block_{i:02d}.onnx",
                        ["hidden_in"], ["hidden_out"],
                        (chw_dyn,),
                        args.opset, log,
                    )
                    gc.collect()

        log("Exporting enc_shell_post")
        export_module(
            EncShellPost(encoder).eval(),
            (torch.randn(*enc_post_input_shape, dtype=torch.float16),),
            args.out_dir / "enc_shell_post.onnx",
            ["hidden"], ["moments"],
            (chw_dyn,),
            args.opset, log,
        )
        gc.collect()

    # ---- Decoder exports ----------------------------------------------------
    if do_dec:
        log("=== DECODER ===")

        log("Exporting dec_shell_pre")
        latent_sample = torch.randn(
            1, LATENT_CHANNELS,
            moments_shape[2], moments_shape[3], moments_shape[4],
            dtype=torch.float16,
        )
        ts_sample = torch.tensor([0.05], dtype=torch.float16)
        export_module(
            DecShellPre(decoder).eval(),
            (latent_sample, ts_sample),
            args.out_dir / "dec_shell_pre.onnx",
            ["latent", "timestep"], ["hidden", "scaled_timestep"],
            (chw_dyn, {}),
            args.opset, log,
        )
        gc.collect()

        if not blocks_only:
            scaled_ts_sample = ts_sample.float() * decoder.timestep_scale_multiplier
            for i, blk in enumerate(decoder.up_blocks):
                sample = dec_block_inputs[i]
                if i in SPLIT_DEC_BLOCKS:
                    if not dec_block_is_mid[i]:
                        raise RuntimeError(
                            f"dec_block_{i:02d} is not a UNetMidBlock3D, can't split"
                        )
                    # Run time_embedder once -> ts_embed; then each ResnetBlock3D
                    # is its own sub-file taking (hidden, ts_embed).
                    n_res = len(blk.res_blocks)
                    log(
                        f"Exporting dec_block_{i:02d} as pre + {n_res} per-resnet sub-files"
                    )
                    export_module(
                        DecMidPre(blk).eval(),
                        (sample, scaled_ts_sample),
                        args.out_dir / f"dec_block_{i:02d}_pre.onnx",
                        ["hidden_in", "scaled_timestep"], ["ts_embed"],
                        (chw_dyn, {}),
                        args.opset, log,
                    )
                    gc.collect()
                    # Trace ts_embed shape for the resnet sub-files.
                    with torch.no_grad():
                        ts_embed_sample = DecMidPre(blk).eval()(
                            sample, scaled_ts_sample
                        )
                    for j, resnet in enumerate(blk.res_blocks):
                        # ts_embed has fully static shape (1, C*4, 1, 1, 1) for
                        # this stage; no dynamic axes needed on it.
                        export_module(
                            DecResnetWrapper(resnet, decoder.causal).eval(),
                            (sample, ts_embed_sample),
                            args.out_dir / f"dec_block_{i:02d}_res_{j}.onnx",
                            ["hidden_in", "ts_embed"], ["hidden_out"],
                            (chw_dyn, {}),
                            args.opset, log,
                        )
                        gc.collect()
                elif dec_block_is_mid[i]:
                    log(f"Exporting dec_block_{i:02d} ({type(blk).__name__})")
                    export_module(
                        DecMidBlockWrapper(blk, decoder.causal).eval(),
                        (sample, scaled_ts_sample),
                        args.out_dir / f"dec_block_{i:02d}.onnx",
                        ["hidden_in", "scaled_timestep"], ["hidden_out"],
                        (chw_dyn, {}),
                        args.opset, log,
                    )
                    gc.collect()
                else:
                    log(f"Exporting dec_block_{i:02d} ({type(blk).__name__})")
                    export_module(
                        DecPlainBlockWrapper(blk, decoder.causal).eval(),
                        (sample,),
                        args.out_dir / f"dec_block_{i:02d}.onnx",
                        ["hidden_in"], ["hidden_out"],
                        (chw_dyn,),
                        args.opset, log,
                    )
                    gc.collect()

        log("Exporting dec_shell_post")
        export_module(
            DecShellPost(decoder).eval(),
            (
                torch.randn(*dec_post_input_shape, dtype=torch.float16),
                ts_sample.float() * decoder.timestep_scale_multiplier,
            ),
            args.out_dir / "dec_shell_post.onnx",
            ["hidden", "scaled_timestep"], ["pixels"],
            (chw_dyn, {}),
            args.opset, log,
        )
        gc.collect()

    log(f"Peak RAM: {ram_peak[0]:.2f} GB")
    log("Done. Next: shard + quantize blocks to q4f16.")


if __name__ == "__main__":
    main()
