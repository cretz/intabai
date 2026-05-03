#!/usr/bin/env python3
"""Export LTX 2B distilled transformer per-block for browser layer-streaming.

Mirrors convert_t5_layers.py: split the monolith into shell_pre + 28 blocks
+ shell_post so the browser can load/run/release one block at a time.
Re-uses the dynamo path that the monolithic convert_transformer.py
established, so no rms_norm or squeeze patches are needed here (dynamo
handles them).

I/O contract (the browser side reassembles the loop):

  shell_pre:
    inputs:
      hidden_states           (1, N_tokens, 128)    fp16
      indices_grid            (1, 3, N_tokens)      float32
      encoder_hidden_states   (1, N_text, 4096)     fp16
      encoder_attention_mask  (1, N_text)           int64
      timestep                (1,)                  fp16
    outputs:
      hidden_proj             (1, N_tokens, 2048)   fp16
      freqs_cos               (1, N_tokens, 2048)   fp16
      freqs_sin               (1, N_tokens, 2048)   fp16
      timestep_mod            (1, 1, 6*2048)        fp16
      embedded_timestep       (1, 1, 2048)          fp16
      encoder_proj            (1, N_text, 2048)     fp16
      encoder_attn_bias       (1, 1, N_text)        fp16

  block_NN (28 of them):
    inputs:
      hidden_states           (1, N_tokens, 2048)   fp16
      freqs_cos, freqs_sin    (1, N_tokens, 2048)   fp16
      timestep_mod            (1, 1, 12288)         fp16
      encoder_proj            (1, N_text, 2048)     fp16
      encoder_attn_bias       (1, 1, N_text)        fp16
    output:
      hidden_states_out       (1, N_tokens, 2048)   fp16

  shell_post:
    inputs:
      hidden_states           (1, N_tokens, 2048)   fp16
      embedded_timestep       (1, 1, 2048)          fp16
    output:
      noise_pred              (1, N_tokens, 128)    fp16

Run:
  cd intabai/web/scripts && uv run python ltx/convert_transformer_layers.py \\
    --space-path C:/work/personal/intabai/notes/ltx-video-distilled-space \\
    --ckpt C:/work/personal/intabai/notes/models/ltx/source/ltxv-2b-0.9.8-distilled.safetensors \\
    --out-dir C:/work/personal/intabai/notes/models/ltx/staging/transformer
"""
import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

# torch.onnx dynamo prints unicode (✅/❌); cp1252 will crash mid-export.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import psutil
import torch
import torch.nn as nn
from torch.export import Dim


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


class ShellPreWrapper(nn.Module):
    """patchify_proj + RoPE + adaln_single + caption_projection + mask->bias."""

    def __init__(self, transformer):
        super().__init__()
        self.tx = transformer

    def forward(
        self,
        hidden_states,
        indices_grid,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
    ):
        # encoder_attention_mask (B, n_text) int64 -> bias (B, 1, n_text) fp16
        enc_bias = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        enc_bias = enc_bias.unsqueeze(1)

        hidden_proj = self.tx.patchify_proj(hidden_states)
        freqs_cos, freqs_sin = self.tx.precompute_freqs_cis(indices_grid)

        # transformer3d.py:418-419 scales the incoming timestep before
        # AdaLN; for LTX 2B distilled the multiplier is 1000.
        if self.tx.timestep_scale_multiplier:
            timestep = self.tx.timestep_scale_multiplier * timestep

        batch_size = hidden_states.shape[0]
        ts_mod, embedded_ts = self.tx.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        ts_mod = ts_mod.view(batch_size, -1, ts_mod.shape[-1])
        embedded_ts = embedded_ts.view(batch_size, -1, embedded_ts.shape[-1])

        encoder_proj = self.tx.caption_projection(encoder_hidden_states)
        encoder_proj = encoder_proj.view(batch_size, -1, hidden_proj.shape[-1])

        return hidden_proj, freqs_cos, freqs_sin, ts_mod, embedded_ts, encoder_proj, enc_bias


class BlockWrapper(nn.Module):
    """One BasicTransformerBlock with explicit (cos, sin) instead of a tuple."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states,
        freqs_cos,
        freqs_sin,
        timestep_mod,
        encoder_proj,
        encoder_attn_bias,
    ):
        return self.block(
            hidden_states,
            freqs_cis=(freqs_cos, freqs_sin),
            attention_mask=None,
            encoder_hidden_states=encoder_proj,
            encoder_attention_mask=encoder_attn_bias,
            timestep=timestep_mod,
            cross_attention_kwargs=None,
            class_labels=None,
            skip_layer_mask=None,
            skip_layer_strategy=None,
        )


class ShellPostWrapper(nn.Module):
    """scale_shift_table + norm_out + proj_out."""

    def __init__(self, transformer):
        super().__init__()
        self.tx = transformer

    def forward(self, hidden_states, embedded_timestep):
        scale_shift = (
            self.tx.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        h = self.tx.norm_out(hidden_states)
        h = h * (1 + scale) + shift
        h = self.tx.proj_out(h)
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ram-budget-gb", type=float, default=14.0)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--only-blocks", type=str, default=None,
                    help="Comma list of block indices to export (default: all)")
    ap.add_argument("--skip-shell", action="store_true",
                    help="Skip shell_pre + shell_post (re-run only blocks)")
    ap.add_argument("--only-shells", action="store_true",
                    help="Only export shell_pre + shell_post (skip blocks)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"RAM budget: {args.ram_budget_gb:.1f} GB")
    ram_peak = start_ram_watchdog(args.ram_budget_gb, log)

    sys.path.insert(0, str(args.space_path))
    from ltx_video.models.transformers.transformer3d import Transformer3DModel

    log(f"Loading transformer from {args.ckpt}")
    tx = Transformer3DModel.from_pretrained(str(args.ckpt))
    tx = tx.to(torch.float16).eval()
    for p in tx.parameters():
        p.requires_grad_(False)
    inner = tx.inner_dim
    num_blocks = len(tx.transformer_blocks)
    log(f"  inner_dim={inner}, blocks={num_blocks}")

    # Representative shapes (axes are dynamic via Dim).
    N, N_text = 128, 64
    hidden_in = torch.randn(1, N, 128, dtype=torch.float16)
    indices_grid = torch.zeros(1, 3, N, dtype=torch.float32)
    encoder_h = torch.randn(1, N_text, 4096, dtype=torch.float16)
    encoder_mask = torch.ones(1, N_text, dtype=torch.int64)
    timestep = torch.tensor([500.0], dtype=torch.float16)

    n_tokens = Dim("n_tokens", min=8, max=200000)
    n_text = Dim("n_text", min=8, max=512)

    if not args.skip_shell:
        # ---- shell_pre -------------------------------------------------------
        log("Exporting shell_pre")
        t0 = time.time()
        shell_pre = ShellPreWrapper(tx).eval()
        with torch.no_grad():
            outs = shell_pre(hidden_in, indices_grid, encoder_h, encoder_mask, timestep)
        log(
            f"  shapes: hp={list(outs[0].shape)}, cos={list(outs[1].shape)}, "
            f"ts_mod={list(outs[3].shape)}, emb={list(outs[4].shape)}, "
            f"enc_proj={list(outs[5].shape)}, enc_bias={list(outs[6].shape)}"
        )
        torch.onnx.export(
            shell_pre,
            (hidden_in, indices_grid, encoder_h, encoder_mask, timestep),
            str(args.out_dir / "shell_pre.onnx"),
            input_names=[
                "hidden_states", "indices_grid", "encoder_hidden_states",
                "encoder_attention_mask", "timestep",
            ],
            output_names=[
                "hidden_proj", "freqs_cos", "freqs_sin",
                "timestep_mod", "embedded_timestep",
                "encoder_proj", "encoder_attn_bias",
            ],
            dynamic_shapes={
                "hidden_states": {1: n_tokens},
                "indices_grid": {2: n_tokens},
                "encoder_hidden_states": {1: n_text},
                "encoder_attention_mask": {1: n_text},
                "timestep": {},
            },
            opset_version=args.opset,
            dynamo=True,
            external_data=True,
        )
        log(f"  shell_pre done in {time.time() - t0:.1f}s")
        del shell_pre
        gc.collect()

        # ---- shell_post ------------------------------------------------------
        log("Exporting shell_post")
        t0 = time.time()
        shell_post = ShellPostWrapper(tx).eval()
        h_post = torch.randn(1, N, inner, dtype=torch.float16)
        emb_ts = torch.randn(1, 1, inner, dtype=torch.float16)
        torch.onnx.export(
            shell_post,
            (h_post, emb_ts),
            str(args.out_dir / "shell_post.onnx"),
            input_names=["hidden_states", "embedded_timestep"],
            output_names=["noise_pred"],
            dynamic_shapes={
                "hidden_states": {1: n_tokens},
                "embedded_timestep": {},
            },
            opset_version=args.opset,
            dynamo=True,
        )
        log(f"  shell_post done in {time.time() - t0:.1f}s")
        del shell_post
        gc.collect()

    # ---- blocks --------------------------------------------------------------
    if args.only_shells:
        block_indices = []
    else:
        block_indices = (
            [int(x) for x in args.only_blocks.split(",")]
            if args.only_blocks else list(range(num_blocks))
        )
    log(f"Exporting {len(block_indices)} blocks")

    h_block = torch.randn(1, N, inner, dtype=torch.float16)
    cos_block = torch.randn(1, N, inner, dtype=torch.float16)
    sin_block = torch.randn(1, N, inner, dtype=torch.float16)
    ts_mod = torch.randn(1, 1, 6 * inner, dtype=torch.float16)
    enc_proj = torch.randn(1, N_text, inner, dtype=torch.float16)
    enc_bias = torch.randn(1, 1, N_text, dtype=torch.float16)

    block_dynamic = {
        "hidden_states": {1: n_tokens},
        "freqs_cos": {1: n_tokens},
        "freqs_sin": {1: n_tokens},
        "timestep_mod": {},
        "encoder_proj": {1: n_text},
        "encoder_attn_bias": {2: n_text},
    }

    for i in block_indices:
        t0 = time.time()
        wrapper = BlockWrapper(tx.transformer_blocks[i]).eval()
        path = args.out_dir / f"block_{i:02d}.onnx"
        torch.onnx.export(
            wrapper,
            (h_block, cos_block, sin_block, ts_mod, enc_proj, enc_bias),
            str(path),
            input_names=[
                "hidden_states", "freqs_cos", "freqs_sin",
                "timestep_mod", "encoder_proj", "encoder_attn_bias",
            ],
            output_names=["hidden_states_out"],
            dynamic_shapes=block_dynamic,
            opset_version=args.opset,
            dynamo=True,
            external_data=True,
        )
        graph_size = path.stat().st_size
        data_path = args.out_dir / f"block_{i:02d}.onnx.data"
        data_size = data_path.stat().st_size if data_path.exists() else 0
        total_mb = (graph_size + data_size) / 1e6
        log(
            f"  block {i}/{num_blocks - 1} -> {path.name} "
            f"({total_mb:.1f} MB graph+data, {time.time() - t0:.1f}s)"
        )
        del wrapper
        gc.collect()

    log(f"Peak RAM: {ram_peak[0]:.2f} GB")
    log("Done. Next: quantize blocks to q4f16.")


if __name__ == "__main__":
    main()
