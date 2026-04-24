#!/usr/bin/env python3
"""Export WanTransformer3DModel from FastWan2.2-TI2V-5B to ONNX.

Produces an fp16 ONNX file with external data layout, suitable for sharding
with shard-onnx-layers.py and subsequent quantization per shard.

The diffusers WanTransformer3DModel.forward() handles RoPE, patch embedding,
timestep/text conditioning, 30 transformer blocks, and unpatchify internally.
We wrap it to accept flat tensors for ONNX:

  Inputs:
    hidden_states            [B, 48, T, H, W]          float16  (noisy latents)
    timestep                 [B, num_timesteps]         int64    (2D for Wan 2.2 TI2V)
    encoder_hidden_states    [B, 512, 4096]             float16  (UMT5 text embeddings)

  Output:
    noise_pred               [B, 48, T, H, W]          float16

Config (from HF):
  - 30 layers, 24 heads x 128 dim = 3072 hidden, FFN 14336
  - patch_size [1, 2, 2], in/out channels 48, text_dim 4096
  - image_dim null, added_kv_proj_dim null (no image cross-attention)
  - 3D RoPE computed internally (freq_dim 256, rope_max_seq_len 1024)
  - Scheduler: UniPCMultistepScheduler, flow matching, flow_shift 5.0, 3 steps

Uses accelerate disk offloading so the ~9.3 GB bf16 model doesn't need to
fit in RAM at once.

Usage:
    uv run export-fastwan-transformer.py <model_path> <output_dir>

    model_path: local path to FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers checkout
    output_dir: where to write the ONNX file + external data

Example:
    uv run export-fastwan-transformer.py \
      ../../notes/models/fastwan/source \
      ../../notes/models/fastwan/onnx
"""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import infer_auto_device_map

from lib.chunk_attn1 import chunk_attn1_file


def patch_rms_norm_for_onnx():
    """Decompose torch.nn.RMSNorm.forward into explicit ops for ONNX export.

    torch.onnx doesn't support aten::rms_norm even at opset 23 (2026-04-16),
    so we replace forward() with the mathematical definition using ops the
    tracer already handles (pow, mean, add, rsqrt, mul).
    """
    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        in_dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(dim=dims, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        out = xf.to(in_dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    torch.nn.RMSNorm.forward = forward


def start_ram_watchdog(max_gb: float, log_fn):
    proc = psutil.Process(os.getpid())
    peak = [0.0]

    def tick():
        while True:
            rss_gb = proc.memory_info().rss / 1e9
            if rss_gb > peak[0]:
                peak[0] = rss_gb
            if rss_gb > max_gb:
                log_fn(f"!!! RAM {rss_gb:.2f} GB > {max_gb:.1f} GB budget, KILLING")
                os._exit(2)
            time.sleep(0.5)

    threading.Thread(target=tick, daemon=True).start()
    return peak


def log_ram(log_fn, label):
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1e9
    log_fn(f"  [ram {label}: {rss_gb:.2f} GB]")


class ShellPreWrapper(nn.Module):
    """Pre-block portion of the transformer: patch embed, RoPE, condition embed.

    Inputs match the monolithic model. Outputs are the tensors each block needs.
    """

    def __init__(self, model):
        super().__init__()
        self.rope = model.rope
        self.patch_embedding = model.patch_embedding
        self.condition_embedder = model.condition_embedder

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        freqs_cos, freqs_sin = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        return hidden_states, encoder_hidden_states, timestep_proj, temb, freqs_cos, freqs_sin


class ShellPostWrapper(nn.Module):
    """Post-block portion: final norm, projection, unpatchify.

    Takes post-block tokens + temb + patch-space dims, returns 5D noise_pred.
    Patch-space dims (ppf, pph, ppw) are passed as int64 scalar tensors from
    the runtime so the graph can do the unpatchify reshape with dynamic shapes.
    """

    def __init__(self, model, patch_size):
        super().__init__()
        self.scale_shift_table = model.scale_shift_table
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.p_t, self.p_h, self.p_w = patch_size

    def forward(self, hidden_states, temb, ppf, pph, ppw):
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            batch_size, ppf, pph, ppw, self.p_t, self.p_h, self.p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return output


class BlockWrapper(nn.Module):
    """Wraps a single WanTransformerBlock, packing the rotary_emb tuple."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin):
        return self.block(hidden_states, encoder_hidden_states, temb, (freqs_cos, freqs_sin))


class WanTransformerWrapper(nn.Module):
    """Wraps WanTransformer3DModel for clean ONNX export.

    The diffusers model returns a Transformer2DModelOutput dict by default.
    We force return_dict=False and extract the tensor output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        result = self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        return result[0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to FastWan2.2-TI2V-5B-FullAttn-Diffusers local checkout",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for ONNX files",
    )
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Text encoder sequence length (default: 512)")
    parser.add_argument("--no-expand-timesteps", action="store_true",
                        help="Use scalar [B] timestep instead of expanded [B, seq_len]")
    parser.add_argument("--max-cpu-gb", type=float, default=4.0,
                        help="Max CPU RAM for model weights (rest offloaded to disk)")
    parser.add_argument("--ram-budget-gb", type=float, default=16.0,
                        help="Process-total RAM budget; kill self if exceeded")
    parser.add_argument("--trace-tiny", action="store_true",
                        help="Use tiny dummy inputs for the trace (real shapes still work via dynamic_axes)")
    parser.add_argument("--sanity-check", action="store_true",
                        help="After export, load ONNX and run dummy inference (WARNING: loads full model into RAM)")
    parser.add_argument("--per-block", action="store_true",
                        help="Export shell_pre.onnx + N block ONNX files + shell_post.onnx instead of monolithic")
    parser.add_argument("--only-shell-pre", action="store_true",
                        help="Stop after shell_pre.onnx is written. Requires --per-block. "
                             "Used when re-exporting only the rope-affected shell after a fix.")
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    transformer_path = args.model_path / "transformer"
    if not transformer_path.exists():
        print(f"Error: transformer subfolder not found at {transformer_path}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "transformer.onnx"

    # Only open a log file if explicitly requested. Default behavior keeps
    # the output dir clean - stdout is already captured by the caller's
    # shell redirect.
    log_file = open(args.log, "w", buffering=1) if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_file is not None:
            log_file.write(line + "\n")

    if log_file is not None:
        log(f"Export log: {args.log}")
    log(f"Target: {args.height}x{args.width}, {args.num_frames} frames")
    log(f"RAM budget: {args.ram_budget_gb:.1f} GB (self-kill if exceeded)")
    ram_peak = start_ram_watchdog(args.ram_budget_gb, log)

    patch_rms_norm_for_onnx()
    log("Patched torch.nn.RMSNorm.forward for ONNX export (aten::rms_norm decomposition)")

    # ---- Load model: shell weights on CPU, block weights on meta ----
    # Accelerate's disk-offload hooks trigger an aten::view bug in the torch
    # JIT tracer when weights are lazily loaded mid-trace. Workaround: load
    # only non-block weights to CPU, leave blocks on meta (never traced through
    # the monolithic model; each block is loaded fresh from safetensors later).
    log(f"Loading WanTransformer3DModel from {transformer_path}...")
    log("  (shell weights on CPU, block weights stay on meta)")
    t0 = time.time()

    from accelerate.utils import set_module_tensor_to_device
    from diffusers import WanTransformer3DModel
    from safetensors import safe_open

    config = WanTransformer3DModel.load_config(str(transformer_path))
    log(f"  config: {config}")

    with init_empty_weights():
        model = WanTransformer3DModel.from_config(config)

    checkpoint_file = transformer_path / "diffusion_pytorch_model.safetensors"
    shell_param_count = 0
    shell_byte_count = 0
    with safe_open(str(checkpoint_file), framework="pt") as f:
        for key in f.keys():
            if key.startswith("blocks."):
                continue
            tensor = f.get_tensor(key).to(torch.float16)
            set_module_tensor_to_device(model, key, "cpu", value=tensor, dtype=torch.float16)
            shell_param_count += 1
            shell_byte_count += tensor.numel() * tensor.element_size()
    model.eval()
    log(f"  loaded {shell_param_count} shell tensors ({shell_byte_count / 1e6:.1f} MB) "
        f"in {time.time() - t0:.1f}s; blocks remain on meta")

    # Re-instantiate WanRotaryPosEmbed outside init_empty_weights context.
    # Its freqs_cos / freqs_sin are registered as persistent=False (see
    # diffusers/models/transformers/transformer_wan.py:392-393), which means
    # they are NOT in the safetensors state_dict. set_module_tensor_to_device
    # above never materializes them - they stay on the meta device from the
    # init_empty_weights context. torch.onnx.export then silently bakes garbage
    # (uninitialized) constants for the RoPE frequencies, which produces
    # per-pixel random-color confetti output (every token gets wrong RoPE,
    # attention computes random-relative dot products, tokens have no spatial
    # coherence at unpatchify time). This was a 10-min-per-test bug hunt; keep
    # the assertion below as a tripwire.
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    attention_head_dim = config.get("attention_head_dim", 128)
    patch_size = tuple(config.get("patch_size", [1, 2, 2]))
    rope_max_seq_len = config.get("rope_max_seq_len", 1024)
    model.rope = WanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=rope_max_seq_len,
    )
    assert model.rope.freqs_cos.device.type != "meta", (
        "rope.freqs_cos on meta device - RoPE will be garbage in exported ONNX"
    )
    assert model.rope.freqs_sin.device.type != "meta", (
        "rope.freqs_sin on meta device - RoPE will be garbage in exported ONNX"
    )
    log(f"  rope freqs materialized: cos {tuple(model.rope.freqs_cos.shape)} "
        f"on {model.rope.freqs_cos.device}, dtype {model.rope.freqs_cos.dtype}")
    offload_dir = None

    wrapper = WanTransformerWrapper(model)

    # ---- Build dummy inputs ----
    # Transformer operates in latent space, not pixel space.
    # Wan 2.2 VAE compression: temporal (T-1)/4+1, spatial /16 each dim.
    B = 1
    C = config.get("in_channels", 48)
    T_latent = (args.num_frames - 1) // 4 + 1
    H_latent = args.height // 16
    W_latent = args.width // 16
    text_dim = config.get("text_dim", 4096)

    p_t, p_h, p_w = config.get("patch_size", [1, 2, 2])

    # Real inference dims (what the pipeline will call the model with).
    real_ts_seq_len = T_latent * (H_latent // p_h) * (W_latent // p_w)

    # Trace dims: optionally tiny to minimize activation memory during tracing.
    # dynamic_axes makes the real shapes work at inference time.
    if args.trace_tiny:
        trace_T, trace_H, trace_W = 1, p_h * 2, p_w * 2  # smallest valid shape after patching
        trace_seq_len = max(1, args.seq_len // 8)
        log(f"  trace mode: TINY (dims {trace_T}f {trace_H}x{trace_W}, text seq {trace_seq_len})")
    else:
        trace_T, trace_H, trace_W = T_latent, H_latent, W_latent
        trace_seq_len = args.seq_len
        log(f"  trace mode: FULL")

    trace_ts_seq_len = trace_T * (trace_H // p_h) * (trace_W // p_w)

    if args.no_expand_timesteps:
        ts_shape = (B,)
        log(f"  timestep mode: scalar [B]")
    else:
        ts_shape = (B, trace_ts_seq_len)
        log(f"  timestep mode: expanded [B, {trace_ts_seq_len}] (per-token, trace); "
            f"real inference uses [B, {real_ts_seq_len}]")

    log(f"  pixel dims: {args.num_frames}f {args.height}x{args.width}")
    log(f"  latent dims (real): {T_latent}f {H_latent}x{W_latent}")
    log(f"  dummy inputs (trace): hidden_states [{B},{C},{trace_T},{trace_H},{trace_W}], "
        f"timestep {list(ts_shape)}, "
        f"encoder_hidden_states [{B},{trace_seq_len},{text_dim}]")

    dummy_hidden = torch.randn(B, C, trace_T, trace_H, trace_W, dtype=torch.float16)
    dummy_timestep = torch.randint(0, 1000, ts_shape, dtype=torch.long)
    dummy_enc_hidden = torch.randn(B, trace_seq_len, text_dim, dtype=torch.float16)

    num_blocks = len(model.blocks)

    if args.per_block:
        log(f"Per-block export mode: shell_pre + {num_blocks} blocks + shell_post")
        import gc

        per_block_dir = args.output_dir / "transformer"
        per_block_dir.mkdir(parents=True, exist_ok=True)

        # Run shell_pre once to capture intermediate tensors used as dummy inputs
        # for block + shell_post exports.
        shell_pre = ShellPreWrapper(model)
        log("  running shell_pre to capture intermediate tensor shapes...")
        t0 = time.time()
        with torch.no_grad():
            tokens, enc_proj, timestep_proj, temb, freqs_cos, freqs_sin = shell_pre(
                dummy_hidden, dummy_timestep, dummy_enc_hidden
            )
        log(f"  shell_pre ran in {time.time() - t0:.1f}s")
        log(f"    tokens:        {list(tokens.shape)}")
        log(f"    enc_proj:      {list(enc_proj.shape)}")
        log(f"    timestep_proj: {list(timestep_proj.shape)}")
        log(f"    temb:          {list(temb.shape)}")
        log(f"    freqs_cos:     {list(freqs_cos.shape)}")
        log(f"    freqs_sin:     {list(freqs_sin.shape)}")

        # ---- Export shell_pre ----
        shell_pre_path = per_block_dir / "shell_pre.onnx"
        log(f"Exporting shell_pre -> {shell_pre_path}")
        t0 = time.time()
        torch.onnx.export(
            shell_pre,
            (dummy_hidden, dummy_timestep, dummy_enc_hidden),
            str(shell_pre_path),
            opset_version=args.opset,
            input_names=["hidden_states", "timestep", "encoder_hidden_states"],
            output_names=["tokens", "enc_proj", "timestep_proj", "temb", "freqs_cos", "freqs_sin"],
            dynamic_axes={
                "hidden_states": {0: "batch", 2: "latent_frames", 3: "latent_height", 4: "latent_width"},
                "timestep": ({0: "batch"} if args.no_expand_timesteps
                             else {0: "batch", 1: "ts_seq_len"}),
                "encoder_hidden_states": {0: "batch", 1: "text_seq"},
                "tokens": {0: "batch", 1: "seq_len"},
                "enc_proj": {0: "batch", 1: "text_seq"},
                "timestep_proj": ({0: "batch"} if args.no_expand_timesteps
                                  else {0: "batch", 1: "seq_len"}),
                "temb": ({0: "batch"} if args.no_expand_timesteps
                         else {0: "batch", 1: "seq_len"}),
                "freqs_cos": {1: "seq_len"},
                "freqs_sin": {1: "seq_len"},
            },
            dynamo=False,
        )
        log(f"  shell_pre exported in {time.time() - t0:.1f}s "
            f"({shell_pre_path.stat().st_size / 1e6:.1f} MB)")
        del shell_pre
        gc.collect()
        log_ram(log, "after shell_pre")

        if args.only_shell_pre:
            log("--only-shell-pre set; skipping shell_post + blocks and exiting.")
            return

        # ---- Export shell_post BEFORE blocks so we can free the monolithic model ----
        p_t, p_h, p_w = config.get("patch_size", [1, 2, 2])
        dummy_ppf = torch.tensor(trace_T // p_t, dtype=torch.int64)
        dummy_pph = torch.tensor(trace_H // p_h, dtype=torch.int64)
        dummy_ppw = torch.tensor(trace_W // p_w, dtype=torch.int64)
        shell_post = ShellPostWrapper(model, (p_t, p_h, p_w))

        shell_post_path = per_block_dir / "shell_post.onnx"
        log(f"Exporting shell_post -> {shell_post_path}")
        t0 = time.time()
        torch.onnx.export(
            shell_post,
            (tokens, temb, dummy_ppf, dummy_pph, dummy_ppw),
            str(shell_post_path),
            opset_version=args.opset,
            input_names=["hidden_states", "temb", "ppf", "pph", "ppw"],
            output_names=["noise_pred"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "temb": ({0: "batch"} if args.no_expand_timesteps
                         else {0: "batch", 1: "seq_len"}),
                "noise_pred": {0: "batch", 2: "latent_frames", 3: "latent_height", 4: "latent_width"},
            },
            dynamo=False,
        )
        log(f"  shell_post exported in {time.time() - t0:.1f}s "
            f"({shell_post_path.stat().st_size / 1e6:.1f} MB)")
        del shell_post

        # Free the monolithic accelerate-loaded model. From here on, each block
        # is loaded fresh from safetensors so no hook/offload state can leak
        # between traces.
        log("Freeing monolithic model before per-block export...")
        del model
        gc.collect()
        if offload_dir is not None:
            import shutil
            shutil.rmtree(offload_dir, ignore_errors=True)
        log_ram(log, "after model freed")

        # ---- Export each block in isolation ----
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        block_dim = config.get("attention_head_dim", 128) * config.get("num_attention_heads", 24)
        block_kwargs = dict(
            dim=block_dim,
            ffn_dim=config.get("ffn_dim", 14336),
            num_heads=config.get("num_attention_heads", 24),
            qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
            cross_attn_norm=config.get("cross_attn_norm", True),
            eps=config.get("eps", 1e-6),
            added_kv_proj_dim=config.get("added_kv_proj_dim", None),
        )
        log(f"  fresh-block kwargs: {block_kwargs}")

        safetensors_path = transformer_path / "diffusion_pytorch_model.safetensors"

        for i in range(num_blocks):
            t0 = time.time()
            prefix = f"blocks.{i}."
            block = WanTransformerBlock(**block_kwargs)
            block.eval()

            state = {}
            with safe_open(str(safetensors_path), framework="pt") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = f.get_tensor(key).to(torch.float16)
            missing, unexpected = block.load_state_dict(state, strict=False)
            if missing or unexpected:
                log(f"  WARN block {i}: missing={missing}, unexpected={unexpected}")
            block = block.to(torch.float16)

            block_wrap = BlockWrapper(block)
            block_path = per_block_dir / f"block_{i:02d}.onnx"
            torch.onnx.export(
                block_wrap,
                (tokens, enc_proj, timestep_proj, freqs_cos, freqs_sin),
                str(block_path),
                opset_version=args.opset,
                input_names=["hidden_states", "encoder_hidden_states", "timestep_proj",
                             "freqs_cos", "freqs_sin"],
                output_names=["hidden_states_out"],
                dynamic_axes={
                    "hidden_states": {0: "batch", 1: "seq_len"},
                    "encoder_hidden_states": {0: "batch", 1: "text_seq"},
                    "timestep_proj": ({0: "batch"} if args.no_expand_timesteps
                                       else {0: "batch", 1: "seq_len"}),
                    "freqs_cos": {1: "seq_len"},
                    "freqs_sin": {1: "seq_len"},
                    "hidden_states_out": {0: "batch", 1: "seq_len"},
                },
                dynamo=False,
            )
            # Seq-chunk attn1 to keep Q*K^T under the 2 GiB WebGPU
            # maxBufferSize cap. See lib/chunk_attn1.py + notes/ort-fp16-bugs.md
            # section 5. CPU/CUDA math is unchanged. Only needed when
            # heads * seq_len^2 * 2 bytes > 2 GiB; below that, chunking adds
            # graph nodes for no reason and breaks at runtime since the lib
            # still hardcodes the 8190-shape split sizes.
            heads = config.get("num_attention_heads", 24)
            qkt_bytes = heads * real_ts_seq_len * real_ts_seq_len * 2
            if qkt_bytes > (2 << 30):
                chunk_attn1_file(block_path, n_chunks=3, seq_len=real_ts_seq_len)
            else:
                log(f"    skipping attn1 chunking (Q*K^T {qkt_bytes/1e9:.2f} GB < 2 GiB cap)")
            size_mb = block_path.stat().st_size / 1e6
            dt = time.time() - t0
            log(f"  block {i+1}/{num_blocks} -> {block_path.name} ({size_mb:.1f} MB, {dt:.1f}s)")
            del block_wrap, block, state
            gc.collect()

        log_ram(log, "after all blocks")

        log(f"Peak RAM observed: {ram_peak[0]:.2f} GB")
        log("Per-block export done. Outputs:")
        log(f"  {per_block_dir}/shell_pre.onnx")
        log(f"  {per_block_dir}/block_00.onnx ... block_{num_blocks-1:02d}.onnx")
        log(f"  {per_block_dir}/shell_post.onnx")
        if log_file is not None:
            log_file.close()
        return

    # ---- Monolithic path below ----
    block_state = {"t_last": time.time(), "t_start": time.time()}

    def make_pre_hook(idx):
        def pre_hook(_mod, _inp):
            now = time.time()
            dt = now - block_state["t_last"]
            total = now - block_state["t_start"]
            log(f"    block {idx+1}/{num_blocks} starting (+{dt:.1f}s, total {total:.1f}s)")
            block_state["t_last"] = now
        return pre_hook

    hook_handles = [b.register_forward_pre_hook(make_pre_hook(i))
                    for i, b in enumerate(model.blocks)]

    # ---- Verify wrapper forward pass ----
    log(f"  verifying forward pass ({num_blocks} blocks)...")
    block_state["t_start"] = time.time()
    block_state["t_last"] = time.time()
    t0 = time.time()
    with torch.no_grad():
        test_out = wrapper(dummy_hidden, dummy_timestep, dummy_enc_hidden)
    log(f"  forward pass OK: output shape {list(test_out.shape)}, took {time.time() - t0:.1f}s")

    for h in hook_handles:
        h.remove()

    # ---- Export to ONNX ----
    log(f"Exporting to ONNX (opset {args.opset})...")
    log(f"  output: {output_path}")
    t0 = time.time()

    block_state["t_start"] = time.time()
    block_state["t_last"] = time.time()
    hook_handles = [b.register_forward_pre_hook(make_pre_hook(i))
                    for i, b in enumerate(model.blocks)]

    torch.onnx.export(
        wrapper,
        (dummy_hidden, dummy_timestep, dummy_enc_hidden),
        str(output_path),
        opset_version=args.opset,
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            "hidden_states": {0: "batch", 2: "latent_frames", 3: "latent_height", 4: "latent_width"},
            "timestep": ({0: "batch"} if args.no_expand_timesteps
                         else {0: "batch", 1: "ts_seq_len"}),
            "encoder_hidden_states": {0: "batch", 1: "seq_len"},
            "noise_pred": {0: "batch", 2: "latent_frames", 3: "latent_height", 4: "latent_width"},
        },
        dynamo=False,
    )
    export_time = time.time() - t0
    log(f"  export completed in {export_time:.1f}s")

    for h in hook_handles:
        h.remove()

    # ---- Convert to external data layout if large ----
    onnx_size = output_path.stat().st_size
    log(f"  ONNX file size: {onnx_size / 1e9:.2f} GB")

    if onnx_size > 2e9:
        log("File is >2 GB. Converting to external data layout...")
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data

        onnx_model = onnx.load(str(output_path))
        data_path = "transformer.onnx_data"
        convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=data_path,
            size_threshold=0,
            convert_attribute=False,
        )
        onnx.save_model(
            onnx_model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path,
        )
        ext_size = (args.output_dir / data_path).stat().st_size
        graph_size = output_path.stat().st_size
        log(f"  graph: {graph_size / 1e6:.1f} MB, data: {ext_size / 1e9:.2f} GB")

    if offload_dir is not None:
        import shutil
        shutil.rmtree(offload_dir, ignore_errors=True)

    if args.sanity_check:
        # ---- Sanity check: load ONNX + dummy run via onnxruntime ----
        # WARNING: loads full ~9 GB ONNX into RAM, will exceed a 10 GB budget.
        del wrapper, model
        import gc
        gc.collect()

        log("Sanity check: loading ONNX with onnxruntime...")
        import onnxruntime as ort_rt
        import numpy as np

        sess_opts = ort_rt.SessionOptions()
        sess_opts.graph_optimization_level = ort_rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            sess = ort_rt.InferenceSession(
                str(output_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            log(f"  session created OK")
            log(f"  inputs:  {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
            log(f"  outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

            real_ts = (B, real_ts_seq_len) if not args.no_expand_timesteps else (B,)
            feeds = {
                "hidden_states": np.zeros((B, C, T_latent, H_latent, W_latent), dtype=np.float16),
                "timestep": np.zeros(real_ts, dtype=np.int64),
                "encoder_hidden_states": np.zeros((B, args.seq_len, text_dim), dtype=np.float16),
            }
            t0 = time.time()
            outputs = sess.run(None, feeds)
            log(f"  inference OK in {time.time() - t0:.1f}s, output shape: {outputs[0].shape}")
            del sess, outputs
        except Exception as e:
            log(f"  sanity check FAILED: {e}")
    else:
        log("Skipping sanity check (pass --sanity-check to enable; loads full ONNX into RAM).")

    log(f"Peak RAM observed: {ram_peak[0]:.2f} GB")
    log("Done. Next steps:")
    log("  1. Inspect:  uv run shard-onnx-layers.py --inspect <output.onnx>")
    log("  2. Shard:    uv run shard-onnx-layers.py <output.onnx> --max-shard-size 1.2GB <out.json>")
    log("  3. Quantize: each shard independently to q4f16")
    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
