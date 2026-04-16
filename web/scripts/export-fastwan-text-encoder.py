#!/usr/bin/env python3
"""Export UMT5-XXL text encoder from FastWan2.2-TI2V-5B to ONNX (per-layer).

Mirrors the transformer export pattern: per-layer ONNX files so we never need
to hold the full ~10.6 GB model in RAM as a single ONNX proto. Each layer is
freshly loaded from safetensors, exported, freed.

The token embedding table is extracted separately as a raw fp16 binary
(`embedding.bin`). At inference time, JS does the token->embedding lookup and
feeds dense [B, seq_len, 4096] embeddings into layer_00.onnx. This avoids
having a single 2.1 GB tensor inside any ONNX file - mobile WebGPU's
maxBufferSize (~256 MB) cannot hold it as one GPU buffer.

Output layout:
  <output_dir>/text-encoder/
    embedding.bin                 # [vocab=256384, d_model=4096] fp16, raw little-endian
    layer_00.onnx ... layer_23.onnx
    shell_post.onnx               # final UMT5LayerNorm

Per layer ONNX:
  Inputs:
    hidden_states        [B, seq_len, 4096]    float16
    attention_mask       [B, 1, 1, seq_len]    float16   (extended mask: 0 / -inf)
  Output:
    hidden_states_out    [B, seq_len, 4096]    float16

shell_post ONNX:
  Inputs:
    hidden_states        [B, seq_len, 4096]    float16
  Output:
    last_hidden_state    [B, seq_len, 4096]    float16

Config (from HF):
  - 24 layers, 64 heads, d_model 4096, d_kv 64, d_ff 10240
  - gated-gelu FFN, vocab 256384, ~10.6 GB bf16 (3 shards)
  - relative_attention_num_buckets=32, untied (each layer computes its own bias)

UMT5LayerNorm is RMSNorm-style but already implemented as pure ops in
transformers source (variance + rsqrt + scale), so the aten::rms_norm patch
the transformer needed does NOT apply here.

Usage:
    uv run export-fastwan-text-encoder.py <model_path> <output_dir>

Example:
    uv run export-fastwan-text-encoder.py \
      ../../notes/models/fastwan/source \
      ../../notes/models/fastwan/hf-repo/onnx
"""

import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn


def patch_umt5_layernorm_for_onnx():
    """Force explicit fp32 variance in UMT5LayerNorm for ONNX export.

    HF UMT5LayerNorm already upcasts to fp32 before pow(2).mean() to avoid
    fp16 overflow, but the ``.to(torch.float32)`` call does not always
    survive ``torch.onnx.export`` tracing under opset 23 (the tracer's
    dtype propagation can fold it away, producing an fp16 graph that
    overflows on inputs with values around ±185, exactly what UMT5's
    embedding table contains). Diagnosed 2026-04-18: layer_00 produced
    22k NaN + -Inf on a real prompt with both q4f16 and fp16 weights;
    input embeds were healthy.

    Replace forward() with an explicit decomposition that keeps the fp32
    casts as distinct ops the tracer can't optimize away.
    """
    from transformers.models.umt5.modeling_umt5 import UMT5LayerNorm

    def forward(self, hidden_states):
        in_dtype = hidden_states.dtype
        x32 = hidden_states.to(torch.float32)
        variance = (x32 * x32).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.variance_epsilon)
        out = x32.to(in_dtype)
        # Explicit final cast: without it, `self.weight * out` traces with a
        # Cast-to-fp32 on the weight and the whole op's output ends up fp32.
        # That's fine mid-block (next op casts down) but makes shell_post's
        # output fp32, which we don't want on the wire (browser expects fp16).
        return (self.weight * out).to(in_dtype)

    UMT5LayerNorm.forward = forward


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


class LayerWrapper(nn.Module):
    """Wraps a single UMT5Block for ONNX export.

    UMT5Block.forward returns a tuple; we extract just hidden_states.
    Position bias is computed internally by self.layer[0].SelfAttention
    (each layer has has_relative_attention_bias=True in untied UMT5).
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask):
        # UMT5Block computes its own relative position bias internally
        # (no position_bias kwarg like T5Block).
        out = self.block(hidden_states, attention_mask=attention_mask)
        return out[0]


class ShellPostWrapper(nn.Module):
    """Final UMT5LayerNorm + identity (dropout is no-op in eval)."""

    def __init__(self, final_layer_norm):
        super().__init__()
        self.final_layer_norm = final_layer_norm

    def forward(self, hidden_states):
        return self.final_layer_norm(hidden_states)


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
        help="Output directory (text-encoder/ subdir will be created)",
    )
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for tracing (default: 512)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--ram-budget-gb", type=float, default=12.0,
                        help="Process-total RAM budget; kill self if exceeded")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding.bin extraction (if already done)")
    parser.add_argument("--only-layers", type=str, default=None,
                        help="Comma-separated layer indices to export (e.g. '0,1,2')")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    text_encoder_path = args.model_path / "text_encoder"
    if not text_encoder_path.exists():
        print(f"Error: text_encoder subfolder not found at {text_encoder_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir / "text-encoder"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only open a log file if explicitly requested. Default behavior keeps
    # hf-repo clean - stdout is already captured by the caller's shell redirect.
    log_file = open(args.log, "w", buffering=1) if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_file is not None:
            log_file.write(line + "\n")

    if log_file is not None:
        log(f"Export log: {args.log}")
    log(f"RAM budget: {args.ram_budget_gb:.1f} GB (self-kill if exceeded)")
    ram_peak = start_ram_watchdog(args.ram_budget_gb, log)

    patch_umt5_layernorm_for_onnx()
    log("Patched UMT5LayerNorm.forward for ONNX export (fp32 variance cast)")

    # ---- Load config + locate safetensors shards ----
    from transformers import AutoConfig
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(str(text_encoder_path))
    log(f"  d_model={config.d_model}, layers={config.num_layers}, "
        f"heads={config.num_heads}, d_ff={config.d_ff}, vocab={config.vocab_size}")

    # Find all safetensors shards. UMT5-XXL is sharded (3 shards typically).
    shard_files = sorted(text_encoder_path.glob("*.safetensors"))
    if not shard_files:
        log(f"  ERROR: no safetensors found in {text_encoder_path}")
        sys.exit(1)
    log(f"  shards: {[f.name for f in shard_files]}")

    # Build a map of {key: shard_path} so we can pull individual tensors lazily.
    key_to_shard = {}
    for shard in shard_files:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                key_to_shard[key] = shard
    log(f"  total tensors: {len(key_to_shard)}")

    def load_tensor(key, dtype=torch.float16):
        shard = key_to_shard[key]
        with safe_open(str(shard), framework="pt") as f:
            return f.get_tensor(key).to(dtype)

    # ---- Step 1: Extract embedding ----
    embedding_path = out_dir / "embedding.bin"
    if args.skip_embedding and embedding_path.exists():
        log(f"Skipping embedding extraction (already at {embedding_path})")
    else:
        log("Extracting token embedding to embedding.bin...")
        t0 = time.time()
        # UMT5 embed_tokens key. Try common locations.
        embed_key = None
        for candidate in ["shared.weight", "encoder.embed_tokens.weight", "embed_tokens.weight"]:
            if candidate in key_to_shard:
                embed_key = candidate
                break
        if embed_key is None:
            log(f"  available top-level keys: {sorted(set(k.split('.')[0] for k in key_to_shard))[:20]}")
            log("  ERROR: could not find embedding tensor key")
            sys.exit(1)
        log(f"  using key: {embed_key}")

        embed = load_tensor(embed_key, dtype=torch.float16).contiguous()
        log(f"  shape: {list(embed.shape)}, dtype: {embed.dtype}, "
            f"size: {embed.numel() * 2 / 1e9:.2f} GB")

        # Write raw little-endian fp16. JS side can map this directly into a
        # Float16Array (or read row-by-row via Range requests).
        with open(embedding_path, "wb") as f:
            f.write(embed.cpu().numpy().tobytes())
        log(f"  wrote {embedding_path.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")
        del embed
        gc.collect()
        log_ram(log, "after embedding")

    # ---- Step 2: Build empty model skeleton on meta, then load just the
    # final layer norm to CPU (needed for shell_post). ----
    log("Building model skeleton on meta...")
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import UMT5EncoderModel
    from transformers.models.umt5.modeling_umt5 import UMT5Block

    with init_empty_weights():
        model = UMT5EncoderModel(config)
    model.eval()

    # Load only encoder.final_layer_norm.weight (small).
    log("Loading final_layer_norm weights to CPU...")
    final_norm_key = "encoder.final_layer_norm.weight"
    if final_norm_key not in key_to_shard:
        log(f"  ERROR: {final_norm_key} not found")
        sys.exit(1)
    set_module_tensor_to_device(
        model, final_norm_key, "cpu",
        value=load_tensor(final_norm_key, dtype=torch.float16),
    )

    # ---- Build dummy inputs for layer + shell_post tracing ----
    B = 1
    d_model = config.d_model
    seq_len = args.seq_len

    dummy_hidden = torch.randn(B, seq_len, d_model, dtype=torch.float16)
    # Extended attention mask: 0 where attended, large negative where masked.
    # All-attended for tracing.
    dummy_mask = torch.zeros(B, 1, 1, seq_len, dtype=torch.float16)

    log(f"Dummy inputs: hidden_states {list(dummy_hidden.shape)}, "
        f"attention_mask {list(dummy_mask.shape)}")

    # ---- Step 3: Export shell_post (final layer norm) ----
    shell_post_path = out_dir / "shell_post.onnx"
    log(f"Exporting shell_post -> {shell_post_path}")
    t0 = time.time()
    shell_post = ShellPostWrapper(model.encoder.final_layer_norm)
    torch.onnx.export(
        shell_post,
        (dummy_hidden,),
        str(shell_post_path),
        opset_version=args.opset,
        input_names=["hidden_states"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
        dynamo=False,
    )
    log(f"  shell_post exported in {time.time() - t0:.1f}s "
        f"({shell_post_path.stat().st_size / 1e6:.2f} MB)")
    del shell_post
    gc.collect()
    log_ram(log, "after shell_post")

    # ---- Step 4: Per-layer export ----
    # Build kwargs for fresh UMT5Block instantiation. UMT5Block needs a config
    # and has_relative_attention_bias flag (True for all encoder layers in
    # untied UMT5).
    num_layers = config.num_layers

    if args.only_layers:
        layer_indices = [int(x) for x in args.only_layers.split(",")]
    else:
        layer_indices = list(range(num_layers))

    log(f"Per-layer export: {len(layer_indices)} layers (indices {layer_indices[:5]}{'...' if len(layer_indices) > 5 else ''})")

    # Free the meta-model's encoder to drop references to per-layer block
    # skeletons (we instantiate fresh UMT5Block instances below).
    del model
    gc.collect()
    log_ram(log, "after model freed")

    for i in layer_indices:
        t0 = time.time()
        prefix = f"encoder.block.{i}."

        # UMT5LayerSelfAttention hardcodes has_relative_attention_bias=True
        # so every block gets its own bias table (untied UMT5).
        block = UMT5Block(config, layer_idx=i)
        block.eval()

        # Load weights for just this block.
        state = {}
        for key, _shard in key_to_shard.items():
            if key.startswith(prefix):
                state[key[len(prefix):]] = load_tensor(key, dtype=torch.float16)
        missing, unexpected = block.load_state_dict(state, strict=False)
        if missing or unexpected:
            log(f"  WARN layer {i}: missing={missing}, unexpected={unexpected}")
        block = block.to(torch.float16)

        wrapper = LayerWrapper(block)
        layer_path = out_dir / f"layer_{i:02d}.onnx"

        torch.onnx.export(
            wrapper,
            (dummy_hidden, dummy_mask),
            str(layer_path),
            opset_version=args.opset,
            input_names=["hidden_states", "attention_mask"],
            output_names=["hidden_states_out"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 3: "seq_len"},
                "hidden_states_out": {0: "batch", 1: "seq_len"},
            },
            dynamo=False,
        )
        size_mb = layer_path.stat().st_size / 1e6
        # If layer is large enough to need external data, ONNX writes a sidecar
        # automatically when the proto exceeds 2 GB. UMT5-XXL layers are
        # ~330 MB each so they fit inline.
        dt = time.time() - t0
        log(f"  layer {i}/{num_layers - 1} -> {layer_path.name} ({size_mb:.1f} MB, {dt:.1f}s)")

        del wrapper, block, state
        gc.collect()

    log_ram(log, "after all layers")
    log(f"Peak RAM observed: {ram_peak[0]:.2f} GB")
    log("Done. Outputs:")
    log(f"  {out_dir}/embedding.bin")
    log(f"  {out_dir}/layer_00.onnx ... layer_{num_layers - 1:02d}.onnx")
    log(f"  {out_dir}/shell_post.onnx")
    log("Next: quantize layers to q4f16 with quantize-fastwan-blocks.py")
    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
