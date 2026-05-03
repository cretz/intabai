#!/usr/bin/env python3
"""Export PixArt-XL T5-XXL text encoder per-layer for browser layer-streaming.

Mirrors fastwan/export-fastwan-text-encoder.py. Each T5Block is exported as a
separate ONNX file so the browser can load+run+release one block at a time
(peak GPU memory = one block, not the whole 9.5 GB encoder).

Differences from the FastWan UMT5 path:
  - T5 (not UMT5): smaller vocab (32128), tied relative-attention-bias
    (only block 0 owns the weight; blocks 1-23 receive position_bias
    from block 0's output).
  - block_00.onnx returns (hidden_states, position_bias).
    block_NN.onnx for N >= 1 takes (hidden, attn_mask, position_bias)
    and returns hidden.
  - shell_post.onnx: final T5LayerNorm.
  - embedding.bin: raw fp16 [vocab, d_model] for JS-side token lookup
    (kept off the GPU; vocab table at fp16 is 263 MB).

Output layout (under <output_dir>/t5/):
  embedding.bin                  raw fp16 [32128, 4096]
  block_00.onnx                  (hidden, attn_mask) -> (hidden, position_bias)
  block_01.onnx ... block_23.onnx (hidden, attn_mask, position_bias) -> hidden
  shell_post.onnx                hidden -> last_hidden_state

Run:
  cd intabai/web/scripts && uv run python ltx/convert_t5_layers.py \\
    --src C:/work/personal/intabai/notes/models/ltx/source/pixart-text-encoder \\
    --out-dir C:/work/personal/intabai/notes/models/ltx/staging
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
import torch.nn as nn


def patch_t5_layernorm_for_onnx():
    """Force explicit fp32 variance in T5LayerNorm for ONNX export.

    Same fix the FastWan UMT5 export needed (see export-fastwan-text-encoder.py
    docstring): the tracer can fold the fp32 cast away and produce an fp16
    graph that overflows on T5's wide embedding values.
    """
    from transformers.models.t5.modeling_t5 import T5LayerNorm

    def forward(self, hidden_states):
        in_dtype = hidden_states.dtype
        x32 = hidden_states.to(torch.float32)
        variance = (x32 * x32).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.variance_epsilon)
        out = x32.to(in_dtype)
        return (self.weight * out).to(in_dtype)

    T5LayerNorm.forward = forward


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


class Block0Wrapper(nn.Module):
    """First T5 encoder block. Owns the relative-attention-bias weights;
    its output position_bias is reused by every later block."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask):
        # T5Block returns (hidden_states, present_key_value_state, position_bias).
        # present_kv is None at inference for encoder; position_bias is the
        # tensor we need to thread to subsequent blocks.
        seq_len = hidden_states.shape[1]
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        out = self.block(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=None,
            use_cache=False,
            cache_position=cache_position,
        )
        # T5Block (current transformers) with use_cache=False returns
        # (hidden_states, position_bias). Older versions returned a longer
        # tuple including present_key_value_state; we don't need that.
        return out[0], out[1]


class BlockNWrapper(nn.Module):
    """T5 encoder blocks 1..23. Take the position_bias from block 0."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask, position_bias):
        seq_len = hidden_states.shape[1]
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        out = self.block(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            use_cache=False,
            cache_position=cache_position,
        )
        return out[0]


class ShellPostWrapper(nn.Module):
    """Final T5LayerNorm."""

    def __init__(self, final_layer_norm):
        super().__init__()
        self.final_layer_norm = final_layer_norm

    def forward(self, hidden_states):
        return self.final_layer_norm(hidden_states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Source dir with text_encoder/ + tokenizer/ subfolders")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Staging dir; t5/ subdir created inside")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="Tracing sequence length (axis is dynamic)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--ram-budget-gb", type=float, default=14.0)
    ap.add_argument("--skip-embedding", action="store_true")
    ap.add_argument("--only-layers", type=str, default=None,
                    help="Comma list, e.g. '0,1,23'")
    args = ap.parse_args()

    text_encoder_path = args.src / "text_encoder"
    if not text_encoder_path.exists():
        print(f"Error: {text_encoder_path} not found", file=sys.stderr)
        sys.exit(1)
    out_dir = args.out_dir / "t5"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"RAM budget: {args.ram_budget_gb:.1f} GB")
    ram_peak = start_ram_watchdog(args.ram_budget_gb, log)

    patch_t5_layernorm_for_onnx()
    log("Patched T5LayerNorm.forward (fp32 variance cast)")

    from transformers import AutoConfig
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(str(text_encoder_path))
    log(f"  d_model={config.d_model}, layers={config.num_layers}, "
        f"heads={config.num_heads}, d_ff={config.d_ff}, vocab={config.vocab_size}")

    shard_files = sorted(text_encoder_path.glob("*.safetensors"))
    if not shard_files:
        log(f"  ERROR: no safetensors in {text_encoder_path}")
        sys.exit(1)

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

    # ---- Embedding -> raw fp16 bin --------------------------------------
    embedding_path = out_dir / "embedding.bin"
    if args.skip_embedding and embedding_path.exists():
        log(f"Skipping embedding (already at {embedding_path})")
    else:
        log("Extracting embedding -> embedding.bin")
        embed_key = None
        for k in ("shared.weight", "encoder.embed_tokens.weight", "embed_tokens.weight"):
            if k in key_to_shard:
                embed_key = k
                break
        if embed_key is None:
            log("  ERROR: could not find embedding tensor key")
            sys.exit(1)
        log(f"  using key: {embed_key}")
        embed = load_tensor(embed_key, dtype=torch.float16).contiguous()
        log(f"  shape={list(embed.shape)} dtype={embed.dtype} "
            f"size={embed.numel() * 2 / 1e6:.1f} MB")
        with open(embedding_path, "wb") as f:
            f.write(embed.cpu().numpy().tobytes())
        log(f"  wrote {embedding_path.stat().st_size / 1e6:.1f} MB")
        del embed
        gc.collect()
        log_ram(log, "after embedding")

    # ---- Build skeleton on meta, load just final_layer_norm -------------
    log("Building T5 skeleton on meta")
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import T5EncoderModel
    from transformers.models.t5.modeling_t5 import T5Block

    with init_empty_weights():
        model = T5EncoderModel(config)
    model.eval()

    final_norm_key = "encoder.final_layer_norm.weight"
    if final_norm_key not in key_to_shard:
        log(f"  ERROR: {final_norm_key} not found")
        sys.exit(1)
    set_module_tensor_to_device(
        model, final_norm_key, "cpu",
        value=load_tensor(final_norm_key, dtype=torch.float16),
    )

    B = 1
    d_model = config.d_model
    seq_len = args.seq_len
    num_heads = config.num_heads
    dummy_hidden = torch.randn(B, seq_len, d_model, dtype=torch.float16)
    # T5 extended attention mask shape: [B, 1, 1, seq_len]; 0 attended,
    # large-negative masked.
    dummy_mask = torch.zeros(B, 1, 1, seq_len, dtype=torch.float16)
    # position_bias for T5 is [B, num_heads, seq_len, seq_len] fp16.
    dummy_position_bias = torch.zeros(
        B, num_heads, seq_len, seq_len, dtype=torch.float16,
    )
    log(f"Dummy: hidden {list(dummy_hidden.shape)}, mask {list(dummy_mask.shape)}, "
        f"pos_bias {list(dummy_position_bias.shape)}")

    # ---- shell_post export ----------------------------------------------
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
    log(f"  shell_post in {time.time() - t0:.1f}s "
        f"({shell_post_path.stat().st_size / 1e6:.2f} MB)")
    del shell_post
    gc.collect()
    log_ram(log, "after shell_post")

    del model
    gc.collect()

    # ---- Per-layer export ------------------------------------------------
    num_layers = config.num_layers
    layer_indices = (
        [int(x) for x in args.only_layers.split(",")]
        if args.only_layers else list(range(num_layers))
    )
    log(f"Per-layer export: {len(layer_indices)} blocks")

    for i in layer_indices:
        t0 = time.time()
        prefix = f"encoder.block.{i}."
        is_first = i == 0

        # Block 0 has has_relative_attention_bias=True (owns the bias table).
        # Blocks 1+ have it False; we feed position_bias as input.
        block = T5Block(config, has_relative_attention_bias=is_first, layer_idx=i)
        block.eval()

        state = {key[len(prefix):]: load_tensor(key, dtype=torch.float16)
                 for key in key_to_shard if key.startswith(prefix)}
        missing, unexpected = block.load_state_dict(state, strict=False)
        if unexpected:
            log(f"  WARN block {i}: unexpected keys: {unexpected}")
        # Block 0 missing nothing; blocks 1+ are missing relative_attention_bias.weight
        # which is fine because has_relative_attention_bias=False has no such
        # parameter. Filter to confirm.
        unexplained = [m for m in missing if "relative_attention_bias" not in m]
        if unexplained:
            log(f"  WARN block {i}: missing keys: {unexplained}")
        block = block.to(torch.float16)

        if is_first:
            wrapper = Block0Wrapper(block)
            inputs = (dummy_hidden, dummy_mask)
            input_names = ["hidden_states", "attention_mask"]
            output_names = ["hidden_states_out", "position_bias"]
            dynamic_axes = {
                "hidden_states": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 3: "seq_len"},
                "hidden_states_out": {0: "batch", 1: "seq_len"},
                "position_bias": {0: "batch", 2: "seq_len", 3: "seq_len"},
            }
        else:
            wrapper = BlockNWrapper(block)
            inputs = (dummy_hidden, dummy_mask, dummy_position_bias)
            input_names = ["hidden_states", "attention_mask", "position_bias"]
            output_names = ["hidden_states_out"]
            dynamic_axes = {
                "hidden_states": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 3: "seq_len"},
                "position_bias": {0: "batch", 2: "seq_len", 3: "seq_len"},
                "hidden_states_out": {0: "batch", 1: "seq_len"},
            }

        layer_path = out_dir / f"block_{i:02d}.onnx"
        torch.onnx.export(
            wrapper,
            inputs,
            str(layer_path),
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        size_mb = layer_path.stat().st_size / 1e6
        log(f"  block {i}/{num_layers - 1} -> {layer_path.name} "
            f"({size_mb:.1f} MB, {time.time() - t0:.1f}s)")

        del wrapper, block, state
        gc.collect()

    log_ram(log, "after all layers")
    log(f"Peak RAM: {ram_peak[0]:.2f} GB")
    log("Done. Next: quantize blocks to q4f16.")


if __name__ == "__main__":
    main()
