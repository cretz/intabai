"""Numerical diff: block_NN.onnx vs PyTorch WanTransformerBlock.

Loads a fresh WanTransformerBlock with the same weights the export script
uses (blocks.{i}.* from the safetensors), wraps in BlockWrapper, and
compares against CPU onnxruntime inference of block_NN.onnx on identical
inputs.

Usage:
    uv run check-block-vs-pytorch.py \\
        ../../../notes/models/fastwan/source \\
        ../../../notes/models/fastwan/hf-repo/onnx/transformer \\
        --block 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin):
        return self.block(hidden_states, encoder_hidden_states, temb, (freqs_cos, freqs_sin))


def patch_rms_norm_for_onnx():
    """Same decomposition the export script applies."""
    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight.float()
        return x.to(in_dtype)
    torch.nn.RMSNorm.forward = forward


def load_block(source_dir: Path, block_idx: int):
    from diffusers import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock
    from safetensors import safe_open

    transformer_path = source_dir / "transformer"
    config = WanTransformer3DModel.load_config(str(transformer_path))

    block_kwargs = dict(
        dim=config.get("attention_head_dim", 128) * config.get("num_attention_heads", 24),
        ffn_dim=config.get("ffn_dim", 14336),
        num_heads=config.get("num_attention_heads", 24),
        qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6),
        added_kv_proj_dim=config.get("added_kv_proj_dim", None),
    )
    block = WanTransformerBlock(**block_kwargs)

    prefix = f"blocks.{block_idx}."
    state = {}
    ckpt = transformer_path / "diffusion_pytorch_model.safetensors"
    with safe_open(str(ckpt), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key).to(torch.float16)
    missing, unexpected = block.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN: missing={missing} unexpected={unexpected}", file=sys.stderr)
    block = block.to(torch.float16).eval()
    return BlockWrapper(block), config


def build_inputs(config, height=480, width=832, num_frames=81, text_seq_len=512, seed=1234):
    rng = np.random.default_rng(seed)
    inner_dim = config.get("attention_head_dim", 128) * config.get("num_attention_heads", 24)
    T = (num_frames - 1) // 4 + 1
    H = height // 16
    W = width // 16
    seq_len = T * (H // 2) * (W // 2)

    # Realistic post-shell_pre magnitudes:
    # hidden_states come out of patch_embedding (small), text comes out of
    # text_embedder Linear (~unit), timestep_proj is ~N(0, 1), freqs are [-1,1].
    hidden = (rng.standard_normal((1, seq_len, inner_dim)) * 0.5).astype(np.float16)
    enc = (rng.standard_normal((1, text_seq_len, inner_dim)) * 0.5).astype(np.float16)
    timestep_proj = (rng.standard_normal((1, seq_len, 6, inner_dim)) * 0.5).astype(np.float16)

    # Freqs are fp32 per the ONNX contract.
    head_dim = config.get("attention_head_dim", 128)
    angles = rng.uniform(-np.pi, np.pi, size=(1, seq_len, 1, head_dim)).astype(np.float32)
    freqs_cos = np.cos(angles).astype(np.float32)
    freqs_sin = np.sin(angles).astype(np.float32)

    return {
        "hidden_states": hidden,
        "encoder_hidden_states": enc,
        "timestep_proj": timestep_proj,
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
    }


def run_pytorch(wrapper, inputs):
    with torch.inference_mode():
        out = wrapper(
            torch.from_numpy(inputs["hidden_states"]),
            torch.from_numpy(inputs["encoder_hidden_states"]),
            torch.from_numpy(inputs["timestep_proj"]),
            torch.from_numpy(inputs["freqs_cos"]),
            torch.from_numpy(inputs["freqs_sin"]),
        )
    return {"hidden_states_out": out.detach().cpu().numpy()}


def run_onnx(onnx_path: Path, inputs):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {k: v for k, v in inputs.items()})
    return dict(zip(out_names, outs))


def compare(pt, ox):
    print(f"{'output':<24} {'shape':<28} {'max_abs':>10} {'mean_abs':>10} "
          f"{'rel_err':>10} {'nan_ox':>8}")
    print("-" * 96)
    worst = 0.0
    for k in pt:
        a = pt[k].astype(np.float32)
        b = ox[k].astype(np.float32)
        if a.shape != b.shape:
            print(f"{k:<24} SHAPE MISMATCH pt={a.shape} onnx={b.shape}")
            continue
        diff = np.abs(a - b)
        mag = np.maximum(np.abs(a), np.abs(b)) + 1e-6
        nan_ox = int(np.isnan(b).sum())
        print(f"{k:<24} {str(a.shape):<28} {diff.max():>10.4g} "
              f"{diff.mean():>10.4g} {(diff / mag).mean():>10.4%} {nan_ox:>8}")
        worst = max(worst, diff.max())
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("transformer_dir", type=Path, help="Directory with block_NN.onnx files")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    onnx_path = args.transformer_dir / f"block_{args.block:02d}.onnx"
    if not onnx_path.exists():
        print(f"ONNX not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    patch_rms_norm_for_onnx()
    print(f"[{time.strftime('%H:%M:%S')}] Loading block {args.block}...")
    t0 = time.time()
    wrapper, config = load_block(args.source_dir, args.block)
    print(f"  loaded in {time.time() - t0:.1f}s")

    inputs = build_inputs(config, seed=args.seed)
    print(f"  hidden {inputs['hidden_states'].shape}  "
          f"enc {inputs['encoder_hidden_states'].shape}  "
          f"timestep_proj {inputs['timestep_proj'].shape}  "
          f"freqs {inputs['freqs_cos'].shape}")

    print(f"[{time.strftime('%H:%M:%S')}] PyTorch forward...")
    t0 = time.time()
    pt = run_pytorch(wrapper, inputs)
    print(f"  done in {time.time() - t0:.1f}s  "
          f"out range [{pt['hidden_states_out'].min():.3f}, {pt['hidden_states_out'].max():.3f}]  "
          f"nan={int(np.isnan(pt['hidden_states_out']).sum())}")

    print(f"[{time.strftime('%H:%M:%S')}] ONNX CPU inference...")
    t0 = time.time()
    ox = run_onnx(onnx_path, inputs)
    print(f"  done in {time.time() - t0:.1f}s  "
          f"out range [{ox['hidden_states_out'].min():.3f}, {ox['hidden_states_out'].max():.3f}]  "
          f"nan={int(np.isnan(ox['hidden_states_out']).sum())}")

    print()
    worst = compare(pt, ox)
    print()
    if worst < 0.5:
        print(f"PASS (fp16 tolerance): worst max_abs={worst:.4g}. "
              f"block_{args.block:02d} export is faithful.")
    else:
        print(f"FAIL: worst max_abs={worst:.4g}. Export drift or tracer bug in block "
              f"{args.block:02d}.")
        sys.exit(2)


if __name__ == "__main__":
    main()
