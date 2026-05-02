"""Numerical diff: shell_pre.onnx vs PyTorch ShellPreWrapper.

Loads the same diffusers WanTransformer3DModel shell weights the export
script uses, wraps them in the same ShellPreWrapper, and compares every
output of a fresh PyTorch forward against a CPU onnxruntime inference of
shell_pre.onnx on identical inputs.

If outputs match within fp16 tolerance: the export is faithful; bug lives
in the JS runtime or per-block exports.

If outputs diverge: the bug is in the export (ShellPreWrapper, RoPE
materialization, condition_embedder path, or torch.onnx tracing).

Usage:
    uv run check-shell-pre-vs-pytorch.py \\
        ../../../notes/models/fastwan/source \\
        ../../../notes/models/fastwan/hf-repo/onnx/transformer/shell_pre.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ShellPreWrapper(nn.Module):
    """Copy of ShellPreWrapper from export-fastwan-transformer.py.

    Kept byte-identical so any divergence from ONNX is a tracer/export bug,
    not a wrapper-drift bug.
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


def load_pytorch_shell(source_dir: Path):
    """Mirror the export script's shell-only load path."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    from safetensors import safe_open

    transformer_path = source_dir / "transformer"
    config = WanTransformer3DModel.load_config(str(transformer_path))

    with init_empty_weights():
        model = WanTransformer3DModel.from_config(config)

    ckpt = transformer_path / "diffusion_pytorch_model.safetensors"
    with safe_open(str(ckpt), framework="pt") as f:
        for key in f.keys():
            if key.startswith("blocks."):
                continue
            tensor = f.get_tensor(key).to(torch.float16)
            set_module_tensor_to_device(model, key, "cpu", value=tensor, dtype=torch.float16)
    model.eval()

    # Re-instantiate rope outside init_empty_weights so persistent=False
    # buffers actually get materialized. Same fix as the export script.
    model.rope = WanRotaryPosEmbed(
        attention_head_dim=config.get("attention_head_dim", 128),
        patch_size=tuple(config.get("patch_size", [1, 2, 2])),
        max_seq_len=config.get("rope_max_seq_len", 1024),
    )
    assert model.rope.freqs_cos.device.type != "meta"
    assert model.rope.freqs_sin.device.type != "meta"

    return ShellPreWrapper(model), config


def build_inputs(config, height=480, width=832, num_frames=81, text_seq_len=512, seed=1234):
    rng = np.random.default_rng(seed)
    C = config.get("in_channels", 48)
    text_dim = config.get("text_dim", 4096)
    T = (num_frames - 1) // 4 + 1
    H = height // 16
    W = width // 16
    seq_len = T * (H // 2) * (W // 2)  # patch_size [1, 2, 2]

    # Match realistic distribution: latent ~N(0,1), text_embeds ~N(0, ~5)
    # (post-scheduler init_noise_sigma=1 for flow matching; text embeds
    # come out of UMT5 with larger magnitude).
    latent = rng.standard_normal((1, C, T, H, W)).astype(np.float16)
    text = (rng.standard_normal((1, text_seq_len, text_dim)) * 3.0).astype(np.float16)

    timestep_scalar = 757  # mid DMD step
    timestep = np.full((1, seq_len), timestep_scalar, dtype=np.int64)
    return {
        "hidden_states": latent,
        "timestep": timestep,
        "encoder_hidden_states": text,
    }, seq_len


def run_pytorch(wrapper, inputs):
    with torch.inference_mode():
        out = wrapper(
            torch.from_numpy(inputs["hidden_states"]),
            torch.from_numpy(inputs["timestep"]),
            torch.from_numpy(inputs["encoder_hidden_states"]),
        )
    # Wrapper returns: hidden, enc_hidden, timestep_proj, temb, freqs_cos, freqs_sin.
    # ONNX output_names: tokens, enc_proj, timestep_proj, temb, freqs_cos, freqs_sin.
    names = ["tokens", "enc_proj", "timestep_proj", "temb", "freqs_cos", "freqs_sin"]
    return {n: t.detach().cpu().numpy() for n, t in zip(names, out)}


def run_onnx(onnx_path: Path, inputs):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    feeds = {
        "hidden_states": inputs["hidden_states"],
        "timestep": inputs["timestep"],
        "encoder_hidden_states": inputs["encoder_hidden_states"],
    }
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    return dict(zip(out_names, outs))


def compare(pt: dict, ox: dict):
    keys = ["tokens", "enc_proj", "timestep_proj", "temb", "freqs_cos", "freqs_sin"]
    print(f"{'output':<16} {'shape':<28} {'dtype_pt':<10} {'dtype_ox':<10} "
          f"{'max_abs':>10} {'mean_abs':>10} {'rel_err':>10}")
    print("-" * 104)
    worst = 0.0
    for k in keys:
        a = pt[k].astype(np.float32)
        b = ox[k].astype(np.float32)
        if a.shape != b.shape:
            print(f"{k:<16} SHAPE MISMATCH pt={a.shape} onnx={b.shape}")
            continue
        diff = np.abs(a - b)
        mag = np.maximum(np.abs(a), np.abs(b)) + 1e-6
        max_abs = diff.max()
        mean_abs = diff.mean()
        rel = (diff / mag).mean()
        print(f"{k:<16} {str(a.shape):<28} {str(pt[k].dtype):<10} "
              f"{str(ox[k].dtype):<10} {max_abs:>10.4g} {mean_abs:>10.4g} {rel:>10.4%}")
        worst = max(worst, max_abs)
    print("-" * 104)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path,
                    help="FastWan2.2-TI2V-5B-FullAttn-Diffusers checkout root (has transformer/)")
    ap.add_argument("onnx_path", type=Path, help="shell_pre.onnx path")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    args = ap.parse_args()

    if not args.onnx_path.exists():
        print(f"ONNX not found: {args.onnx_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] Loading PyTorch shell...")
    t0 = time.time()
    wrapper, config = load_pytorch_shell(args.source_dir)
    print(f"  loaded in {time.time() - t0:.1f}s")

    inputs, seq_len = build_inputs(
        config, args.height, args.width, args.num_frames, seed=args.seed
    )
    print(f"  latent {inputs['hidden_states'].shape}  "
          f"text {inputs['encoder_hidden_states'].shape}  "
          f"timestep {inputs['timestep'].shape}  seq_len={seq_len}")

    print(f"[{time.strftime('%H:%M:%S')}] Running PyTorch forward...")
    t0 = time.time()
    pt = run_pytorch(wrapper, inputs)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Running ONNX inference (CPU)...")
    t0 = time.time()
    ox = run_onnx(args.onnx_path, inputs)
    print(f"  done in {time.time() - t0:.1f}s")

    print()
    worst = compare(pt, ox)
    print()
    if worst < 1e-2:
        print(f"PASS: outputs match within fp16 tolerance (worst max_abs={worst:.4g}).")
        print("Bug is NOT in shell_pre export. Look at per-block exports or JS runtime.")
        sys.exit(0)
    elif worst < 1.0:
        print(f"SOFT MISMATCH: worst max_abs={worst:.4g}. Possibly fp16 drift, "
              f"possibly real. Inspect per-output numbers above.")
        sys.exit(0)
    else:
        print(f"FAIL: worst max_abs={worst:.4g}. shell_pre export is diverging "
              f"from PyTorch reference.")
        sys.exit(2)


if __name__ == "__main__":
    main()
