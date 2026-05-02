"""Numerical diff: shell_post.onnx vs PyTorch ShellPostWrapper."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ShellPostWrapper(nn.Module):
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
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import WanTransformer3DModel
    from safetensors import safe_open

    transformer_path = args.source_dir / "transformer"
    config = WanTransformer3DModel.load_config(str(transformer_path))

    with init_empty_weights():
        model = WanTransformer3DModel.from_config(config)
    ckpt = transformer_path / "diffusion_pytorch_model.safetensors"
    want = ("scale_shift_table", "norm_out", "proj_out")
    with safe_open(str(ckpt), framework="pt") as f:
        for key in f.keys():
            if any(key.startswith(w) or key == w for w in want):
                t = f.get_tensor(key).to(torch.float16)
                set_module_tensor_to_device(model, key, "cpu", value=t, dtype=torch.float16)
    model.eval()

    patch_size = tuple(config.get("patch_size", [1, 2, 2]))
    wrapper = ShellPostWrapper(model, patch_size)

    rng = np.random.default_rng(args.seed)
    inner_dim = config.get("attention_head_dim", 128) * config.get("num_attention_heads", 24)
    ppf, pph, ppw = 21, 15, 26
    seq_len = ppf * pph * ppw

    hidden = (rng.standard_normal((1, seq_len, inner_dim)) * 1.5).astype(np.float16)
    temb = (rng.standard_normal((1, seq_len, inner_dim)) * 0.5).astype(np.float16)

    inputs = {
        "hidden_states": hidden,
        "temb": temb,
        "ppf": np.array(ppf, dtype=np.int64),
        "pph": np.array(pph, dtype=np.int64),
        "ppw": np.array(ppw, dtype=np.int64),
    }

    print(f"[{time.strftime('%H:%M:%S')}] PyTorch forward...")
    t0 = time.time()
    with torch.inference_mode():
        pt_out = wrapper(
            torch.from_numpy(hidden),
            torch.from_numpy(temb),
            torch.tensor(ppf),
            torch.tensor(pph),
            torch.tensor(ppw),
        ).detach().cpu().numpy()
    print(f"  done in {time.time() - t0:.1f}s  shape {pt_out.shape}  "
          f"range [{pt_out.min():.3f}, {pt_out.max():.3f}]  nan={int(np.isnan(pt_out).sum())}")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(args.onnx_path), so, providers=["CPUExecutionProvider"])
    print(f"[{time.strftime('%H:%M:%S')}] ONNX CPU inference...")
    t0 = time.time()
    ox_out = sess.run(None, inputs)[0]
    print(f"  done in {time.time() - t0:.1f}s  shape {ox_out.shape}  "
          f"range [{ox_out.min():.3f}, {ox_out.max():.3f}]  nan={int(np.isnan(ox_out).sum())}")

    a = pt_out.astype(np.float32)
    b = ox_out.astype(np.float32)
    diff = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b)) + 1e-6
    print(f"\nmax_abs={diff.max():.4g}  mean_abs={diff.mean():.4g}  "
          f"rel_err={(diff / mag).mean():.4%}")
    if diff.max() < 0.5:
        print("PASS: shell_post export is faithful.")
    else:
        print("FAIL: shell_post export diverges.")
        sys.exit(2)


if __name__ == "__main__":
    main()
