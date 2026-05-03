#!/usr/bin/env python3
"""Convert LTX 2B distilled transformer to ONNX via the dynamo exporter.

Inputs (pipeline does patchify + indices_grid in JS):
  hidden_states          (1, N_tokens, 128)    fp16
  indices_grid           (1, 3, N_tokens)      float32  (RoPE positions)
  encoder_hidden_states  (1, N_text, 4096)     fp16     (T5-XXL)
  encoder_attention_mask (1, N_text)           int64
  timestep               (1,)                  fp16

Batch is concrete=1 (we never run B>1 in browser); making it symbolic only
adds constraint-solver work for no gain. Two free Dims: n_tokens, n_text.

Run:
  cd intabai/web/scripts && uv run python ltx/convert_transformer.py \
    --space-path /abs/path/to/ltx-video-distilled-space \
    --ckpt /abs/path/to/ltxv-2b-0.9.8-distilled.safetensors \
    --out-dir /abs/path/to/notes/models/ltx/onnx-staging
"""
import argparse
import sys
from pathlib import Path

# torch.onnx dynamo logs ✅/❌ chars; force UTF-8 so Windows cp1252 doesn't crash mid-export.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.export import Dim


class TransformerWrapper(torch.nn.Module):
    """Adapts Transformer3DModel.forward to flat tensor I/O for ONNX."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        indices_grid,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
    ):
        out = self.transformer(
            hidden_states=hidden_states,
            indices_grid=indices_grid,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            return_dict=False,
        )
        return out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.space_path))
    from ltx_video.models.transformers.transformer3d import Transformer3DModel

    print(f"Loading transformer from {args.ckpt} ...")
    transformer = Transformer3DModel.from_pretrained(str(args.ckpt))
    print(f"  loaded dtype: {next(transformer.parameters()).dtype}; casting to fp16")
    transformer = transformer.to(torch.float16)
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    wrapper = TransformerWrapper(transformer).eval()

    # Representative shapes for tracing. Pick values comfortably above any
    # likely constant the model might bake (avoids n_tokens collapsing to a
    # small constant during shape inference).
    N, N_text = 128, 64
    hidden_states = torch.randn(1, N, 128, dtype=torch.float16)
    indices_grid = torch.zeros(1, 3, N, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, N_text, 4096, dtype=torch.float16)
    encoder_attention_mask = torch.ones(1, N_text, dtype=torch.int64)
    timestep = torch.tensor([500.0], dtype=torch.float16)

    n_tokens = Dim("n_tokens", min=8, max=200000)
    n_text = Dim("n_text", min=8, max=512)

    dynamic_shapes = {
        "hidden_states": {1: n_tokens},
        "indices_grid": {2: n_tokens},
        "encoder_hidden_states": {1: n_text},
        "encoder_attention_mask": {1: n_text},
        "timestep": {},
    }

    onnx_path = args.out_dir / "transformer.onnx"
    print(f"Exporting (dynamo) to {onnx_path} ...")
    torch.onnx.export(
        wrapper,
        (hidden_states, indices_grid, encoder_hidden_states, encoder_attention_mask, timestep),
        str(onnx_path),
        input_names=[
            "hidden_states",
            "indices_grid",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "timestep",
        ],
        output_names=["sample"],
        dynamic_shapes=dynamic_shapes,
        opset_version=args.opset,
        dynamo=True,
        external_data=True,
    )
    size = onnx_path.stat().st_size
    # Dynamo writes external data as sibling files; sum non-.onnx entries.
    ext_total = sum(
        p.stat().st_size for p in args.out_dir.iterdir()
        if p.is_file() and p.suffix != ".onnx"
    )
    print(f"OK\n  {onnx_path}: {size/1e6:.1f} MB graph")
    print(f"  external data total: {ext_total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
