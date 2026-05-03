#!/usr/bin/env python3
"""Convert PixArt-XL's T5-XXL text encoder to ONNX via the dynamo exporter.

Inputs:
  input_ids       (1, N_text)  int64
  attention_mask  (1, N_text)  int64

Output:
  last_hidden_state (1, N_text, 4096) fp16 -- feeds transformer's
  encoder_hidden_states.

B is concrete=1 (browser only encodes one prompt at a time). One free Dim:
n_text (max 512, the LTX pipeline truncates/pads to <=256 in practice).

Run:
  cd intabai/web/scripts && uv run python ltx/convert_t5.py \\
    --src C:/work/personal/intabai/notes/models/ltx/source/pixart-text-encoder \\
    --out-dir C:/work/personal/intabai/notes/models/ltx/staging/t5
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.export import Dim
from transformers import T5EncoderModel


class T5Wrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="dir with text_encoder/ + tokenizer/ subfolders")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading T5 encoder from {args.src}/text_encoder ...")
    encoder = T5EncoderModel.from_pretrained(
        str(args.src / "text_encoder"),
        torch_dtype=torch.float16,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  loaded: {n_params/1e9:.2f}B params, dtype={next(encoder.parameters()).dtype}")

    wrapper = T5Wrapper(encoder).eval()

    N_text = 64
    input_ids = torch.zeros(1, N_text, dtype=torch.int64)
    attention_mask = torch.ones(1, N_text, dtype=torch.int64)

    n_text = Dim("n_text", min=8, max=512)
    dynamic_shapes = {
        "input_ids": {1: n_text},
        "attention_mask": {1: n_text},
    }

    onnx_path = args.out_dir / "t5.onnx"
    print(f"Exporting (dynamo) to {onnx_path} ...")
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_shapes=dynamic_shapes,
        opset_version=args.opset,
        dynamo=True,
        external_data=True,
    )
    size = onnx_path.stat().st_size
    ext_total = sum(
        p.stat().st_size for p in args.out_dir.iterdir()
        if p.is_file() and p.suffix != ".onnx"
    )
    print(f"OK\n  {onnx_path}: {size/1e6:.1f} MB graph")
    print(f"  external data total: {ext_total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
