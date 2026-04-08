#!/usr/bin/env python3
"""Export ZImageTransformer2DModel to ONNX with clean layer structure.

Produces an fp16 ONNX file with external data layout, suitable for sharding
with shard-onnx-layers.py and subsequent q4f16 quantization per shard.

The PyTorch model's forward() takes nested lists of tensors (omni_mode etc.).
This script wraps it to accept flat tensors matching the ONNX runtime contract
that webnn/Z-Image-Turbo established:

  Inputs:
    hidden_states         [B, 16, 1, H/8, W/8]  float32
    timestep              [B]                     float32
    encoder_hidden_states [B, seq_len, 2560]      float32

  Output:
    unified_results       [16, 1, H/8, W/8]      float32

Uses accelerate disk offloading so the full ~23 GB model doesn't need to
fit in RAM at once. Peak memory is roughly the size of the largest single
layer plus export overhead.

Usage:
    uv run export-zimage-transformer.py <model_path> <output_dir>

    model_path: local path or HF model ID for Tongyi-MAI/Z-Image-Turbo
    output_dir: where to write the ONNX file + external data

Example:
    uv run export-zimage-transformer.py \
      ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/<hash> \
      ./output
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn


class ZImageTransformerWrapper(nn.Module):
    """Wraps ZImageTransformer2DModel to accept flat tensors for ONNX export.

    The original model.forward expects:
        x:         list of [C, F, H, W] tensors (one per batch item, no batch dim)
        t:         [B] timestep tensor
        cap_feats: list of [seq_len, hidden_dim] tensors (one per batch item)

    The ONNX contract expects:
        hidden_states:          [B, 16, 1, H/8, W/8]
        timestep:               [B]
        encoder_hidden_states:  [B, seq_len, 2560]

    Returns:
        unified_results:        [16, 1, H/8, W/8]  (squeezed from batch dim)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        B = hidden_states.shape[0]

        # Unbatch into lists (the model expects list-per-batch-item)
        x = [hidden_states[i] for i in range(B)]             # list of [16, 1, H, W]
        cap_feats = [encoder_hidden_states[i] for i in range(B)]  # list of [seq, 2560]

        result = self.model(
            x=x,
            t=timestep,
            cap_feats=cap_feats,
            return_dict=False,
        )

        # result is a tuple; first element is the output tensor or list of tensors
        output = result[0]
        if isinstance(output, list):
            output = torch.stack(output)
        if output.dim() == 5:
            # [B, C, F, H, W] -> squeeze batch
            output = output.squeeze(0)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", type=Path, help="Path to Tongyi-MAI/Z-Image-Turbo local checkout or HF model ID")
    parser.add_argument("output_dir", type=Path, help="Output directory for ONNX files")
    parser.add_argument("--height", type=int, default=512, help="Image height for dummy input (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Image width for dummy input (default: 512)")
    parser.add_argument("--seq-len", type=int, default=113, help="Sequence length for dummy input (default: 113, matches MS reference)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18)")
    parser.add_argument("--log", type=Path, default=None, help="Log file path (default: <output_dir>/export.log)")
    args = parser.parse_args()

    transformer_path = args.model_path / "transformer"
    if not transformer_path.exists():
        print(f"Error: transformer subfolder not found at {transformer_path}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "transformer_model.onnx"

    # Set up logging to both console and file
    log_path = args.log or (args.output_dir / "export.log")
    log_file = open(log_path, "w", buffering=1)  # line-buffered

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Export log: {log_path}")

    # ---- Load model with disk offloading ----
    # The full model is ~23 GB in fp16. Using accelerate's disk offloading
    # keeps only the active layer in RAM during the ONNX trace.
    log(f"Loading ZImageTransformer2DModel from {transformer_path}...")
    log("  (using accelerate disk offloading to limit RAM usage)")
    t0 = time.time()

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from diffusers import ZImageTransformer2DModel

    # First, get the config so we can build dummy inputs
    config = ZImageTransformer2DModel.load_config(str(transformer_path))

    # Init empty model skeleton (no RAM cost)
    with init_empty_weights():
        model = ZImageTransformer2DModel.from_config(config)

    # Dispatch with offloading - weights load from disk per-layer during forward
    offload_dir = tempfile.mkdtemp(prefix="zimage_offload_")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(transformer_path),
        device_map="auto",
        offload_folder=offload_dir,
        dtype=torch.float16,
        no_split_module_classes=["ZImageTransformerBlock"],
    )
    model.eval()
    log(f"  loaded in {time.time() - t0:.1f}s")

    wrapper = ZImageTransformerWrapper(model)

    # ---- Build dummy inputs ----
    B = 1
    C = config.get("in_channels", 16)
    F = 1   # temporal frames
    latent_h = args.height // 8
    latent_w = args.width // 8
    hidden_dim = config.get("cap_feat_dim", 2560)

    log(f"  dummy inputs: hidden_states [{B},{C},{F},{latent_h},{latent_w}], "
        f"timestep [{B}], encoder_hidden_states [{B},{args.seq_len},{hidden_dim}]")

    dummy_hidden = torch.randn(B, C, F, latent_h, latent_w, dtype=torch.float16)
    dummy_timestep = torch.tensor([0.5], dtype=torch.float16)
    dummy_enc_hidden = torch.randn(B, args.seq_len, hidden_dim, dtype=torch.float16)

    # ---- Verify wrapper forward pass ----
    log("  verifying forward pass...")
    t0 = time.time()
    with torch.no_grad():
        test_out = wrapper(dummy_hidden, dummy_timestep, dummy_enc_hidden)
    log(f"  forward pass OK: output shape {list(test_out.shape)}, took {time.time() - t0:.1f}s")

    # ---- Export to ONNX ----
    log(f"Exporting to ONNX (opset {args.opset})...")
    log(f"  output: {output_path}")
    t0 = time.time()

    torch.onnx.export(
        wrapper,
        (dummy_hidden, dummy_timestep, dummy_enc_hidden),
        str(output_path),
        opset_version=args.opset,
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["unified_results"],
        dynamic_axes={
            "hidden_states": {0: "batch", 3: "height", 4: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence_length"},
            "unified_results": {1: "frames", 2: "height", 3: "width"},
        },
    )
    export_time = time.time() - t0
    log(f"  export completed in {export_time:.1f}s")

    # ---- Convert to external data layout if large ----
    onnx_size = output_path.stat().st_size
    log(f"  ONNX file size: {onnx_size / 1e9:.2f} GB")

    if onnx_size > 2e9:
        log("File is >2 GB. Converting to external data layout...")
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data

        onnx_model = onnx.load(str(output_path))
        data_path = "transformer_model.onnx_data"
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

    # Clean up offload dir
    import shutil
    shutil.rmtree(offload_dir, ignore_errors=True)

    log("Done. Next steps:")
    log("  1. Shard:    python shard-onnx-layers.py <output.onnx> --max-shard-size 1.2GB")
    log("  2. Quantize: each shard independently to q4f16")
    log_file.close()


if __name__ == "__main__":
    main()
