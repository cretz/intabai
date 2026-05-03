"""Export LTX spatial latent upscaler (482 MB safetensors) to ONNX fp16.

LatentUpsampler config: in_channels=128, mid_channels=512,
num_blocks_per_stage=4, dims=3, spatial_upsample=True. Forward shape:
  in:  [1, 128, T, H, W]
  out: [1, 128, T, 2H, 2W]

Doubles spatial only (temporal_upsample=False). Internal middle layer
rearranges to 2D for the upsampler conv+pixelshuffle, which dynamo
handles via Reshape ops -- no manual surgery needed.

Output: notes/models/ltx/hf-repo/onnx/upscaler.onnx (+ .onnx.data).
Monolithic; ~500 MB total, well under the wasm32 4 GB cap so no shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]  # web/scripts/ltx/file -> intabai/
SPACE = ROOT / "notes" / "ltx-video-distilled-space"
sys.path.insert(0, str(SPACE))

from ltx_video.models.autoencoders.latent_upsampler import (  # noqa: E402
    LatentUpsampler,
)

SRC = ROOT / "notes" / "models" / "ltx" / "source" / "ltxv-spatial-upscaler-0.9.8.safetensors"
OUT_DIR = ROOT / "notes" / "models" / "ltx" / "hf-repo" / "onnx"
OUT = OUT_DIR / "upscaler.onnx"


def main() -> None:
    print(f"loading upscaler from {SRC}")
    model = LatentUpsampler.from_pretrained(str(SRC))
    model = model.eval().to(torch.float16)
    print(
        f"  in_channels={model.in_channels} mid_channels={model.mid_channels} "
        f"num_blocks_per_stage={model.num_blocks_per_stage} dims={model.dims}"
    )

    # Sanity forward pass.
    x = torch.randn(1, 128, 2, 8, 8, dtype=torch.float16)
    with torch.no_grad():
        y = model(x)
    print(f"sanity forward: {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape == (1, 128, 2, 16, 16), y.shape

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"exporting to {OUT}")

    T = torch.export.Dim("T", min=2, max=64)
    H = torch.export.Dim("H", min=4, max=256)
    W = torch.export.Dim("W", min=4, max=256)
    dynamic_shapes = {
        "latent": {0: None, 1: None, 2: T, 3: H, 4: W},
    }

    torch.onnx.export(
        model,
        (x,),
        str(OUT),
        input_names=["latent"],
        output_names=["upsampled"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=True,
        opset_version=20,
    )
    print("done")


if __name__ == "__main__":
    main()
