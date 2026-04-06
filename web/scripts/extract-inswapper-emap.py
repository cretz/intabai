#!/usr/bin/env python3
"""Extract the embedding mapping matrix ("emap") from inswapper_128.onnx.

inswapper expects its source input to be an ArcFace embedding mapped through
this 512x512 matrix:

    source_input = (arcface_embedding @ emap) / norm(arcface_embedding)

We extract it once here so the browser doesn't need to parse the .onnx
protobuf at runtime - it just downloads the resulting raw float32 .bin
alongside the model.

Usage:
    pip install onnx numpy
    python extract-inswapper-emap.py <inswapper_128.onnx> <output.bin>

The inswapper_128.onnx file can be downloaded from (any of these mirrors):
  https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
  https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
"""
import sys
from pathlib import Path

import numpy
import onnx
from onnx import numpy_helper


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    onnx_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print(f"loading {onnx_path}...")
    model = onnx.load(str(onnx_path))

    # Standard inswapper names this initializer "emap". Fall back to the
    # last initializer in the graph if the name differs (some forks rename it).
    emap = None
    for init in model.graph.initializer:
        if init.name == "emap":
            emap = init
            break
    if emap is None:
        emap = model.graph.initializer[-1]
        print(f"  warning: no initializer named 'emap', falling back to last: {emap.name}")

    arr = numpy_helper.to_array(emap).astype(numpy.float32)
    print(f"  emap: name={emap.name!r} shape={arr.shape} dtype={arr.dtype}")

    if arr.shape != (512, 512):
        print(f"  warning: expected shape (512, 512), got {arr.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
