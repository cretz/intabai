"""Build minimal Conv3D ONNX probes to isolate which op/shape TDRs the WebGPU
device inside the Wan VAE decoder.

Matches the real decoder's Conv3D convention (kernel=3x3x3, pad=0, explicit
upstream Pad). Each probe is a single Conv node; the input shape includes the
+2 H,W,T border the upstream Pad would have added.

All probes ship fp16 weights so the output is directly comparable to the real
graph's kernels. Writes to <out_dir>/conv3d_probe_{tag}.onnx.

The five shapes cover the heaviest Convs in the decoder step graph at T=2:

  A  up_blocks.1  [1,1024,2,62,106]  -> [1,1024,2,60,104]   1024->1024
  B  up_blocks.2t [1,1024,2,122,210] -> [1, 512,2,120,208]  1024->512 transition
  C  up_blocks.2  [1, 512,2,122,210] -> [1, 512,2,120,208]   512->512
  D  up_blocks.3t [1, 512,2,242,418] -> [1, 256,2,240,416]   512->256 transition
  E  up_blocks.3  [1, 256,2,242,418] -> [1, 256,2,240,416]   256->256
  F  conv_out     [1, 256,2,242,418] -> [1,   3,2,240,416]   256->3

(E was already probed as "untiled" in the earlier run — ran clean at 5.6s.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# (tag, in_ch, out_ch, T_in, H_in, W_in). T_in=4 so output T=2 matches step.
PROBES: list[tuple[str, int, int, int, int, int]] = [
    ("Btile_1024to512_15x208", 1024, 512, 4, 17, 210),
    ("G_1024to1024_30x52",  1024, 1024, 4,  32,  54),
    ("A_1024to1024_60x104", 1024, 1024, 4,  62, 106),
    ("B_1024to512_120x208", 1024,  512, 4, 122, 210),
    ("C_512to512_120x208",   512,  512, 4, 122, 210),
    ("D_512to256_240x416",   512,  256, 4, 242, 418),
    ("F_256to3_240x416",     256,    3, 4, 242, 418),
]

KERNEL = [3, 3, 3]
PADS = [0, 0, 0, 0, 0, 0]


def build_one(out_path: Path, in_ch: int, out_ch: int, T: int, H: int, W: int) -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, size=(out_ch, in_ch, *KERNEL)).astype(np.float16)
    b = np.zeros((out_ch,), dtype=np.float16)
    w_init = numpy_helper.from_array(w, name="W")
    b_init = numpy_helper.from_array(b, name="B")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, in_ch, T, H, W])
    out_T = T - 2
    out_H = H - 2
    out_W = W - 2
    y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT16, [1, out_ch, out_T, out_H, out_W]
    )
    conv = helper.make_node(
        "Conv",
        inputs=["x", "W", "B"],
        outputs=["y"],
        kernel_shape=KERNEL,
        strides=[1, 1, 1],
        pads=PADS,
        dilations=[1, 1, 1],
        group=1,
    )
    graph = helper.make_graph(
        nodes=[conv],
        name=f"conv3d_probe_{in_ch}to{out_ch}",
        inputs=[x],
        outputs=[y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, str(out_path))
    weight_mb = (w.nbytes + b.nbytes) / 1e6
    print(
        f"wrote {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB, "
        f"weights {weight_mb:.1f} MB, in={[1,in_ch,T,H,W]}, out={[1,out_ch,out_T,out_H,out_W]})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, in_ch, out_ch, T, H, W in PROBES:
        build_one(args.out_dir / f"conv3d_probe_{tag}.onnx", in_ch, out_ch, T, H, W)


if __name__ == "__main__":
    sys.exit(main() or 0)
