"""Chain N Conv3Ds back-to-back in one graph to test cumulative-submit TDR.

If a single Conv3D at shape S runs clean but N of them in one session.run
TDRs, Chrome is hitting per-submit TDR rather than per-dispatch. That means
tiling the real VAE's Convs doesn't help: we'd need to force ORT-web to
submit work in smaller chunks.

All Convs are 512->512, kernel 3x3x3, pad=1 (shape-preserving so we can
chain trivially). Input [1, 512, 2, 60, 104] -> same. Per-conv FLOPs
= 512*512*27*2*60*104 = ~88 Gop, matching the Btile probe that ran clean
at 2.9s wall. Tested N = 1, 3, 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


IN_CH = 512
T = 2
H = 60
W = 104
KERNEL = [3, 3, 3]
PADS = [1, 1, 1, 1, 1, 1]


def build_chain(out_path: Path, n_convs: int) -> None:
    rng = np.random.default_rng(0)
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    for i in range(n_convs):
        w = rng.normal(0, 0.02, size=(IN_CH, IN_CH, *KERNEL)).astype(np.float16)
        b = np.zeros((IN_CH,), dtype=np.float16)
        inits.append(numpy_helper.from_array(w, name=f"W{i}"))
        inits.append(numpy_helper.from_array(b, name=f"B{i}"))
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, IN_CH, T, H, W])
    prev = "x"
    for i in range(n_convs):
        out_name = "y" if i == n_convs - 1 else f"h{i}"
        nodes.append(helper.make_node(
            "Conv",
            inputs=[prev, f"W{i}", f"B{i}"],
            outputs=[out_name],
            name=f"conv_{i}",
            kernel_shape=KERNEL,
            strides=[1, 1, 1],
            pads=PADS,
            dilations=[1, 1, 1],
            group=1,
        ))
        prev = out_name
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, IN_CH, T, H, W])
    graph = helper.make_graph(
        nodes=nodes,
        name=f"conv3d_chain_n{n_convs}",
        inputs=[x],
        outputs=[y],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, str(out_path))
    print(f"wrote {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB, n={n_convs})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in [1, 3, 5]:
        build_chain(args.out_dir / f"conv3d_chain_n{n}.onnx", n)


if __name__ == "__main__":
    sys.exit(main() or 0)
