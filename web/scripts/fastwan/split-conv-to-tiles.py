"""Split a single-Conv3D ONNX graph into N per-tile sub-graphs.

Each output tile becomes its own ONNX file containing {Slice, Conv}. The
caller runs them as separate session.runs (JS await between) and concats
outputs in JS. This is what actually dodges Windows D3D12 TDR: mapAsync at
session boundary drains the GPU queue, resetting the watchdog.

Input graph must contain exactly one Conv node with kernel=3x3x3, pad=0,
stride=1, group=1. Weights/bias are shared by name across every tile's
ONNX (same file copy in each).

Usage:
    uv run split-conv-to-tiles.py <conv.onnx> --n-tiles 6 \
        --input-shape name=1,1024,3,62,106
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--n-tiles", type=int, required=True)
    ap.add_argument(
        "--input-shape", action="append", default=[], required=True,
        help="name=d1,d2,... the Conv's input shape (5D)",
    )
    args = ap.parse_args()

    shape_overrides = {}
    for spec in args.input_shape:
        k, v = spec.split("=", 1)
        shape_overrides[k] = [int(x) for x in v.split(",")]

    log(f"Loading {args.input}")
    model = onnx.load(str(args.input), load_external_data=False)
    g = model.graph
    conv_nodes = [n for n in g.node if n.op_type == "Conv"]
    if len(conv_nodes) != 1:
        raise SystemExit(f"expected 1 Conv, found {len(conv_nodes)}")
    conv = conv_nodes[0]
    w_name = conv.input[1]
    b_name = conv.input[2] if len(conv.input) >= 3 else ""

    # Apply input-shape override and read H.
    if len(g.input) != 1:
        raise SystemExit(f"expected 1 graph input, got {len(g.input)}")
    gin = g.input[0]
    if gin.name not in shape_overrides:
        raise SystemExit(f"must pass --input-shape for {gin.name}")
    in_shape = shape_overrides[gin.name]
    if len(in_shape) != 5:
        raise SystemExit(f"expected 5D input, got {in_shape}")
    for d, v in zip(gin.type.tensor_type.shape.dim, in_shape):
        d.Clear()
        d.dim_value = v
    in_h = in_shape[3]
    out_h = in_h - 2
    n = args.n_tiles
    if out_h % 1 != 0 or out_h < n:
        raise SystemExit(f"out_h={out_h} incompatible with n={n}")

    # Per-tile output H (distribute remainder).
    base = out_h // n
    rem = out_h - base * n
    tile_out_hs = [base + (1 if i < rem else 0) for i in range(n)]
    assert sum(tile_out_hs) == out_h
    log(f"Tiling out_h={out_h} into {n}: {tile_out_hs}")

    # Collect initializers for weight/bias.
    init_map = {init.name: init for init in g.initializer}
    if w_name not in init_map:
        raise SystemExit(f"weight {w_name} not in initializers")
    w_init = init_map[w_name]
    b_init = init_map.get(b_name) if b_name else None

    # Conv output name (reused per tile, each sub-graph's single output).
    orig_out_name = conv.output[0]
    # Output dtype
    out_vi = None
    for vi in g.output:
        if vi.name == orig_out_name:
            out_vi = vi
            break
    if out_vi is None:
        raise SystemExit("conv output not in graph outputs")
    elem_type = out_vi.type.tensor_type.elem_type

    cursor = 0
    for i, toh in enumerate(tile_out_hs):
        slice_start = cursor
        slice_end = cursor + toh + 2
        cursor += toh

        axes = helper.make_tensor(f"tile{i}_axes", TensorProto.INT64, [1], [3])
        starts = helper.make_tensor(f"tile{i}_starts", TensorProto.INT64, [1], [slice_start])
        ends = helper.make_tensor(f"tile{i}_ends", TensorProto.INT64, [1], [slice_end])

        slice_out = f"tile{i}_sliced"
        conv_out = f"tile{i}_out"
        slice_node = helper.make_node(
            "Slice",
            inputs=[gin.name, starts.name, ends.name, axes.name],
            outputs=[slice_out],
            name=f"tile{i}_slice",
        )
        conv_inputs = [slice_out, w_name]
        if b_name:
            conv_inputs.append(b_name)
        conv_node_i = helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_out],
            name=f"tile{i}_conv",
            kernel_shape=[3, 3, 3],
            strides=[1, 1, 1],
            pads=[0, 0, 0, 0, 0, 0],
            dilations=[1, 1, 1],
            group=1,
        )

        out_shape_i = [in_shape[0], in_shape[1], in_shape[2] - 2, toh, in_shape[4] - 2]
        # Channels out from weight: first dim of W.
        out_channels = list(w_init.dims)[0]
        out_shape_i[1] = out_channels
        out_vi_i = helper.make_tensor_value_info(conv_out, elem_type, out_shape_i)

        in_vi_i = helper.make_tensor_value_info(gin.name, gin.type.tensor_type.elem_type, in_shape)

        kept_inits = [axes, starts, ends, w_init]
        if b_init is not None:
            kept_inits.append(b_init)

        sub_graph = helper.make_graph(
            nodes=[slice_node, conv_node_i],
            name=f"{g.name}_tile{i}",
            inputs=[in_vi_i],
            outputs=[out_vi_i],
            initializer=kept_inits,
        )
        sub_model = helper.make_model(
            sub_graph,
            opset_imports=list(model.opset_import),
            ir_version=model.ir_version,
        )
        onnx.checker.check_model(sub_model, full_check=False)

        tile_path = args.input.with_name(f"{args.input.stem}_tile{i}of{n}.onnx")
        onnx.save_model(sub_model, str(tile_path))
        log(
            f"  wrote {tile_path.name} ({tile_path.stat().st_size/1e6:.1f} MB): "
            f"slice[{slice_start}:{slice_end}] -> out H={toh}"
        )

    log("Done.")


if __name__ == "__main__":
    main()
