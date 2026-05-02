#!/usr/bin/env python3
"""Enumerate Conv nodes in an ONNX graph that operate on 5D tensors (Conv3D).
Reports shape/kernel/stride/pad/dilation/groups so we can confirm they're all
the same 3x3x3-pad1-stride1 case before writing the H-axis tiling patch.

Usage:
    uv run analyze-vae-conv3d.py <onnx_path>
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import onnx


def attr_val(attr):
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    if attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode()
    return f"<type {attr.type}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", type=Path)
    args = ap.parse_args()

    model = onnx.load(str(args.onnx_path), load_external_data=False)
    g = model.graph

    # Build initializer shape lookup (for weight W to determine kernel).
    init_shape = {}
    for init in g.initializer:
        init_shape[init.name] = list(init.dims)

    # Shape inference on tensors for input spatial dims.
    inferred = None
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as e:
        print(f"shape_inference failed: {e}", file=sys.stderr)

    value_shape = {}
    if inferred is not None:
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
            t = vi.type.tensor_type
            shape = []
            for d in t.shape.dim:
                if d.dim_value:
                    shape.append(d.dim_value)
                elif d.dim_param:
                    shape.append(d.dim_param)
                else:
                    shape.append("?")
            value_shape[vi.name] = shape

    conv_nodes = []
    for node in g.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2:
            continue
        w_name = node.input[1]
        w_shape = init_shape.get(w_name)
        # Conv3D weight shape: [out_ch, in_ch/groups, kT, kH, kW] -> 5 dims
        if w_shape is None or len(w_shape) != 5:
            continue
        conv_nodes.append((node, w_shape))

    print(f"Found {len(conv_nodes)} Conv3D nodes")
    print()

    buckets = Counter()
    big_spatial_count = 0
    for node, w_shape in conv_nodes:
        attrs = {a.name: attr_val(a) for a in node.attribute}
        kernel = attrs.get("kernel_shape", w_shape[2:])
        strides = attrs.get("strides", [1, 1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0, 0, 0])
        dilations = attrs.get("dilations", [1, 1, 1])
        group = attrs.get("group", 1)
        in_shape = value_shape.get(node.input[0], ["?"] * 5)
        out_shape = value_shape.get(node.output[0], ["?"] * 5)
        # w_shape: [Cout, Cin/g, kT, kH, kW]
        cout, cin_per_g, *_ = w_shape
        sig = (
            tuple(kernel),
            tuple(strides),
            tuple(pads),
            tuple(dilations),
            group,
        )
        buckets[sig] += 1
        # Flag spatially big ones (H or W >= 120).
        try:
            h = in_shape[3] if isinstance(in_shape[3], int) else 0
            w = in_shape[4] if isinstance(in_shape[4], int) else 0
            if h >= 120 or w >= 120:
                big_spatial_count += 1
        except Exception:
            pass

    print("Per-(kernel, strides, pads, dilations, group) bucket counts:")
    for sig, n in buckets.most_common():
        kernel, strides, pads, dilations, group = sig
        print(
            f"  n={n:3d}  kernel={list(kernel)} strides={list(strides)} "
            f"pads={list(pads)} dilations={list(dilations)} group={group}"
        )
    print()
    print(f"Conv3D nodes with in-H>=120 or in-W>=120: {big_spatial_count}")

    print()
    print("Detail (input->output shapes and weight shape):")
    for i, (node, w_shape) in enumerate(conv_nodes):
        attrs = {a.name: attr_val(a) for a in node.attribute}
        in_shape = value_shape.get(node.input[0], ["?"] * 5)
        out_shape = value_shape.get(node.output[0], ["?"] * 5)
        pads = attrs.get("pads", [0, 0, 0, 0, 0, 0])
        strides = attrs.get("strides", [1, 1, 1])
        print(
            f"  [{i:3d}] {node.name or '<unnamed>'}"
        )
        print(
            f"         in={in_shape} out={out_shape} W={w_shape} "
            f"strides={strides} pads={pads}"
        )


if __name__ == "__main__":
    main()
