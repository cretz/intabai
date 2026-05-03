#!/usr/bin/env python3
"""Run onnx.checker + shape inference on the LTX transformer to find
the node ORT is rejecting for rank mismatch.

Run:
  cd intabai/web/scripts && uv run python ltx/check_onnx.py \
    --onnx /abs/path/to/staging/transformer.onnx
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import onnx
import onnx.shape_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    args = ap.parse_args()

    # Load graph only for header/IO inspection (full load would copy
    # 3.85 GB of weights into RAM and re-serialization for the checker
    # would hit protobuf's 2 GB limit anyway).
    print(f"Loading graph (no external data) from {args.onnx} ...")
    model = onnx.load(str(args.onnx), load_external_data=False)
    print(f"  ir_version={model.ir_version} producer={model.producer_name}/{model.producer_version}")
    print(f"  opset domains: {[(o.domain, o.version) for o in model.opset_import]}")
    print(f"  inputs:")
    for vi in model.graph.input:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else str(d.dim_value))
        print(f"    {vi.name}: rank={len(dims)} shape=[{','.join(dims)}] elem={vi.type.tensor_type.elem_type}")
    print(f"  outputs:")
    for vi in model.graph.output:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else str(d.dim_value))
        print(f"    {vi.name}: rank={len(dims)} shape=[{','.join(dims)}] elem={vi.type.tensor_type.elem_type}")
    print(f"  nodes: {len(model.graph.node)}")

    print("\n--- Bad Squeeze node sample (named node_squeeze*) ---")
    vi_by_name = {vi.name: vi for vi in model.graph.value_info}
    init_by_name = {t.name: t for t in model.graph.initializer}
    target_names = {"node_squeeze"} | {f"node_squeeze_{i}" for i in range(0, 5)}
    for i, n in enumerate(model.graph.node):
        if n.name not in target_names:
            continue
        print(f"  [{i}] {n.name} op={n.op_type}")
        print(f"      inputs: {list(n.input)}")
        print(f"      outputs: {list(n.output)}")
        for attr in n.attribute:
            print(f"      attr {attr.name}: type={attr.type} val={attr.i if attr.type==2 else list(attr.ints) if attr.type==7 else attr.s}")
        for inp in n.input:
            if inp in init_by_name:
                t = init_by_name[inp]
                dims = list(t.dims)
                vals = list(onnx.numpy_helper.to_array(t).flatten()[:8])
                print(f"      init {inp}: dims={dims} dtype={t.data_type} vals={vals}")
            elif inp in vi_by_name:
                vi = vi_by_name[inp]
                dims = []
                for d in vi.type.tensor_type.shape.dim:
                    dims.append(d.dim_param if d.dim_param else str(d.dim_value))
                print(f"      vi   {inp}: shape=[{','.join(dims)}]")
        for out in n.output:
            if out in vi_by_name:
                vi = vi_by_name[out]
                dims = []
                for d in vi.type.tensor_type.shape.dim:
                    dims.append(d.dim_param if d.dim_param else str(d.dim_value))
                print(f"      out  {out}: shape=[{','.join(dims)}]")

    # Path-based variants stream external data and avoid the 2 GB
    # protobuf serialization cap.
    print("\n--- onnx.checker.check_model (path, full_check=True) ---")
    try:
        onnx.checker.check_model(str(args.onnx), full_check=True)
        print("OK")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")

    print("\n--- onnx.shape_inference.infer_shapes_path (strict, data_prop) ---")
    out_path = args.onnx.with_suffix(".inferred.onnx")
    try:
        onnx.shape_inference.infer_shapes_path(
            str(args.onnx), str(out_path),
            check_type=True, strict_mode=True, data_prop=True,
        )
        print(f"OK -> {out_path}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
