"""Rewrite an LTX transformer block ONNX to expose every intermediate
tensor as a graph output. Used for op-level bisection of the WebGPU vs
WASM divergence (see notes/attic/ort-fp16-bugs.md and notes/worklog.md
"Active bug" section).

The original block_NN.onnx is untouched. The rewritten copy lands at
block_NN_tapped.onnx in the SAME directory and references the existing
block_NN.onnx.data sidecar via a relative path. We deliberately load
topology-only (`load_external_data=False`) and save without
`save_as_external_data=True` to avoid the well-known weight-corruption
roundtrip bug.

Usage:
  uv run --with onnx python expose_block_intermediates.py \\
      ../../../notes/models/ltx/hf-repo/onnx/transformer-fp16/block_00.onnx
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import onnx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "onnx_path",
        type=Path,
        help="path to block_NN.onnx",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (defaults to <stem>_tapped.onnx in the same dir)",
    )
    ap.add_argument(
        "--n-tokens",
        type=int,
        default=128,
        help="concrete n_tokens dim used for shape inference + size filter",
    )
    ap.add_argument(
        "--n-text",
        type=int,
        default=64,
        help="concrete n_text dim used for shape inference + size filter",
    )
    ap.add_argument(
        "--skip-ops",
        default=(
            "Constant,Shape,Gather,Range,ConstantOfShape,Unsqueeze,Expand,"
            "Concat,Equal,Where,Slice,Cast,Reshape,Transpose,Squeeze,Split,"
            "Mod,ScatterND,If,Div,Neg"
        ),
        help=(
            "comma-separated ONNX op types to skip tapping. Default skips "
            "shape-metadata and plumbing ops; without skipping, ORT-web "
            "memory planner OOMs at session create."
        ),
    )
    ap.add_argument(
        "--max-elements",
        type=int,
        default=4_000_000,
        help=(
            "skip taps whose output exceeds this element count. fp16 tap "
            "size is max-elements * 2 bytes. Default 4M = 8 MB/tap."
        ),
    )
    args = ap.parse_args()
    skip_ops = {s.strip() for s in args.skip_ops.split(",") if s.strip()}

    if not args.onnx_path.exists():
        print(f"error: {args.onnx_path} does not exist", file=sys.stderr)
        return 2

    out_path = args.output or args.onnx_path.with_name(
        f"{args.onnx_path.stem}_tapped.onnx"
    )

    # Topology-only load: leaves initializer external_data refs intact so
    # we can re-save without re-serializing weights (avoids the
    # save_as_external_data corruption bug, see memory note
    # feedback_onnx_external_data_roundtrip).
    print(f"loading {args.onnx_path} (topology only)", flush=True)
    model = onnx.load(str(args.onnx_path), load_external_data=False)

    # Pin dynamic dims to concrete values so shape inference can size taps.
    # Auto-pick variant by filename: shell_pre and transformer block have
    # different shapes for `hidden_states`.
    is_shell_pre = "shell_pre" in args.onnx_path.stem
    if is_shell_pre:
        dim_overrides = {
            "hidden_states": (1, args.n_tokens, 128),
            "indices_grid": (1, 3, args.n_tokens),
            "encoder_hidden_states": (1, args.n_text, 4096),
            "encoder_attention_mask": (1, args.n_text),
            "timestep": (1,),
        }
    else:
        dim_overrides = {
            "hidden_states": (1, args.n_tokens, 2048),
            "freqs_cos": (1, args.n_tokens, 2048),
            "freqs_sin": (1, args.n_tokens, 2048),
            "timestep_mod": (1, 1, 12288),
            "encoder_proj": (1, args.n_text, 2048),
            "encoder_attn_bias": (1, 1, args.n_text),
        }
    for inp in model.graph.input:
        if inp.name in dim_overrides:
            tt = inp.type.tensor_type
            tt.shape.ClearField("dim")
            for d in dim_overrides[inp.name]:
                tt.shape.dim.add().dim_value = d

    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as e:
        print(f"shape inference failed: {e}; will skip --max-elements filter", flush=True)
    graph = model.graph

    elt_count: dict[str, int] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = []
        ok = True
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                ok = False
                break
        if ok and dims:
            n = 1
            for x in dims:
                n *= x
            elt_count[vi.name] = n

    existing_outputs = {o.name for o in graph.output}
    existing_inputs = {i.name for i in graph.input}
    initializers = {t.name for t in graph.initializer}

    src_op = {}
    for node in graph.node:
        for o in node.output:
            if o:
                src_op[o] = node.op_type

    added = 0
    skipped = 0
    skipped_by_op_filter = 0
    skipped_by_size = 0
    total_elts = 0
    taps_by_op: dict[str, int] = {}
    for node in graph.node:
        if node.op_type in skip_ops:
            skipped_by_op_filter += len([o for o in node.output if o])
            continue
        for o in node.output:
            if not o:
                continue
            if o in existing_outputs or o in existing_inputs or o in initializers:
                skipped += 1
                continue
            n = elt_count.get(o)
            if n is None or n > args.max_elements:
                skipped_by_size += 1
                continue
            total_elts += n
            vi = onnx.ValueInfoProto()
            vi.name = o
            graph.output.append(vi)
            existing_outputs.add(o)
            added += 1
            taps_by_op[node.op_type] = taps_by_op.get(node.op_type, 0) + 1

    print(
        f"added {added} intermediate outputs, skipped {skipped} "
        f"(+{skipped_by_op_filter} from --skip-ops, "
        f"+{skipped_by_size} over --max-elements={args.max_elements}). "
        f"total tap bytes (fp16 worst case) ~= "
        f"{total_elts * 2 / (1024*1024):.1f} MB",
        flush=True,
    )
    print("taps per op type:", flush=True)
    for op, n in sorted(taps_by_op.items(), key=lambda kv: -kv[1]):
        print(f"  {op}: {n}", flush=True)

    # Save without re-serializing weights. Initializer external_data
    # entries still point at the sidecar's relative filename, which
    # resolves alongside the new .onnx since we put it in the same dir.
    print(f"writing {out_path}", flush=True)
    if out_path.exists():
        out_path.unlink()
    onnx.save(model, str(out_path))
    graph_kb = out_path.stat().st_size / 1024
    print(f"done (graph {graph_kb:.1f} KB; weights stay in existing .onnx.data sidecar)", flush=True)

    sidecar = args.onnx_path.with_suffix(".onnx.data")
    if sidecar.exists():
        print(
            f"sidecar present at {sidecar.name} "
            f"({sidecar.stat().st_size / (1024*1024):.1f} MB) - "
            f"tapped graph references it via relative path, no copy needed",
            flush=True,
        )
    else:
        print(f"WARNING: sidecar {sidecar} not found", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
