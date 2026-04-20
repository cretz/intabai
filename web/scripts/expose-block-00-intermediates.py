"""Rewrite block_00.onnx to expose every intermediate tensor as a graph
output. Used for op-level bisection of the WebGPU-vs-CPU divergence we
see after patching MatMul/Softmax accumulators (see notes/ort-fp16-bugs.md
section 2b). The original block_00.onnx is untouched; the rewritten copy
lands at block_00_debug.onnx next to it so the vite proxy can serve it.

Usage:
  uv run expose-block-00-intermediates.py \
      ../../../notes/models/fastwan/hf-repo/onnx/transformer/block_00.onnx

Writes block_00_debug.onnx in the same directory.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import onnx


def sanitize_tap_name(raw: str) -> str:
    # Browser console can handle any string, but keep taps identifiable
    # and URL-safe-ish. ONNX producer names are usually like
    # "/blocks.0/attn1/to_q/MatMul_output_0" - forward-slash-prefixed,
    # slash-separated. Leave them as-is; the browser/python code dumps
    # them verbatim.
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "onnx_path",
        type=Path,
        help="path to block_00.onnx (or any single-block ONNX)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (defaults to <stem>_debug.onnx in the same dir)",
    )
    ap.add_argument(
        "--skip-ops",
        default=(
            "Constant,Shape,Gather,Range,ConstantOfShape,Unsqueeze,Expand,"
            "Concat,Equal,Where,Slice,Cast,Reshape,Transpose,Squeeze,Split,"
            "Mod,ScatterND,If,Div"
        ),
        help=(
            "comma-separated ONNX op types to skip tapping. Default skips "
            "shape-metadata and plumbing ops. With them NOT skipped, "
            "session creation OOMs: ORT can't constant-fold shape paths "
            "whose outputs are exposed, so WebGPU has to allocate buffers "
            "for every live Reshape/Mod/If intermediate."
        ),
    )
    ap.add_argument(
        "--only-nodes",
        default=None,
        help=(
            "comma-separated list of node NAMES (not output names). When set, "
            "only those nodes' outputs are exposed as graph outputs and all "
            "other filters (--skip-ops, --max-elements) are ignored. Use for "
            "surgical tap sets that preserve attention fusion (skip anything "
            "between attn{1,2}/MatMul and attn{1,2}/MatMul_1)."
        ),
    )
    ap.add_argument(
        "--max-elements",
        type=int,
        default=30_000_000,
        help=(
            "skip taps whose output exceeds this element count. fp16 tap "
            "size is max-elements * 2 bytes. Default 30M = 60 MB/tap. "
            "Total across all taps must fit in the ORT-web wasm 4 GB cap "
            "and the WebGPU buffer budget; previous 214-tap run with no "
            "size cap OOM'd because timestep_proj broadcast [1,8190,6,3072] "
            "= 151M elements / 302 MB per tensor times ~40 such taps."
        ),
    )
    args = ap.parse_args()
    skip_ops = {s.strip() for s in args.skip_ops.split(",") if s.strip()}

    if not args.onnx_path.exists():
        print(f"error: {args.onnx_path} does not exist", file=sys.stderr)
        return 2

    out_path = args.output or args.onnx_path.with_name(
        f"{args.onnx_path.stem}_debug.onnx"
    )

    print(f"loading {args.onnx_path}", flush=True)
    model = onnx.load(str(args.onnx_path), load_external_data=True)

    # Run shape inference with the block's known input shapes substituted
    # so --max-elements can filter on numeric element counts rather than
    # symbolic dims. Inputs (from transformer.ts): hidden_states
    # [1,8190,3072] fp16, encoder_hidden_states [1,512,3072] fp16,
    # timestep_proj [1,8190,6,3072] fp16, freqs_cos/sin [1,8190,1,128] fp32.
    dim_overrides = {
        "hidden_states": (1, 8190, 3072),
        "encoder_hidden_states": (1, 512, 3072),
        "timestep_proj": (1, 8190, 6, 3072),
        "freqs_cos": (1, 8190, 1, 128),
        "freqs_sin": (1, 8190, 1, 128),
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

    # Index inferred element counts for every named value.
    elt_count: dict[str, int] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = []
        ok = True
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                ok = False; break
        if ok and dims:
            n = 1
            for x in dims: n *= x
            elt_count[vi.name] = n

    existing_outputs = {o.name for o in graph.output}
    existing_inputs = {i.name for i in graph.input}
    initializers = {t.name for t in graph.initializer}

    # Index each value's source so we can report op type alongside the tap
    # name later (helps the human reading the diff).
    src_op = {}
    for node in graph.node:
        for o in node.output:
            if o:
                src_op[o] = node.op_type

    only_nodes = None
    if args.only_nodes:
        only_nodes = {s.strip() for s in args.only_nodes.split(",") if s.strip()}

    added = 0
    skipped = 0
    skipped_by_op_filter = 0
    skipped_by_size = 0
    total_elts = 0
    taps_by_op = {}
    for node in graph.node:
        if only_nodes is not None:
            if node.name not in only_nodes:
                continue
        elif node.op_type in skip_ops:
            skipped_by_op_filter += len([o for o in node.output if o])
            continue
        for o in node.output:
            if not o:
                continue
            if o in existing_outputs:
                skipped += 1
                continue
            if o in existing_inputs or o in initializers:
                skipped += 1
                continue
            if only_nodes is None:
                # Size filter only applies in auto-tap mode; in --only-nodes
                # mode the human picked the nodes deliberately.
                n = elt_count.get(o)
                if n is None or n > args.max_elements:
                    skipped_by_size += 1
                    continue
                total_elts += n
            else:
                total_elts += elt_count.get(o, 0)
            # Use a ValueInfoProto with no shape/type so ORT infers from
            # the graph. That avoids having to run shape inference here
            # (which can be flaky on custom ops / dynamic shapes).
            vi = onnx.ValueInfoProto()
            vi.name = sanitize_tap_name(o)
            graph.output.append(vi)
            existing_outputs.add(o)
            added += 1
            taps_by_op[node.op_type] = taps_by_op.get(node.op_type, 0) + 1

    if only_nodes is not None:
        missing = only_nodes - {n.name for n in graph.node}
        if missing:
            print(f"WARNING: --only-nodes referenced unknown nodes: {sorted(missing)}", flush=True)

    print(
        f"added {added} intermediate outputs, skipped {skipped} "
        f"(+{skipped_by_op_filter} from --skip-ops, "
        f"+{skipped_by_size} over --max-elements={args.max_elements}). "
        f"total tap bytes (fp16 worst case) ~= "
        f"{total_elts * 2 / (1024*1024):.0f} MB",
        flush=True,
    )
    print("taps per op type (top 15):", flush=True)
    for op, n in sorted(taps_by_op.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {op}: {n}", flush=True)

    print(f"writing {out_path}", flush=True)
    # Use external data so the tiny graph proto loads into wasm memory
    # quickly and the 327 MB of weights stream to GPU via the sidecar.
    # Monolithic failed with std::bad_alloc during session creation when
    # shell_pre (180 MB fp16) was already resident — wasm-side ONNX proto
    # parsing held the full 327 MB in linear memory at init time.
    data_path = out_path.with_suffix(".onnx.data")
    # onnx.save appends to existing sidecar by default, doubling the file
    # silently — delete both first.
    for p in (out_path, data_path):
        if p.exists():
            p.unlink()
    onnx.save(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    graph_mb = out_path.stat().st_size / (1024 * 1024)
    data_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"done (graph {graph_mb:.1f} MB + data {data_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
