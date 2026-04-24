"""Split a half-resnet sub-ONNX at its Pad output into two sub-graphs.

Input:  decoder_init_part_NN_..._a.onnx  (one Pad -> one Conv3D)
Output: <same>_pre.onnx   - everything up to and including the Pad node, plus
                            any extra outputs (e.g. cache_out slices) that
                            branched off before the Pad.
        <same>_conv.onnx  - just the Conv3D (and anything after it, if present)

The cut tensor is auto-discovered: the single Conv node's input[0] is the Pad
output, and that's where we cut. If a graph has multiple Pad+Conv chains
(unlikely for a half-resnet) we bail.

The cut lets the caller run the two halves as separate ORT-web session.runs
with a JS await between them. Since the Conv3D is what dominates GPU time,
splitting it away from the ~18 element-wise pre-ops halves the cumulative
GPU-busy time per session.run, keeping each session under Windows D3D12 TDR.

The input graph may already have concrete input shapes embedded (from the
--input-shape pass of patch-vae-conv3d-tile.py or the upstream splitter). If
not, pass --input-shape name=d1,d2,...

Weights stay embedded (the part files are fp16 and small; no external data).

Usage:
    uv run split-half-resnet-at-pad.py <input.onnx> \
        [--input-shape name=1,1024,1,60,104] \
        [--skip-verify]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_cut_tensor(model: onnx.ModelProto) -> str:
    """The cut tensor is the input to the (single) Conv node."""
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    if len(conv_nodes) != 1:
        raise SystemExit(
            f"expected exactly one Conv, found {len(conv_nodes)}: "
            f"{[n.name for n in conv_nodes]}"
        )
    return conv_nodes[0].input[0]


def transitive_descendants(model: onnx.ModelProto, source: str) -> set:
    """Return the set of node names that transitively consume `source` as an
    input (strict descendants — excludes the producer of `source`)."""
    consumers_of: dict = {}
    for n in model.graph.node:
        for inp in n.input:
            consumers_of.setdefault(inp, []).append(n)

    visited_nodes: set = set()
    frontier = [source]
    while frontier:
        t = frontier.pop()
        for n in consumers_of.get(t, []):
            if n.name in visited_nodes:
                continue
            visited_nodes.add(n.name)
            for out in n.output:
                frontier.append(out)
    return visited_nodes


def tensor_type_of(model: onnx.ModelProto, name: str):
    """Look up the ONNX type (elem_type, shape) for a tensor. Prefers
    shape-inferred value_info / outputs / inputs; falls back to fp16 5D with
    symbolic dims."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(
        inferred.graph.output
    ):
        if vi.name == name:
            return vi.type
    # Fallback
    return None


def build_subgraph(
    model: onnx.ModelProto,
    node_names: set,
    graph_inputs: list,
    graph_outputs: list,
    graph_name: str,
) -> onnx.ModelProto:
    """Build a new ModelProto containing only `node_names` nodes."""
    g = model.graph
    nodes = [n for n in g.node if n.name in node_names or (not n.name and False)]
    # Preserve order from original.
    kept_outputs = {out for n in nodes for out in n.output}

    # Initializers used by kept nodes.
    used_init_names = set()
    for n in nodes:
        for inp in n.input:
            used_init_names.add(inp)
    kept_inits = [
        init for init in g.initializer if init.name in used_init_names
    ]

    new_graph = helper.make_graph(
        nodes=nodes,
        name=graph_name,
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=kept_inits,
    )
    new_model = helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    new_model.producer_name = model.producer_name or "split-half-resnet-at-pad"
    return new_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument(
        "--input-shape", action="append", default=[],
        help="name=d1,d2,... override a graph input's shape, repeatable",
    )
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    shape_overrides = {}
    for spec in args.input_shape:
        if "=" not in spec:
            raise SystemExit(f"bad --input-shape: {spec!r}")
        k, v = spec.split("=", 1)
        shape_overrides[k] = [int(x) for x in v.split(",")]

    log(f"Loading {args.input}")
    model = onnx.load(str(args.input), load_external_data=False)
    g = model.graph

    # Apply shape overrides to graph inputs (improves downstream verify feeds).
    for vi in g.input:
        if vi.name in shape_overrides:
            dims = shape_overrides[vi.name]
            tshape = vi.type.tensor_type.shape
            if len(tshape.dim) != len(dims):
                raise SystemExit(
                    f"rank mismatch on override for {vi.name}"
                )
            for d, v in zip(tshape.dim, dims):
                d.Clear()
                d.dim_value = v
            log(f"  override input {vi.name} -> {dims}")

    cut = find_cut_tensor(model)
    log(f"Cut tensor: {cut}")

    conv_node_names = transitive_descendants(model, cut)
    all_node_names = {n.name for n in g.node if n.name}
    pre_node_names = all_node_names - conv_node_names
    log(f"  pre side: {len(pre_node_names)} nodes")
    log(f"  conv side: {len(conv_node_names)} nodes")

    # Verify shape for the cut tensor is known (needed as a graph input for
    # the _conv subgraph).
    cut_type = tensor_type_of(model, cut)
    if cut_type is None:
        raise SystemExit(f"could not infer type for cut tensor {cut}")

    # Graph inputs for _pre: same as original.
    pre_inputs = list(g.input)

    # Graph outputs for _pre: the cut tensor, plus any original graph outputs
    # whose producer is in pre_node_names (e.g. cache_out slices).
    pre_outputs: list = []
    pre_output_names: set = set()
    # 1. The cut tensor becomes a new output of _pre.
    pre_outputs.append(helper.make_tensor_value_info(cut, 0, None))
    pre_outputs[-1].type.CopyFrom(cut_type)
    pre_output_names.add(cut)
    # 2. Any original graph output produced on the pre side.
    producer_of = {out: n for n in g.node for out in n.output}
    for orig_out in g.output:
        n = producer_of.get(orig_out.name)
        if n is not None and n.name in pre_node_names:
            if orig_out.name not in pre_output_names:
                pre_outputs.append(orig_out)
                pre_output_names.add(orig_out.name)

    # Graph inputs for _conv: the cut tensor.
    conv_input_vi = helper.make_tensor_value_info(cut, 0, None)
    conv_input_vi.type.CopyFrom(cut_type)
    conv_inputs = [conv_input_vi]

    # Graph outputs for _conv: any original graph output produced on the conv
    # side.
    conv_outputs: list = []
    for orig_out in g.output:
        n = producer_of.get(orig_out.name)
        if n is not None and n.name in conv_node_names:
            conv_outputs.append(orig_out)

    pre_model = build_subgraph(
        model, pre_node_names, pre_inputs, pre_outputs,
        graph_name=f"{g.name}_pre",
    )
    conv_model = build_subgraph(
        model, conv_node_names, conv_inputs, conv_outputs,
        graph_name=f"{g.name}_conv",
    )

    log("Checking models...")
    for tag, m in (("pre", pre_model), ("conv", conv_model)):
        try:
            onnx.checker.check_model(m, full_check=False)
            log(f"  {tag}: check_model ok")
        except Exception as e:
            log(f"  {tag}: check_model WARNING: {e}")

    pre_path = args.input.with_name(args.input.stem + "_pre.onnx")
    conv_path = args.input.with_name(args.input.stem + "_conv.onnx")
    onnx.save_model(pre_model, str(pre_path))
    onnx.save_model(conv_model, str(conv_path))
    log(f"Wrote {pre_path.name} ({pre_path.stat().st_size / 1e6:.1f} MB)")
    log(f"Wrote {conv_path.name} ({conv_path.stat().st_size / 1e6:.1f} MB)")

    if args.skip_verify:
        log("Skipping verify (--skip-verify)")
        return

    log("Verifying: pre then conv composed equals original (CPU ORT, zeros)...")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_orig = ort.InferenceSession(str(args.input), so, providers=["CPUExecutionProvider"])
    sess_pre = ort.InferenceSession(str(pre_path), so, providers=["CPUExecutionProvider"])
    sess_conv = ort.InferenceSession(str(conv_path), so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    feeds = {}
    # Prefer concrete shapes from sess_pre (which got the override applied).
    pre_inp_by_name = {i.name: i for i in sess_pre.get_inputs()}
    for inp in sess_orig.get_inputs():
        ref = pre_inp_by_name.get(inp.name, inp)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in ref.shape]
        dtype = np.float16 if "float16" in inp.type else np.float32
        feeds[inp.name] = (rng.standard_normal(shape) * 0.01).astype(dtype)

    t0 = time.time()
    out_orig = {o.name: arr for o, arr in zip(sess_orig.get_outputs(), sess_orig.run(None, feeds))}
    log(f"  orig run {time.time()-t0:.1f}s")

    t0 = time.time()
    pre_results = {o.name: arr for o, arr in zip(sess_pre.get_outputs(), sess_pre.run(None, feeds))}
    log(f"  pre  run {time.time()-t0:.1f}s")

    conv_feeds = {}
    for inp in sess_conv.get_inputs():
        if inp.name in pre_results:
            conv_feeds[inp.name] = pre_results[inp.name]
        elif inp.name in feeds:
            conv_feeds[inp.name] = feeds[inp.name]
        else:
            raise SystemExit(f"can't satisfy conv input {inp.name}")
    t0 = time.time()
    conv_results = {o.name: arr for o, arr in zip(sess_conv.get_outputs(), sess_conv.run(None, conv_feeds))}
    log(f"  conv run {time.time()-t0:.1f}s")

    combined = {**pre_results, **conv_results}
    worst = 0.0
    worst_name = ""
    for name, arr in out_orig.items():
        if name not in combined:
            raise SystemExit(f"output {name} missing from split outputs")
        diff = float(np.max(np.abs(arr.astype(np.float32) - combined[name].astype(np.float32))))
        log(f"  diff[{name}] = {diff:.6g}  shape={arr.shape}")
        if diff > worst:
            worst = diff
            worst_name = name

    if worst > 1e-2:
        raise SystemExit(f"verify FAILED: worst diff {worst} on {worst_name}")
    log(f"Verify OK: worst diff {worst:.6g} on {worst_name or '(none)'}")


if __name__ == "__main__":
    main()
