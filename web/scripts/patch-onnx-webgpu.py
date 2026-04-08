#!/usr/bin/env python3
"""Patch ONNX models to remove ops that fall back to CPU on the onnxruntime-web
WebGPU JS execution provider.

The ORT WebGPU JS EP is missing implementations for a handful of common ops
(notably `Max` and `PRelu`). When these appear in a graph, ORT silently falls
back to CPU for those nodes, which (because every fallback round-trips tensors
between GPU and CPU) can cost hundreds of ms per inference. We rewrite each
unsupported op as an equivalent subgraph using only ops the WebGPU EP supports.

Rewrites:
  Max(a, b)     -> Add(Relu(Sub(a, b)), b)
  PRelu(x, s)   -> Sub(Relu(x), Mul(s, Relu(Neg(x))))

Both decompositions are exact (no numerical approximation) and use only
elementwise ops that the WebGPU EP handles natively.

Commands:
  list-incompatible-ops <model.onnx>
      Print every node whose op_type is in the unsupported set, with input
      info. For Max nodes, additionally report whether one input is a constant
      zero (i.e. the node is really a ReLU in disguise).

  patch-incompatible-ops <input.onnx> <output.onnx>
      Rewrite all unsupported ops and save to <output.onnx>. Refuses to write
      if the output already exists.

  diff-onnx <original.onnx> <patched.onnx> <out.json>
      Produce a forward-applicable binary diff from <original> to <patched>.
      The output JSON has shape:
        {
          "srcSha256": "...", "srcLen": N,
          "dstSha256": "...", "dstLen": M,
          "edits": [{"offset": int, "delete": int, "insert": "hex"}, ...]
        }
      Edits are sorted ascending by offset and reference offsets in the
      ORIGINAL file. To apply forward, walk edits in order maintaining a
      running delta = (sum of insert_len - delete_len so far). Bytes from
      original at position p are written to output at position p + delta_so_far.

Usage:
    pip install onnx numpy
    python patch-onnx-webgpu.py list-incompatible-ops xseg_1.onnx
    python patch-onnx-webgpu.py patch-incompatible-ops xseg_1.onnx xseg_1.patched.onnx
    python patch-onnx-webgpu.py diff-onnx xseg_1.onnx xseg_1.patched.onnx xseg_1.patch.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy
import onnx
from onnx import helper, numpy_helper


UNSUPPORTED_OPS = {"Max", "PRelu", "GlobalAveragePool"}


def _initializer_map(graph):
    return {init.name: init for init in graph.initializer}


def _is_constant_zero(name, inits):
    init = inits.get(name)
    if init is None:
        return False
    arr = numpy_helper.to_array(init)
    return arr.size > 0 and bool(numpy.all(arr == 0))


def cmd_list(model_path: Path) -> None:
    print(f"loading {model_path}...")
    model = onnx.load(str(model_path))
    graph = model.graph
    inits = _initializer_map(graph)

    counts = {}
    for node in graph.node:
        if node.op_type not in UNSUPPORTED_OPS:
            continue
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        print(f"  {node.op_type} node={node.name!r}")
        for i, inp in enumerate(node.input):
            tag = ""
            if inp in inits:
                init = inits[inp]
                arr = numpy_helper.to_array(init)
                tag = f" [initializer shape={tuple(arr.shape)} dtype={arr.dtype}"
                if arr.size <= 4:
                    tag += f" value={arr.tolist()}"
                tag += "]"
            print(f"    input[{i}] = {inp!r}{tag}")
        if node.op_type == "Max":
            zero_inputs = [i for i, inp in enumerate(node.input) if _is_constant_zero(inp, inits)]
            if zero_inputs:
                print(f"    -> ReLU-equivalent (input[{zero_inputs[0]}] is constant zero)")

    print()
    if not counts:
        print("no unsupported ops found")
    else:
        print("summary:")
        for op, n in sorted(counts.items()):
            print(f"  {op}: {n}")


class _NameGen:
    def __init__(self, existing):
        self._existing = set(existing)
        self._n = 0

    def make(self, prefix):
        while True:
            self._n += 1
            name = f"{prefix}_{self._n}"
            if name not in self._existing:
                self._existing.add(name)
                return name


class _Ctx:
    """Shared state passed to each rewriter: name generator, shape info,
    opset, and a place to stash new initializers (e.g. ReduceMean axes
    constants for opset >= 18)."""

    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        existing = set()
        for n in self.graph.node:
            existing.update(n.input)
            existing.update(n.output)
            if n.name:
                existing.add(n.name)
        for init in self.graph.initializer:
            existing.add(init.name)
        self.names = _NameGen(existing)
        self.new_initializers = []
        self.opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            default=13,
        )
        # Run shape inference so rewriters can look up tensor shapes by name.
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            self.shapes = {}
            for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(
                inferred.graph.output
            ):
                shape = []
                for d in vi.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value > 0 else None)
                self.shapes[vi.name] = shape
        except Exception as e:
            print(f"  warning: shape inference failed: {e}")
            self.shapes = {}


def _rewrite_max(node, ctx):
    a, b = node.input[0], node.input[1]
    out = node.output[0]
    sub = ctx.names.make("wgpu_max_sub")
    relu = ctx.names.make("wgpu_max_relu")
    return [
        helper.make_node("Sub", [a, b], [sub], name=ctx.names.make("wgpu_max_sub_n")),
        helper.make_node("Relu", [sub], [relu], name=ctx.names.make("wgpu_max_relu_n")),
        helper.make_node("Add", [relu, b], [out], name=ctx.names.make("wgpu_max_add_n")),
    ]


def _rewrite_prelu(node, ctx):
    x, slope = node.input[0], node.input[1]
    out = node.output[0]
    pos = ctx.names.make("wgpu_prelu_pos")
    neg = ctx.names.make("wgpu_prelu_neg")
    negrelu = ctx.names.make("wgpu_prelu_negrelu")
    scaled = ctx.names.make("wgpu_prelu_scaled")
    return [
        helper.make_node("Relu", [x], [pos], name=ctx.names.make("wgpu_prelu_pos_n")),
        helper.make_node("Neg", [x], [neg], name=ctx.names.make("wgpu_prelu_neg_n")),
        helper.make_node("Relu", [neg], [negrelu], name=ctx.names.make("wgpu_prelu_negrelu_n")),
        helper.make_node("Mul", [slope, negrelu], [scaled], name=ctx.names.make("wgpu_prelu_mul_n")),
        helper.make_node("Sub", [pos, scaled], [out], name=ctx.names.make("wgpu_prelu_sub_n")),
    ]


def _rewrite_global_average_pool(node, ctx):
    inp = node.input[0]
    out = node.output[0]
    in_shape = ctx.shapes.get(inp)
    out_shape = ctx.shapes.get(out)
    if in_shape is None or out_shape is None:
        raise RuntimeError(
            f"GlobalAveragePool {node.name!r}: missing shape info for input/output; "
            "cannot determine which axes to reduce. Re-export the model with shape "
            "info or run onnx.shape_inference first."
        )
    if len(in_shape) != len(out_shape):
        raise RuntimeError(
            f"GlobalAveragePool {node.name!r}: rank mismatch in={in_shape} out={out_shape}"
        )
    axes = []
    for i, (a, b) in enumerate(zip(in_shape, out_shape)):
        if b == 1 and (a is None or a != 1):
            axes.append(i)
        elif a != b:
            raise RuntimeError(
                f"GlobalAveragePool {node.name!r}: unexpected shape change at axis {i}: "
                f"in={a} out={b}"
            )
    if not axes:
        raise RuntimeError(
            f"GlobalAveragePool {node.name!r}: no reduced axes detected (in={in_shape} out={out_shape})"
        )
    # ReduceMean: opset >= 18 takes axes as an input tensor (initializer);
    # earlier opsets take axes as a node attribute.
    if ctx.opset >= 18:
        axes_name = ctx.names.make("wgpu_gap_axes")
        ctx.new_initializers.append(
            numpy_helper.from_array(numpy.array(axes, dtype=numpy.int64), name=axes_name)
        )
        return [
            helper.make_node(
                "ReduceMean",
                [inp, axes_name],
                [out],
                keepdims=1,
                name=ctx.names.make("wgpu_gap_n"),
            )
        ]
    return [
        helper.make_node(
            "ReduceMean",
            [inp],
            [out],
            axes=axes,
            keepdims=1,
            name=ctx.names.make("wgpu_gap_n"),
        )
    ]


REWRITES = {
    "Max": _rewrite_max,
    "PRelu": _rewrite_prelu,
    "GlobalAveragePool": _rewrite_global_average_pool,
}


def cmd_patch(in_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"refusing to overwrite existing file: {out_path}")
        sys.exit(1)

    print(f"loading {in_path}...")
    model = onnx.load(str(in_path))
    ctx = _Ctx(model)
    graph = ctx.graph

    new_nodes = []
    counts = {}
    for node in graph.node:
        rewriter = REWRITES.get(node.op_type)
        if rewriter is None:
            new_nodes.append(node)
            continue
        replacement = rewriter(node, ctx)
        new_nodes.extend(replacement)
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    if not counts:
        print("no unsupported ops found, nothing to patch")
        sys.exit(1)

    del graph.node[:]
    graph.node.extend(new_nodes)
    if ctx.new_initializers:
        graph.initializer.extend(ctx.new_initializers)

    print("rewrote:")
    for op, n in sorted(counts.items()):
        print(f"  {op}: {n} node(s)")

    print(f"checking patched model...")
    onnx.checker.check_model(model)

    print(f"writing {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _diff_bytes(a: bytes, b: bytes, window: int = 32):
    """Forward-applicable binary diff producing {offset, delete, insert} edits.

    Strategy: use numpy to peel off the longest matching prefix and longest
    matching suffix in two O(N) C-level passes. That leaves a tiny middle
    region (typically <100KB for our ONNX rewrites where only a handful of
    nodes change), on which we run a straightforward byte-level diff with
    a hash-anchor resync. All offsets emitted reference the original file.
    """
    la, lb = len(a), len(b)
    a_np = numpy.frombuffer(a, dtype=numpy.uint8)
    b_np = numpy.frombuffer(b, dtype=numpy.uint8)

    # Longest common prefix.
    n = min(la, lb)
    diff = a_np[:n] != b_np[:n]
    idxs = numpy.flatnonzero(diff)
    prefix = int(idxs[0]) if idxs.size else n

    # Longest common suffix (must not overlap the prefix on either side).
    max_suffix = min(la - prefix, lb - prefix)
    if max_suffix > 0:
        ar = a_np[la - max_suffix : la]
        br = b_np[lb - max_suffix : lb]
        diff = ar != br
        idxs = numpy.flatnonzero(diff)
        suffix = max_suffix if not idxs.size else max_suffix - 1 - int(idxs[-1])
    else:
        suffix = 0

    # Middle slices to actually diff.
    a_mid = bytes(a_np[prefix : la - suffix])
    b_mid = bytes(b_np[prefix : lb - suffix])

    edits = []
    if not a_mid and not b_mid:
        return edits
    if not a_mid:
        edits.append({"offset": prefix, "delete": 0, "insert": b_mid.hex()})
        return edits
    if not b_mid:
        edits.append({"offset": prefix, "delete": len(a_mid), "insert": ""})
        return edits

    # Byte-level diff on the small middle. Same two-pointer + hash-anchor
    # approach as before, but bounded to a tiny region so it's fast.
    lam, lbm = len(a_mid), len(b_mid)
    am_np = numpy.frombuffer(a_mid, dtype=numpy.uint8)
    bm_np = numpy.frombuffer(b_mid, dtype=numpy.uint8)
    ai = bi = 0
    while ai < lam and bi < lbm:
        # Match run via numpy.
        n = min(lam - ai, lbm - bi)
        d = am_np[ai : ai + n] != bm_np[bi : bi + n]
        ix = numpy.flatnonzero(d)
        m = int(ix[0]) if ix.size else n
        ai += m
        bi += m
        if ai >= lam or bi >= lbm:
            break

        # Build index of every window in remaining a_mid.
        index = {}
        i = ai
        while i + window <= lam:
            key = bytes(am_np[i : i + window])
            if key not in index:
                index[key] = i
            i += 1

        # Scan b_mid for a matching window.
        sync_a = sync_b = None
        j = bi
        while j + window <= lbm:
            key = bytes(bm_np[j : j + window])
            hit = index.get(key)
            if hit is not None:
                sync_a, sync_b = hit, j
                break
            j += 1

        if sync_a is None:
            edits.append(
                {"offset": prefix + ai, "delete": lam - ai, "insert": b_mid[bi:lbm].hex()}
            )
            ai, bi = lam, lbm
            break

        edits.append(
            {
                "offset": prefix + ai,
                "delete": sync_a - ai,
                "insert": b_mid[bi:sync_b].hex(),
            }
        )
        ai, bi = sync_a, sync_b

    if ai < lam or bi < lbm:
        edits.append(
            {"offset": prefix + ai, "delete": lam - ai, "insert": b_mid[bi:lbm].hex()}
        )

    return edits


def cmd_diff(src_path: Path, dst_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"refusing to overwrite existing file: {out_path}")
        sys.exit(1)

    print(f"loading {src_path}...")
    src = src_path.read_bytes()
    print(f"loading {dst_path}...")
    dst = dst_path.read_bytes()

    print(f"hashing...")
    src_sha = _sha256(src)
    dst_sha = _sha256(dst)

    print(f"diffing ({len(src)} -> {len(dst)} bytes)...")
    edits = _diff_bytes(src, dst)

    # Sanity check: forward-apply with running delta and confirm result.
    print(f"verifying patch...")
    out = bytearray()
    cursor = 0
    for e in edits:
        if e["offset"] < cursor:
            raise RuntimeError(f"edits not sorted: offset {e['offset']} < cursor {cursor}")
        out.extend(src[cursor : e["offset"]])
        out.extend(bytes.fromhex(e["insert"]))
        cursor = e["offset"] + e["delete"]
    out.extend(src[cursor:])
    if bytes(out) != dst:
        raise RuntimeError("patch verification failed: applied output != dst")
    print(f"  ok ({len(edits)} edit(s))")

    patch = {
        "srcSha256": src_sha,
        "srcLen": len(src),
        "dstSha256": dst_sha,
        "dstLen": len(dst),
        "edits": edits,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patch, indent=2))
    print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "list-incompatible-ops":
        if len(sys.argv) != 3:
            print(__doc__)
            sys.exit(1)
        cmd_list(Path(sys.argv[2]))
    elif cmd == "patch-incompatible-ops":
        if len(sys.argv) != 4:
            print(__doc__)
            sys.exit(1)
        cmd_patch(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "diff-onnx":
        if len(sys.argv) != 5:
            print(__doc__)
            sys.exit(1)
        cmd_diff(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
