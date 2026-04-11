#!/usr/bin/env python3
"""Inspect and shard a DiT-style ONNX transformer at layer boundaries.

  --inspect: Analyze the ONNX graph structure to find transformer blocks,
    their weight sizes, and natural split points.

  (default): Split the ONNX into N sub-graph shards by --max-shard-size.
    Writes shard ONNX files + a manifest JSON. Each shard is a valid ONNX
    model that can be loaded independently by ORT. The external data file
    (_data sidecar) is shared across all shards unchanged.

Uses raw protobuf wire-format scanning (no onnx.load for the source) so
memory usage is ~1x file size (flat byte buffer).

Usage:
    python shard-onnx-layers.py --inspect <transformer.onnx>
    python shard-onnx-layers.py <transformer.onnx> --max-shard-size 1.2GB <out.json>

Example:
    python shard-onnx-layers.py --inspect \\
        ../../notes/models/zimage/transformer_model_q4f16.onnx

    python shard-onnx-layers.py \\
        ../../notes/models/zimage/transformer_model_q4f16.onnx \\
        --max-shard-size 1.2GB \\
        ../../notes/models/zimage/transformer_shards.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from lib.onnx_patch_common import (
    WIRE_LEN,
    WIRE_VARINT,
    encode_len_prefixed,
    encode_string_field,
    encode_tag,
    encode_varint,
    read_string_field,
    read_varint,
    scan_fields,
    sha256,
)

# ONNX protobuf field numbers
FIELD_MODEL_GRAPH = 7
FIELD_MODEL_OPSET_IMPORT = 8
FIELD_MODEL_IR_VERSION = 1

FIELD_GRAPH_NODE = 1
FIELD_GRAPH_INITIALIZER = 5
FIELD_GRAPH_INPUT = 11
FIELD_GRAPH_OUTPUT = 12

FIELD_NODE_NAME = 3
FIELD_NODE_OP_TYPE = 4
FIELD_NODE_INPUT = 1
FIELD_NODE_OUTPUT = 2

FIELD_TENSOR_NAME = 8
FIELD_TENSOR_RAW_DATA = 9
FIELD_TENSOR_EXTERNAL_DATA = 13
FIELD_TENSOR_DATA_LOCATION = 14

FIELD_VALUE_INFO_NAME = 1
FIELD_VALUE_INFO_TYPE = 2
FIELD_TYPE_PROTO_TENSOR = 1
FIELD_TENSOR_TYPE_ELEM_TYPE = 1
FIELD_TENSOR_TYPE_SHAPE = 2
FIELD_SHAPE_DIM = 1
FIELD_DIM_VALUE = 1
FIELD_DIM_PARAM = 2

FIELD_STRING_STRING_KEY = 1
FIELD_STRING_STRING_VALUE = 2

FIELD_OPSET_DOMAIN = 1
FIELD_OPSET_VERSION = 2

DTYPE_NAMES = {
    0: "UNDEFINED", 1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16",
    5: "INT16", 6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL",
    10: "FLOAT16", 11: "DOUBLE", 12: "UINT32", 13: "UINT64",
    14: "COMPLEX64", 15: "COMPLEX128", 16: "BFLOAT16",
    17: "FLOAT8E4M3FN", 18: "FLOAT8E4M3FNUZ",
    19: "FLOAT8E5M2", 20: "FLOAT8E5M2FNUZ",
    21: "UINT4", 22: "INT4",
}


def parse_size(s: str) -> int:
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def extract_block_prefix(name: str) -> str | None:
    m = re.match(r"/?([a-zA-Z_]+(?:\.[a-zA-Z_]+)*\.(\d+))", name)
    return m.group(1) if m else None


def block_sort_key(name: str):
    nums = re.findall(r"(\d+)", name)
    return (name.split(".")[0], int(nums[-1]) if nums else 0)


def read_varint_field_value(buf, field):
    val, _ = read_varint(buf, field.data_start)
    return val


def parse_value_info(buf, field):
    inner = scan_fields(buf, field.data_start, field.data_end)
    name = ""
    dtype_int = 0
    shape = []
    for f in inner:
        if f.field_number == FIELD_VALUE_INFO_NAME:
            name = read_string_field(buf, f)
        elif f.field_number == FIELD_VALUE_INFO_TYPE:
            type_fields = scan_fields(buf, f.data_start, f.data_end)
            for tf in type_fields:
                if tf.field_number == FIELD_TYPE_PROTO_TENSOR:
                    tensor_fields = scan_fields(buf, tf.data_start, tf.data_end)
                    for ttf in tensor_fields:
                        if ttf.field_number == FIELD_TENSOR_TYPE_ELEM_TYPE:
                            dtype_int = read_varint_field_value(buf, ttf)
                        elif ttf.field_number == FIELD_TENSOR_TYPE_SHAPE:
                            dim_fields = scan_fields(buf, ttf.data_start, ttf.data_end)
                            for df in dim_fields:
                                if df.field_number == FIELD_SHAPE_DIM:
                                    dim_inner = scan_fields(buf, df.data_start, df.data_end)
                                    for di in dim_inner:
                                        if di.field_number == FIELD_DIM_VALUE:
                                            shape.append(("val", read_varint_field_value(buf, di)))
                                        elif di.field_number == FIELD_DIM_PARAM:
                                            shape.append(("param", read_string_field(buf, di)))
    return name, dtype_int, shape


def build_value_info_bytes(name: str, elem_type: int, shape=None):
    """Build a complete graph-level input/output ValueInfoProto field."""
    # TensorTypeProto
    tt = encode_tag(FIELD_TENSOR_TYPE_ELEM_TYPE, WIRE_VARINT) + encode_varint(elem_type)
    if shape:
        dims = bytearray()
        for kind, val in shape:
            if kind == "val":
                dim = encode_tag(FIELD_DIM_VALUE, WIRE_VARINT) + encode_varint(val)
            else:
                dim = encode_string_field(FIELD_DIM_PARAM, val)
            dims.extend(encode_tag(FIELD_SHAPE_DIM, WIRE_LEN) + encode_len_prefixed(dim))
        tt += encode_tag(FIELD_TENSOR_TYPE_SHAPE, WIRE_LEN) + encode_len_prefixed(bytes(dims))
    # TypeProto
    tp = encode_tag(FIELD_TYPE_PROTO_TENSOR, WIRE_LEN) + encode_len_prefixed(tt)
    # ValueInfoProto
    vi = encode_string_field(FIELD_VALUE_INFO_NAME, name) + \
         encode_tag(FIELD_VALUE_INFO_TYPE, WIRE_LEN) + encode_len_prefixed(tp)
    return vi


# ---------------------------------------------------------------------------
# Graph parsing (shared between inspect and shard modes)
# ---------------------------------------------------------------------------

def parse_graph(model_path: Path) -> dict:
    """Parse ONNX file via wire-format scanning. Returns dict of parsed data."""
    print(f"loading {model_path} ({model_path.stat().st_size:,} bytes)...")
    buf = model_path.read_bytes()

    model_fields = scan_fields(buf, 0, len(buf))
    graph_field = None
    non_graph_fields = []

    for f in model_fields:
        if f.field_number == FIELD_MODEL_GRAPH:
            graph_field = f
        else:
            non_graph_fields.append(f)

    if not graph_field:
        print("error: no graph found in model")
        sys.exit(1)

    graph_inner = scan_fields(buf, graph_field.data_start, graph_field.data_end)

    node_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_NODE]
    init_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_INITIALIZER]
    input_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_INPUT]
    output_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_OUTPUT]

    # Parse graph inputs/outputs
    graph_inputs = []  # (name, elem_type, shape, field)
    for f in input_fields:
        name, dtype_int, shape = parse_value_info(buf, f)
        graph_inputs.append((name, dtype_int, shape, f))
    graph_input_names = {gi[0] for gi in graph_inputs}

    graph_outputs = []
    for f in output_fields:
        name, dtype_int, shape = parse_value_info(buf, f)
        graph_outputs.append((name, dtype_int, shape, f))

    # Parse initializers
    print("scanning initializers...")
    inits = []  # (name, byte_size, field)
    for init_f in init_fields:
        tensor_fields = scan_fields(buf, init_f.data_start, init_f.data_end)
        name = ""
        raw_data_size = 0
        for tf in tensor_fields:
            if tf.field_number == FIELD_TENSOR_NAME:
                name = read_string_field(buf, tf)
            elif tf.field_number == FIELD_TENSOR_RAW_DATA:
                raw_data_size = tf.data_end - tf.data_start
            elif tf.field_number == FIELD_TENSOR_EXTERNAL_DATA:
                entry_fields = scan_fields(buf, tf.data_start, tf.data_end)
                key = value = ""
                for ef in entry_fields:
                    if ef.field_number == FIELD_STRING_STRING_KEY:
                        key = read_string_field(buf, ef)
                    elif ef.field_number == FIELD_STRING_STRING_VALUE:
                        value = read_string_field(buf, ef)
                if key == "length":
                    raw_data_size = int(value)
        inits.append((name, raw_data_size, init_f))

    init_names_set = {i[0] for i in inits}
    init_sizes = {name: size for name, size, _ in inits}

    # Parse nodes
    print("scanning nodes and tracing graph edges...")
    nodes = []  # (name, inputs, outputs, field)
    init_consumers = defaultdict(list)  # init_name -> [block, ...]

    for node_f in node_fields:
        node_inner = scan_fields(buf, node_f.data_start, node_f.data_end)
        node_name = ""
        inputs = []
        outputs = []
        for nf in node_inner:
            if nf.field_number == FIELD_NODE_NAME:
                node_name = read_string_field(buf, nf)
            elif nf.field_number == FIELD_NODE_INPUT:
                inputs.append(read_string_field(buf, nf))
            elif nf.field_number == FIELD_NODE_OUTPUT:
                outputs.append(read_string_field(buf, nf))
        nodes.append((node_name, inputs, outputs, node_f))

        block = extract_block_prefix(node_name) or "(non-block)"
        for inp in inputs:
            if inp and inp in init_names_set:
                init_consumers[inp].append(block)

    # Attribute initializers to blocks (first consumer wins for primary attribution)
    init_to_block = {}
    for inp_name, consumers in init_consumers.items():
        init_to_block[inp_name] = consumers[0]
    for name in init_names_set:
        if name not in init_to_block:
            block = extract_block_prefix(name)
            init_to_block[name] = block if block else "(non-block)"

    # Block weights
    block_weights = defaultdict(int)
    block_init_count = defaultdict(int)
    for name, size, _ in inits:
        block = init_to_block[name]
        block_weights[block] += size
        block_init_count[block] += 1

    return {
        "buf": buf,
        "model_fields": model_fields,
        "non_graph_fields": non_graph_fields,
        "graph_field": graph_field,
        "graph_inner": graph_inner,
        "node_fields": node_fields,
        "init_fields": init_fields,
        "nodes": nodes,
        "inits": inits,
        "init_sizes": init_sizes,
        "init_names_set": init_names_set,
        "init_consumers": init_consumers,
        "init_to_block": init_to_block,
        "graph_inputs": graph_inputs,
        "graph_input_names": graph_input_names,
        "graph_outputs": graph_outputs,
        "block_weights": block_weights,
        "block_init_count": block_init_count,
    }


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------

def inspect_model(model_path: Path) -> None:
    pg = parse_graph(model_path)
    buf = pg["buf"]
    block_weights = pg["block_weights"]
    block_init_count = pg["block_init_count"]
    init_to_block = pg["init_to_block"]
    init_sizes = pg["init_sizes"]
    nodes = pg["nodes"]
    graph_inputs = pg["graph_inputs"]
    graph_outputs = pg["graph_outputs"]

    non_block_weights = block_weights.pop("(non-block)", 0)
    non_block_count = block_init_count.pop("(non-block)", 0)

    total_weight_bytes = sum(init_sizes.values())
    graph_traced = sum(1 for b in init_to_block.values() if b != "(non-block)")
    name_traced = sum(1 for n in init_sizes if init_to_block.get(n) != "(non-block)" and extract_block_prefix(n) is None)
    print(f"  {graph_traced} inits attributed to blocks ({name_traced} via graph tracing)")

    # Print graph I/O
    print(f"\ngraph inputs:")
    for name, dt, shape, _ in graph_inputs:
        dims = ", ".join(str(v) if k == "val" else v for k, v in shape)
        print(f"  {name}: {DTYPE_NAMES.get(dt, '?')} [{dims}]")
    print(f"\ngraph outputs:")
    for name, dt, shape, _ in graph_outputs:
        dims = ", ".join(str(v) if k == "val" else v for k, v in shape)
        print(f"  {name}: {DTYPE_NAMES.get(dt, '?')} [{dims}]")

    print(f"\ntotal weight data: {format_size(total_weight_bytes)}")

    # Block families
    sorted_blocks = sorted(block_weights.keys(), key=block_sort_key)
    families = defaultdict(list)
    for block in sorted_blocks:
        parts = block.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isdigit():
            families[parts[0]].append(block)
        else:
            families[block].append(block)

    print(f"\n{'='*70}")
    print(f"BLOCK STRUCTURE (graph-traced attribution)")
    print(f"{'='*70}")

    for family_name, blocks in sorted(families.items(), key=lambda x: block_sort_key(x[1][0])):
        if len(blocks) > 1:
            total = sum(block_weights[b] for b in blocks)
            avg = total / len(blocks)
            print(f"\n{family_name} ({len(blocks)} blocks, {format_size(total)} total, ~{format_size(int(avg))}/block):")
            for b in blocks:
                print(f"  {b}: {format_size(block_weights[b])} ({block_init_count[b]} params)")
        else:
            b = blocks[0]
            print(f"\n{b}: {format_size(block_weights[b])} ({block_init_count[b]} params)")

    non_block_names = [(n, init_sizes[n]) for n in init_sizes if init_to_block[n] == "(non-block)"]
    if non_block_names:
        print(f"\nnon-block initializers ({non_block_count}, {format_size(non_block_weights)} total):")
        for name, size in sorted(non_block_names, key=lambda x: -x[1])[:20]:
            print(f"  {name}: {format_size(size)}")
        if non_block_count > 20:
            print(f"  ... and {non_block_count - 20} more")

    # Execution order
    print(f"\n{'='*70}")
    print(f"EXECUTION ORDER (node file positions)")
    print(f"{'='*70}")

    block_first_idx = {}
    block_last_idx = {}
    for i, (node_name, _, _, _) in enumerate(nodes):
        block = extract_block_prefix(node_name)
        if block:
            if block not in block_first_idx:
                block_first_idx[block] = i
            block_last_idx[block] = i

    for bname in sorted(block_first_idx.keys(), key=lambda b: block_first_idx[b]):
        weight = block_weights.get(bname, 0)
        print(f"  {bname}: nodes [{block_first_idx[bname]}..{block_last_idx[bname]}] ({format_size(weight)})")

    # Sharding analysis
    print(f"\n{'='*70}")
    print(f"SHARDING ANALYSIS")
    print(f"{'='*70}")

    main_family = max(
        ((fn, sum(block_weights[b] for b in bl)) for fn, bl in families.items() if len(bl) > 1),
        key=lambda x: x[1], default=(None, 0)
    )[0]

    if main_family:
        blocks = families[main_family]
        per_block_sizes = [block_weights[b] for b in blocks]
        print(f"\nmain block family: {main_family} ({len(blocks)} blocks)")
        print(f"  per-block weight size: ~{format_size(sum(per_block_sizes) // len(per_block_sizes))}")
        print(f"  total block weights: {format_size(sum(per_block_sizes))}")
        print(f"  non-block overhead: {format_size(non_block_weights)}")

        # Inter-block tensors
        node_block_outputs = defaultdict(set)
        node_block_inputs = defaultdict(set)
        for node_name, inputs, outputs, _ in nodes:
            block = extract_block_prefix(node_name)
            if not block:
                continue
            for inp in inputs:
                if inp and inp not in pg["init_names_set"] and inp not in pg["graph_input_names"]:
                    node_block_inputs[block].add(inp)
            for out in outputs:
                if out:
                    node_block_outputs[block].add(out)

        print(f"\ninter-block tensors (first 3 boundaries):")
        for i in range(min(3, len(blocks) - 1)):
            crossing = node_block_outputs[blocks[i]] & node_block_inputs[blocks[i + 1]]
            if crossing:
                print(f"  {blocks[i]} -> {blocks[i+1]}: {sorted(crossing)[:3]}")


# ---------------------------------------------------------------------------
# Shard mode
# ---------------------------------------------------------------------------

def shard_model(model_path: Path, max_shard_size: int, output_path: Path) -> None:
    pg = parse_graph(model_path)
    buf = pg["buf"]
    nodes = pg["nodes"]
    inits = pg["inits"]
    init_sizes = pg["init_sizes"]
    init_names_set = pg["init_names_set"]
    init_to_block = pg["init_to_block"]
    init_consumers = pg["init_consumers"]
    graph_input_names = pg["graph_input_names"]
    graph_inputs = pg["graph_inputs"]
    graph_outputs = pg["graph_outputs"]
    block_weights = pg["block_weights"]

    # --- Topological sort ---
    # The ONNX optimizer hoists ops out of block boundaries (e.g. Rotary
    # embedding pre/post nodes), so block-name-based partitioning creates
    # unsolvable back-edges. Instead, topologically sort ALL nodes and
    # greedily partition in topo order by weight budget.
    print("\ntopological sorting nodes...")

    # Build adjacency: node index -> set of successor node indices
    output_to_node: dict[str, int] = {}
    for i, (_, _, outputs, _) in enumerate(nodes):
        for out in outputs:
            if out:
                output_to_node[out] = i

    n_nodes = len(nodes)
    in_degree = [0] * n_nodes
    successors: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, (_, inputs, _, _) in enumerate(nodes):
        for inp in inputs:
            if not inp or inp in init_names_set or inp in graph_input_names:
                continue
            producer = output_to_node.get(inp)
            if producer is not None:
                successors[producer].append(i)
                in_degree[i] += 1

    # Kahn's algorithm
    from collections import deque
    queue = deque(i for i in range(n_nodes) if in_degree[i] == 0)
    topo_order: list[int] = []
    while queue:
        node_idx = queue.popleft()
        topo_order.append(node_idx)
        for succ in successors[node_idx]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    assert len(topo_order) == n_nodes, f"cycle detected: {n_nodes - len(topo_order)} nodes in cycle"
    print(f"  {n_nodes} nodes sorted")

    # --- Compute per-node weight contribution ---
    # Each initializer's weight is attributed to the FIRST node in topo
    # order that consumes it. This way, greedy shard filling by topo order
    # accurately tracks cumulative weight.
    init_first_consumer: dict[str, int] = {}  # init name -> topo position
    topo_pos = [0] * n_nodes  # node_idx -> topo position
    for pos, node_idx in enumerate(topo_order):
        topo_pos[node_idx] = pos
    for pos, node_idx in enumerate(topo_order):
        _, inputs, _, _ = nodes[node_idx]
        for inp in inputs:
            if inp in init_names_set and inp not in init_first_consumer:
                init_first_consumer[inp] = pos

    node_weight = [0] * n_nodes  # indexed by topo position
    for init_name, pos in init_first_consumer.items():
        node_weight[pos] += init_sizes.get(init_name, 0)

    # --- Greedy partition in topo order ---
    node_shard = [0] * n_nodes  # indexed by original node index
    shard_weights_list = [0]
    current_shard = 0

    for pos, node_idx in enumerate(topo_order):
        w = node_weight[pos]
        if shard_weights_list[current_shard] + w > max_shard_size and shard_weights_list[current_shard] > 0:
            current_shard += 1
            shard_weights_list.append(0)
        node_shard[node_idx] = current_shard
        shard_weights_list[current_shard] += w

    n_shards = current_shard + 1
    print(f"\npartition: {n_shards} shards (max target: {format_size(max_shard_size)})")
    for i in range(n_shards):
        print(f"  shard {i}: {format_size(shard_weights_list[i])}")

    # --- Assign initializers to shards ---
    # An init goes to every shard that has a node consuming it.
    init_shards: dict[str, set[int]] = defaultdict(set)
    for i, (_, inputs, _, _) in enumerate(nodes):
        shard_idx = node_shard[i]
        for inp in inputs:
            if inp in init_names_set:
                init_shards[inp].add(shard_idx)
    for name in init_names_set:
        if name not in init_shards:
            init_shards[name] = {0}

    dup_count = sum(1 for s in init_shards.values() if len(s) > 1)
    dup_bytes = sum(init_sizes[n] * (len(s) - 1) for n, s in init_shards.items() if len(s) > 1)
    print(f"{dup_count} initializers duplicated across shards ({format_size(dup_bytes)} extra)")

    # Recompute shard weight totals after reassignment
    actual_shard_weights = [0] * n_shards
    for name, size in init_sizes.items():
        for s in init_shards[name]:
            actual_shard_weights[s] += size
    print("actual shard weights after reassignment:")
    for i in range(n_shards):
        print(f"  shard {i}: {format_size(actual_shard_weights[i])}")

    # --- Compute shard I/O ---
    print("\ncomputing shard I/O boundaries...")
    tensor_producer: dict[str, int] = {}
    for i, (_, _, outputs, _) in enumerate(nodes):
        for out in outputs:
            if out:
                tensor_producer[out] = node_shard[i]

    shard_inputs: list[set[str]] = [set() for _ in range(n_shards)]
    shard_outputs: list[set[str]] = [set() for _ in range(n_shards)]

    for i, (_, inputs, _, _) in enumerate(nodes):
        my_shard = node_shard[i]
        for inp in inputs:
            if not inp or inp in init_names_set:
                continue
            if inp in graph_input_names:
                shard_inputs[my_shard].add(inp)
                continue
            producer_shard = tensor_producer.get(inp)
            if producer_shard is not None and producer_shard != my_shard:
                shard_inputs[my_shard].add(inp)
                shard_outputs[producer_shard].add(inp)

    for i in range(n_shards):
        print(f"  shard {i}: {len(shard_inputs[i])} inputs, {len(shard_outputs[i])} outputs")
        for t in sorted(shard_inputs[i]):
            src = "graph" if t in graph_input_names else f"shard {tensor_producer.get(t, '?')}"
            print(f"    in:  {t} (from {src})")
        for t in sorted(shard_outputs[i]):
            print(f"    out: {t}")

    # --- Sanity check: no circular dependencies between shards ---
    print("\nsanity checking shard dependencies...")
    # Build dependency graph: shard X depends on shard Y if X has an input produced by Y
    shard_deps: dict[int, set[int]] = defaultdict(set)
    for i in range(n_shards):
        for t in shard_inputs[i]:
            if t in graph_input_names:
                continue
            producer = tensor_producer.get(t)
            if producer is not None and producer != i:
                shard_deps[i].add(producer)

    # Check for cycles via topological sort
    visited = set()
    in_stack = set()
    order = []
    has_cycle = False

    def visit(s):
        nonlocal has_cycle
        if s in in_stack:
            has_cycle = True
            return
        if s in visited:
            return
        in_stack.add(s)
        for dep in shard_deps.get(s, set()):
            visit(dep)
        in_stack.remove(s)
        visited.add(s)
        order.append(s)

    for s in range(n_shards):
        visit(s)

    if has_cycle:
        print("  FAIL: circular dependency detected between shards!")
        print("  shard dependency graph:")
        for i in range(n_shards):
            deps = shard_deps.get(i, set())
            if deps:
                print(f"    shard {i} depends on: {sorted(deps)}")
        # Show which tensors cause the back-edges
        for i in range(n_shards):
            for t in shard_inputs[i]:
                if t in graph_input_names:
                    continue
                producer = tensor_producer.get(t)
                if producer is not None and producer > i:
                    print(f"    BACK-EDGE: shard {i} needs '{t}' from shard {producer}")
        print("\n  The graph has interleaved nodes: some nodes named /layers.N/...")
        print("  were split across shards by block name, but the ONNX optimizer")
        print("  reordered them so they produce/consume tensors across shard boundaries.")
        print("  Need to reassign these cross-shard nodes to resolve cycles.")
        sys.exit(1)
    else:
        print(f"  ok: valid execution order {order}")

    # --- Resolve types for boundary tensors ---
    # Look up types from graph inputs; for intermediate tensors, use FLOAT as default.
    graph_input_types = {name: (dt, shape) for name, dt, shape, _ in graph_inputs}
    graph_output_types = {name: (dt, shape) for name, dt, shape, _ in graph_outputs}

    def get_tensor_type(tensor_name):
        if tensor_name in graph_input_types:
            return graph_input_types[tensor_name]
        if tensor_name in graph_output_types:
            return graph_output_types[tensor_name]
        return (1, [])  # FLOAT, no shape (ORT will infer)

    # --- Build shard ONNX files ---
    print("\nbuilding shard files...")
    shard_files = []
    model_header_bytes = bytearray()
    for mf in pg["non_graph_fields"]:
        model_header_bytes.extend(buf[mf.tag_start:mf.data_end])
    model_header_bytes = bytes(model_header_bytes)

    for shard_idx in range(n_shards):
        # Collect node fields for this shard
        shard_node_fields = [
            nodes[i][3] for i in range(len(nodes)) if node_shard[i] == shard_idx
        ]
        # Collect init fields for this shard
        shard_init_fields = [
            field for name, _, field in inits if shard_idx in init_shards[name]
        ]
        # Build graph input ValueInfoProto entries
        input_vis = bytearray()
        for tensor_name in sorted(shard_inputs[shard_idx]):
            # Copy original graph inputs verbatim (they have correct types)
            copied = False
            for gi_name, _, _, gi_field in graph_inputs:
                if gi_name == tensor_name:
                    input_vis.extend(buf[gi_field.tag_start:gi_field.data_end])
                    copied = True
                    break
            if not copied:
                # Cross-shard boundary: name-only ValueInfoProto, let ORT infer type
                vi = encode_string_field(FIELD_VALUE_INFO_NAME, tensor_name)
                input_vis.extend(encode_tag(FIELD_GRAPH_INPUT, WIRE_LEN) + encode_len_prefixed(vi))

        # Build graph output ValueInfoProto entries
        output_vis = bytearray()
        # Final shard gets the original graph outputs (correct types)
        if shard_idx == n_shards - 1:
            for _, _, _, go_field in graph_outputs:
                output_vis.extend(buf[go_field.tag_start:go_field.data_end])
        # Cross-shard outputs: name-only, let ORT infer type
        for tensor_name in sorted(shard_outputs[shard_idx]):
            vi = encode_string_field(FIELD_VALUE_INFO_NAME, tensor_name)
            output_vis.extend(encode_tag(FIELD_GRAPH_OUTPUT, WIRE_LEN) + encode_len_prefixed(vi))

        # Assemble graph body
        graph_body = bytearray()
        graph_body.extend(input_vis)
        graph_body.extend(output_vis)
        for nf in shard_node_fields:
            graph_body.extend(buf[nf.tag_start:nf.data_end])
        for inf in shard_init_fields:
            graph_body.extend(buf[inf.tag_start:inf.data_end])

        # Wrap in graph field
        graph_bytes = encode_tag(FIELD_MODEL_GRAPH, WIRE_LEN) + encode_len_prefixed(bytes(graph_body))

        # Full model
        shard_data = model_header_bytes + graph_bytes
        shard_files.append(shard_data)
        print(f"  shard {shard_idx}: {format_size(len(shard_data))} "
              f"({len(shard_node_fields)} nodes, {len(shard_init_fields)} inits)")

    # --- Write shard files ---
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem

    shard_paths = []
    for i, shard_data in enumerate(shard_files):
        shard_path = output_dir / f"{stem}_shard{i}.onnx"
        shard_path.write_bytes(shard_data)
        shard_paths.append(shard_path)
        print(f"  wrote {shard_path} ({format_size(len(shard_data))})")

    # --- Infer types and fix up shards ---
    # ORT requires type info on graph inputs. We run shape inference on each
    # shard incrementally (shard 0 first, using known graph input types, then
    # each subsequent shard using types learned from prior shards' outputs).
    # Each shard is loaded one at a time (~500 MB max) to keep RAM low.
    #
    # In the same pass we also emit a per-shard external data file: we walk
    # the shard's external initializers, copy their bytes out of the source
    # _data file into a fresh per-shard buffer, and rewrite each initializer's
    # `external_data` entries to reference the new per-shard file at its new
    # offset. This means each shard session only mmaps/loads the bytes it
    # actually needs (~150-350 MB) instead of the full 2 GB shared file.
    print("\ninferring boundary tensor types (incremental shape inference)...")
    import onnx
    from onnx import shape_inference

    # Cache of open source data file handles (keyed by `location` string
    # written in the source TensorProto.external_data entries). Opened once
    # per unique location to avoid re-opening the 2 GB file per shard.
    source_data_handles: dict[str, object] = {}
    source_dir = model_path.parent

    def get_source_handle(location: str):
        if location not in source_data_handles:
            source_data_handles[location] = open(source_dir / location, "rb")
        return source_data_handles[location]

    per_shard_data_sizes: list[int] = []

    # Seed with original graph input/output types
    known_types: dict[str, tuple[int, list]] = {}
    for name, dt, shape, _ in graph_inputs:
        known_types[name] = (dt, shape)
    for name, dt, shape, _ in graph_outputs:
        known_types[name] = (dt, shape)

    for i, shard_path in enumerate(shard_paths):
        model = onnx.load(str(shard_path), load_external_data=False)

        # Set input types from known_types
        for inp in model.graph.input:
            if inp.name in known_types:
                dt, shape = known_types[inp.name]
                inp.type.tensor_type.elem_type = dt
                if shape:
                    inp.type.tensor_type.shape.Clear()
                    for kind, val in shape:
                        dim = inp.type.tensor_type.shape.dim.add()
                        if kind == "val":
                            dim.dim_value = val
                        else:
                            dim.dim_param = val

        # Run shape inference to fill in output types
        try:
            model = shape_inference.infer_shapes(model, check_type=False, data_prop=False)
        except Exception as e:
            print(f"  shard {i}: shape inference failed: {e}")
            sys.exit(1)

        # Collect inferred types for subsequent shards
        for vi in model.graph.value_info:
            if vi.type.tensor_type.elem_type:
                known_types[vi.name] = (vi.type.tensor_type.elem_type, [])
        for out in model.graph.output:
            if out.type.tensor_type.elem_type:
                known_types[out.name] = (out.type.tensor_type.elem_type, [])

        # Fix remaining UNDEFINED types on inputs/outputs. Shape inference
        # can't see through custom ops (MatMulNBits, etc.) so some boundary
        # tensors stay UNDEFINED. Default to FLOAT16 since this is a q4f16
        # model with fp16 compute.
        n_fixed = 0
        for inp in model.graph.input:
            if not inp.type.tensor_type.elem_type:
                inp.type.tensor_type.elem_type = 10  # FLOAT16
                known_types[inp.name] = (10, [])
                n_fixed += 1
        for out in model.graph.output:
            if not out.type.tensor_type.elem_type:
                out.type.tensor_type.elem_type = 10  # FLOAT16
                known_types[out.name] = (10, [])
                n_fixed += 1
        if n_fixed:
            print(f"  shard {i}: defaulted {n_fixed} UNDEFINED types to FLOAT16")

        # Extract a per-shard external data file. For each external
        # initializer, read `length` bytes from the source data file at the
        # source `offset`, append to this shard's buffer, then rewrite the
        # initializer's external_data entries so `location` points at the
        # new per-shard file and `offset` points at the new position.
        shard_data_bytes = bytearray()
        n_ext = 0
        for init in model.graph.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL:
                continue
            src_loc: str | None = None
            src_offset = 0
            length = 0
            for entry in init.external_data:
                if entry.key == "location":
                    src_loc = entry.value
                elif entry.key == "offset":
                    src_offset = int(entry.value)
                elif entry.key == "length":
                    length = int(entry.value)
            if src_loc is None:
                print(f"  shard {i}: initializer {init.name!r} has no external location")
                sys.exit(1)
            handle = get_source_handle(src_loc)
            handle.seek(src_offset)
            data = handle.read(length)
            if len(data) != length:
                print(f"  shard {i}: short read for {init.name!r} "
                      f"({len(data)} of {length} bytes)")
                sys.exit(1)
            new_offset = len(shard_data_bytes)
            shard_data_bytes.extend(data)
            new_location = f"{stem}_shard{i}.onnx_data"
            for entry in init.external_data:
                if entry.key == "location":
                    entry.value = new_location
                elif entry.key == "offset":
                    entry.value = str(new_offset)
                # length stays the same
            n_ext += 1

        shard_data_path = shard_path.parent / f"{stem}_shard{i}.onnx_data"
        shard_data_path.write_bytes(bytes(shard_data_bytes))
        per_shard_data_sizes.append(len(shard_data_bytes))
        print(f"  shard {i}: wrote {shard_data_path.name} "
              f"({format_size(len(shard_data_bytes))}, {n_ext} initializers)")

        # Re-save with correct types and rewritten external_data references.
        onnx.save(model, str(shard_path))
        shard_files[i] = shard_path.read_bytes()
        del model
        del shard_data_bytes

        n_in = sum(1 for _ in onnx.load(str(shard_path), load_external_data=False).graph.input)
        n_out = sum(1 for _ in onnx.load(str(shard_path), load_external_data=False).graph.output)
        print(f"  shard {i}: {format_size(len(shard_files[i]))} ({n_in} in, {n_out} out)")

    # Done with source data files; close the handles.
    for h in source_data_handles.values():
        h.close()
    source_data_handles.clear()

    # Per-shard data summary and duplication stats.
    total_shard_data = sum(per_shard_data_sizes)
    unique_init_bytes = sum(init_sizes.values())
    extra = max(0, total_shard_data - unique_init_bytes)
    dup_pct = (100.0 * extra / unique_init_bytes) if unique_init_bytes else 0.0
    print("\nper-shard data summary:")
    for i, size in enumerate(per_shard_data_sizes):
        print(f"  shard {i}: {format_size(size)}")
    print(f"  total across shards: {format_size(total_shard_data)}")
    print(f"  unique initializer bytes: {format_size(unique_init_bytes)}")
    print(f"  duplication overhead: {format_size(extra)} ({dup_pct:.1f}%)")

    # --- Verify each shard ---
    print("\nverifying shards...")
    for i, shard_path in enumerate(shard_paths):
        try:
            model = onnx.load(str(shard_path), load_external_data=False)
            n_nodes = len(model.graph.node)
            n_inits = len(model.graph.initializer)
            n_in = len(model.graph.input)
            n_out = len(model.graph.output)
            print(f"  shard {i}: valid ONNX ({n_nodes} nodes, {n_inits} inits, {n_in} in, {n_out} out)")
            for inp in model.graph.input:
                dt = DTYPE_NAMES.get(inp.type.tensor_type.elem_type, "?")
                print(f"    in:  {inp.name} ({dt})")
            for out in model.graph.output:
                dt = DTYPE_NAMES.get(out.type.tensor_type.elem_type, "?")
                print(f"    out: {out.name} ({dt})")
            del model
        except Exception as e:
            print(f"  shard {i}: FAILED - {e}")
            sys.exit(1)

    # --- Write manifest ---
    print("\nbuilding manifest...")
    manifest = {
        "srcSha256": sha256(buf),
        "srcLen": len(buf),
        "shards": [],
    }
    for i, shard_data in enumerate(shard_files):
        node_ranges = []
        for ni in range(len(nodes)):
            if node_shard[ni] == i:
                nf = nodes[ni][3]
                node_ranges.append({"offset": nf.tag_start, "length": nf.data_end - nf.tag_start})
        init_ranges = []
        for name, _, field in inits:
            if i in init_shards[name]:
                init_ranges.append({"offset": field.tag_start, "length": field.data_end - field.tag_start})

        manifest["shards"].append({
            "fileId": f"{stem}_shard{i}",
            "sha256": sha256(shard_data),
            "len": len(shard_data),
            "nodeRanges": node_ranges,
            "initRanges": init_ranges,
        })

    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"\ndone: {n_shards} shards, {sum(len(s) for s in shard_files):,} bytes total")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and shard DiT-style ONNX transformers at layer boundaries."
    )
    parser.add_argument("model", type=Path, help="Input ONNX transformer file")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect graph structure and report block layout")
    parser.add_argument("--max-shard-size", type=str, default="1.2GB",
                        help="Maximum weight size per shard (default: 1.2GB)")
    parser.add_argument("output", type=Path, nargs="?",
                        help="Output manifest JSON (required unless --inspect)")
    args = parser.parse_args()

    if args.inspect:
        inspect_model(args.model)
    else:
        if not args.output:
            print("error: output path required when not using --inspect")
            sys.exit(1)
        max_size = parse_size(args.max_shard_size)
        shard_model(args.model, max_size, args.output)


if __name__ == "__main__":
    main()
