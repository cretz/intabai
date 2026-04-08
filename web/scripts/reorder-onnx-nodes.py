#!/usr/bin/env python3
"""Reorder nodes in an interleaved ONNX graph so each block's ops are contiguous.

MS's ORT graph optimizer hoists MatMulNBits dequantize ops across layer
boundaries for fusion. This makes the graph un-shardable because node ranges
overlap (e.g. layers.0 spans nodes [194..1736] instead of a tight range).

This script topologically re-sorts nodes so each block's ops are grouped
together, producing a clean graph that shard-onnx-layers.py can split.

The reordering does NOT change the computation - ORT builds its own execution
plan from graph topology, not file order. All fused ops (MultiHeadAttention,
SimplifiedLayerNormalization, etc.) are preserved as-is.

Usage:
    python reorder-onnx-nodes.py <input.onnx> <output.onnx>
    python reorder-onnx-nodes.py --verify <reordered.onnx>

The --verify flag runs the inspect analysis on the output to confirm blocks
are now contiguous (tight node ranges).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the shared protobuf parsing infrastructure
from lib.onnx_patch_common import (
    WIRE_LEN,
    encode_len_prefixed,
    encode_tag,
    read_string_field,
    scan_fields,
)

# ONNX protobuf field numbers (same as shard-onnx-layers.py)
FIELD_MODEL_GRAPH = 7
FIELD_GRAPH_NODE = 1
FIELD_GRAPH_INITIALIZER = 5
FIELD_GRAPH_INPUT = 11
FIELD_GRAPH_OUTPUT = 12
FIELD_NODE_NAME = 3
FIELD_NODE_INPUT = 1
FIELD_NODE_OUTPUT = 2


def extract_block_prefix(name: str) -> str | None:
    m = re.match(r"/?([a-zA-Z_]+(?:\.[a-zA-Z_]+)*\.(\d+))", name)
    return m.group(1) if m else None


def block_sort_key(name: str):
    """Sort blocks by family name then numeric index."""
    nums = re.findall(r"(\d+)", name)
    return (name.split(".")[0], int(nums[-1]) if nums else 0)


def parse_nodes(buf, node_fields):
    """Parse node names, inputs, outputs from raw protobuf fields."""
    nodes = []
    for node_f in node_fields:
        node_inner = scan_fields(buf, node_f.data_start, node_f.data_end)
        name = ""
        inputs = []
        outputs = []
        for nf in node_inner:
            if nf.field_number == FIELD_NODE_NAME:
                name = read_string_field(buf, nf)
            elif nf.field_number == FIELD_NODE_INPUT:
                inputs.append(read_string_field(buf, nf))
            elif nf.field_number == FIELD_NODE_OUTPUT:
                outputs.append(read_string_field(buf, nf))
        nodes.append({
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "field": node_f,
            "block": extract_block_prefix(name),
        })
    return nodes


def topological_reorder(nodes, graph_input_names, init_names):
    """Reorder nodes so each block's ops are contiguous, respecting data deps.

    Strategy:
    1. Build a dependency graph (node -> set of predecessor nodes).
    2. Group nodes by block.
    3. Determine block execution order from inter-block data dependencies.
    4. Within each block, topologically sort nodes.
    5. Emit non-block nodes as early as possible (before the first block that
       needs them).
    """
    # Map output tensor -> producing node index
    output_to_node = {}
    for i, node in enumerate(nodes):
        for out in node["outputs"]:
            output_to_node[out] = i

    # Available tensors (graph inputs + initializers don't need a producing node)
    available = set(graph_input_names) | set(init_names)

    # Build predecessor map: node_idx -> set of node_idx it depends on
    predecessors = defaultdict(set)
    for i, node in enumerate(nodes):
        for inp in node["inputs"]:
            if inp and inp not in available and inp in output_to_node:
                pred = output_to_node[inp]
                if pred != i:
                    predecessors[i].add(pred)

    # Group node indices by block
    block_nodes = defaultdict(list)  # block_name -> [node_idx, ...]
    non_block_nodes = []
    for i, node in enumerate(nodes):
        if node["block"]:
            block_nodes[node["block"]].append(i)
        else:
            non_block_nodes.append(i)

    # Determine block execution order from data dependencies between blocks
    block_names = sorted(block_nodes.keys(), key=block_sort_key)

    # For each block, find which other blocks it depends on
    block_deps = defaultdict(set)  # block -> set of blocks it depends on
    for block in block_names:
        for node_idx in block_nodes[block]:
            for pred_idx in predecessors[node_idx]:
                pred_block = nodes[pred_idx]["block"]
                if pred_block and pred_block != block:
                    block_deps[block].add(pred_block)

    # Topological sort of blocks
    ordered_blocks = []
    visited = set()
    temp_visited = set()

    def visit_block(b):
        if b in visited:
            return
        if b in temp_visited:
            # Circular dependency between blocks - this shouldn't happen
            # if the graph is valid, but fall back to sorted order
            print(f"  WARNING: circular dependency involving block {b}", file=sys.stderr)
            return
        temp_visited.add(b)
        for dep in block_deps[b]:
            if dep in block_nodes:
                visit_block(dep)
        temp_visited.discard(b)
        visited.add(b)
        ordered_blocks.append(b)

    for b in block_names:
        visit_block(b)

    # Topological sort of nodes within each block
    def topo_sort_indices(indices):
        """Topologically sort a subset of node indices."""
        idx_set = set(indices)
        local_order = []
        local_visited = set()
        local_temp = set()

        def visit(i):
            if i in local_visited:
                return
            if i in local_temp:
                return  # cycle within block, keep original order
            local_temp.add(i)
            for pred in predecessors[i]:
                if pred in idx_set:
                    visit(pred)
            local_temp.discard(i)
            local_visited.add(i)
            local_order.append(i)

        for i in indices:
            visit(i)
        return local_order

    # For non-block nodes, figure out which block first needs each one
    non_block_needed_before = {}  # node_idx -> first block that needs it
    for block in ordered_blocks:
        for node_idx in block_nodes[block]:
            for pred_idx in predecessors[node_idx]:
                if nodes[pred_idx]["block"] is None and pred_idx not in non_block_needed_before:
                    non_block_needed_before[pred_idx] = block

    # Also handle non-block nodes that depend on other non-block nodes
    # but aren't directly needed by any block (rare, but handle it)
    remaining_non_block = [i for i in non_block_nodes if i not in non_block_needed_before]

    # Build final order
    result = []

    # First: non-block nodes needed before any block, plus those with no
    # block dependency (emit them at the start)
    pre_block_non_block = [i for i in non_block_nodes
                           if i not in non_block_needed_before]
    # Also collect non-block nodes needed before each block
    non_block_before = defaultdict(list)  # block -> [non_block_idx, ...]
    for idx, block in non_block_needed_before.items():
        non_block_before[block].append(idx)

    # Emit pre-block non-block nodes (topologically sorted)
    result.extend(topo_sort_indices(pre_block_non_block))

    # Emit blocks in order, with their prerequisite non-block nodes
    for block in ordered_blocks:
        # Non-block nodes this block needs
        needed = non_block_before.get(block, [])
        if needed:
            result.extend(topo_sort_indices(needed))
        # Block's own nodes
        result.extend(topo_sort_indices(block_nodes[block]))

    # Sanity check
    if len(result) != len(nodes):
        # Some nodes may have been missed - add them at the end
        emitted = set(result)
        for i in range(len(nodes)):
            if i not in emitted:
                result.append(i)
        print(f"  WARNING: {len(result) - len(nodes)} nodes were not placed by "
              f"the reorder algorithm and were appended at the end", file=sys.stderr)

    return result


def reorder(input_path: Path, output_path: Path):
    """Reorder nodes in an ONNX file so blocks are contiguous."""
    print(f"loading {input_path} ({input_path.stat().st_size:,} bytes)...")
    buf = input_path.read_bytes()

    # Parse top-level model fields
    model_fields = scan_fields(buf, 0, len(buf))
    graph_field = None
    non_graph_ranges = []

    for f in model_fields:
        if f.field_number == FIELD_MODEL_GRAPH:
            graph_field = f
        else:
            non_graph_ranges.append((f.tag_start, f.data_end))

    if not graph_field:
        print("error: no graph found in model", file=sys.stderr)
        sys.exit(1)

    # Parse graph contents
    graph_inner = scan_fields(buf, graph_field.data_start, graph_field.data_end)
    node_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_NODE]
    non_node_fields = [f for f in graph_inner if f.field_number != FIELD_GRAPH_NODE]

    print(f"  {len(node_fields)} nodes, {len(non_node_fields)} other graph fields")

    # Parse nodes
    nodes = parse_nodes(buf, node_fields)

    # Collect graph input names and initializer names for dependency resolution
    graph_input_names = set()
    init_names = set()
    for f in graph_inner:
        if f.field_number == FIELD_GRAPH_INPUT:
            inner = scan_fields(buf, f.data_start, f.data_end)
            for sf in inner:
                if sf.field_number == 1:  # name field
                    graph_input_names.add(read_string_field(buf, sf))
        elif f.field_number == FIELD_GRAPH_INITIALIZER:
            inner = scan_fields(buf, f.data_start, f.data_end)
            for sf in inner:
                if sf.field_number == 8:  # tensor name field
                    init_names.add(read_string_field(buf, sf))

    # Show current state
    block_ranges = defaultdict(lambda: [float("inf"), 0])
    for i, node in enumerate(nodes):
        block = node["block"] or "(non-block)"
        block_ranges[block][0] = min(block_ranges[block][0], i)
        block_ranges[block][1] = max(block_ranges[block][1], i)

    print("\ncurrent node ranges (showing interleaving):")
    for block in sorted(block_ranges.keys(), key=lambda b: block_ranges[b][0]):
        lo, hi = block_ranges[block]
        span = hi - lo + 1
        count = sum(1 for n in nodes if (n["block"] or "(non-block)") == block)
        tight = "tight" if span == count else f"INTERLEAVED (span {span}, only {count} nodes)"
        print(f"  {block}: [{lo}..{hi}] ({count} nodes) {tight}")

    # Reorder
    print("\nreordering nodes...")
    new_order = topological_reorder(nodes, graph_input_names, init_names)

    # Show new state
    reordered_nodes = [nodes[i] for i in new_order]
    new_ranges = defaultdict(lambda: [float("inf"), 0])
    for i, node in enumerate(reordered_nodes):
        block = node["block"] or "(non-block)"
        new_ranges[block][0] = min(new_ranges[block][0], i)
        new_ranges[block][1] = max(new_ranges[block][1], i)

    print("\nnew node ranges:")
    all_tight = True
    for block in sorted(new_ranges.keys(), key=lambda b: new_ranges[b][0]):
        lo, hi = new_ranges[block]
        span = hi - lo + 1
        count = sum(1 for n in reordered_nodes if (n["block"] or "(non-block)") == block)
        tight = "tight" if span == count else f"STILL INTERLEAVED (span {span}, only {count} nodes)"
        if span != count:
            all_tight = False
        print(f"  {block}: [{lo}..{hi}] ({count} nodes) {tight}")

    if not all_tight:
        print("\nWARNING: some blocks are still interleaved after reordering.",
              file=sys.stderr)
        print("This may indicate real cross-block data dependencies that",
              file=sys.stderr)
        print("cannot be resolved by reordering alone.", file=sys.stderr)

    # Build output: non-graph model fields + reordered graph
    print(f"\nwriting {output_path}...")
    parts = []

    # Copy non-graph model fields as-is (ir_version, opset_import, etc.)
    for start, end in non_graph_ranges:
        parts.append(buf[start:end])

    # Build new graph: non-node fields (inputs, outputs, initializers) +
    # reordered node fields
    graph_parts = []

    # Non-node graph fields in their original order
    for f in non_node_fields:
        graph_parts.append(buf[f.tag_start:f.data_end])

    # Nodes in new order
    for idx in new_order:
        f = nodes[idx]["field"]
        graph_parts.append(buf[f.tag_start:f.data_end])

    graph_bytes = b"".join(graph_parts)

    # Wrap graph in its field tag
    graph_encoded = encode_tag(FIELD_MODEL_GRAPH, WIRE_LEN) + encode_len_prefixed(graph_bytes)
    parts.append(graph_encoded)

    output_bytes = b"".join(parts)
    output_path.write_bytes(output_bytes)
    print(f"  wrote {len(output_bytes):,} bytes ({len(output_bytes) / 1e9:.2f} GB)")

    if all_tight:
        print("\nAll blocks are now contiguous. Ready for sharding:")
        print(f"  python shard-onnx-layers.py {output_path} --max-shard-size 1.2GB <manifest.json>")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input ONNX file (interleaved)")
    parser.add_argument("output", type=Path, nargs="?", help="Output ONNX file (reordered)")
    parser.add_argument("--verify", action="store_true",
                        help="Just verify the input is already contiguous (no output)")
    args = parser.parse_args()

    if args.verify or args.output is None:
        # Just inspect/verify mode
        if args.output:
            print("--verify mode: ignoring output path", file=sys.stderr)
        print(f"loading {args.input} ({args.input.stat().st_size:,} bytes)...")
        buf = args.input.read_bytes()
        model_fields = scan_fields(buf, 0, len(buf))
        graph_field = next(f for f in model_fields if f.field_number == FIELD_MODEL_GRAPH)
        graph_inner = scan_fields(buf, graph_field.data_start, graph_field.data_end)
        node_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_NODE]
        nodes = parse_nodes(buf, node_fields)

        block_ranges = defaultdict(lambda: [float("inf"), 0])
        block_counts = defaultdict(int)
        for i, node in enumerate(nodes):
            block = node["block"] or "(non-block)"
            block_ranges[block][0] = min(block_ranges[block][0], i)
            block_ranges[block][1] = max(block_ranges[block][1], i)
            block_counts[block] += 1

        all_tight = True
        for block in sorted(block_ranges.keys(), key=lambda b: block_ranges[b][0]):
            lo, hi = block_ranges[block]
            span = hi - lo + 1
            count = block_counts[block]
            status = "tight" if span == count else f"INTERLEAVED (span {span}, only {count} nodes)"
            if span != count:
                all_tight = False
            print(f"  {block}: [{lo}..{hi}] ({count} nodes) {status}")

        if all_tight:
            print("\nAll blocks are contiguous.")
        else:
            print("\nSome blocks are interleaved. Run without --verify to reorder.")
        sys.exit(0 if all_tight else 1)

    reorder(args.input, args.output)


if __name__ == "__main__":
    main()
