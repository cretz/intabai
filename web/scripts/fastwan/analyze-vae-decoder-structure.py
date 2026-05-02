#!/usr/bin/env python3
"""Analyze decoder_init.onnx / decoder_step.onnx topology to find cut points
for per-up_block session splitting.

Goal: identify the tensors produced at the boundary between conv_in, mid_block,
up_blocks.0..3, and conv_out so a splitter can use them as sub-graph I/O.

Prints:
  - graph I/O summary
  - node-name prefix histogram (which `up_blocks.N.*` groups exist)
  - for each boundary, the candidate output tensor name(s) — the value that
    feeds the *next* block's first node, which is the cut tensor.

Usage:
    uv run analyze-vae-decoder-structure.py <path/to/decoder_init.onnx>
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import onnx


def node_group(name: str, depth: int = 2) -> str:
    """Bucket a node name by its leading dotted path prefix.
    depth=2: up_blocks.0, mid_block
    depth=3: up_blocks.0/resnets.0, mid_block/attentions.0
    """
    if not name:
        return "<unnamed>"
    parts = name.split("/")
    # depth=2 path: first macro block
    macro = None
    macro_idx = -1
    for i, p in enumerate(parts):
        if p.startswith("up_blocks."):
            macro = ".".join(p.split(".")[:2])
            macro_idx = i
            break
        if p in ("mid_block", "conv_in", "conv_out", "conv_norm_out", "conv_act"):
            macro = p
            macro_idx = i
            break
    if macro is None:
        return "/".join(parts[:3]) if len(parts) >= 3 else name
    if depth <= 2:
        return macro
    # depth=3: append next non-empty path segment
    if macro_idx + 1 < len(parts) and parts[macro_idx + 1]:
        return f"{macro}/{parts[macro_idx + 1]}"
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("--depth", type=int, default=2, help="grouping depth (2=macro, 3=sub)")
    args = ap.parse_args()

    print(f"loading {args.onnx_path} ...", flush=True)
    model = onnx.load(str(args.onnx_path), load_external_data=False)
    g = model.graph

    print(f"\n=== graph I/O ===")
    print(f"inputs ({len(g.input)}):")
    for t in g.input:
        dims = [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]
        print(f"  {t.name}  {dims}  dtype={t.type.tensor_type.elem_type}")
    print(f"outputs ({len(g.output)}):")
    for t in g.output[:5]:
        dims = [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]
        print(f"  {t.name}  {dims}  dtype={t.type.tensor_type.elem_type}")
    if len(g.output) > 5:
        print(f"  ... ({len(g.output) - 5} more outputs)")

    print(f"\n=== node count: {len(g.node)} ===")

    groups = Counter()
    first_node_in_group: dict[str, int] = {}
    last_node_in_group: dict[str, int] = {}
    for i, n in enumerate(g.node):
        grp = node_group(n.name, args.depth)
        groups[grp] += 1
        if grp not in first_node_in_group:
            first_node_in_group[grp] = i
        last_node_in_group[grp] = i

    # order by first appearance
    ordered = sorted(groups.keys(), key=lambda k: first_node_in_group[k])
    print(f"\n=== node groups (in topological order) ===")
    for grp in ordered:
        first = first_node_in_group[grp]
        last = last_node_in_group[grp]
        print(f"  {grp:40s}  count={groups[grp]:5d}  nodes[{first}..{last}]")

    # For each group boundary, print the output tensor of the last node of that group
    # (candidate cut tensor) — but only if that tensor is consumed by a node in a later group.
    print(f"\n=== candidate cut tensors (group-last outputs consumed by later groups) ===")
    # Build index: tensor -> consumers (node indices)
    consumers: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(g.node):
        for inp in n.input:
            consumers[inp].append(i)

    for grp in ordered:
        last_idx = last_node_in_group[grp]
        last_node = g.node[last_idx]
        for out in last_node.output:
            consumer_idxs = consumers.get(out, [])
            later = [ci for ci in consumer_idxs if ci > last_idx]
            if later:
                first_later_group = node_group(g.node[later[0]].name, args.depth)
                print(f"  {grp} -> {first_later_group}: tensor '{out}' (consumed by node {later[0]})")


if __name__ == "__main__":
    main()
