#!/usr/bin/env python3
"""Inspect layer_00.onnx: list ops, look for Cast nodes around RMSNorm."""
import onnx, sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "../../../notes/models/fastwan/hf-repo/onnx/text-encoder/layer_00.onnx")
m = onnx.load(str(p))
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("op counts:")
for k in sorted(ops): print(f"  {k}: {ops[k]}")
print(f"\ntotal nodes: {len(m.graph.node)}")

# Find first 30 nodes and the Cast nodes
print("\nfirst 20 nodes:")
for n in m.graph.node[:20]:
    print(f"  {n.op_type:20s} {list(n.input)} -> {list(n.output)}")

print("\nall Cast nodes (to dtype):")
for n in m.graph.node:
    if n.op_type == "Cast":
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        # 1=fp32, 10=fp16, 16=bf16
        dtype = {1: "fp32", 10: "fp16", 16: "bf16"}.get(to, f"dtype{to}")
        print(f"  {list(n.input)} -> {list(n.output)}  to={dtype}")

print("\nall Pow and ReduceMean nodes (RMSNorm signature):")
for n in m.graph.node:
    if n.op_type in ("Pow", "ReduceMean", "Sqrt", "Rsqrt"):
        print(f"  {n.op_type:12s} {list(n.input)} -> {list(n.output)}")
