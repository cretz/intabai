#!/usr/bin/env python3
"""Dump nodes and their outputs for a given node-index range, so we can find
intra-resnet cut tensors (for splitting a too-big resnet session.run)."""

import argparse
import sys
from pathlib import Path

import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    args = ap.parse_args()

    model = onnx.load(str(args.onnx_path), load_external_data=False)
    g = model.graph
    for i in range(args.start, args.end + 1):
        n = g.node[i]
        outs = ", ".join(n.output)
        print(f"{i:5d}  op={n.op_type:20s}  name={n.name}  outputs={outs}")


if __name__ == "__main__":
    main()
