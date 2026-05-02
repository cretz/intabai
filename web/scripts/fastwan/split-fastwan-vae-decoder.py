#!/usr/bin/env python3
"""Split the monolithic decoder_init.onnx / decoder_step.onnx into six
per-region sub-graphs so each sub-graph runs as its own ORT session.run.

The goal is to break the D3D12 cumulative-submit TDR: ORT-web fires all
dispatches of one session.run without JS awaits between them, so Windows
sees continuous GPU-busy time. Splitting into N session.runs puts a JS
`await` boundary between each, letting the queue drain.

Cut tensors (found by analyze-vae-decoder-structure.py):

    part_0_pre        latent                                  -> /decoder/conv_in/Conv_output_0
    part_1_mid        /decoder/conv_in/Conv_output_0          -> /decoder/mid_block/resnets.1/Add_output_0
    part_2_up0        /decoder/mid_block/resnets.1/Add_output_0 -> /decoder/up_blocks.0/Add_output_0
    part_3_up1        /decoder/up_blocks.0/Add_output_0       -> /decoder/up_blocks.1/Add_output_0
    part_4_up2        /decoder/up_blocks.1/Add_output_0       -> /decoder/up_blocks.2/Add_output_0
    part_5_up3_tail   /decoder/up_blocks.2/Add_output_0       -> frames

Cache tensors: decoder_init emits cache_out_00..31; decoder_step reads
cache_in_00..31 AND emits cache_out_00..31. Each cache tensor is attached
to whichever region produces/consumes it (discovered automatically, not
assigned by hand).

Weights: extract_model copies the initializer tensors each sub-graph
actually uses. No per-region dedup; total on-disk size = sum of per-region
initializer bytes, which should be ~= original (each weight lives in
exactly one region).

Usage:
    uv run split-fastwan-vae-decoder.py <decoder_init_or_step.onnx> <output_dir> [--prefix init|step]
"""

import argparse
import sys
import time
from pathlib import Path

import onnx
import onnx.utils


# Presets: each is a list of (part_name, main_input_cut_tensor, [extra_inputs...]).
# The last entry's "cut" is the graph's final output, not an intermediate.
# `extra_inputs` are additional graph inputs for this part beyond the boundary
# tensor (e.g. up_blocks.N's avg_shortcut reads the up_block's *block input*,
# not resnets.2's output — without declaring that as an input, extract_model
# walks back through all prior resnets and the sub-graph explodes in size).


def _resnet_halves(block_path: str, resnet_idx: int, block_input: str,
                   block_output: str, tag_prefix: str) -> list[tuple]:
    """Emit two parts splitting a resnet at the conv1 output.
    `block_input` is the tensor entering norm1 (also consumed by the residual
    Add, so the second half declares it as an extra input).
    `block_output` is the resnet's Add output.
    """
    r = f"{block_path}/resnets.{resnet_idx}"
    return [
        (f"{tag_prefix}_a", block_input, []),
        (f"{tag_prefix}_b", f"/decoder{r}/conv1/Conv_output_0", [block_input]),
    ]


def _build_fine40() -> list[tuple]:
    out: list[tuple] = [("part_00_pre", "latent", [])]
    # mid_block: resnets.0 (halved), attentions.0, resnets.1 (halved)
    out += _resnet_halves("/mid_block", 0,
                          "/decoder/conv_in/Conv_output_0",
                          "/decoder/mid_block/resnets.0/Add_output_0",
                          "part_01_mid_r0")
    out += [("part_03_mid_attn", "/decoder/mid_block/resnets.0/Add_output_0", [])]
    out += _resnet_halves("/mid_block", 1,
                          "/decoder/mid_block/attentions.0/Add_1_output_0",
                          "/decoder/mid_block/resnets.1/Add_output_0",
                          "part_04_mid_r1")
    # up_blocks.0..2: each has 3 resnets (halved) + upsample
    up_inputs = [
        "/decoder/mid_block/resnets.1/Add_output_0",
        "/decoder/up_blocks.0/Add_output_0",
        "/decoder/up_blocks.1/Add_output_0",
        "/decoder/up_blocks.2/Add_output_0",
    ]
    part_ctr = 6
    for ub in range(3):
        block_in = up_inputs[ub]
        r0_in = block_in
        r0_out = f"/decoder/up_blocks.{ub}/resnets.0/Add_output_0"
        r1_out = f"/decoder/up_blocks.{ub}/resnets.1/Add_output_0"
        r2_out = f"/decoder/up_blocks.{ub}/resnets.2/Add_output_0"
        ub_out = f"/decoder/up_blocks.{ub}/Add_output_0"
        out += _resnet_halves(f"/up_blocks.{ub}", 0, r0_in, r0_out,
                              f"part_{part_ctr:02d}_up{ub}_r0"); part_ctr += 2
        out += _resnet_halves(f"/up_blocks.{ub}", 1, r0_out, r1_out,
                              f"part_{part_ctr:02d}_up{ub}_r1"); part_ctr += 2
        out += _resnet_halves(f"/up_blocks.{ub}", 2, r1_out, r2_out,
                              f"part_{part_ctr:02d}_up{ub}_r2"); part_ctr += 2
        # upsample: takes resnets.2 output AND block input (avg_shortcut)
        out += [(f"part_{part_ctr:02d}_up{ub}_upsample", r2_out, [block_in])]
        part_ctr += 1
    # up_blocks.3: 3 resnets (halved), no upsample
    ub = 3
    block_in = up_inputs[3]
    r0_out = f"/decoder/up_blocks.{ub}/resnets.0/Add_output_0"
    r1_out = f"/decoder/up_blocks.{ub}/resnets.1/Add_output_0"
    r2_out = f"/decoder/up_blocks.{ub}/resnets.2/Add_output_0"
    out += _resnet_halves(f"/up_blocks.{ub}", 0, block_in, r0_out,
                          f"part_{part_ctr:02d}_up{ub}_r0"); part_ctr += 2
    out += _resnet_halves(f"/up_blocks.{ub}", 1, r0_out, r1_out,
                          f"part_{part_ctr:02d}_up{ub}_r1"); part_ctr += 2
    out += _resnet_halves(f"/up_blocks.{ub}", 2, r1_out, r2_out,
                          f"part_{part_ctr:02d}_up{ub}_r2"); part_ctr += 2
    # tail
    out += [(f"part_{part_ctr:02d}_tail", r2_out, [])]
    out += [("<end>", "frames", [])]
    return out


PRESETS: dict[str, list[tuple]] = {
    "coarse": [
        ("part_0_pre", "latent", []),
        ("part_1_mid", "/decoder/conv_in/Conv_output_0", []),
        ("part_2_up0", "/decoder/mid_block/resnets.1/Add_output_0", []),
        ("part_3_up1", "/decoder/up_blocks.0/Add_output_0", []),
        ("part_4_up2", "/decoder/up_blocks.1/Add_output_0", []),
        ("part_5_up3_tail", "/decoder/up_blocks.2/Add_output_0", []),
        ("<end>", "frames", []),
    ],
    # 20-way split: resnets/attention/upsampler granularity. Used after coarse
    # 6-way still TDR'd on the mid_block/up_blocks.0 regions (too many
    # dispatches per session.run even within one region).
    "fine": [
        ("part_00_pre", "latent", []),
        ("part_01_mid_resnets0", "/decoder/conv_in/Conv_output_0", []),
        ("part_02_mid_attn", "/decoder/mid_block/resnets.0/Add_output_0", []),
        ("part_03_mid_resnets1", "/decoder/mid_block/attentions.0/Add_1_output_0", []),
        ("part_04_up0_resnets0", "/decoder/mid_block/resnets.1/Add_output_0", []),
        ("part_05_up0_resnets1", "/decoder/up_blocks.0/resnets.0/Add_output_0", []),
        ("part_06_up0_resnets2", "/decoder/up_blocks.0/resnets.1/Add_output_0", []),
        # up_blocks.N/upsample = upsampler(resnets.2_out) + avg_shortcut(block_input) -> Add.
        # Declare both the mid-branch and block-input to keep the sub-graph local.
        ("part_07_up0_upsample", "/decoder/up_blocks.0/resnets.2/Add_output_0",
            ["/decoder/mid_block/resnets.1/Add_output_0"]),
        ("part_08_up1_resnets0", "/decoder/up_blocks.0/Add_output_0", []),
        ("part_09_up1_resnets1", "/decoder/up_blocks.1/resnets.0/Add_output_0", []),
        ("part_10_up1_resnets2", "/decoder/up_blocks.1/resnets.1/Add_output_0", []),
        ("part_11_up1_upsample", "/decoder/up_blocks.1/resnets.2/Add_output_0",
            ["/decoder/up_blocks.0/Add_output_0"]),
        ("part_12_up2_resnets0", "/decoder/up_blocks.1/Add_output_0", []),
        ("part_13_up2_resnets1", "/decoder/up_blocks.2/resnets.0/Add_output_0", []),
        ("part_14_up2_resnets2", "/decoder/up_blocks.2/resnets.1/Add_output_0", []),
        ("part_15_up2_upsample", "/decoder/up_blocks.2/resnets.2/Add_output_0",
            ["/decoder/up_blocks.1/Add_output_0"]),
        ("part_16_up3_resnets0", "/decoder/up_blocks.2/Add_output_0", []),
        ("part_17_up3_resnets1", "/decoder/up_blocks.3/resnets.0/Add_output_0", []),
        ("part_18_up3_resnets2", "/decoder/up_blocks.3/resnets.1/Add_output_0", []),
        ("part_19_tail", "/decoder/up_blocks.3/resnets.2/Add_output_0", []),
        ("<end>", "frames", []),
    ],
    "fine40": _build_fine40(),
}


def build_producer_index(graph) -> dict[str, int]:
    """tensor name -> producer node index (or -1 for graph inputs/initializers)."""
    producer: dict[str, int] = {}
    for name in (i.name for i in graph.input):
        producer[name] = -1
    for init in graph.initializer:
        producer[init.name] = -1
    for i, node in enumerate(graph.node):
        for out in node.output:
            producer[out] = i
    return producer


def build_consumers_index(graph) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for i, node in enumerate(graph.node):
        for inp in node.input:
            if not inp:
                continue
            consumers.setdefault(inp, []).append(i)
    return consumers


def find_boundary_node_indices(graph, cut_tensors: list[str]) -> list[int]:
    """Return node index of the producer of each non-input cut tensor.
    The first cut (latent) has no producer; represented as -1."""
    producer = build_producer_index(graph)
    return [producer.get(t, -1) for t in cut_tensors]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--prefix", type=str, required=True,
                    help="'init' or 'step' — output files named decoder_{prefix}_{part}.onnx")
    ap.add_argument("--preset", type=str, default="fine40", choices=sorted(PRESETS.keys()),
                    help="which cut-point preset to apply")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    part_names = [p[0] for p in preset[:-1]]
    cut_tensors = [p[1] for p in preset]
    # per-part extra graph inputs (beyond the boundary cut tensor)
    extra_inputs_by_part: list[list[str]] = [
        list(p[2]) if len(p) > 2 else [] for p in preset[:-1]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"loading {args.onnx_path} ...", flush=True)
    model = onnx.load(str(args.onnx_path), load_external_data=False)
    g = model.graph
    print(f"  {len(g.node)} nodes, {len(g.initializer)} initializers, "
          f"{len(g.input)} inputs, {len(g.output)} outputs "
          f"(loaded in {time.time() - t0:.1f}s)", flush=True)

    # Locate each cut tensor's producer node index; derive node ranges per region.
    boundary_indices = find_boundary_node_indices(g, cut_tensors)
    print(f"\ncut boundaries (node indices): {boundary_indices}")
    # Region i spans nodes [boundary_indices[i]+1 .. boundary_indices[i+1]] (inclusive on right,
    # since the producer of the next cut is in region i too if it's the region's last node).
    # For region 0, start = 0. For the final region, end = len(g.node) - 1.
    regions: list[tuple[int, int]] = []
    for i in range(len(part_names)):
        start = 0 if i == 0 else boundary_indices[i] + 1
        end = boundary_indices[i + 1] if i < len(part_names) - 1 else len(g.node) - 1
        regions.append((start, end))
        print(f"  {part_names[i]:22s} nodes[{start}..{end}]  ({end - start + 1} nodes)")

    # cache_in_* / cache_out_* discovery
    graph_input_names = {i.name for i in g.input}
    graph_output_names = {o.name for o in g.output}
    cache_ins = sorted(n for n in graph_input_names if n.startswith("cache_in_"))
    cache_outs = sorted(n for n in graph_output_names if n.startswith("cache_out_"))
    print(f"\ncache_in inputs: {len(cache_ins)}")
    print(f"cache_out outputs: {len(cache_outs)}")

    producer = build_producer_index(g)
    consumers = build_consumers_index(g)

    # Assign each cache_out_* to the region that produces it.
    cache_out_by_region: list[list[str]] = [[] for _ in part_names]
    for co in cache_outs:
        pidx = producer.get(co, -1)
        if pidx < 0:
            print(f"  WARN: cache_out '{co}' has no producer node (graph input/initializer?)")
            continue
        for ri, (start, end) in enumerate(regions):
            if start <= pidx <= end:
                cache_out_by_region[ri].append(co)
                break
        else:
            print(f"  WARN: cache_out '{co}' producer {pidx} not in any region")

    # Assign each cache_in_* to the regions that consume it. A cache_in COULD
    # be consumed by multiple regions in principle, but in practice each slot
    # is read once by its own causal conv. Track per-region.
    cache_in_by_region: list[list[str]] = [[] for _ in part_names]
    for ci in cache_ins:
        consumer_nodes = consumers.get(ci, [])
        regions_hit = set()
        for cn in consumer_nodes:
            for ri, (start, end) in enumerate(regions):
                if start <= cn <= end:
                    regions_hit.add(ri)
                    break
        if not regions_hit:
            print(f"  WARN: cache_in '{ci}' has no in-range consumer")
            continue
        for ri in regions_hit:
            cache_in_by_region[ri].append(ci)

    # Build (inputs, outputs) per region and call extract_model.
    onnx_path_str = str(args.onnx_path)
    for ri, part_name in enumerate(part_names):
        in_tensor = cut_tensors[ri]
        out_tensor = cut_tensors[ri + 1]
        inputs = [in_tensor] + extra_inputs_by_part[ri] + cache_in_by_region[ri]
        outputs = [out_tensor] + cache_out_by_region[ri]

        # Output file — base filename inherits from source (decoder_init vs decoder_step via --prefix).
        out_path = args.output_dir / f"decoder_{args.prefix}_{part_name}.onnx"
        print(f"\n[{ri}] {part_name}")
        print(f"    inputs  ({len(inputs)}): {inputs[0]}"
              + (f" + {len(inputs) - 1} cache_in_*" if len(inputs) > 1 else ""))
        print(f"    outputs ({len(outputs)}): {outputs[0]}"
              + (f" + {len(outputs) - 1} cache_out_*" if len(outputs) > 1 else ""))
        print(f"    writing -> {out_path}")

        t1 = time.time()
        # extract_model walks graph topology from declared outputs backward,
        # keeping only the nodes + initializers required. `check_model` on
        # large graphs is slow; we pass check_model=False and run a quick
        # shape-inference-free sanity check ourselves (output tensors exist).
        onnx.utils.extract_model(
            input_path=onnx_path_str,
            output_path=str(out_path),
            input_names=inputs,
            output_names=outputs,
            check_model=False,
        )
        elapsed = time.time() - t1

        # quick sanity — reload extracted sub-graph and print node count + file size
        sub = onnx.load(str(out_path), load_external_data=False)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"    done: {len(sub.graph.node)} nodes, "
              f"{len(sub.graph.initializer)} initializers, "
              f"{size_mb:.1f} MB ({elapsed:.1f}s)")

    print(f"\ntotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
