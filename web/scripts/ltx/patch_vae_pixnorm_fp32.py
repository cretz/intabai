#!/usr/bin/env python3
"""Insert fp32 Cast nodes around PixelNorm chains in LTX VAE decoder shards.

LTX VAE residual blocks contain a PixelNorm of the form

    y = x / sqrt(mean(x ** 2, axis=channels))

which on fp16 saturates the squared term to +Inf when |x| > sqrt(65504).
The 2B distilled decoder hits |x| ~ 680 at dec_block_00_res_0.conv3d, so
x**2 ~ 462k -> +Inf in 62k of 800k positions. ORT-web's WebGPU EP and
WASM EP disagree on how ReduceMean treats those Infs (wasm produces
finite-but-saturated results, webgpu produces NaN), and the divergence
cascades into green-tile garbage at the decoder's output.

This patch keeps weights in fp16 but routes the Pow -> ReduceMean ->
Sqrt -> Div chain through fp32. Both Pow inputs are cast to fp32, the
Div numerator (same tensor as Pow's first input) reads from the f32
cast, and the Div output is cast back to fp16 so downstream consumers
remain unchanged. No new initializers, so the .onnx.data sidecar is
reused as-is.

Run:
  cd intabai/web/scripts && uv run python ltx/patch_vae_pixnorm_fp32.py \\
    --vae-dir C:/work/personal/intabai/notes/models/ltx/staging/vae

Outputs <stem>_pixnorm_fp32.onnx alongside each input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, TensorProto

PATCHED_SUFFIX = "_pixnorm_fp32"


def find_chains(graph):
    """Yield (pow, reducemean, sqrt, div) tuples for every PixelNorm chain.

    Chain shape: Pow(x, e) -> ReduceMean -> Sqrt -> Div(x, sqrt_out).
    Only matches chains where Div's numerator is bit-identical to Pow's
    first input (same tensor name).
    """
    consumers: dict[str, list] = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    def sole(tensor_name: str, op_type: str):
        cs = consumers.get(tensor_name, [])
        if len(cs) != 1 or cs[0].op_type != op_type:
            return None
        return cs[0]

    for pow_node in graph.node:
        if pow_node.op_type != "Pow" or len(pow_node.output) != 1:
            continue
        red = sole(pow_node.output[0], "ReduceMean")
        if red is None:
            continue
        sqrt = sole(red.output[0], "Sqrt")
        if sqrt is None:
            continue
        # Div may share Sqrt's output with other ops in pathological
        # graphs; require exact PixelNorm shape.
        div = None
        for c in consumers.get(sqrt.output[0], []):
            if (
                c.op_type == "Div"
                and len(c.input) >= 2
                and c.input[1] == sqrt.output[0]
                and c.input[0] == pow_node.input[0]
            ):
                div = c
                break
        if div is None:
            continue
        yield pow_node, red, sqrt, div


def patch_chain(chain, idx: int):
    pow_node, _red, _sqrt, div = chain
    x = pow_node.input[0]
    exp = pow_node.input[1]
    div_out = div.output[0]

    sfx = f"__pixnorm_fp32_{idx}"
    x_f32 = f"{x}{sfx}_xf32"
    exp_f32 = f"{exp}{sfx}_expf32"
    div_out_f32 = f"{div_out}{sfx}_outf32"

    cast_x = helper.make_node(
        "Cast", inputs=[x], outputs=[x_f32],
        to=TensorProto.FLOAT, name=f"PixNormCastX{sfx}",
    )
    cast_exp = helper.make_node(
        "Cast", inputs=[exp], outputs=[exp_f32],
        to=TensorProto.FLOAT, name=f"PixNormCastExp{sfx}",
    )
    cast_out = helper.make_node(
        "Cast", inputs=[div_out_f32], outputs=[div_out],
        to=TensorProto.FLOAT16, name=f"PixNormCastOut{sfx}",
    )

    pow_node.input[0] = x_f32
    pow_node.input[1] = exp_f32
    div.input[0] = x_f32
    div.output[0] = div_out_f32

    return [cast_x, cast_exp, cast_out]


def patch_file(src: Path, dst: Path) -> int:
    model = onnx.load(str(src), load_external_data=False)
    graph = model.graph

    chains = list(find_chains(graph))
    if not chains:
        return 0

    new_nodes = []
    for i, chain in enumerate(chains):
        new_nodes.extend(patch_chain(chain, i))
    for n in new_nodes:
        graph.node.append(n)

    # Pow/ReduceMean/Sqrt outputs are now fp32. Drop stale value_info so
    # ORT re-infers; Div's output keeps its original name and is produced
    # by the new f16 Cast, so its value_info (if any) is still correct.
    retyped = set()
    for pow_node, red, sqrt, _div in chains:
        retyped.add(pow_node.output[0])
        retyped.add(red.output[0])
        retyped.add(sqrt.output[0])
    keep = [vi for vi in graph.value_info if vi.name not in retyped]
    del graph.value_info[:]
    graph.value_info.extend(keep)

    # Save graph proto only. Initializers retain their existing
    # external_data refs to <stem>.onnx.data; we touched no weights, so
    # the sidecar is reused unchanged.
    onnx.save_model(model, str(dst), save_as_external_data=False)
    return len(chains)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vae-dir", type=Path, required=True,
                    help="Directory holding <stem>.onnx + <stem>.onnx.data")
    args = ap.parse_args()

    vae_dir: Path = args.vae_dir
    if not vae_dir.is_dir():
        ap.error(f"--vae-dir not a directory: {vae_dir}")

    print(f"Scanning {vae_dir}")
    total_chains = 0
    patched_files = 0
    skipped = 0
    for src in sorted(vae_dir.glob("*.onnx")):
        stem = src.stem
        if stem.endswith(PATCHED_SUFFIX):
            continue
        dst = vae_dir / f"{stem}{PATCHED_SUFFIX}.onnx"
        n = patch_file(src, dst)
        if n == 0:
            skipped += 1
            continue
        patched_files += 1
        total_chains += n
        print(f"  {stem}: {n} chain(s) -> {dst.name}")

    print(f"\nPatched {patched_files} files, {total_chains} chains total "
          f"({skipped} files had no PixelNorm chains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
