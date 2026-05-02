"""Minimal batched-attention probe for WebGPU-vs-CPU divergence diagnosis.

Builds a tiny ONNX that runs the exact op pattern of the failing block_00
attn1: Q -> K^T -> MatMul -> Scale -> Softmax -> MatMul(.V). Inputs are
deterministic fp16 fills (sin-based, reproducible in JS) so the smoke
candidate can feed byte-identical values.

Usage:
  uv run probe-attn-matmul.py --seq 2048 --heads 24 --head-dim 128 \
      --out ../../../notes/models/fastwan/hf-repo/onnx/probe/probe-2048.onnx

The model-smoke page loads the same .onnx and fills inputs with the same
deterministic formula, then diffs output first-32 hex against the log
this script emits.

Shapes follow FastWan block_00 attn1:
  Q: [1, heads, seq, head_dim]   fp16
  K: [1, heads, seq, head_dim]   fp16 (we transpose inside the graph)
  V: [1, heads, seq, head_dim]   fp16
  -> out: [1, heads, seq, head_dim] fp16

At seq=8190, the Q.K^T intermediate is 1*heads*seq*seq fp16 =
heads*seq^2*2 bytes. heads=24,seq=8190: 24*8190^2*2 = 3.22 GB. That's
larger than most WebGPU maxBufferSize limits. Test seq values on either
side of the cliff to localize.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def deterministic_fp16(name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Deterministic fp16 fill. Uses sin(i*k + offset) with a name-derived
    offset so Q, K, V get different but reproducible values. Both Python
    and JS compute sin() in IEEE double, then cast to fp32 then fp16 with
    round-to-nearest-even - byte-identical across platforms.
    """
    n = int(np.prod(shape))
    # name-derived offset keeps inputs distinct but stable
    offset_map = {"q": 0.0, "k": 1.1, "v": 2.2}
    off = offset_map.get(name.lower(), 0.0)
    idx = np.arange(n, dtype=np.float64)
    # small-amplitude signal (similar in magnitude to Q/K/V real values
    # post-RoPE at ~±10): use sin() * 8
    vals = (np.sin(idx * 0.0017 + off) * 8.0).astype(np.float32).astype(np.float16)
    return vals.reshape(shape)


def build_onnx(seq: int, heads: int, head_dim: int, out_path: Path) -> None:
    """Graph:  Transpose(K) -> MatMul(Q, Kt) -> Mul(scale) -> Softmax -> MatMul(sm, V) -> out
    No RoPE, no projections. Just the failing chain.
    """
    B = 1
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT16, [B, heads, seq, head_dim])
    k = helper.make_tensor_value_info("k", TensorProto.FLOAT16, [B, heads, seq, head_dim])
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT16, [B, heads, seq, head_dim])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [B, heads, seq, head_dim])

    scale_val = 1.0 / math.sqrt(head_dim)
    scale_init = helper.make_tensor(
        "scale", TensorProto.FLOAT16,
        [1], np.array([scale_val], dtype=np.float16).tobytes(), raw=True,
    )

    nodes = [
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2], name="T_k"),
        helper.make_node("MatMul", ["q", "kt"], ["scores_raw"], name="MM_qkt"),
        helper.make_node("Mul", ["scores_raw", "scale"], ["scores"], name="Scale"),
        helper.make_node("Softmax", ["scores"], ["weights"], axis=-1, name="SM"),
        helper.make_node("MatMul", ["weights", "v"], ["out"], name="MM_scoreV"),
    ]

    graph = helper.make_graph(
        nodes, "probe_attn", [q, k, v], [out], initializer=[scale_init],
    )
    # opset 23 matches the transformer export
    opset = helper.make_opsetid("", 23)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))


def run_cpu_ref(onnx_path: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"q": q, "k": k, "v": v})[0]
    return out


def stats(name: str, a: np.ndarray) -> str:
    flat = a.astype(np.float32).ravel()
    n = flat.size
    nan = int(np.isnan(flat).sum())
    zero = int((flat == 0).sum())
    return (
        f"stats[{name}] n={n} min={flat.min():.4f} max={flat.max():.4f} "
        f"mean={flat.mean():.6f} nan={nan} zeros={zero}"
    )


def first32_hex(a: np.ndarray) -> str:
    """First 32 fp16 values of flattened tensor as uint16 hex, matching
    browser dump format (copyF16Bits)."""
    bits = a.astype(np.float16).view(np.uint16).ravel()[:32]
    return ",".join(f"{b:04x}" for b in bits)


def first32_f32(a: np.ndarray) -> str:
    flat = a.astype(np.float32).ravel()[:32]
    return "[" + ",".join(f"{x:.4f}" for x in flat) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True, help="path to write probe ONNX")
    args = ap.parse_args()

    shape = (1, args.heads, args.seq, args.head_dim)
    print(f"probe shape Q/K/V = {shape}")
    print(f"building {args.out}")
    build_onnx(args.seq, args.heads, args.head_dim, args.out)

    print("generating deterministic fp16 inputs")
    q = deterministic_fp16("q", shape)
    k = deterministic_fp16("k", shape)
    v = deterministic_fp16("v", shape)
    print(stats("q", q))
    print(stats("k", k))
    print(stats("v", v))
    print(f"q[0:32] hex={first32_hex(q)}")
    print(f"k[0:32] hex={first32_hex(k)}")
    print(f"v[0:32] hex={first32_hex(v)}")

    # Intermediate size warning
    scores_bytes = args.heads * args.seq * args.seq * 2
    print(
        f"scores[1,{args.heads},{args.seq},{args.seq}] fp16 = "
        f"{scores_bytes / 1024 / 1024:.1f} MiB (intermediate; WebGPU must fit this in one buffer)"
    )

    print("running CPU ORT reference (graphOpt disabled)")
    out = run_cpu_ref(args.out, q, k, v)
    print(stats("out", out))
    print(f"out[tok=0,0:32] f32={first32_f32(out[0, 0, 0])}")
    print(f"out[tok=0,0:32] hex={first32_hex(out[0, 0, 0])}")


if __name__ == "__main__":
    main()
