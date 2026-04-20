"""Seq-chunk attn1 in a FastWan transformer block ONNX.

Replaces the single attn1 core triad:

    /block/attn1/MatMul    (Q @ Kt)
    /block/attn1/Softmax
    /block/attn1/MatMul_1  (probs @ V)

with an N-way Q-split-Softmax-Concat variant. Concat output is renamed to
/block/attn1/MatMul_1_output_0 so downstream wiring is untouched.

Rationale: Q*K^T at seq=8190, heads=24 is 3.07 GiB fp16, exceeding the
2 GiB WebGPU maxBufferSize cap. See notes/ort-fp16-bugs.md section 5.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper


ATTN1_MATMUL = "/block/attn1/MatMul"
ATTN1_SOFTMAX = "/block/attn1/Softmax"
ATTN1_MATMUL_1 = "/block/attn1/MatMul_1"
ATTN1_OUTPUT_NAME = "/block/attn1/MatMul_1_output_0"

Q_NAME = "/block/attn1/Mul_24_output_0"
KT_NAME = "/block/attn1/Mul_25_output_0"
V_NAME = "/block/attn1/Transpose_1_output_0"

SEQ_LEN = 8190


def chunk_attn1(model: onnx.ModelProto, n_chunks: int = 3) -> onnx.ModelProto:
    if SEQ_LEN % n_chunks != 0:
        raise ValueError(
            f"seq_len {SEQ_LEN} not divisible by n_chunks {n_chunks}"
        )
    chunk = SEQ_LEN // n_chunks
    split_sizes = [chunk] * n_chunks

    graph = model.graph
    nodes = list(graph.node)

    attn1_nodes = {ATTN1_MATMUL, ATTN1_SOFTMAX, ATTN1_MATMUL_1}
    found = {n.name for n in nodes if n.name in attn1_nodes}
    missing = attn1_nodes - found
    if missing:
        raise RuntimeError(f"missing expected attn1 nodes: {sorted(missing)}")

    matmul_idx = next(i for i, n in enumerate(nodes) if n.name == ATTN1_MATMUL)
    kept = [n for n in nodes if n.name not in attn1_nodes]

    split_init_name = "/block/attn1/chunk_split_sizes"
    split_init = numpy_helper.from_array(
        np.array(split_sizes, dtype=np.int64), name=split_init_name
    )

    q_chunk_names = [f"/block/attn1/chunk_q_{i}" for i in range(n_chunks)]
    scores_names = [f"/block/attn1/chunk_scores_{i}" for i in range(n_chunks)]
    probs_names = [f"/block/attn1/chunk_probs_{i}" for i in range(n_chunks)]
    out_names = [f"/block/attn1/chunk_out_{i}" for i in range(n_chunks)]

    new_nodes = [
        helper.make_node(
            "Split",
            inputs=[Q_NAME, split_init_name],
            outputs=q_chunk_names,
            name="/block/attn1/chunk_split_q",
            axis=2,
        )
    ]
    for i in range(n_chunks):
        new_nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[q_chunk_names[i], KT_NAME],
                outputs=[scores_names[i]],
                name=f"/block/attn1/chunk_matmul_qk_{i}",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Softmax",
                inputs=[scores_names[i]],
                outputs=[probs_names[i]],
                name=f"/block/attn1/chunk_softmax_{i}",
                axis=-1,
            )
        )
        new_nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[probs_names[i], V_NAME],
                outputs=[out_names[i]],
                name=f"/block/attn1/chunk_matmul_pv_{i}",
            )
        )
    new_nodes.append(
        helper.make_node(
            "Concat",
            inputs=out_names,
            outputs=[ATTN1_OUTPUT_NAME],
            name="/block/attn1/chunk_concat",
            axis=2,
        )
    )

    rebuilt = kept[:matmul_idx] + new_nodes + kept[matmul_idx:]

    del graph.node[:]
    graph.node.extend(rebuilt)
    graph.initializer.append(split_init)

    stale = {
        "/block/attn1/MatMul_output_0",
        "/block/attn1/Softmax_output_0",
    }
    kept_vi = [v for v in graph.value_info if v.name not in stale]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)

    return model


def chunk_attn1_file(path, n_chunks: int = 3) -> None:
    """Load, rewrite, save in place. Path is str or Path."""
    path = str(path)
    model = onnx.load(path)
    model = chunk_attn1(model, n_chunks)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)
