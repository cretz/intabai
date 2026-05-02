"""Verify cross-attention in block_NN.onnx actually depends on
encoder_hidden_states. Feeds the ONNX two runs with IDENTICAL hidden_states
and modulation tensors but DIFFERENT encoder_hidden_states. If cross-attn
is wired correctly, outputs should differ significantly. If they don't,
the export silently ignored the text path."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import onnxruntime as ort

    rng = np.random.default_rng(args.seed)
    seq_len = 8190
    text_seq = 512
    inner = 3072
    head_dim = 128

    hidden = (rng.standard_normal((1, seq_len, inner)) * 0.5).astype(np.float16)
    timestep_proj = (rng.standard_normal((1, seq_len, 6, inner)) * 0.5).astype(np.float16)
    angles = rng.uniform(-np.pi, np.pi, (1, seq_len, 1, head_dim)).astype(np.float32)
    freqs_cos = np.cos(angles).astype(np.float32)
    freqs_sin = np.sin(angles).astype(np.float32)

    enc_a = (rng.standard_normal((1, text_seq, inner)) * 0.5).astype(np.float16)
    enc_b = (rng.standard_normal((1, text_seq, inner)) * 0.5).astype(np.float16)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(args.onnx_path), so, providers=["CPUExecutionProvider"])

    feeds_common = {
        "hidden_states": hidden,
        "timestep_proj": timestep_proj,
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
    }

    print("Run A (enc_a)...")
    out_a = sess.run(None, {**feeds_common, "encoder_hidden_states": enc_a})[0]
    print("Run B (enc_b)...")
    out_b = sess.run(None, {**feeds_common, "encoder_hidden_states": enc_b})[0]

    a = out_a.astype(np.float32)
    b = out_b.astype(np.float32)
    diff = np.abs(a - b)
    print(f"\nout A range [{a.min():.3f}, {a.max():.3f}] mean {a.mean():.5f}")
    print(f"out B range [{b.min():.3f}, {b.max():.3f}] mean {b.mean():.5f}")
    print(f"|A - B|: max={diff.max():.4g}  mean={diff.mean():.4g}")
    if diff.max() < 1e-3:
        print("\nFAIL: outputs are identical. Cross-attention ignores encoder_hidden_states.")
    else:
        print(f"\nPASS: outputs differ substantially (max diff {diff.max():.3f}). "
              f"Cross-attention is live.")

    # Also test: enc all zeros vs random
    enc_zero = np.zeros_like(enc_a)
    print("\nRun C (enc_zero)...")
    out_c = sess.run(None, {**feeds_common, "encoder_hidden_states": enc_zero})[0]
    c = out_c.astype(np.float32)
    diff2 = np.abs(a - c)
    print(f"out C range [{c.min():.3f}, {c.max():.3f}] mean {c.mean():.5f}")
    print(f"|A - C|: max={diff2.max():.4g}  mean={diff2.mean():.4g}")
    if diff2.max() < 1e-3:
        print("FAIL: zero-text produces same output as random-text.")


if __name__ == "__main__":
    main()
