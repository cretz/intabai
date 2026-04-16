"""End-to-end diff: run our ONNX text encoder layers on CPU with the same
tokens as the PyTorch reference, zero-pad padded positions, and compare
element-wise. If outputs diverge significantly, the text encoder export
is faulty (not the stats-close-but-content-wrong trap).

Uses the q4f16 layers by default (what the browser actually ships). Pass
--fp16 to test the pre-quant layers instead."""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch


def load_embedding_from_bin(q8_path: Path, scales_path: Path):
    """Dequantize the JS-side int8 row embedding back to fp16."""
    q8 = np.fromfile(q8_path, dtype=np.int8).reshape(256384, 4096)
    scales = np.fromfile(scales_path, dtype=np.float16).reshape(256384)
    return q8, scales


def lookup_embedding_js(ids: np.ndarray, q8, scales):
    """Match embedding.ts lookup exactly."""
    B, L = ids.shape
    out = np.zeros((B, L, 4096), dtype=np.float16)
    for b in range(B):
        for i in range(L):
            tid = int(ids[b, i])
            row_q = q8[tid].astype(np.float32)
            s = float(scales[tid])
            out[b, i] = (row_q * s).astype(np.float16)
    return out


def run_onnx_chain(onnx_dir: Path, precision: str, hidden, mask):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    subdir = "text-encoder-q4f16" if precision == "q4f16" else "text-encoder"
    base = onnx_dir / subdir
    hidden_fp16 = hidden.astype(np.float16)
    mask_fp16 = mask.astype(np.float16).reshape(1, 1, 1, -1)
    print(f"  loading and running 24 layers from {base}...")
    for i in range(24):
        layer_path = base / f"layer_{i:02d}.onnx"
        sess = ort.InferenceSession(str(layer_path), so, providers=["CPUExecutionProvider"])
        t0 = time.time()
        out = sess.run(None, {"hidden_states": hidden_fp16, "attention_mask": mask_fp16})
        hidden_fp16 = out[0]
        if i in (0, 11, 23):
            a = hidden_fp16.astype(np.float32)
            print(f"    layer_{i:02d}: {time.time() - t0:.2f}s  "
                  f"range [{a.min():.3f}, {a.max():.3f}] nan={int(np.isnan(a).sum())}")
        del sess

    shell_path = base / "shell_post.onnx"
    sess = ort.InferenceSession(str(shell_path), so, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"hidden_states": hidden_fp16})
    return out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path,
                    help="FastWan source (has text_encoder/ and tokenizer/)")
    ap.add_argument("onnx_dir", type=Path,
                    help="hf-repo/onnx directory containing text-encoder-q4f16/")
    ap.add_argument("--prompt", default="a calm zen garden at dawn")
    ap.add_argument("--precision", choices=["q4f16", "fp16"], default="q4f16")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--use-js-embedding", action="store_true",
                    help="Dequantize JS int8 embedding (matches browser path).")
    args = ap.parse_args()

    # Tokenize exactly like diffusers WanPipeline / FastVideo t5_postprocess.
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(str(args.source_dir / "tokenizer"))
    ti = tok([args.prompt], padding="max_length", max_length=args.max_length,
             truncation=True, add_special_tokens=True,
             return_attention_mask=True, return_tensors="pt")
    ids = ti.input_ids  # [1, 512]
    mask = ti.attention_mask
    valid_length = int(mask.sum().item())
    print(f"Prompt: {args.prompt!r}")
    print(f"  tokenized to {valid_length} real tokens")

    # ---- Reference path: HF model end to end ----
    print("\n[REF] Loading UMT5EncoderModel fp16...")
    model = UMT5EncoderModel.from_pretrained(str(args.source_dir / "text_encoder"),
                                             torch_dtype=torch.float16).eval()
    print("[REF] Running forward...")
    with torch.inference_mode():
        ref_out = model(ids, mask).last_hidden_state  # fp16 [1, 512, 4096]
    ref = ref_out.cpu().numpy()
    # Same zero-pad as diffusers / t5_postprocess.
    ref[0, valid_length:] = 0
    print(f"[REF] range [{ref.astype(np.float32).min():.4f}, "
          f"{ref.astype(np.float32).max():.4f}]")

    # ---- Embedding: either HF model's lookup or JS int8 dequant ----
    if args.use_js_embedding:
        q8_path = args.onnx_dir / "text-encoder-q4f16" / "embedding_q8.bin"
        scales_path = args.onnx_dir / "text-encoder-q4f16" / "embedding_scales.bin"
        if not q8_path.exists():
            raise SystemExit(f"Missing {q8_path}")
        print(f"\n[ONNX] Using JS int8 embedding from {q8_path}")
        q8, scales = load_embedding_from_bin(q8_path, scales_path)
        hidden_in = lookup_embedding_js(ids.numpy(), q8, scales)
    else:
        print("\n[ONNX] Using HF model's shared embedding for fair layer comparison")
        with torch.inference_mode():
            hidden_in = model.shared(ids).detach().cpu().numpy().astype(np.float16)

    # Mask as fp16 additive (matches text-encoder.ts).
    additive_mask = np.where(mask.numpy() > 0, 0.0, -65504.0).astype(np.float32)

    # ---- Run our ONNX ----
    onnx_out = run_onnx_chain(args.onnx_dir, args.precision, hidden_in, additive_mask)
    # Same zero-pad the JS pipeline applies.
    onnx_out[0, valid_length:] = 0

    # ---- Compare ----
    a = ref.astype(np.float32)
    b = onnx_out.astype(np.float32)
    diff = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b)) + 1e-6
    print("\n=== Valid-token slice diff ===")
    av = a[0, :valid_length]
    bv = b[0, :valid_length]
    dv = np.abs(av - bv)
    print(f"ref valid   range [{av.min():.4f}, {av.max():.4f}] mean {av.mean():.6f}")
    print(f"onnx valid  range [{bv.min():.4f}, {bv.max():.4f}] mean {bv.mean():.6f}")
    print(f"|diff|      max={dv.max():.4g}  mean={dv.mean():.4g}  "
          f"rel={np.mean(dv / (np.maximum(np.abs(av), np.abs(bv)) + 1e-6)):.4%}")
    # Per-token cosine-ish correlation
    def cos(x, y):
        xn = x / (np.linalg.norm(x) + 1e-9)
        yn = y / (np.linalg.norm(y) + 1e-9)
        return float(np.dot(xn, yn))
    per_tok_cos = [cos(av[i], bv[i]) for i in range(valid_length)]
    print(f"per-valid-token cosine: "
          f"min={min(per_tok_cos):.4f}  mean={np.mean(per_tok_cos):.4f}  "
          f"max={max(per_tok_cos):.4f}")

    # Overall tolerance check
    if dv.max() < 0.1:
        print("\nPASS: ONNX text encoder matches PyTorch reference.")
    elif np.mean(per_tok_cos) > 0.95:
        print("\nSOFT MATCH: direction matches but magnitudes drift "
              f"(max_abs={dv.max():.4g}). Quantization drift.")
    else:
        print(f"\nFAIL: substantial divergence. max_abs={dv.max():.4g} "
              f"mean_cos={np.mean(per_tok_cos):.4f}. "
              f"This could cause prompt-invariant output.")


if __name__ == "__main__":
    main()
