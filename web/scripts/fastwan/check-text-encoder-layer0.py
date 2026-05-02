#!/usr/bin/env python3
"""Reference check: run layer_00.onnx on CPU with fp32 and compare against
the browser's NaN output. Fast diagnosis for the gray-output regression
(worklog 2026-04-18).

Inputs are synthesized to match the browser's feed:
  - tokenize "a red panda eating bamboo in a sunlit forest" with UMT5
  - look up embeddings from embedding.bin (fp16)
  - build additive attention mask (0 at valid, -65504 at pad)
  - run layer_00.onnx

Reports min/max/mean/NaN/Inf per output. If clean in Python, the issue is
browser-side (feed dtype, shape, byteorder). If NaN in Python, the graph
is genuinely broken and we need to patch the export.
"""

import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path


def stats(name, arr):
    flat = arr.reshape(-1).astype(np.float32)
    nan = int(np.isnan(flat).sum())
    inf = int(np.isinf(flat).sum())
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        print(f"{name}: n={flat.size} ALL non-finite (nan={nan} inf={inf})")
        return
    print(
        f"{name}: n={flat.size} min={finite.min():.4f} max={finite.max():.4f} "
        f"mean={finite.mean():.4f} nan={nan} inf={inf}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", type=Path,
                    help="notes/models/fastwan/hf-repo/onnx root")
    ap.add_argument("--tokenizer-dir", type=Path, default=None,
                    help="UMT5 tokenizer dir (defaults to ../tokenizer under repo_root)")
    ap.add_argument("--prompt", type=str,
                    default="a red panda eating bamboo in a sunlit forest")
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    root = args.repo_root
    layer_path = root / "text-encoder" / "layer_00.onnx"
    embed_path = root / "text-encoder" / "embedding.bin"
    if not layer_path.exists():
        raise SystemExit(f"missing {layer_path}")
    if not embed_path.exists():
        raise SystemExit(f"missing {embed_path}")

    # Tokenize using raw tokenizer.json (no config.json required).
    from tokenizers import Tokenizer
    tokenizer_dir = args.tokenizer_dir or (root.parent / "tokenizer")
    tok_file = tokenizer_dir / "tokenizer.json"
    print(f"loading tokenizer from {tok_file}")
    tok = Tokenizer.from_file(str(tok_file))
    enc = tok.encode(args.prompt)
    raw_ids = enc.ids
    # UMT5 uses pad_token_id=0, appends eos manually. transformers.js mirrors HF:
    # tokenize -> append eos=1 -> pad with 0 to seq_len.
    ids = list(raw_ids) + [1]  # eos
    if len(ids) > args.seq_len:
        ids = ids[:args.seq_len]
    valid_len = len(ids)
    while len(ids) < args.seq_len:
        ids.append(0)
    ids = np.array(ids, dtype=np.int64)
    print(f"tokens: {ids[:valid_len].tolist()} (valid_len={valid_len})")

    # Embedding lookup from embedding.bin (fp16 raw [vocab, 4096]).
    VOCAB = 256384
    HIDDEN = 4096
    embed_raw = np.fromfile(embed_path, dtype=np.float16)
    assert embed_raw.size == VOCAB * HIDDEN, \
        f"embedding.bin size {embed_raw.size} != {VOCAB*HIDDEN}"
    embed_table = embed_raw.reshape(VOCAB, HIDDEN)
    hidden = embed_table[ids].reshape(1, args.seq_len, HIDDEN)
    stats("embed.out[all]", hidden)
    stats("embed.out[valid]", hidden[0, :valid_len])

    # Additive fp16 mask: 0 at valid, -65504 at pad. Shape [1,1,1,seq_len].
    mask = np.where(np.arange(args.seq_len) < valid_len,
                    np.float16(0.0), np.float16(-65504.0))
    mask = mask.reshape(1, 1, 1, args.seq_len).astype(np.float16)
    stats("mask", mask)

    print(f"\nloading {layer_path}")
    sess = ort.InferenceSession(str(layer_path), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  input {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  output {o.name}: {o.shape} {o.type}")

    out = sess.run(None, {
        "hidden_states": hidden.astype(np.float16),
        "attention_mask": mask,
    })
    print()
    for o_spec, o_val in zip(sess.get_outputs(), out):
        stats(f"{o_spec.name}[all]", o_val)
        if o_val.ndim >= 2:
            stats(f"{o_spec.name}[valid:0..{valid_len}]",
                  o_val[0, :valid_len] if o_val.ndim == 3 else o_val)


if __name__ == "__main__":
    main()
