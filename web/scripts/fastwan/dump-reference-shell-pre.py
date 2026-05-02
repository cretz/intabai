"""Reference shell_pre outputs for byte-diffing against the browser.

Reproduces the same inputs the browser feeds shell_pre on step 1:
- noise_init: mulberry32 seeded Gaussian (same as src/image-gen/generate-utils.ts)
- text_embeds: PyTorch UMT5EncoderModel on the given prompt (fp16)
- timestep: scalar 1000 expanded to [1, 8190] int64

Then runs hf-repo/onnx/transformer/shell_pre.onnx (fp16) via CPU ORT and
prints first 32 values of each output tensor in fp16 hex — the same format
the browser's transformer.ts dumpHex writes. Byte-diff to localize
shell_pre WebGPU divergence.

Usage:
  uv run dump-reference-shell-pre.py \
      ../../../notes/models/fastwan/source \
      ../../../notes/models/fastwan/hf-repo/onnx \
      --seed 3355567881 --prompt "a calm zen garden at dawn"
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch


def mulberry32(seed: int):
    """Bit-exact replica of src/image-gen/generate-utils.ts mulberry32."""
    a = seed & 0xFFFFFFFF

    def next_() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        # Math.imul(x, y) = int32 multiply, low 32 bits, signed.
        def imul(x: int, y: int) -> int:
            r = (x * y) & 0xFFFFFFFF
            if r & 0x80000000:
                r -= 0x100000000
            return r & 0xFFFFFFFF
        t = imul(t ^ (t >> 15), t | 1)
        t = (t ^ (t + imul(t ^ (t >> 7), t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return next_


def gaussian_noise(n: int, rand) -> np.ndarray:
    """Bit-exact replica of gaussianNoise in image-gen/generate-utils.ts."""
    out = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        u1 = rand()
        u2 = rand()
        r = math.sqrt(-2 * math.log(u1 or 1e-9))
        theta = 2 * math.pi * u2
        out[i] = r * math.cos(theta)
        if i + 1 < n:
            out[i + 1] = r * math.sin(theta)
        i += 2
    return out


def dump_hex(label: str, bits_u16: np.ndarray):
    """Print first 32 fp16 bit patterns as hex + decoded f32 (matches browser)."""
    slice_ = bits_u16[:32]
    f32 = slice_.view(np.float16).astype(np.float32)
    hex_str = ",".join(format(int(b), "04x") for b in slice_)
    f32_str = ",".join(f"{v:.4f}" for v in f32)
    print(f"{label} f32=[{f32_str}]")
    print(f"{label} hex={hex_str}")


def dump_hex_f32(label: str, values: np.ndarray):
    """Print first 32 fp32 bit patterns as hex32 + f32 values."""
    slice_ = values.ravel()[:32].astype(np.float32)
    u32 = slice_.view(np.uint32)
    hex_str = ",".join(format(int(b), "08x") for b in u32)
    f32_str = ",".join(f"{v:.6f}" for v in slice_)
    print(f"{label} f32=[{f32_str}]")
    print(f"{label} hex32={hex_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("onnx_dir", type=Path)
    ap.add_argument("--seed", type=int, default=3355567881)
    ap.add_argument("--prompt", default="a calm zen garden at dawn")
    ap.add_argument("--timestep", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    # 1. Seeded noise [1, 48, 21, 30, 52] - matches FASTWAN_LATENT_* in code.
    n_elts = 1 * 48 * 21 * 30 * 52
    rand = mulberry32(args.seed)
    noise = gaussian_noise(n_elts, rand)
    noise_fp16 = noise.astype(np.float16).reshape(1, 48, 21, 30, 52)
    print("noise_init stats: "
          f"min={noise.min():.4f} max={noise.max():.4f} mean={noise.mean():.6f}")
    dump_hex_f32("noise_init[0:32]", noise[:32])
    print()

    # 2. Text embeds via PyTorch UMT5 (matches what our wasm text encoder
    # should reproduce - confirmed cosine 0.9998 to this).
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(str(args.source_dir / "tokenizer"))
    ti = tok([args.prompt], padding="max_length", max_length=args.max_length,
             truncation=True, add_special_tokens=True,
             return_attention_mask=True, return_tensors="pt")
    valid = int(ti.attention_mask.sum().item())
    print(f"tokenized to {valid} real tokens")

    model = UMT5EncoderModel.from_pretrained(
        str(args.source_dir / "text_encoder"), dtype=torch.float16).eval()
    with torch.inference_mode():
        ref = model(ti.input_ids, ti.attention_mask).last_hidden_state
    text_embeds = ref.cpu().numpy()
    # Zero-pad like diffusers / t5_postprocess / our JS.
    text_embeds[0, valid:] = 0
    print(f"text_embeds range [{text_embeds.astype(np.float32).min():.4f}, "
          f"{text_embeds.astype(np.float32).max():.4f}]")
    print()

    # 3. Run shell_pre via CPU ORT.
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    shell_pre_path = args.onnx_dir / "transformer" / "shell_pre.onnx"
    if not shell_pre_path.exists():
        # fall back to q4 variant
        shell_pre_path = args.onnx_dir / "transformer-q4f16" / "shell_pre.onnx"
    print(f"Loading {shell_pre_path}...")
    sess = ort.InferenceSession(str(shell_pre_path), so,
                                providers=["CPUExecutionProvider"])

    timestep = np.full((1, 8190), args.timestep, dtype=np.int64)
    feeds = {
        "hidden_states": noise_fp16,
        "timestep": timestep,
        "encoder_hidden_states": text_embeds.astype(np.float16),
    }
    print("Running shell_pre on CPU...")
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    named = dict(zip(names, outs))
    for k, v in named.items():
        a = v.astype(np.float32)
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"range [{a.min():.4f}, {a.max():.4f}] "
              f"mean {a.mean():.6f} nan={int(np.isnan(a).sum())}")
    print()

    # 4. Byte-diff dumps for the first token, matching browser dumpHex exactly.
    def first32_f16_bits(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1).astype(np.float16)
        return flat[:32].view(np.uint16)

    dump_hex("shell_pre.tokens[tok=0,0:32]", first32_f16_bits(named["tokens"]))
    dump_hex("shell_pre.enc_proj[tok=0,0:32]", first32_f16_bits(named["enc_proj"]))
    dump_hex("shell_pre.temb[tok=0,0:32]", first32_f16_bits(named["temb"]))
    dump_hex_f32("shell_pre.freqs_cos[tok=0,0:32]",
                 named["freqs_cos"].reshape(-1)[:32])
    dump_hex_f32("shell_pre.freqs_sin[tok=0,0:32]",
                 named["freqs_sin"].reshape(-1)[:32])


if __name__ == "__main__":
    main()
