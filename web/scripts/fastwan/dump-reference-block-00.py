"""Reference block_00 output for byte-diffing against the browser.

RAM notes:
- Without `--text-embeds-cache`: peak RAM ~7 GB (UMT5 fp16 ~5.3 GB +
  ORT sessions + tap materialisation). Python doesn't return pages to
  the OS after `del`, so gc alone isn't enough.
- With `--text-embeds-cache path.npz`: first run still ~7 GB but
  subsequent runs skip PyTorch entirely. Peak ~2 GB.

If you're iterating, always pass `--text-embeds-cache` so you only pay
the 7 GB cost once per (prompt, seed, max_length) combination.


Pipeline:
  1. Reproduce the browser's step-1 noise_init via mulberry32 + Box-Muller.
  2. Compute PyTorch UMT5 text embeds (matches the wasm text encoder to
     ~1 ULP, per check-text-encoder-vs-pytorch.py).
  3. Run shell_pre.onnx on CPU ORT to produce tokens / enc_proj /
     timestep_proj / freqs_cos / freqs_sin at t=1000.
  4. Feed those into block_00.onnx on CPU ORT.
  5. Dump block_00.out[tok=0,0:32] in the same fp16-hex format the
     browser writes from transformer.ts.

The browser run with `?textfp16=1&textwasm=1` + seed 3355567881 + prompt
"a calm zen garden at dawn" produces a matching line from transformer.ts
at step 1. Byte-diff localises whether the WebGPU block kernel diverges.

Usage:
  uv run dump-reference-block-00.py \
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
    a = seed & 0xFFFFFFFF

    def next_() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a

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
    slice_ = bits_u16[:32]
    f32 = slice_.view(np.float16).astype(np.float32)
    hex_str = ",".join(format(int(b), "04x") for b in slice_)
    f32_str = ",".join(f"{v:.4f}" for v in f32)
    print(f"{label} f32=[{f32_str}]")
    print(f"{label} hex={hex_str}")


def first32_f16_bits(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1).astype(np.float16)
    return flat[:32].view(np.uint16)


def stats_line(name: str, arr: np.ndarray):
    a = arr.astype(np.float32)
    print(
        f"  {name}: shape={arr.shape} dtype={arr.dtype} "
        f"range [{a.min():.4f}, {a.max():.4f}] "
        f"mean {a.mean():.6f} nan={int(np.isnan(a).sum())}"
    )


def dump_tap(name: str, arr: np.ndarray):
    """Mirror the browser's per-tap format.

    stats[NAME] n=... min=... max=... mean=... nan=... zeros=...
    NAME[tok=0,0:32] hex=...   (fp16: 4-char hex per value)
    NAME[tok=0,0:32] hex32=... (fp32: 8-char hex per value)
    For int/bool tensors: prints raw ints in an ints= line instead.
    """
    flat = arr.reshape(-1)
    a = flat.astype(np.float32, copy=False) if flat.dtype != np.float32 else flat
    finite = a[np.isfinite(a)]
    if finite.size:
        mn = float(finite.min()); mx = float(finite.max())
        mean = float(finite.mean())
    else:
        mn = mx = mean = float("nan")
    nan = int(np.isnan(a).sum())
    zeros = int((a == 0).sum())
    print(
        f"stats[{name}] n={flat.size} min={mn:.4f} max={mx:.4f} "
        f"mean={mean:.4f} nan={nan} zeros={zeros}"
    )
    head = flat[:32]
    label = f"{name}[tok=0,0:32]"
    if arr.dtype == np.float16:
        bits = head.view(np.uint16)
        f32 = head.astype(np.float32)
        f32_str = ",".join(f"{v:.4f}" for v in f32)
        hex_str = ",".join(format(int(b), "04x") for b in bits)
        print(f"{label} f32=[{f32_str}]")
        print(f"{label} hex={hex_str}")
    elif arr.dtype == np.float32:
        f32 = head.astype(np.float32)
        bits = f32.view(np.uint32)
        f32_str = ",".join(f"{v:.6f}" for v in f32)
        hex_str = ",".join(format(int(b), "08x") for b in bits)
        print(f"{label} f32=[{f32_str}]")
        print(f"{label} hex32={hex_str}")
    else:
        ints = ",".join(str(int(v)) for v in head.tolist())
        print(f"{label} dtype={arr.dtype} ints=[{ints}]")


def find_onnx(onnx_dir: Path, rel_q4: str, rel_fp16: str) -> Path:
    # Prefer fp16 since the browser uses fp16 shell_pre (q4 shell_pre had
    # a MatMulNBits NaN bug on WebGPU). For block_00, q4 is the shipped
    # path so use that. Caller specifies preference via the rel args.
    for rel in (rel_fp16, rel_q4):
        p = onnx_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"neither {rel_fp16} nor {rel_q4} under {onnx_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("onnx_dir", type=Path)
    ap.add_argument("--seed", type=int, default=3355567881)
    ap.add_argument("--prompt", default="a calm zen garden at dawn")
    ap.add_argument("--timestep", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--block-variant",
        choices=("q4f16", "fp16"),
        default="q4f16",
        help="which block_00 to run; browser ships q4f16",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help=(
            "run block_00_debug.onnx (every intermediate exposed as output) "
            "and dump stats+hex per tap in browser-matching format"
        ),
    )
    ap.add_argument(
        "--text-embeds-cache",
        type=Path,
        default=None,
        help=(
            "path to a .npy file that caches text embeds. On first run "
            "(file missing) we compute via UMT5 PyTorch then save; on "
            "later runs we load from disk and skip PyTorch entirely. "
            "Cuts peak RAM from ~7 GB to ~2 GB. Cache key is (prompt, "
            "seed, max_length) encoded in filename — user's job to pick "
            "a descriptive path."
        ),
    )
    args = ap.parse_args()

    # 1. Seeded noise [1, 48, 21, 30, 52].
    n_elts = 1 * 48 * 21 * 30 * 52
    rand = mulberry32(args.seed)
    noise = gaussian_noise(n_elts, rand)
    noise_fp16 = noise.astype(np.float16).reshape(1, 48, 21, 30, 52)
    print(
        f"noise_init: min={noise.min():.4f} max={noise.max():.4f} "
        f"mean={noise.mean():.6f}"
    )

    # 2. PyTorch text embeds (or load cached embeds from disk).
    cache = args.text_embeds_cache
    if cache is not None and cache.exists():
        loaded = np.load(str(cache))
        text_embeds = loaded["text_embeds"]
        valid = int(loaded["valid"])
        print(f"loaded text_embeds from cache: {cache} (valid={valid})")
    else:
        from transformers import AutoTokenizer, UMT5EncoderModel

        tok = AutoTokenizer.from_pretrained(str(args.source_dir / "tokenizer"))
        ti = tok(
            [args.prompt],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        valid = int(ti.attention_mask.sum().item())
        print(f"tokenized to {valid} real tokens")

        model = UMT5EncoderModel.from_pretrained(
            str(args.source_dir / "text_encoder"), dtype=torch.float16
        ).eval()
        with torch.inference_mode():
            ref = model(ti.input_ids, ti.attention_mask).last_hidden_state
        text_embeds = ref.cpu().numpy()
        text_embeds[0, valid:] = 0

        if cache is not None:
            np.savez(str(cache), text_embeds=text_embeds, valid=valid)
            print(f"saved text_embeds cache: {cache}")

        import gc
        del model, ref, tok, ti
        gc.collect()

    # 3. shell_pre on CPU ORT. Browser runs fp16 shell_pre (see worklog
    # q4 shell_pre NaN bug), so mirror that.
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3

    shell_pre_path = find_onnx(
        args.onnx_dir,
        "transformer-q4f16/shell_pre.onnx",
        "transformer/shell_pre.onnx",
    )
    print(f"shell_pre: {shell_pre_path}")
    shell_pre = ort.InferenceSession(
        str(shell_pre_path), so, providers=["CPUExecutionProvider"]
    )

    timestep = np.full((1, 8190), args.timestep, dtype=np.int64)
    shell_pre_feeds = {
        "hidden_states": noise_fp16,
        "timestep": timestep,
        "encoder_hidden_states": text_embeds.astype(np.float16),
    }
    sp_outs = shell_pre.run(None, shell_pre_feeds)
    sp_names = [o.name for o in shell_pre.get_outputs()]
    sp = dict(zip(sp_names, sp_outs))
    print("shell_pre outputs:")
    for k, v in sp.items():
        stats_line(k, v)

    # 4. block_00 on CPU ORT.
    if args.debug:
        # --debug forces fp16 path since block_00_debug.onnx is only produced
        # from the monolithic fp16 block (q4 has external data).
        block_path = args.onnx_dir / "transformer" / "block_00_debug.onnx"
    elif args.block_variant == "q4f16":
        block_path = args.onnx_dir / "transformer-q4f16" / "block_00.onnx"
    else:
        block_path = args.onnx_dir / "transformer" / "block_00.onnx"
    print(f"block_00: {block_path}")
    block = ort.InferenceSession(
        str(block_path), so, providers=["CPUExecutionProvider"]
    )
    block_feeds = {
        "hidden_states": sp["tokens"],
        "encoder_hidden_states": sp["enc_proj"],
        "timestep_proj": sp["timestep_proj"],
        "freqs_cos": sp["freqs_cos"],
        "freqs_sin": sp["freqs_sin"],
    }
    block_outs = block.run(None, block_feeds)
    block_names = [o.name for o in block.get_outputs()]
    bo = dict(zip(block_names, block_outs))

    if args.debug:
        # Per-tap dumps matching browser transformer.ts format. The debug
        # graph has ~800 intermediate outputs; we emit every one in a stable
        # order so `diff` against the browser log localises the first
        # diverging op.
        print(f"block_00_debug: {len(block_names)} outputs")
        for k in block_names:
            dump_tap(k, bo[k])
    else:
        print("block_00 outputs:")
        for k, v in bo.items():
            stats_line(k, v)
        # Byte-diff dump. Browser writes "block_00.out[tok=0,0:32]".
        out_name = "hidden_states_out" if "hidden_states_out" in bo else block_names[0]
        dump_hex("block_00.out[tok=0,0:32]", first32_f16_bits(bo[out_name]))


if __name__ == "__main__":
    main()
