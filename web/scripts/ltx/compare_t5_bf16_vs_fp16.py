"""Compare PixArt T5-XXL encoder in bf16 (Lightricks-recommended) vs fp16
(what we ship in the browser).

bf16 is the canonical precision per the LTX-Video model card. fp16 is what
WebGPU/ORT-web supports. This script answers: when we drop bf16 -> fp16,
where does numerical drift enter the encoder, and how much does it
amplify across the 24 T5 blocks?

Output (per-block maxAbs / meanAbs / std of bf16-vs-fp16 hidden_states)
plus saved fp32 dumps of both final last_hidden_state tensors.

Run from intabai/web/scripts:
  uv run python ltx/compare_t5_bf16_vs_fp16.py \\
    --src C:/work/personal/intabai/notes/models/ltx/source/pixart-text-encoder \\
    --out-dir C:/work/personal/intabai/notes/compare-t5 \\
    --prompt "a cat playing piano in a jazz club, cinematic" \\
    --n-text 256
"""
import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import struct

import numpy as np
import psutil
import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, T5Config, T5EncoderModel


SAFETENSORS_DTYPE_MAP = {
    "F32": (torch.float32, np.float32, 4),
    "F16": (torch.float16, np.float16, 2),
    "BF16": (torch.bfloat16, None, 2),
    "I64": (torch.int64, np.int64, 8),
    "I32": (torch.int32, np.int32, 4),
    "I8":  (torch.int8,  np.int8,  1),
    "U8":  (torch.uint8, np.uint8, 1),
    "BOOL":(torch.bool,  np.bool_, 1),
}


def iter_safetensors_no_mmap(path: Path):
    """Yield (tensor_name, torch.Tensor) by reading the safetensors file
    with stdio I/O (no mmap). Each tensor is read as bytes, converted to a
    torch tensor, yielded; caller must consume/free before the next yield
    to keep peak bounded to one tensor at a time."""
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        # Stable order by data_offsets so reads are sequential.
        items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
        items.sort(key=lambda kv: kv[1]["data_offsets"][0])
        for tname, info in items:
            off_start, off_end = info["data_offsets"]
            shape = info["shape"]
            dtype_str = info["dtype"]
            torch_dt, np_dt, _ = SAFETENSORS_DTYPE_MAP[dtype_str]
            nbytes = off_end - off_start
            f.seek(data_start + off_start)
            buf = f.read(nbytes)
            if np_dt is not None:
                arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
                t = torch.from_numpy(arr.copy())
            else:
                # bf16 has no numpy dtype: load as int16 bits, view as bf16.
                arr = np.frombuffer(buf, dtype=np.int16).reshape(shape)
                t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            del buf, arr
            yield tname, t


def start_ram_watchdog(max_gb: float):
    proc = psutil.Process(os.getpid())
    peak = [0.0]

    def tick():
        while True:
            rss = proc.memory_info().rss / 1e9
            if rss > peak[0]:
                peak[0] = rss
            if rss > max_gb:
                print(f"!!! RAM {rss:.2f} GB > {max_gb:.1f} GB, KILLING", flush=True)
                os._exit(2)
            time.sleep(0.5)

    threading.Thread(target=tick, daemon=True).start()
    return peak


def _set_module_tensor(model, tname, tensor):
    """Replace a meta-init parameter or buffer with a real tensor."""
    parts = tname.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    leaf = parts[-1]
    cur = getattr(mod, leaf)
    if isinstance(cur, torch.nn.Parameter):
        new = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(mod, leaf, new)
    else:
        # Buffer (e.g. relative_attention_bias is a parameter actually,
        # but other buffers go through this path).
        mod._buffers[leaf] = tensor


def encode_with_taps(src_dir: Path, dtype: torch.dtype, input_ids, attn_mask):
    """Load T5 at given dtype via streaming shard reader, run forward with
    per-block hooks, return (last_hidden_state_fp32,
    [block_out_fp32 for each of 24 blocks]).

    Streaming load avoids the fp32 transient that from_pretrained() spikes
    when source shards are fp32 on disk: we open one shard, read each
    tensor (fp32 ~<= 256 MB peak), cast to dtype, assign in-place, and
    free the source tensor before the next read."""
    enc_dir = src_dir / "text_encoder"
    print(f"Loading T5 ({dtype}) from {enc_dir} (streaming) ...")
    t0 = time.time()
    config = T5Config.from_pretrained(str(enc_dir))
    with init_empty_weights():
        encoder = T5EncoderModel(config)
    encoder.eval()

    index_path = enc_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shard_to_tensors: dict[str, list[str]] = {}
    for tname, shard in weight_map.items():
        shard_to_tensors.setdefault(shard, []).append(tname)

    seen = set()
    for shard_name in shard_to_tensors:
        shard_path = enc_dir / shard_name
        wanted = set(shard_to_tensors[shard_name])
        n_loaded = 0
        for tname, src in iter_safetensors_no_mmap(shard_path):
            if tname not in wanted:
                del src
                continue
            tgt = src.to(dtype).contiguous() if src.dtype != dtype else src.contiguous()
            del src
            _set_module_tensor(encoder, tname, tgt)
            seen.add(tname)
            n_loaded += 1
        gc.collect()
        rss = psutil.Process().memory_info().rss / 1e9
        print(f"  shard {shard_name}: {n_loaded} tensors, RSS={rss:.2f} GB")

    missing = set(weight_map) - seen
    if missing:
        raise RuntimeError(f"unfilled tensors: {sorted(missing)[:5]}...")

    for p in encoder.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  loaded {n_params/1e9:.2f}B params in {time.time()-t0:.1f}s")

    blocks = encoder.encoder.block
    print(f"  num encoder blocks: {len(blocks)}")

    block_outs: list[torch.Tensor] = [None] * len(blocks)

    def make_hook(i):
        def _hook(module, inputs, output):
            # T5Block returns a tuple (hidden_states, ...).
            h = output[0] if isinstance(output, tuple) else output
            block_outs[i] = h.detach().to(torch.float32).cpu().clone()
        return _hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]

    print(f"  forward (n_text={input_ids.shape[1]}) ...")
    t0 = time.time()
    with torch.no_grad():
        out = encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
    last_hidden = out.last_hidden_state.detach().to(torch.float32).cpu().clone()
    print(f"  forward done in {time.time()-t0:.1f}s")
    print(f"  last_hidden_state: shape={tuple(last_hidden.shape)} "
          f"mean={last_hidden.mean():.5f} std={last_hidden.std():.5f} "
          f"min={last_hidden.min():.5f} max={last_hidden.max():.5f}")

    for h in handles:
        h.remove()

    del encoder
    gc.collect()

    return last_hidden, block_outs


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a - b).abs()
    s = b.float().std().item() + 1e-12
    return {
        "max": d.max().item(),
        "mean": d.mean().item(),
        "std_b": b.float().std().item(),
        "rel_mean": d.mean().item() / s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="dir with text_encoder/ + tokenizer/ subfolders")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=str,
                    default="a cat playing piano in a jazz club, cinematic")
    ap.add_argument("--n-text", type=int, default=256,
                    help="pad/truncate length (LTX uses 256)")
    ap.add_argument("--ram-budget-gb", type=float, default=14.8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ram_peak = start_ram_watchdog(args.ram_budget_gb)
    print(f"RAM budget: {args.ram_budget_gb:.1f} GB")

    print(f"Loading tokenizer from {args.src/'tokenizer'} ...")
    tok = AutoTokenizer.from_pretrained(str(args.src / "tokenizer"))
    enc = tok(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=args.n_text,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    n_real = int(attn_mask.sum().item())
    print(f"  prompt: {args.prompt!r}")
    print(f"  input_ids shape={tuple(input_ids.shape)} real_tokens={n_real}")

    print("\n=== bf16 pass ===")
    bf_last, bf_blocks = encode_with_taps(args.src, torch.bfloat16, input_ids, attn_mask)
    np.save(args.out_dir / "t5_bf16_last_hidden.npy", bf_last.numpy())
    print(f"  saved {args.out_dir/'t5_bf16_last_hidden.npy'}")

    print("\n=== fp16 pass ===")
    fp_last, fp_blocks = encode_with_taps(args.src, torch.float16, input_ids, attn_mask)
    np.save(args.out_dir / "t5_fp16_last_hidden.npy", fp_last.numpy())
    print(f"  saved {args.out_dir/'t5_fp16_last_hidden.npy'}")

    # Per-block divergence (real tokens only - padding is masked out
    # numerically in attention but the hidden_states column still exists,
    # and depending on T5 internals padded positions can have arbitrary
    # values that drown the real signal in mean/max stats).
    print("\n=== per-block bf16 vs fp16 (real tokens only) ===")
    print(f"{'block':>5}  {'maxAbs':>10}  {'meanAbs':>10}  "
          f"{'std(bf16)':>10}  {'meanAbs/std':>12}")
    summary_lines = ["block,maxAbs,meanAbs,std_bf16,rel_mean"]
    for i, (b, f) in enumerate(zip(bf_blocks, fp_blocks)):
        b_real = b[:, :n_real, :]
        f_real = f[:, :n_real, :]
        s = diff_stats(b_real, f_real)
        print(f"  {i:>3d}    {s['max']:>10.4g}  {s['mean']:>10.4g}  "
              f"{s['std_b']:>10.4g}  {s['rel_mean']:>12.4g}")
        summary_lines.append(
            f"{i},{s['max']:.6g},{s['mean']:.6g},"
            f"{s['std_b']:.6g},{s['rel_mean']:.6g}"
        )

    s_last = diff_stats(bf_last[:, :n_real, :], fp_last[:, :n_real, :])
    print(f"\nlast_hidden_state (real tokens):")
    print(f"  maxAbs={s_last['max']:.4g} meanAbs={s_last['mean']:.4g} "
          f"std(bf16)={s_last['std_b']:.4g} rel_mean={s_last['rel_mean']:.4g}")
    summary_lines.append(
        f"last,{s_last['max']:.6g},{s_last['mean']:.6g},"
        f"{s_last['std_b']:.6g},{s_last['rel_mean']:.6g}"
    )

    (args.out_dir / "t5_diff_summary.csv").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(f"\nsaved {args.out_dir/'t5_diff_summary.csv'}")
    print(f"Peak RAM: {ram_peak[0]:.2f} GB")


if __name__ == "__main__":
    main()
