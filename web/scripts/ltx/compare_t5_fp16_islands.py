"""Search for the minimum set of T5 sub-ops that need to stay in fp32
to close the bf16-vs-fp16 gap.

T5-XXL fp16 overflows past block 7 (FFN activations exceed fp16's +/-65504
range). Diffusers/HF ship a runtime clamp workaround. For browser ONNX
deployment we want a static structural answer: which sub-modules can we
wrap with Cast(fp32)...Cast(fp16) islands so that fp16 matches bf16?

This script monkey-patches selected submodule.forward methods to upcast
the leading tensor input to fp32, run forward, and downcast the result
back to the original dtype. Internally those modules then compute in
fp32 (since their weights remain fp16, but matmul accumulates fp32 when
the input is fp32 and weight is fp16... actually we cast weights too,
to mirror what an ONNX fp32 island would do).

Run from intabai/web/scripts:
  uv run python ltx/compare_t5_fp16_islands.py \\
    --src C:/work/personal/intabai/notes/models/ltx/source/pixart-text-encoder \\
    --out-dir C:/work/personal/intabai/notes/compare-t5 \\
    --prompt "a cat playing piano in a jazz club, cinematic" \\
    --n-text 256 \\
    --fp32 ffn

Sites supported (pass comma-separated to --fp32):
  ffn       - T5DenseGatedActDense (entire FFN: wi_0, wi_1, gelu, wo)
  ffn_wo    - just the wo projection inside the FFN
  attn_o    - T5Attention output projection (`o` linear)
  attn      - entire T5Attention forward (q,k,v,scores,softmax,o)
  layer_ff  - T5LayerFF (FFN + residual). Wraps the whole FFN sub-block.
  layer_sa  - T5LayerSelfAttention (attn + residual).
"""
import argparse
import gc
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import psutil
import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, T5Config, T5EncoderModel


SAFETENSORS_DTYPE_MAP = {
    "F32": (torch.float32, np.float32),
    "F16": (torch.float16, np.float16),
    "BF16": (torch.bfloat16, None),
    "I64": (torch.int64, np.int64),
    "I32": (torch.int32, np.int32),
    "I8":  (torch.int8,  np.int8),
    "U8":  (torch.uint8, np.uint8),
    "BOOL":(torch.bool,  np.bool_),
}


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


def iter_safetensors_no_mmap(path: Path):
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
        items.sort(key=lambda kv: kv[1]["data_offsets"][0])
        for tname, info in items:
            off_start, off_end = info["data_offsets"]
            shape = info["shape"]
            torch_dt, np_dt = SAFETENSORS_DTYPE_MAP[info["dtype"]]
            f.seek(data_start + off_start)
            buf = f.read(off_end - off_start)
            if np_dt is not None:
                arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
                t = torch.from_numpy(arr.copy())
            else:
                arr = np.frombuffer(buf, dtype=np.int16).reshape(shape)
                t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            del buf, arr
            yield tname, t


def _set_module_tensor(model, tname, tensor):
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
        mod._buffers[leaf] = tensor


def load_t5_streaming(src_dir: Path, dtype: torch.dtype):
    enc_dir = src_dir / "text_encoder"
    config = T5Config.from_pretrained(str(enc_dir))
    with init_empty_weights():
        encoder = T5EncoderModel(config)
    encoder.eval()

    index = json.loads((enc_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shard_to_tensors: dict[str, set] = {}
    for tname, shard in weight_map.items():
        shard_to_tensors.setdefault(shard, set()).add(tname)

    seen = set()
    for shard_name, wanted in shard_to_tensors.items():
        n_loaded = 0
        for tname, src in iter_safetensors_no_mmap(enc_dir / shard_name):
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
    return encoder


def install_taps(encoder):
    """Per-block forward hook. Returns block_outs list (filled at fwd
    time) and a remove() callable."""
    blocks = encoder.encoder.block
    block_outs: list[torch.Tensor] = [None] * len(blocks)

    def make_hook(i):
        def _hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            block_outs[i] = h.detach().to(torch.float32).cpu().clone()
        return _hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]

    def remove():
        for h in handles:
            h.remove()

    return block_outs, remove


def _wrap_module_fp32(module: torch.nn.Module):
    """Replace module.forward with a wrapper that, per call: swaps this
    module's params/buffers to fp32, casts low-precision tensor inputs to
    fp32, runs the original forward, restores params, downcasts outputs
    to the original dtype.

    Per-call swap (instead of permanent upcast) keeps memory bounded:
    only the currently-running block's island holds fp32 weights at a
    time, since blocks run sequentially inside encoder.forward."""
    orig_forward = module.forward
    orig_dtype = next(module.parameters()).dtype

    def _swap_to_fp32():
        saved_p, saved_b = [], []
        for p in module.parameters(recurse=True):
            if p.dtype in (torch.float16, torch.bfloat16):
                saved_p.append((p, p.data))
                p.data = p.data.to(torch.float32)
        for owner_name, owner in module.named_modules():
            for bname, buf in list(owner._buffers.items()):
                if buf is not None and buf.dtype in (torch.float16, torch.bfloat16):
                    saved_b.append((owner, bname, buf))
                    owner._buffers[bname] = buf.to(torch.float32)
        return saved_p, saved_b

    def _restore(saved_p, saved_b):
        for p, orig in saved_p:
            p.data = orig
        for owner, bname, orig in saved_b:
            owner._buffers[bname] = orig

    def _cast_args(a):
        if isinstance(a, torch.Tensor) and a.dtype in (torch.float16, torch.bfloat16):
            return a.to(torch.float32)
        return a

    _wrapped_debug = {"called": False}

    def _wrapped(*args, **kwargs):
        saved_p, saved_b = _swap_to_fp32()
        try:
            new_args = [_cast_args(a) for a in args]
            new_kwargs = {k: _cast_args(v) for k, v in kwargs.items()}
            if not _wrapped_debug["called"]:
                _wrapped_debug["called"] = True
                in_dt = next(
                    (a.dtype for a in new_args if isinstance(a, torch.Tensor)),
                    None,
                )
                w_dt = next(
                    (p.dtype for p in module.parameters(recurse=True)),
                    None,
                )
                print(f"  [wrap {type(module).__name__}] first-call "
                      f"in_dtype={in_dt} weight_dtype={w_dt}", flush=True)
            out = orig_forward(*new_args, **new_kwargs)
            if not _wrapped_debug.get("logged_out"):
                _wrapped_debug["logged_out"] = True
                if isinstance(out, torch.Tensor):
                    print(f"  [wrap {type(module).__name__}] out: "
                          f"dtype={out.dtype} max={out.abs().max().item():.4g}",
                          flush=True)
                elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                    print(f"  [wrap {type(module).__name__}] out[0]: "
                          f"dtype={out[0].dtype} max={out[0].abs().max().item():.4g}",
                          flush=True)
        finally:
            _restore(saved_p, saved_b)

        def _down(t):
            if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                # Clamp to target-dtype finite range before downcast so
                # extreme fp32 values don't become +/-Inf in fp16. This
                # mirrors what an ONNX Clip(...) before Cast(fp16) would
                # do, and matches HF's documented T5 fp16 workaround.
                fmax = torch.finfo(orig_dtype).max
                t = t.clamp_(min=-fmax, max=fmax)
                return t.to(orig_dtype)
            return t
        if isinstance(out, torch.Tensor):
            return _down(out)
        if isinstance(out, tuple):
            return tuple(_down(o) for o in out)
        return out

    module.forward = _wrapped


def apply_islands(encoder, sites: list[str]):
    """Apply fp32 islands to T5 sub-modules."""
    blocks = encoder.encoder.block
    n_patched = {s: 0 for s in sites}
    for blk in blocks:
        # T5 v1.1 layers: blk.layer[0] = T5LayerSelfAttention, blk.layer[1] = T5LayerFF.
        sa = blk.layer[0]
        ff = blk.layer[1]
        attn = sa.SelfAttention
        ffn = ff.DenseReluDense  # T5DenseGatedActDense for v1.1

        for site in sites:
            if site == "ffn":
                _wrap_module_fp32(ffn)
                n_patched[site] += 1
            elif site == "ffn_wo":
                _wrap_module_fp32(ffn.wo)
                n_patched[site] += 1
            elif site == "attn_o":
                _wrap_module_fp32(attn.o)
                n_patched[site] += 1
            elif site == "attn":
                _wrap_module_fp32(attn)
                n_patched[site] += 1
            elif site == "layer_ff":
                _wrap_module_fp32(ff)
                n_patched[site] += 1
            elif site == "layer_sa":
                _wrap_module_fp32(sa)
                n_patched[site] += 1
            else:
                raise ValueError(f"unknown site: {site}")
    print(f"  islands applied: {n_patched}")


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a - b).abs()
    s = b.float().std().item() + 1e-12
    return {
        "max": d.max().item(),
        "mean": d.mean().item(),
        "std_b": b.float().std().item(),
        "rel_mean": d.mean().item() / s,
    }


def run_pass(label, encoder, input_ids, attn_mask):
    print(f"  forward [{label}] ...")
    block_outs, remove = install_taps(encoder)
    t0 = time.time()
    with torch.no_grad():
        out = encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
    last = out.last_hidden_state.detach().to(torch.float32).cpu().clone()
    print(f"  forward done in {time.time()-t0:.1f}s")
    print(f"  last_hidden: shape={tuple(last.shape)} "
          f"mean={last.mean():.5f} std={last.std():.5f} "
          f"min={last.min():.5f} max={last.max():.5f}")
    remove()
    return last, block_outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=str,
                    default="a cat playing piano in a jazz club, cinematic")
    ap.add_argument("--n-text", type=int, default=256)
    ap.add_argument("--fp32", type=str, default="ffn",
                    help="comma-separated sites: ffn,ffn_wo,attn_o,attn,layer_ff,layer_sa")
    ap.add_argument("--ram-budget-gb", type=float, default=14.8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sites = [s.strip() for s in args.fp32.split(",") if s.strip()]
    label = "fp16+" + "+".join(sites) if sites else "fp16"
    print(f"sites = {sites}  (label: {label})")

    ram_peak = start_ram_watchdog(args.ram_budget_gb)
    print(f"RAM budget: {args.ram_budget_gb:.1f} GB")

    print(f"Loading tokenizer from {args.src/'tokenizer'} ...")
    tok = AutoTokenizer.from_pretrained(str(args.src / "tokenizer"))
    enc = tok(args.prompt, padding="max_length", truncation=True,
              max_length=args.n_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    n_real = int(attn_mask.sum().item())
    print(f"  prompt: {args.prompt!r}  real_tokens={n_real}")

    print("\n=== bf16 baseline ===")
    enc_bf = load_t5_streaming(args.src, torch.bfloat16)
    bf_last, bf_blocks = run_pass("bf16", enc_bf, input_ids, attn_mask)
    del enc_bf
    gc.collect()

    print(f"\n=== {label} ===")
    enc_fp = load_t5_streaming(args.src, torch.float16)
    if sites:
        apply_islands(enc_fp, sites)
    fp_last, fp_blocks = run_pass(label, enc_fp, input_ids, attn_mask)
    del enc_fp
    gc.collect()

    np.save(args.out_dir / f"t5_{label.replace('+','_')}_last_hidden.npy",
            fp_last.numpy())

    print(f"\n=== per-block bf16 vs {label} (real tokens only) ===")
    print(f"{'block':>5}  {'maxAbs':>10}  {'meanAbs':>10}  "
          f"{'std(bf16)':>10}  {'meanAbs/std':>12}")
    summary = ["block,maxAbs,meanAbs,std_bf16,rel_mean"]
    for i, (b, f) in enumerate(zip(bf_blocks, fp_blocks)):
        s = diff_stats(b[:, :n_real, :], f[:, :n_real, :])
        print(f"  {i:>3d}    {s['max']:>10.4g}  {s['mean']:>10.4g}  "
              f"{s['std_b']:>10.4g}  {s['rel_mean']:>12.4g}")
        summary.append(f"{i},{s['max']:.6g},{s['mean']:.6g},"
                       f"{s['std_b']:.6g},{s['rel_mean']:.6g}")

    s_last = diff_stats(bf_last[:, :n_real, :], fp_last[:, :n_real, :])
    print(f"\nlast_hidden_state (real tokens):")
    print(f"  maxAbs={s_last['max']:.4g} meanAbs={s_last['mean']:.4g} "
          f"std(bf16)={s_last['std_b']:.4g} rel_mean={s_last['rel_mean']:.4g}")
    summary.append(f"last,{s_last['max']:.6g},{s_last['mean']:.6g},"
                   f"{s_last['std_b']:.6g},{s_last['rel_mean']:.6g}")

    csv = args.out_dir / f"t5_diff_{label.replace('+','_')}.csv"
    csv.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nsaved {csv}")
    print(f"Peak RAM: {ram_peak[0]:.2f} GB")


if __name__ == "__main__":
    main()
