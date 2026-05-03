#!/usr/bin/env python3
"""Pack ONNX per-initializer external-data files into N target-sized shards.

The legacy torch.onnx.export emits transformer.onnx plus one raw file per
initializer (each protobuf init's external_data.location names its file).
Wasm32 has a 4 GB single-file cap and mobile parallel downloads prefer many
small shards, so we repack into ~target-mb shards.

This script:
  1. Loads the protobuf with load_external_data=False (graph only).
  2. For each external initializer, reads raw bytes from
     <staging-dir>/<location> verbatim.
  3. Appends those bytes to one of N output shard files, rolling over when
     a shard would exceed target-mb.
  4. Rewrites each init's external_data (location, offset, length) to point
     at the new shard, saves ONLY the protobuf (no save_model).

This avoids `onnx.save_model(save_as_external_data=True)` on an already
externally-loaded model, which can corrupt weights past an offset.

Run:
  cd intabai/web/scripts && uv run python ltx/shard_external_data.py \
    --onnx /abs/.../staging/transformer.onnx \
    --staging-dir /abs/.../staging \
    --out-dir /abs/.../hf-repo/onnx \
    --target-mb 600
"""
import argparse
from pathlib import Path

import onnx
from onnx.external_data_helper import _get_all_tensors


def get_external_kv(initializer) -> dict:
    return {entry.key: entry.value for entry in initializer.external_data}


def set_external_kv(initializer, kv: dict) -> None:
    del initializer.external_data[:]
    for k, v in kv.items():
        e = initializer.external_data.add()
        e.key = k
        e.value = str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--staging-dir", type=Path, required=True,
                    help="Directory holding the per-initializer external-data files")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-mb", type=int, default=600)
    ap.add_argument("--prefix", type=str, default="transformer")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = args.target_mb * 1024 * 1024

    print(f"Loading protobuf (graph only) from {args.onnx} ...")
    model = onnx.load(str(args.onnx), load_external_data=False)
    # Walk all TensorProtos: graph.initializer + Constant op attribute values.
    all_tensors = list(_get_all_tensors(model))
    ext_tensors = [t for t in all_tensors if t.data_location == onnx.TensorProto.EXTERNAL]
    print(f"  tensors: {len(all_tensors)} total, {len(ext_tensors)} external")

    shards: list[list] = [[]]
    sizes: list[int] = [0]
    for t in ext_tensors:
        kv = get_external_kv(t)
        src_path = args.staging_dir / kv["location"]
        src_off = int(kv.get("offset", 0))
        length = int(kv.get("length", src_path.stat().st_size - src_off))
        if sizes[-1] > 0 and sizes[-1] + length > target_bytes:
            shards.append([])
            sizes.append(0)
        shards[-1].append((t, src_path, src_off, length))
        sizes[-1] += length

    print(f"  shard plan: {len(shards)} shards, sizes (MB): "
          + ", ".join(f"{s/1e6:.0f}" for s in sizes))

    for shard_idx, items in enumerate(shards):
        shard_name = f"{args.prefix}.shard{shard_idx:02d}.bin"
        shard_path = args.out_dir / shard_name
        print(f"  writing {shard_name} ({sizes[shard_idx]/1e6:.1f} MB, {len(items)} tensors)")
        with open(shard_path, "wb") as dst:
            running = 0
            for t, src_path, src_off, length in items:
                with open(src_path, "rb") as src:
                    src.seek(src_off)
                    remaining = length
                    while remaining > 0:
                        chunk = src.read(min(remaining, 64 * 1024 * 1024))
                        if not chunk:
                            raise RuntimeError(
                                f"unexpected EOF reading {t.name} from {src_path}"
                            )
                        dst.write(chunk)
                        remaining -= len(chunk)
                set_external_kv(t, {
                    "location": shard_name,
                    "offset": running,
                    "length": length,
                })
                running += length

    out_onnx = args.out_dir / f"{args.prefix}.onnx"
    print(f"Saving rewritten protobuf to {out_onnx} ...")
    # Save protobuf only; no external-data side effects.
    with open(out_onnx, "wb") as f:
        f.write(model.SerializeToString())

    print("OK")


if __name__ == "__main__":
    main()
