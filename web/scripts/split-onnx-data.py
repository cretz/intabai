#!/usr/bin/env python3
"""Produce a patch JSON that splits a monolithic ONNX file into a slim graph
+ external-data sidecar during download.

The output JSON uses the same Patch format as patch-onnx-webgpu.py's diff-onnx
command, extended with a `sidecar` field. The TS PatchApplier applies both edits
(proto field rewrites) and sidecar ranges (weight byte redirects) in a single
streaming pass during download.

How it works:
  1. Scan the raw protobuf wire format to locate every TensorProto initializer
     whose raw_data exceeds a size threshold.
  2. For each large initializer, produce:
     - A sidecar range: the byte range of raw_data payload in the source file.
     - Edits: rewrite the raw_data field header to external_data entries, and
       fix up all parent length-prefix varints (TensorProto wrapper and
       GraphProto wrapper) to account for the size change.
  3. Verify the result by forward-applying the patch and loading the graph
     with the onnx library (only the ~77 MB graph, not the full model).

Usage:
    pip install onnx
    python split-onnx-data.py <model.onnx> <sidecar_cache_id> <out.json> [--threshold BYTES]

    sidecar_cache_id: the OPFS cache key the TS runtime will use for the
      sidecar file (e.g. "janus_pro_1b_language_model_q4f16_data").

    --threshold: minimum raw_data size in bytes to externalize (default 1MB).
    --sidecar-filename: the location string written into external_data entries
      (default: sidecar_cache_id). Must match what ORT-web / transformers.js
      passes as the externalData path.

Example:
    python split-onnx-data.py language_model_q4f16.onnx \\
        janus_pro_1b_language_model_q4f16_data \\
        language_model_q4f16.split.json \\
        --sidecar-filename language_model_q4f16.onnx_data
"""
import argparse
import json
import sys
from pathlib import Path

from lib.onnx_patch_common import (
    WIRE_LEN,
    WIRE_VARINT,
    encode_string_field,
    encode_tag,
    encode_varint,
    read_string_field,
    read_varint,
    scan_fields,
    sha256,
    verify_patch,
)

# ONNX protobuf field numbers.
FIELD_MODEL_GRAPH = 7
FIELD_GRAPH_INITIALIZER = 5
FIELD_TENSOR_NAME = 8
FIELD_TENSOR_RAW_DATA = 9
FIELD_TENSOR_EXTERNAL_DATA = 13
FIELD_TENSOR_DATA_LOCATION = 14

DEFAULT_THRESHOLD = 1_048_576  # 1 MB


def build_external_data_bytes(sidecar_filename, offset, length):
    """Construct the protobuf wire bytes for external_data entries + data_location."""
    out = bytearray()

    def string_string_entry(key, value):
        return encode_string_field(1, key) + encode_string_field(2, value)

    for key, val in [("location", sidecar_filename),
                     ("offset", str(offset)),
                     ("length", str(length))]:
        entry = string_string_entry(key, val)
        out.extend(encode_tag(FIELD_TENSOR_EXTERNAL_DATA, WIRE_LEN))
        out.extend(encode_varint(len(entry)))
        out.extend(entry)

    out.extend(encode_tag(FIELD_TENSOR_DATA_LOCATION, WIRE_VARINT))
    out.extend(encode_varint(1))
    return bytes(out)


def varint_len(value):
    """How many bytes a varint encoding of value takes."""
    return len(encode_varint(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Input monolithic ONNX file")
    parser.add_argument("sidecar_cache_id", help="OPFS cache key for the sidecar file")
    parser.add_argument("output", type=Path, help="Output patch JSON file")
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Minimum raw_data size to externalize (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--sidecar-filename", default=None,
        help="Filename for external_data location field (default: sidecar_cache_id)",
    )
    args = parser.parse_args()

    if args.output.exists():
        print(f"refusing to overwrite existing file: {args.output}")
        sys.exit(1)

    sidecar_filename = args.sidecar_filename or args.sidecar_cache_id

    print(f"loading {args.model} ({args.model.stat().st_size:,} bytes)...")
    src = args.model.read_bytes()

    # --- Scan the protobuf structure ---
    print("scanning protobuf structure...")
    model_fields = scan_fields(src, 0, len(src))
    graph_field = next(f for f in model_fields if f.field_number == FIELD_MODEL_GRAPH)
    graph_inner = scan_fields(src, graph_field.data_start, graph_field.data_end)
    init_fields = [f for f in graph_inner if f.field_number == FIELD_GRAPH_INITIALIZER]

    print(f"  {len(init_fields)} initializer(s) in graph")

    # For each initializer, find name, raw_data, and any existing data_location field.
    # Record: (name, init_field, raw_data_field, existing_data_location_field_or_None)
    large_inits = []
    for init_f in init_fields:
        tensor_fields = scan_fields(src, init_f.data_start, init_f.data_end)
        name_fields = [f for f in tensor_fields if f.field_number == FIELD_TENSOR_NAME]
        rd_fields = [f for f in tensor_fields if f.field_number == FIELD_TENSOR_RAW_DATA]
        dl_fields = [f for f in tensor_fields if f.field_number == FIELD_TENSOR_DATA_LOCATION]
        if not rd_fields:
            continue
        rd = rd_fields[0]
        payload_size = rd.data_end - rd.data_start
        if payload_size < args.threshold:
            continue
        name = read_string_field(src, name_fields[0]) if name_fields else "<unnamed>"
        # There may be an existing data_location=0 (DEFAULT) field that we need to delete,
        # otherwise protobuf "last wins" will override our inserted data_location=1.
        existing_dl = dl_fields[-1] if dl_fields else None
        large_inits.append((name, init_f, rd, existing_dl))

    print(f"  {len(large_inits)} to externalize:")
    for name, _, rd, _ in large_inits:
        print(f"    {name}: {rd.data_end - rd.data_start:,} bytes")

    if not large_inits:
        print("nothing to split")
        sys.exit(1)

    # --- Build edits and sidecar ranges ---
    # For each large initializer we need three edits:
    #   1. Fix the GraphProto length prefix (it wraps all initializers)
    #   2. Fix the TensorProto (initializer) length prefix
    #   3. Replace the raw_data field tag+varint with external_data fields
    # Plus one sidecar range for the raw_data payload.
    #
    # The graph length prefix changes once (cumulative delta from all inits).
    # Each initializer length prefix changes individually.

    edits = []
    sidecar_ranges = []
    sidecar_offset = 0
    cumulative_graph_delta = 0  # total bytes removed - bytes added inside graph

    for name, init_f, rd, existing_dl in large_inits:
        payload_size = rd.data_end - rd.data_start
        raw_data_field_size = rd.data_end - rd.tag_start  # tag + varint + payload

        ext_data_bytes = build_external_data_bytes(sidecar_filename, sidecar_offset, payload_size)

        # The raw_data field (tag + varint_len + payload) is replaced by ext_data_bytes.
        # But the payload goes to sidecar, so in the primary output we:
        #   - Delete: tag + varint header of raw_data field
        #   - Insert: ext_data_bytes
        #   - Sidecar: the payload bytes
        header_size = rd.data_start - rd.tag_start  # just tag + varint, not payload

        # Size change inside this TensorProto:
        # removed: raw_data field entirely (header + payload)
        # added: ext_data_bytes
        inner_delta = len(ext_data_bytes) - raw_data_field_size

        # If the TensorProto already has a data_location field (typically =0/DEFAULT),
        # delete it. Otherwise protobuf "last wins" means the existing field at the end
        # overrides our inserted data_location=1 (EXTERNAL).
        existing_dl_size = 0
        if existing_dl is not None:
            existing_dl_size = existing_dl.data_end - existing_dl.tag_start
            edits.append({
                "offset": existing_dl.tag_start,
                "delete": existing_dl_size,
                "insert": "",
            })
            inner_delta -= existing_dl_size

        # Fix TensorProto (initializer) length prefix.
        old_init_len = init_f.data_end - init_f.data_start
        new_init_len = old_init_len + inner_delta
        old_init_header = src[init_f.tag_start:init_f.data_start]
        init_tag_bytes = encode_tag(FIELD_GRAPH_INITIALIZER, WIRE_LEN)
        new_init_header = init_tag_bytes + encode_varint(new_init_len)

        if old_init_header != new_init_header:
            edits.append({
                "offset": init_f.tag_start,
                "delete": len(old_init_header),
                "insert": new_init_header.hex(),
            })

        # Replace raw_data tag+varint with external_data fields.
        edits.append({
            "offset": rd.tag_start,
            "delete": header_size,
            "insert": ext_data_bytes.hex(),
        })

        # Sidecar range: the raw_data payload.
        sidecar_ranges.append({
            "offset": rd.data_start,
            "length": payload_size,
        })

        # Track cumulative size change for the graph length prefix.
        init_header_delta = len(new_init_header) - len(old_init_header)
        cumulative_graph_delta += inner_delta + init_header_delta

        sidecar_offset += payload_size

    # Fix GraphProto length prefix.
    old_graph_len = graph_field.data_end - graph_field.data_start
    new_graph_len = old_graph_len + cumulative_graph_delta
    old_graph_header = src[graph_field.tag_start:graph_field.data_start]
    graph_tag_bytes = encode_tag(FIELD_MODEL_GRAPH, WIRE_LEN)
    new_graph_header = graph_tag_bytes + encode_varint(new_graph_len)

    if old_graph_header != new_graph_header:
        edits.append({
            "offset": graph_field.tag_start,
            "delete": len(old_graph_header),
            "insert": new_graph_header.hex(),
        })

    # Sort all edits by offset (required by PatchApplier).
    edits.sort(key=lambda e: e["offset"])

    # --- Forward-apply to compute sizes and hashes ---
    print("\nforward-applying patch...")
    actions = []
    for e in edits:
        actions.append(("edit", e["offset"], e))
    for r in sidecar_ranges:
        actions.append(("sidecar", r["offset"], r))
    actions.sort(key=lambda a: a[1])

    primary = bytearray()
    sidecar = bytearray()
    cursor = 0

    for kind, offset, item in actions:
        if offset > cursor:
            primary.extend(src[cursor:offset])
            cursor = offset
        if kind == "edit":
            primary.extend(bytes.fromhex(item["insert"]))
            cursor = offset + item["delete"]
        else:
            sidecar.extend(src[cursor:cursor + item["length"]])
            cursor = offset + item["length"]
    primary.extend(src[cursor:])

    print(f"  source:  {len(src):,} bytes")
    print(f"  graph:   {len(primary):,} bytes")
    print(f"  sidecar: {len(sidecar):,} bytes")

    patch = {
        "srcSha256": sha256(src),
        "srcLen": len(src),
        "dstSha256": sha256(bytes(primary)),
        "dstLen": len(primary),
        "edits": edits,
        "sidecar": {
            "fileId": args.sidecar_cache_id,
            "sha256": sha256(bytes(sidecar)),
            "len": len(sidecar),
            "ranges": sidecar_ranges,
        },
    }

    # --- Verification ---
    print("\nverifying patch (forward-apply)...")
    verify_patch(src, patch)

    print("verifying graph is valid onnx protobuf...")
    import onnx
    try:
        loaded = onnx.load_from_string(bytes(primary))
        ext_count = sum(1 for i in loaded.graph.initializer if i.data_location == 1)
        inline_big = sum(
            1 for i in loaded.graph.initializer
            if len(i.raw_data) >= args.threshold
        )
        print(f"  ok: {ext_count} external, {inline_big} remaining large inline")
        if inline_big > 0:
            print("  WARNING: some large initializers were not externalised")
        if ext_count != len(large_inits):
            print(f"  WARNING: expected {len(large_inits)} external, got {ext_count}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("verifying sidecar matches original raw_data...")
    sc_cursor = 0
    for name, _, rd, _ in large_inits:
        length = rd.data_end - rd.data_start
        if src[rd.data_start:rd.data_end] != sidecar[sc_cursor:sc_cursor + length]:
            print(f"  FAILED: mismatch for {name}")
            sys.exit(1)
        sc_cursor += length
    if sc_cursor != len(sidecar):
        print(f"  FAILED: sidecar length {len(sidecar)} != consumed {sc_cursor}")
        sys.exit(1)
    print(f"  ok: all {len(large_inits)} payloads match")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(patch, indent=2))
    print(f"\nwrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
