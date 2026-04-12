"""Shared utilities for ONNX patch/split scripts.

Provides:
  - SHA-256 hashing
  - Protobuf wire format reading/writing (varint, tags, field scanning)
  - Patch JSON verification (forward-apply edits + sidecar extraction)
"""
import hashlib
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Protobuf wire format
# ---------------------------------------------------------------------------

WIRE_VARINT = 0
WIRE_LEN = 2

def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode a protobuf varint at `pos`. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if (b & 0x80) == 0:
            return result, pos
        shift += 7


def encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def encode_tag(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def encode_len_prefixed(data: bytes) -> bytes:
    """Varint length prefix + data."""
    return encode_varint(len(data)) + data


def encode_string_field(field_number: int, value: str) -> bytes:
    """Encode a protobuf string field (tag + length + utf8 bytes)."""
    encoded = value.encode("utf-8")
    return encode_tag(field_number, WIRE_LEN) + encode_len_prefixed(encoded)


class Field(NamedTuple):
    """A parsed protobuf field in a message."""
    field_number: int
    wire_type: int
    # Byte offset of the tag in the buffer.
    tag_start: int
    # Byte offset just past the tag+length header (start of payload).
    data_start: int
    # Byte offset just past the payload.
    data_end: int


def scan_fields(buf: bytes, start: int, end: int) -> list[Field]:
    """Walk a protobuf message region and return all top-level fields."""
    fields = []
    pos = start
    while pos < end:
        tag_start = pos
        tag, pos = read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == WIRE_VARINT:
            data_start = pos
            _, pos = read_varint(buf, pos)
            fields.append(Field(field_number, wire_type, tag_start, data_start, pos))
        elif wire_type == WIRE_LEN:
            length, data_start = read_varint(buf, pos)
            pos = data_start + length
            fields.append(Field(field_number, wire_type, tag_start, data_start, pos))
        elif wire_type == 0:  # WIRE_VARINT (already handled, but for safety)
            raise ValueError(f"unexpected wire type {wire_type} at offset {tag_start}")
        elif wire_type == 5:  # 32-bit fixed
            fields.append(Field(field_number, wire_type, tag_start, pos, pos + 4))
            pos += 4
        elif wire_type == 1:  # 64-bit fixed
            fields.append(Field(field_number, wire_type, tag_start, pos, pos + 8))
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wire_type} at offset {tag_start}")
    return fields


def read_bytes_field(buf: bytes, field: Field) -> bytes:
    """Extract the raw payload bytes of a length-delimited field."""
    assert field.wire_type == WIRE_LEN
    return buf[field.data_start : field.data_end]


def read_string_field(buf: bytes, field: Field) -> str:
    return read_bytes_field(buf, field).decode("utf-8")


# ---------------------------------------------------------------------------
# Patch verification
# ---------------------------------------------------------------------------

def verify_patch(src: bytes, patch: dict) -> None:
    """Forward-apply a patch dict to src bytes and verify correctness.

    Handles both plain edit patches and patches with sidecar ranges.
    Raises on any mismatch.
    """
    edits = patch["edits"]
    sidecar_spec = patch.get("sidecar")

    # Build a merged action list sorted by offset, same as the TS PatchApplier.
    actions = []
    for e in edits:
        actions.append(("edit", e["offset"], e))
    if sidecar_spec:
        for r in sidecar_spec["ranges"]:
            actions.append(("sidecar", r["offset"], r))
    actions.sort(key=lambda a: a[1])

    primary = bytearray()
    sidecar = bytearray()
    cursor = 0

    for kind, offset, item in actions:
        # Passthrough bytes before this action.
        if offset > cursor:
            primary.extend(src[cursor:offset])
            cursor = offset
        if kind == "edit":
            primary.extend(bytes.fromhex(item["insert"]))
            cursor = offset + item["delete"]
        else:
            length = item["length"]
            sidecar.extend(src[cursor : cursor + length])
            cursor = offset + length

    # Trailing passthrough.
    primary.extend(src[cursor:])

    # Verify primary output.
    if len(primary) != patch["dstLen"]:
        raise RuntimeError(
            f"primary length mismatch: got {len(primary)}, expected {patch['dstLen']}"
        )
    got_sha = sha256(bytes(primary))
    if got_sha != patch["dstSha256"]:
        raise RuntimeError(
            f"primary SHA mismatch: got {got_sha}, expected {patch['dstSha256']}"
        )

    # Verify sidecar if present.
    if sidecar_spec:
        if len(sidecar) != sidecar_spec["len"]:
            raise RuntimeError(
                f"sidecar length mismatch: got {len(sidecar)}, expected {sidecar_spec['len']}"
            )
        got_sc_sha = sha256(bytes(sidecar))
        if got_sc_sha != sidecar_spec["sha256"]:
            raise RuntimeError(
                f"sidecar SHA mismatch: got {got_sc_sha}, expected {sidecar_spec['sha256']}"
            )

    print(f"  verified: {len(edits)} edit(s)", end="")
    if sidecar_spec:
        print(f", {len(sidecar_spec['ranges'])} sidecar range(s)", end="")
    print()
