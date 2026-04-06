// Streaming binary patch applier for ONNX model rewrites.
//
// Patch JSONs are produced by `scripts/patch-onnx-webgpu.py diff-onnx`. Each
// patch is a forward-applicable list of edits sorted ascending by offset
// (offsets reference the ORIGINAL source file). We apply edits as bytes flow
// in from the network — no buffering of the full source — by maintaining a
// running source-byte cursor and a small amount of state for an edit's
// pending delete that may straddle a chunk boundary.

export interface PatchEdit {
  /** Offset in the original (pre-patch) source byte stream. */
  offset: number;
  /** Number of source bytes to drop at this offset. */
  delete: number;
  /** Bytes to insert at this offset (hex-encoded in the JSON). */
  insert: string;
}

export interface Patch {
  srcSha256: string;
  srcLen: number;
  dstSha256: string;
  dstLen: number;
  edits: PatchEdit[];
}

interface DecodedEdit {
  offset: number;
  delete: number;
  insert: Uint8Array;
}

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length >> 1);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substr(i << 1, 2), 16);
  }
  return out;
}

export class PatchApplier {
  private readonly edits: DecodedEdit[];
  private editIdx = 0;
  private srcPos = 0;
  private pendingDelete = 0;
  private dstWritten = 0;

  constructor(private readonly patch: Patch) {
    this.edits = patch.edits.map((e) => ({
      offset: e.offset,
      delete: e.delete,
      insert: hexToBytes(e.insert),
    }));
  }

  /** Feed a chunk of source bytes; emits patched bytes via `write`. */
  async apply(chunk: Uint8Array, write: (bytes: Uint8Array) => Promise<void>): Promise<void> {
    let off = 0;
    while (off < chunk.length) {
      // Drain any pending delete carried over from a previous chunk.
      if (this.pendingDelete > 0) {
        const skip = Math.min(this.pendingDelete, chunk.length - off);
        off += skip;
        this.srcPos += skip;
        this.pendingDelete -= skip;
        continue;
      }
      // Apply any edits whose trigger offset is exactly here.
      if (this.editIdx < this.edits.length && this.srcPos === this.edits[this.editIdx].offset) {
        const e = this.edits[this.editIdx++];
        if (e.insert.length > 0) {
          await write(e.insert);
          this.dstWritten += e.insert.length;
        }
        this.pendingDelete = e.delete;
        continue;
      }
      // Pass through src bytes up to (but not including) the next edit.
      const nextOffset =
        this.editIdx < this.edits.length ? this.edits[this.editIdx].offset : Infinity;
      const room = chunk.length - off;
      const untilEdit = nextOffset - this.srcPos;
      const passThrough = Math.min(room, untilEdit);
      const slice = chunk.subarray(off, off + passThrough);
      await write(slice);
      this.dstWritten += slice.length;
      off += passThrough;
      this.srcPos += passThrough;
    }
  }

  /** Flush any tail edits whose offset is at end-of-source. */
  async finish(write: (bytes: Uint8Array) => Promise<void>): Promise<void> {
    while (
      this.editIdx < this.edits.length &&
      this.edits[this.editIdx].offset === this.srcPos &&
      this.pendingDelete === 0
    ) {
      const e = this.edits[this.editIdx++];
      if (e.insert.length > 0) {
        await write(e.insert);
        this.dstWritten += e.insert.length;
      }
      // delete must be 0 at EOF
      if (e.delete !== 0) {
        throw new Error(`patch trailing edit has nonzero delete=${e.delete}`);
      }
    }
    if (this.srcPos !== this.patch.srcLen) {
      throw new Error(
        `patch srcLen mismatch: consumed ${this.srcPos}, expected ${this.patch.srcLen}`,
      );
    }
    if (this.editIdx !== this.edits.length) {
      throw new Error(`patch has ${this.edits.length - this.editIdx} unapplied edit(s)`);
    }
    if (this.dstWritten !== this.patch.dstLen) {
      throw new Error(
        `patch dstLen mismatch: wrote ${this.dstWritten}, expected ${this.patch.dstLen}`,
      );
    }
  }
}

/** Hex-encode a hash digest for comparison against patch JSON fields. */
export async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const view = new Uint8Array(digest);
  let out = "";
  for (let i = 0; i < view.length; i++) {
    out += view[i].toString(16).padStart(2, "0");
  }
  return out;
}
