// Streaming binary patch applier for ONNX model rewrites.
//
// Patch JSONs are produced offline (e.g. by `scripts/patch-onnx-webgpu.py
// diff-onnx`). Each patch is a forward-applicable list of edits sorted
// ascending by offset (offsets reference the ORIGINAL source file). We apply
// edits as bytes flow in from the network - no buffering of the full source -
// by maintaining a running source-byte cursor and a small amount of state for
// an edit's pending delete that may straddle a chunk boundary.
//
// Sidecar support: a patch can optionally declare byte ranges in the source
// that should be redirected to a separate OPFS file instead of the primary
// output. This enables splitting a monolithic ONNX file into a slim graph +
// external-data sidecar during download. Edits and sidecar ranges are
// interleaved in a single sorted pass through the source stream.

import type { ModelFileTransform, TransformWrite } from "./model-cache";

export interface PatchEdit {
  /** Offset in the original (pre-patch) source byte stream. */
  offset: number;
  /** Number of source bytes to drop at this offset. */
  delete: number;
  /** Bytes to insert at this offset (hex-encoded in the JSON). */
  insert: string;
}

export interface PatchSidecar {
  /** OPFS cache key for the sidecar file. */
  fileId: string;
  /** SHA-256 of the expected sidecar output. */
  sha256: string;
  /** Expected total sidecar size in bytes. */
  len: number;
  /** Byte ranges in the source to redirect. Sorted ascending by offset. */
  ranges: { offset: number; length: number }[];
}

export interface Patch {
  srcSha256: string;
  srcLen: number;
  dstSha256: string;
  dstLen: number;
  edits: PatchEdit[];
  sidecar?: PatchSidecar;
}

/** Decoded edit with raw bytes instead of hex string. */
interface DecodedEdit {
  offset: number;
  delete: number;
  insert: Uint8Array;
}

/** A source-offset action: either an edit or a sidecar range redirect. */
type Action =
  | { type: "edit"; offset: number; edit: DecodedEdit }
  | { type: "sidecar"; offset: number; length: number; fileId: string };

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length >> 1);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substr(i << 1, 2), 16);
  }
  return out;
}

/** Build a merged, sorted action list from edits and sidecar ranges. */
function buildActions(patch: Patch): Action[] {
  const actions: Action[] = [];
  for (const e of patch.edits) {
    actions.push({
      type: "edit",
      offset: e.offset,
      edit: { offset: e.offset, delete: e.delete, insert: hexToBytes(e.insert) },
    });
  }
  if (patch.sidecar) {
    for (const r of patch.sidecar.ranges) {
      actions.push({
        type: "sidecar",
        offset: r.offset,
        length: r.length,
        fileId: patch.sidecar.fileId,
      });
    }
  }
  actions.sort((a, b) => a.offset - b.offset);
  return actions;
}

export class PatchApplier {
  private readonly actions: Action[];
  private actionIdx = 0;
  private srcPos = 0;
  private dstWritten = 0;
  private sidecarWritten = 0;

  // State for an in-progress delete (from an edit) or sidecar redirect that
  // straddles a chunk boundary.
  private pendingDelete = 0;
  private pendingSidecar: { fileId: string; remaining: number } | null = null;

  constructor(private readonly patch: Patch) {
    this.actions = buildActions(patch);
  }

  /** Feed a chunk of source bytes; emits patched bytes via `write`. */
  async apply(chunk: Uint8Array, write: TransformWrite): Promise<void> {
    let off = 0;
    while (off < chunk.length) {
      // Drain pending sidecar redirect carried over from a previous chunk.
      if (this.pendingSidecar) {
        const n = Math.min(this.pendingSidecar.remaining, chunk.length - off);
        await write(chunk.subarray(off, off + n), this.pendingSidecar.fileId);
        this.sidecarWritten += n;
        off += n;
        this.srcPos += n;
        this.pendingSidecar.remaining -= n;
        if (this.pendingSidecar.remaining === 0) this.pendingSidecar = null;
        continue;
      }

      // Drain pending delete carried over from a previous chunk.
      if (this.pendingDelete > 0) {
        const skip = Math.min(this.pendingDelete, chunk.length - off);
        off += skip;
        this.srcPos += skip;
        this.pendingDelete -= skip;
        continue;
      }

      // Apply any action whose trigger offset is exactly here.
      if (
        this.actionIdx < this.actions.length &&
        this.srcPos === this.actions[this.actionIdx].offset
      ) {
        const a = this.actions[this.actionIdx++];
        if (a.type === "edit") {
          if (a.edit.insert.length > 0) {
            await write(a.edit.insert);
            this.dstWritten += a.edit.insert.length;
          }
          this.pendingDelete = a.edit.delete;
        } else {
          // Sidecar redirect: start streaming source bytes to the sidecar file.
          this.pendingSidecar = { fileId: a.fileId, remaining: a.length };
        }
        continue;
      }

      // Pass through src bytes up to the next action.
      const nextOffset =
        this.actionIdx < this.actions.length ? this.actions[this.actionIdx].offset : Infinity;
      const room = chunk.length - off;
      const untilAction = nextOffset - this.srcPos;
      const passThrough = Math.min(room, untilAction);
      const slice = chunk.subarray(off, off + passThrough);
      await write(slice);
      this.dstWritten += slice.length;
      off += passThrough;
      this.srcPos += passThrough;
    }
  }

  /** Flush any tail actions whose offset is at end-of-source. */
  async finish(write: TransformWrite): Promise<void> {
    while (
      this.actionIdx < this.actions.length &&
      this.actions[this.actionIdx].offset === this.srcPos &&
      this.pendingDelete === 0 &&
      !this.pendingSidecar
    ) {
      const a = this.actions[this.actionIdx++];
      if (a.type === "edit") {
        if (a.edit.insert.length > 0) {
          await write(a.edit.insert);
          this.dstWritten += a.edit.insert.length;
        }
        if (a.edit.delete !== 0) {
          throw new Error(`patch trailing edit has nonzero delete=${a.edit.delete}`);
        }
      } else {
        throw new Error(`sidecar range at EOF (offset=${a.offset}) is invalid`);
      }
    }
    if (this.srcPos !== this.patch.srcLen) {
      throw new Error(
        `patch srcLen mismatch: consumed ${this.srcPos}, expected ${this.patch.srcLen}`,
      );
    }
    if (this.actionIdx !== this.actions.length) {
      throw new Error(`patch has ${this.actions.length - this.actionIdx} unapplied action(s)`);
    }
    if (this.dstWritten !== this.patch.dstLen) {
      throw new Error(
        `patch dstLen mismatch: wrote ${this.dstWritten}, expected ${this.patch.dstLen}`,
      );
    }
    if (this.patch.sidecar && this.sidecarWritten !== this.patch.sidecar.len) {
      throw new Error(
        `sidecar len mismatch: wrote ${this.sidecarWritten}, expected ${this.patch.sidecar.len}`,
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

/**
 * Wrap a Patch as a ModelFileTransform the shared cache can run during
 * download. Works for both plain edit patches and split patches (with sidecar).
 * The cache key suffix encodes the expected output hash so changing the patch
 * naturally invalidates the old cached blob.
 */
export function patchTransform(patch: Patch): ModelFileTransform {
  const tag = patch.sidecar ? "split" : "patched";
  return {
    cacheKeySuffix: `.${tag}.${patch.dstSha256.slice(0, 8)}`,
    createApplier() {
      const applier = new PatchApplier(patch);
      return {
        apply: (chunk: Uint8Array, write: TransformWrite) => applier.apply(chunk, write),
        finish: (write: TransformWrite) => applier.finish(write),
        async verify(bytes) {
          const got = await sha256Hex(bytes);
          if (got !== patch.dstSha256) {
            throw new Error(`${tag} hash mismatch (got ${got}, expected ${patch.dstSha256})`);
          }
        },
      };
    },
  };
}
