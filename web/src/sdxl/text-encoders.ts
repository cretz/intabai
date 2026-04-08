// SDXL dual text encoder. Wraps two ORT-web InferenceSessions, one for the
// CLIP-L encoder (text_encoder) and one for the OpenCLIP-bigG encoder
// (text_encoder_2), and combines their outputs into the form the SDXL
// UNet's cross-attention expects:
//
//   encoder_hidden_states = concat(
//       penultimate_hidden_state(text_encoder),    // [B, 77, 768]
//       penultimate_hidden_state(text_encoder_2),  // [B, 77, 1280]
//     ) along the last axis -> [B, 77, 2048]
//
//   pooled_text_embeds = pooled_output(text_encoder_2)  // [B, 1280]
//
// Diffusers/optimum SDXL exports include `hidden_states.N` outputs for
// every transformer layer plus the input embedding (CLIP-L: 13 of them
// indexed 0..12; bigG: 33 indexed 0..32) when `output_hidden_states=True`
// was set during export. SDXL specifically wants the penultimate layer
// (index N-1 where N is the largest exported index), NOT `last_hidden_state`.
// We probe session.outputNames at load time and pick the right one.
//
// For the pooled output we look for the conventional names `text_embeds`,
// `pooled_output`, or `pooler_output` and use whichever the export provides.
//
// CFG: caller invokes encode() twice, once with the empty string for the
// unconditional embedding and once with the real prompt, same as SD1.5.
//
// RAM policy: load both -> encode -> release both. Both encoders together
// are ~1.55 GB on disk; we don't separate them because they always run as
// a pair and the bigG sidecar dominates the budget.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { CLIP_CONTEXT_LENGTH, type TokenizedPrompt } from "../sd15/tokenizer";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { f16ToF32Array } from "../sd15/fp16";
import { SDXL_CONTEXT_LENGTH, SDXL_HIDDEN_SIZE, SDXL_POOLED_SIZE } from "./unet";

const CLIP_L_HIDDEN = 768;
const CLIP_BIGG_HIDDEN = 1280;

function defaultProviders(): string[] {
  // Unlike SD1.5's CLIP-L (which hits the WebGPU Attention-mask kernel
  // gap and is forced to wasm), the diffusers/optimum SDXL exports for
  // both text_encoder and text_encoder_2 use a different Attention shape
  // that ORT-web's WebGPU kernel handles. Smoke session 2026-04-08
  // verified session.create on this exact bundle via [webgpu, wasm].
  // Wasm stays in the list as a fallback for stacks without WebGPU.
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

/** One pair of SDXL embeddings ready to feed the UNet. */
export interface SdxlEmbeddings {
  /** Concatenated penultimate hidden states, [77, 2048] flattened. */
  hiddenStates: Float32Array;
  /** Pooled bigG output, [1280]. */
  pooledTextEmbeds: Float32Array;
}

interface EncoderHandles {
  session: ort.InferenceSession;
  inputName: string;
  /** Resolved input dtype. Probed at load time via a dummy session.run
   *  with a 77-element zeros tensor: try int64 first, retry int32 on
   *  dtype-mismatch error, pin. Even within a single SDXL bundle the two
   *  encoders can disagree (segmind-vega: CLIP-L int64, bigG int32;
   *  webnn/sdxl-turbo: the reverse). session.inputMetadata is unreliable
   *  in the ORT-web version we are on so the dummy-run probe is the
   *  authoritative answer. ~10ms one-time cost per encoder. */
  inputIsInt64: boolean;
  /** Output name for the penultimate hidden state. */
  hiddenStateOutput: string;
  /** Output name for the pooled output, or null if this encoder does not
   *  expose one (CLIP-L typically does not need to in the SDXL pipeline). */
  pooledOutput: string | null;
}

function pickPenultimateHiddenState(outputNames: readonly string[]): string {
  // Look for `hidden_states.N` outputs, pick the one with the second-largest N.
  const hsIndexed: Array<{ name: string; idx: number }> = [];
  for (const n of outputNames) {
    const m = /^hidden_states\.(\d+)$/.exec(n);
    if (m) hsIndexed.push({ name: n, idx: parseInt(m[1], 10) });
  }
  if (hsIndexed.length >= 2) {
    hsIndexed.sort((a, b) => a.idx - b.idx);
    // Penultimate = second to last = index length-2.
    return hsIndexed[hsIndexed.length - 2].name;
  }
  // Some exports (e.g. webnn/sdxl-turbo q4f16) prune all intermediate
  // hidden_states and only ship the penultimate layer. If there is exactly
  // one hidden_states.N output, trust that the export already selected the
  // right layer.
  if (hsIndexed.length === 1) {
    return hsIndexed[0].name;
  }
  throw new Error(
    `SDXL text encoder ONNX is missing per-layer hidden_states outputs ` +
      `(need hidden_states.0..N). Have: [${outputNames.join(", ")}]`,
  );
}

function pickPooledOutput(outputNames: readonly string[]): string | null {
  for (const candidate of ["text_embeds", "pooled_output", "pooler_output"]) {
    if (outputNames.includes(candidate)) return candidate;
  }
  return null;
}

async function loadEncoder(
  cache: ModelCache,
  file: OrtModelFile,
  needsPooled: boolean,
): Promise<EncoderHandles> {
  // graphOptimizationLevel: "disabled" sidesteps an ORT-web optimizer bug
  // observed on the segmind-vega text encoders: the SimplifiedLayerNormFusion
  // pass references "InsertedPrecisionFreeCast_*" node args that an earlier
  // pass already removed, throwing "Attempting to get index by a name which
  // does not exist" at session.create. The diffusers/optimum exports are
  // already pre-optimized at export time, so disabling runtime fusion costs
  // ~5-15% on the text encoder pass and the encoder runs once per generation.
  // Cheap insurance.
  const session = await createSession(cache, file, defaultProviders(), {
    graphOptimizationLevel: "disabled",
  });
  if (session.inputNames.length === 0) {
    throw new Error("SDXL text encoder ONNX has no inputs");
  }
  const inputName = session.inputNames[0];
  const inputIsInt64 = await probeInputDtype(session, inputName);
  const hiddenStateOutput = pickPenultimateHiddenState(session.outputNames);
  const pooledOutput = needsPooled ? pickPooledOutput(session.outputNames) : null;
  if (needsPooled && !pooledOutput) {
    throw new Error(
      `SDXL bigG text encoder is missing a pooled output ` +
        `(need text_embeds / pooled_output / pooler_output). ` +
        `Have: [${session.outputNames.join(", ")}]`,
    );
  }

  return { session, inputName, inputIsInt64, hiddenStateOutput, pooledOutput };
}

/** Probe whether the encoder wants int64 or int32 input_ids. Tries int64
 *  first; on dtype-mismatch error, retries int32. Either way the dummy
 *  inputs are a 77-element zeros tensor (the encoder's own padding token
 *  if zero is special is fine - we throw the result away). */
async function probeInputDtype(session: ort.InferenceSession, inputName: string): Promise<boolean> {
  const dims = [1, CLIP_CONTEXT_LENGTH];
  try {
    await session.run({
      [inputName]: new ort.Tensor("int64", new BigInt64Array(CLIP_CONTEXT_LENGTH), dims),
    });
    return true;
  } catch (err) {
    const msg = (err as Error).message ?? "";
    if (!/data type|tensor\(int|expected/i.test(msg)) {
      // Not a dtype error - rethrow so the caller sees the real failure
      // (e.g. missing kernel, malformed graph) instead of silently
      // falling back and getting a confusing int32 mismatch later.
      throw err;
    }
  }
  await session.run({
    [inputName]: new ort.Tensor("int32", new Int32Array(CLIP_CONTEXT_LENGTH), dims),
  });
  return false;
}

/** Build the input_ids tensor in the dtype the encoder accepts. Resolved
 *  per-encoder at load time via probeInputDtype(). */
function buildIdsTensor(prompt: TokenizedPrompt, isInt64: boolean): ort.Tensor {
  const dims = [1, CLIP_CONTEXT_LENGTH];
  if (isInt64) {
    const big = new BigInt64Array(CLIP_CONTEXT_LENGTH);
    for (let i = 0; i < CLIP_CONTEXT_LENGTH; i++) big[i] = BigInt(prompt.ids[i]);
    return new ort.Tensor("int64", big, dims);
  }
  return new ort.Tensor("int32", new Int32Array(prompt.ids), dims);
}

/** Convert one ORT output tensor to a Float32Array, handling fp16/fp32 and
 *  the Float16Array vs Uint16Array detection dance. */
function tensorToF32(tensor: ort.Tensor): Float32Array {
  const ctorName = (tensor.data as ArrayLike<number>).constructor.name;
  if (ctorName === "Float16Array") {
    return new Float32Array(tensor.data as ArrayLike<number>);
  }
  if (tensor.type === "float16") {
    return f16ToF32Array(tensor.data as Uint16Array);
  }
  return new Float32Array(tensor.data as Float32Array);
}

export class SdxlDualTextEncoder {
  private constructor(
    private clipL: EncoderHandles | null,
    private bigG: EncoderHandles | null,
  ) {}

  static async load(
    cache: ModelCache,
    clipLFile: OrtModelFile,
    bigGFile: OrtModelFile,
  ): Promise<SdxlDualTextEncoder> {
    // Load sequentially so peak memory during session create is one
    // encoder at a time, not both. They are still both held resident
    // afterwards because we need both for every generation.
    const clipL = await loadEncoder(cache, clipLFile, false);
    const bigG = await loadEncoder(cache, bigGFile, true);
    return new SdxlDualTextEncoder(clipL, bigG);
  }

  /**
   * Encode one prompt through both encoders and assemble the SDXL UNet
   * conditioning tensors. Caller invokes twice (uncond + cond) per
   * generation, then feeds each pair to SdxlUnet.predictNoise.
   */
  async encode(prompt: TokenizedPrompt): Promise<SdxlEmbeddings> {
    if (!this.clipL || !this.bigG) {
      throw new Error("SDXL text encoders have been released");
    }
    if (prompt.ids.length !== CLIP_CONTEXT_LENGTH) {
      throw new Error(`expected ${CLIP_CONTEXT_LENGTH} input ids, got ${prompt.ids.length}`);
    }

    // CLIP-L: penultimate hidden states only.
    const clipLFeeds: Record<string, ort.Tensor> = {
      [this.clipL.inputName]: buildIdsTensor(prompt, this.clipL.inputIsInt64),
    };
    const clipLOut = await this.clipL.session.run(clipLFeeds);
    const clipLHidden = tensorToF32(clipLOut[this.clipL.hiddenStateOutput]);
    if (clipLHidden.length !== CLIP_CONTEXT_LENGTH * CLIP_L_HIDDEN) {
      throw new Error(
        `CLIP-L hidden state length ${clipLHidden.length} != expected ${CLIP_CONTEXT_LENGTH * CLIP_L_HIDDEN}`,
      );
    }

    // bigG: penultimate hidden states + pooled output.
    const bigGFeeds: Record<string, ort.Tensor> = {
      [this.bigG.inputName]: buildIdsTensor(prompt, this.bigG.inputIsInt64),
    };
    const bigGOut = await this.bigG.session.run(bigGFeeds);
    const bigGHidden = tensorToF32(bigGOut[this.bigG.hiddenStateOutput]);
    if (bigGHidden.length !== CLIP_CONTEXT_LENGTH * CLIP_BIGG_HIDDEN) {
      throw new Error(
        `bigG hidden state length ${bigGHidden.length} != expected ${CLIP_CONTEXT_LENGTH * CLIP_BIGG_HIDDEN}`,
      );
    }
    const pooled = tensorToF32(bigGOut[this.bigG.pooledOutput!]);
    if (pooled.length !== SDXL_POOLED_SIZE) {
      throw new Error(`bigG pooled output length ${pooled.length} != expected ${SDXL_POOLED_SIZE}`);
    }

    // Concat along the last axis: for each of 77 token positions, lay
    // CLIP-L's 768 floats followed by bigG's 1280 floats. Output is
    // row-major [77, 2048].
    const hiddenStates = new Float32Array(SDXL_CONTEXT_LENGTH * SDXL_HIDDEN_SIZE);
    for (let t = 0; t < SDXL_CONTEXT_LENGTH; t++) {
      const dstRow = t * SDXL_HIDDEN_SIZE;
      const lSrc = t * CLIP_L_HIDDEN;
      const gSrc = t * CLIP_BIGG_HIDDEN;
      for (let i = 0; i < CLIP_L_HIDDEN; i++) {
        hiddenStates[dstRow + i] = clipLHidden[lSrc + i];
      }
      for (let i = 0; i < CLIP_BIGG_HIDDEN; i++) {
        hiddenStates[dstRow + CLIP_L_HIDDEN + i] = bigGHidden[gSrc + i];
      }
    }

    return { hiddenStates, pooledTextEmbeds: pooled };
  }

  async release(): Promise<void> {
    if (this.clipL) {
      await this.clipL.session.release();
      this.clipL = null;
    }
    if (this.bigG) {
      await this.bigG.session.release();
      this.bigG = null;
    }
  }
}
