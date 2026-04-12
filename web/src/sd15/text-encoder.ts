// CLIP text encoder for SD1.5. Wraps an ORT-web InferenceSession around the
// text_encoder/model.onnx file from tlwu/stable-diffusion-v1-5-onnxruntime.
// The encoder maps a length-77 token id sequence (produced by ClipTokenizer)
// to a [77, 768] embedding tensor that the SD1.5 UNet's cross-attention
// layers consume.
//
// RAM policy: this class owns exactly one ORT session. Callers must follow
// the load -> encode -> release pattern so the GPU memory the encoder uses
// is freed before the UNet session is created. The four SD1.5 networks
// never run concurrently, so peak GPU memory is max(stage_size) rather than
// sum(stage_sizes). See web/src/sd15/models.ts header comment.
//
// Classifier-free guidance: SD1.5 needs both an unconditional ("") and a
// conditional embedding per generation, concatenated along the batch axis
// before being fed to the UNet. We expose a single-prompt encode() and let
// the caller invoke it twice. Batching the two calls into a single batch=2
// session run is a possible optimization, but the text encoder is the
// fastest stage by an order of magnitude (~50ms), so it is not worth the
// API complexity yet.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { CLIP_CONTEXT_LENGTH, type TokenizedPrompt } from "./tokenizer";
import { createSession, type OrtModelFile } from "./ort-helpers";

/** SD1.5's CLIP text encoder hidden size. Hard-coded by the model. */
export const CLIP_HIDDEN_SIZE = 768;

/**
 * One [77, 768] embedding tensor, flattened row-major. Length is always
 * CLIP_CONTEXT_LENGTH * CLIP_HIDDEN_SIZE = 59136 floats.
 */
export type TextEmbedding = Float32Array;

function defaultProviders(): string[] {
  // ORT-web's WebGPU Attention kernel rejects the fused Attention op in
  // tlwu's CLIP text encoder export with "Mask not supported" - it doesn't
  // handle the causal mask variant CLIP uses, and the failure happens at
  // run() time so the wasm fallback in a [webgpu, wasm] provider list is
  // never engaged. Until we either patch the ONNX (face-swap precedent) or
  // ORT-web ships a fix, the text encoder runs on wasm only. Cost is
  // ~500ms vs ~50ms on WebGPU; the encoder runs once per generation so
  // this is acceptable. UNet and VAE will keep webgpu via their own load
  // calls.
  return ["wasm"];
}

export class TextEncoder {
  private constructor(
    private session: ort.InferenceSession | null,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly inputIsInt64: boolean,
  ) {}

  /** Load the text encoder ONNX from the cache and create an ORT session. */
  static async load(cache: ModelCache, file: OrtModelFile): Promise<TextEncoder> {
    const session = await createSession(cache, file, defaultProviders());

    // SD1.5 ONNX exports name the input "input_ids" and the output we care
    // about "last_hidden_state". Some exports also expose "pooler_output";
    // SD1.5 ignores it. Look the names up dynamically rather than hard-
    // coding them so a finetune with slightly different export names still
    // works.
    if (session.inputNames.length === 0) {
      throw new Error("text encoder ONNX has no inputs");
    }
    const inputName = session.inputNames[0];

    let outputName = "last_hidden_state";
    if (!session.outputNames.includes(outputName)) {
      // Fall back to the first output. The hidden states tensor is always
      // the first output in every SD1.5 export I've seen.
      outputName = session.outputNames[0];
    }

    // Some exports take int64 ids, some take int32. Probe by trying to read
    // the input metadata; if it's missing, we default to int32 (the
    // tlwu/stable-diffusion-v1-5-onnxruntime export uses int32).
    const meta = (
      session as unknown as {
        inputMetadata?: Record<string, { type?: string }>;
      }
    ).inputMetadata;
    const inputType = meta?.[inputName]?.type;
    const inputIsInt64 = inputType === "int64";

    return new TextEncoder(session, inputName, outputName, inputIsInt64);
  }

  /** Encode a single tokenized prompt to a [77, 768] embedding. */
  async encode(prompt: TokenizedPrompt): Promise<TextEmbedding> {
    if (!this.session) {
      throw new Error("text encoder has been released");
    }
    if (prompt.ids.length !== CLIP_CONTEXT_LENGTH) {
      throw new Error(`expected ${CLIP_CONTEXT_LENGTH} input ids, got ${prompt.ids.length}`);
    }

    // Build the input tensor with shape [1, 77]. ORT-web wants either an
    // Int32Array or a BigInt64Array depending on the model's declared dtype.
    const dims = [1, CLIP_CONTEXT_LENGTH];
    let inputTensor: ort.Tensor;
    if (this.inputIsInt64) {
      const big = new BigInt64Array(CLIP_CONTEXT_LENGTH);
      for (let i = 0; i < CLIP_CONTEXT_LENGTH; i++) {
        big[i] = BigInt(prompt.ids[i]);
      }
      inputTensor = new ort.Tensor("int64", big, dims);
    } else {
      // Copy into a fresh Int32Array so ORT owns it cleanly. Reusing the
      // caller's buffer is fine in practice but makes lifetimes confusing.
      inputTensor = new ort.Tensor("int32", new Int32Array(prompt.ids), dims);
    }

    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: inputTensor,
    };
    const results = await this.session.run(feeds);
    const out = results[this.outputName];
    if (!out) {
      throw new Error(`text encoder did not produce output "${this.outputName}"`);
    }
    const data = out.data as Float32Array;
    const expectedLen = CLIP_CONTEXT_LENGTH * CLIP_HIDDEN_SIZE;
    if (data.length !== expectedLen) {
      throw new Error(`text encoder output has ${data.length} floats, expected ${expectedLen}`);
    }
    // Copy out so we can release the session without invalidating the
    // returned buffer (ORT-web's WebGPU EP can back tensor data with GPU
    // memory that goes away on release).
    return new Float32Array(data);
  }

  /** Release the ORT session and any GPU memory it holds. The instance
   *  becomes unusable; create a new one to encode again. */
  async release(): Promise<void> {
    if (!this.session) return;
    await this.session.release();
    this.session = null;
  }
}
