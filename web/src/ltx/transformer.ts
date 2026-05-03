// LTX 2B distilled transformer, run as 1 shell_pre + 28 per-block sessions
// + 1 shell_post per denoising step.
//
// shell_pre: patchify_proj + RoPE + adaln_single + caption_projection +
//   encoder_attention_mask -> bias. Loaded once, run once per step (timestep
//   feeds in here). Outputs a bundle of conditioning tensors that the
//   blocks reuse.
// blocks (28 of them): each loaded, run, released before the next block
//   loads, so peak GPU = 1 block (~38 MB q4f16) + the shared conditioning
//   tensors. hidden_states is the only ping-pong buffer.
// shell_post: scale_shift + norm_out + proj_out -> noise_pred (B, N, 128).
//
// io-binding is intentionally skipped here while the pipeline is being
// validated end-to-end; CPU round-trip per block costs a few MB per
// transfer and is fine for first-pass correctness. Promote to the fastwan
// io-binding pattern if profiling shows it dominates.

import * as ort from "onnxruntime-web";

import type { ModelCache } from "../shared/model-cache";
import { createSession, type OrtModelFile } from "../sd15/ort-helpers";
import { copyF16Bits } from "../sd15/fp16";
import { LTX_TX_NUM_BLOCKS } from "./models";
import {
  assertArrayType,
  assertF16View,
  assertLength,
  assertOrtOutput,
} from "./validate";

/** Internal hidden size. */
export const LTX_TX_HIDDEN = 2048;
/** Patch-space input/output channels. */
export const LTX_TX_PATCH_CHANNELS = 128;

export interface LtxTxFiles {
  shellPre: OrtModelFile;
  blocks: OrtModelFile[];
  shellPost: OrtModelFile;
}

export interface LtxTxBlockTiming {
  blockIndex: number;
  totalBlocks: number;
  loadMs: number;
  runMs: number;
  releaseMs: number;
}

export type LtxTxBlockProgress = (info: LtxTxBlockTiming) => void;

/** Optional per-component tap for op-level bisection. Called after
 *  shell_pre, every block, and shell_post with the fp16 output buffer.
 *  Labels: "shell_pre.hidden", "block_00".."block_27", "shell_post.noise_pred".
 *  Caller owns the buffer; do not retain past the callback. */
export type LtxTxTap = (label: string, data: Uint16Array) => void;

export interface LtxTxStepInputs {
  /** [1, N_tokens, 128] fp16 (already patchified by JS). */
  hiddenStates: Uint16Array;
  /** [1, 3, N_tokens] fp32 RoPE indices. */
  indicesGrid: Float32Array;
  /** [1, N_text, 4096] fp16 (T5 output). */
  encoderHiddenStates: Uint16Array;
  /** [1, N_text] int64 attention mask (1 = real token, 0 = pad). */
  encoderAttentionMask: BigInt64Array;
  /** [1] fp16 current sigma/timestep. */
  timestep: Uint16Array;
  /** Number of patch tokens. */
  nTokens: number;
  /** Length of the text sequence. */
  nText: number;
}

export interface LtxTxStepResult {
  /** [1, N_tokens, 128] fp16 noise prediction. */
  noisePred: Uint16Array;
}

export class LtxTransformer {
  private readonly shellProviders: ("webgpu" | "wasm")[];
  constructor(
    private readonly cache: ModelCache,
    private readonly files: LtxTxFiles,
    private readonly providers: ("webgpu" | "wasm")[] = ["webgpu", "wasm"],
    shellPreProviders?: ("webgpu" | "wasm")[],
  ) {
    this.shellProviders = shellPreProviders ?? providers;
    if (files.blocks.length !== LTX_TX_NUM_BLOCKS) {
      throw new Error(
        `expected ${LTX_TX_NUM_BLOCKS} transformer blocks, got ${files.blocks.length}`,
      );
    }
  }

  /** Run one denoising step (shell_pre + 28 blocks + shell_post). */
  async runStep(
    inputs: LtxTxStepInputs,
    onBlock?: LtxTxBlockProgress,
    tap?: LtxTxTap,
  ): Promise<LtxTxStepResult> {
    const { nTokens, nText } = inputs;

    // Boundary checks: typed-array ctors and lengths must match what we
    // declare to ORT below. If any of these are wrong, ORT either errors
    // or (worse) silently misinterprets bytes - the EPs disagree on which.
    assertArrayType("runStep.hiddenStates", inputs.hiddenStates, Uint16Array);
    assertLength("runStep.hiddenStates", inputs.hiddenStates, nTokens * LTX_TX_PATCH_CHANNELS);
    assertArrayType("runStep.indicesGrid", inputs.indicesGrid, Float32Array);
    assertLength("runStep.indicesGrid", inputs.indicesGrid, 3 * nTokens);
    assertArrayType("runStep.encoderHiddenStates", inputs.encoderHiddenStates, Uint16Array);
    assertLength("runStep.encoderHiddenStates", inputs.encoderHiddenStates, nText * 4096);
    assertArrayType("runStep.encoderAttentionMask", inputs.encoderAttentionMask, BigInt64Array);
    assertLength("runStep.encoderAttentionMask", inputs.encoderAttentionMask, nText);
    assertArrayType("runStep.timestep", inputs.timestep, Uint16Array);
    assertLength("runStep.timestep", inputs.timestep, 1);

    // ---- shell_pre ----
    if (this.shellProviders !== this.providers) {
      console.log("[ltx-diag] shell_pre providers override:", this.shellProviders);
    }
    const shellPreSession = await createSession(
      this.cache,
      this.files.shellPre,
      this.shellProviders,
    );

    let hidden: Uint16Array;
    let freqsCos: Uint16Array;
    let freqsSin: Uint16Array;
    let timestepMod: Uint16Array;
    let embeddedTimestep: Uint16Array;
    let encoderProj: Uint16Array;
    let encoderAttnBias: Uint16Array;
    let timestepModLastDim: number;
    let embeddedTimestepLastDim: number;

    {
      const feeds: Record<string, ort.Tensor> = {
        hidden_states: new ort.Tensor("float16", inputs.hiddenStates, [1, nTokens, LTX_TX_PATCH_CHANNELS]),
        indices_grid: new ort.Tensor("float32", inputs.indicesGrid, [1, 3, nTokens]),
        encoder_hidden_states: new ort.Tensor("float16", inputs.encoderHiddenStates, [1, nText, 4096]),
        encoder_attention_mask: new ort.Tensor("int64", inputs.encoderAttentionMask, [1, nText]),
        timestep: new ort.Tensor("float16", inputs.timestep, [1]),
      };
      try {
        const r = await shellPreSession.run(feeds);
        // Every shell_pre output is fp16 in our export. If any EP returns
        // an output as fp32 (e.g. graph type inference promoted RoPE freqs)
        // copyF16Bits would reinterpret 4-byte fp32 as two 2-byte fp16s and
        // produce numerical garbage - a candidate explanation for the green-
        // tile webgpu output. assertF16View enforces strict ctor + 2-byte
        // width per element.
        assertOrtOutput("shell_pre.hidden_proj", r.hidden_proj as ort.Tensor, {
          type: "float16",
          dims: [1, nTokens, LTX_TX_HIDDEN],
        });
        assertOrtOutput("shell_pre.freqs_cos", r.freqs_cos as ort.Tensor, {
          type: "float16",
          dims: [1, nTokens, LTX_TX_HIDDEN],
        });
        assertOrtOutput("shell_pre.freqs_sin", r.freqs_sin as ort.Tensor, {
          type: "float16",
          dims: [1, nTokens, LTX_TX_HIDDEN],
        });
        assertOrtOutput("shell_pre.timestep_mod", r.timestep_mod as ort.Tensor, {
          type: "float16",
        });
        assertOrtOutput("shell_pre.embedded_timestep", r.embedded_timestep as ort.Tensor, {
          type: "float16",
        });
        assertOrtOutput("shell_pre.encoder_proj", r.encoder_proj as ort.Tensor, {
          type: "float16",
          dims: [1, nText, LTX_TX_HIDDEN],
        });
        assertOrtOutput("shell_pre.encoder_attn_bias", r.encoder_attn_bias as ort.Tensor, {
          type: "float16",
        });
        const tsModLast = (r.timestep_mod.dims as number[])[2];
        const embTsLast = (r.embedded_timestep.dims as number[])[2];
        const encBiasLast = (r.encoder_attn_bias.dims as number[])[2];
        const hiddenView = assertF16View("shell_pre.hidden_proj.data", r.hidden_proj.data, nTokens * LTX_TX_HIDDEN);
        const freqsCosView = assertF16View("shell_pre.freqs_cos.data", r.freqs_cos.data, nTokens * LTX_TX_HIDDEN);
        const freqsSinView = assertF16View("shell_pre.freqs_sin.data", r.freqs_sin.data, nTokens * LTX_TX_HIDDEN);
        const tsModView = assertF16View("shell_pre.timestep_mod.data", r.timestep_mod.data, tsModLast);
        const embTsView = assertF16View("shell_pre.embedded_timestep.data", r.embedded_timestep.data, embTsLast);
        const encProjView = assertF16View("shell_pre.encoder_proj.data", r.encoder_proj.data, nText * LTX_TX_HIDDEN);
        const encBiasView = assertF16View("shell_pre.encoder_attn_bias.data", r.encoder_attn_bias.data, encBiasLast);
        hidden = copyF16Bits(hiddenView);
        freqsCos = copyF16Bits(freqsCosView);
        freqsSin = copyF16Bits(freqsSinView);
        timestepMod = copyF16Bits(tsModView);
        embeddedTimestep = copyF16Bits(embTsView);
        encoderProj = copyF16Bits(encProjView);
        encoderAttnBias = copyF16Bits(encBiasView);
        timestepModLastDim = tsModLast;
        embeddedTimestepLastDim = embTsLast;
        for (const k in r) (r[k] as ort.Tensor).dispose?.();
      } finally {
        for (const k in feeds) feeds[k].dispose?.();
        await shellPreSession.release();
      }
    }
    tap?.("shell_pre.hidden", hidden);
    tap?.("shell_pre.freqs_cos", freqsCos);
    tap?.("shell_pre.freqs_sin", freqsSin);
    tap?.("shell_pre.timestep_mod", timestepMod);
    tap?.("shell_pre.encoder_proj", encoderProj);
    tap?.("shell_pre.encoder_attn_bias", encoderAttnBias);

    // ---- 28 blocks ----
    const hiddenDims = [1, nTokens, LTX_TX_HIDDEN];
    const freqsDims = [1, nTokens, LTX_TX_HIDDEN];
    const tsModDims = [1, 1, timestepModLastDim];
    const encProjDims = [1, nText, LTX_TX_HIDDEN];
    const encBiasDims = [1, 1, nText];

    for (let i = 0; i < LTX_TX_NUM_BLOCKS; i++) {
      const tLoad = performance.now();
      if (i === 0) console.log("[ltx-diag] block session opts: graphOpt=disabled, providers=", this.providers);
      const session = await createSession(this.cache, this.files.blocks[i], this.providers, {
        graphOptimizationLevel: "disabled",
      });
      const loadMs = performance.now() - tLoad;

      const feeds: Record<string, ort.Tensor> = {
        hidden_states: new ort.Tensor("float16", hidden, hiddenDims),
        freqs_cos: new ort.Tensor("float16", freqsCos, freqsDims),
        freqs_sin: new ort.Tensor("float16", freqsSin, freqsDims),
        timestep_mod: new ort.Tensor("float16", timestepMod, tsModDims),
        encoder_proj: new ort.Tensor("float16", encoderProj, encProjDims),
        encoder_attn_bias: new ort.Tensor("float16", encoderAttnBias, encBiasDims),
      };

      let runMs = 0;
      let releaseMs = 0;
      try {
        const tRun = performance.now();
        const r = await session.run(feeds);
        runMs = performance.now() - tRun;
        const out = r.hidden_states_out ?? r[Object.keys(r)[0]];
        assertOrtOutput(`block_${i}.hidden_states_out`, out, {
          type: "float16",
          dims: [1, nTokens, LTX_TX_HIDDEN],
        });
        const view = assertF16View(`block_${i}.hidden_states_out.data`, out.data, nTokens * LTX_TX_HIDDEN);
        hidden = copyF16Bits(view);
        for (const k in r) (r[k] as ort.Tensor).dispose?.();
      } finally {
        for (const k in feeds) feeds[k].dispose?.();
        const tRel = performance.now();
        await session.release();
        releaseMs = performance.now() - tRel;
      }

      onBlock?.({ blockIndex: i, totalBlocks: LTX_TX_NUM_BLOCKS, loadMs, runMs, releaseMs });
      tap?.(`block_${String(i).padStart(2, "0")}`, hidden);
    }

    // ---- shell_post ----
    const shellPostSession = await createSession(
      this.cache,
      this.files.shellPost,
      this.providers,
    );
    let noisePred: Uint16Array;
    {
      const feeds: Record<string, ort.Tensor> = {
        hidden_states: new ort.Tensor("float16", hidden, hiddenDims),
        embedded_timestep: new ort.Tensor("float16", embeddedTimestep, [1, 1, embeddedTimestepLastDim]),
      };
      try {
        const r = await shellPostSession.run(feeds);
        const out = r.noise_pred ?? r[Object.keys(r)[0]];
        assertOrtOutput("shell_post.noise_pred", out, {
          type: "float16",
          dims: [1, nTokens, LTX_TX_PATCH_CHANNELS],
        });
        const view = assertF16View("shell_post.noise_pred.data", out.data, nTokens * LTX_TX_PATCH_CHANNELS);
        noisePred = copyF16Bits(view);
        for (const k in r) (r[k] as ort.Tensor).dispose?.();
      } finally {
        for (const k in feeds) feeds[k].dispose?.();
        await shellPostSession.release();
      }
    }
    tap?.("shell_post.noise_pred", noisePred);

    return { noisePred };
  }
}
