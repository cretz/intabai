# video-gen

Live at [intabai.dev/tools/video-gen/](https://intabai.dev/tools/video-gen/).

In-browser text-to-video generation. Model downloads once into your
browser (OPFS) and runs locally on WebGPU via onnxruntime-web. Nothing
leaves your tab.

## Models

- [FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers) - DMD-distilled Wan 2.2 TI2V-5B, 3 denoising steps, 480x832 @ 16fps, 81 frames (5 seconds). Sharded + quantized ONNX build hosted at `cretz/FastWan2.2-TI2V-5B-ONNX` (~6.5 GB total).

## Features

- Runs fully in-tab: no upload, no server, no install
- Model cached to OPFS after first download
- Sequential three-stage pipeline: UMT5-XXL text encoder, 30-layer DiT
  transformer, LightTAE VAE decoder
- Per-layer / per-block int4 quantization keeps peak GPU residency under
  the ~4 GB browser tab budget
- MP4 output via WebCodecs VideoEncoder + mp4-muxer
- Progressive VAE previews after step 1 and step 2 so you see the clip
  forming, not a blank screen for ten minutes
- Seeded deterministic generation, persisted settings, debug log
- ETA + per-stage progress

## Problems overcome

Video generation in a browser tab on stock onnxruntime-web is at the
absolute edge of what the stack supports. Most of the work was making
that edge hold.

### Memory and hosting

- **4 GB wasm32 linear-memory ceiling.** onnxruntime-web requests wasm
  initial memory equal to the `.onnx.data` sidecar size before handing
  tensors to the WebGPU EP. Any single external-data file over ~3.95 GB
  refuses to load. Solution: per-block and per-layer shards, loaded one
  at a time.
- **2.1 GB embedding table exceeds WebGPU `maxBufferSize` on mobile.**
  Solution: extract the UMT5 embedding as a raw binary, quantize it to
  per-row int8 (2.1 GB fp16 -> 1.05 GB int8 + 0.5 MB fp16 scales), and
  do the lookup in JS instead of ONNX.
- **Peak GPU residency during denoising.** Thirty transformer blocks
  cannot live on the GPU simultaneously. Solution: a double-buffer loop
  that compiles block N+1 while block N runs, so only two block
  sessions are resident at once (~184 MB q4 plus activations).

### The `maxBufferSize` cliff (the big one)

- attn1 in every block wants to materialize a Q.K^T intermediate of
  shape [1, 24 heads, 8190, 8190] fp16, which is 3.07 GiB: larger than
  the 2 GiB `maxBufferSize` every WebGPU device we tested reports. The
  allocation silently short-allocates, only ~128 of 25M output
  positions get written, and every downstream block sees garbage.
- Solution: rewrite attn1 at export time to split Q along the sequence
  axis into three chunks of 2730, run three `MatMul + Softmax + MatMul`
  pairs, and `Concat` the results. Same math, same FLOPs, fits.

### ORT-web kernel bugs we had to patch

- **MatMul packed kernel accumulates in fp16.** Produces 5-15% error
  vs PyTorch on the text encoder before patching, 1-5 ULP after.
  Patched to fp32 accumulation. Load-bearing: without this, the text
  encoder output is wrong enough to poison the whole pipeline.
- **Softmax kernel accumulates in fp16.** Same fix, fp32 accumulation.
- **MatMulNBits at `accuracy_level=4` produces NaN** in shell_pre's
  small q4 MatMuls, at every token position, on WebGPU only. CPU ORT
  at both fp16 and q4 was clean. Workaround: ship shell_pre as
  unquantized fp16 (180 MB) instead of q4 (52 MB). The full patch is
  applied via patch-package on `npm install`.

### Float16Array gotcha

- Modern Chrome (147+) exposes native `Float16Array`, and
  onnxruntime-web hands back fp16 tensors using it. The natural
  `new Uint16Array(tensor.data)` *numerically converts* fp16 to uint16
  (rounds, clamps, loses signs) instead of reinterpreting the bits,
  silently corrupting every fp16 readback. Solution: a `copyF16Bits`
  helper that handles the Float16Array path correctly. Now mandatory
  for every fp16 tensor read.

### Pipeline correctness

- **VAE latent denormalization was missing.** Wan 2.2's VAE expects
  `x = x * latents_std[c] + latents_mean[c]` per channel before decode.
  Without it the decoder sees normalized-space tensors it was never
  trained on. Applied in both preview and final decode paths.
- **Flow-matching sigma was double-shifted.** The obvious formula
  `sigma = flow_shift * s / (1 + (shift-1)*s)` applies the shift a
  second time because the FastVideo scheduler has already baked it
  into the timesteps. Correct formula: `sigma = t / 1000`.
- **UniPC sampler produced bland pastel output** with DMD-distilled
  weights (the model expects direct x0 prediction, not higher-order
  solver steps). Replaced with the inlined DMD loop:
  `x0 = latent - sigma_t * noise_pred`, re-noise between steps.
- **Timesteps for TI2V-5B are `[1000, 757, 522]`**, not the A14B
  variant's `[1000, 750, 500, 250]`. Wrong constants also produced
  pastel output. Source of truth is in FastVideo's pipeline configs.
- **Timestep is a 2D `[B, 8190]` tensor**, not a scalar: every token
  gets its own timestep value for Wan 2.2 TI2V.

### Export-time landmines

- **opset 23 `aten::rms_norm` unsupported by ONNX exporter.** Monkey-
  patched RMSNorm to the decomposed form before tracing.
- **UMT5 LayerNorm overflowed fp16 variance** during export traces on
  long sequences. Patched to a fp32 decomposition for the variance
  reduction, cast back to fp16.
- **Accelerate's disk-offload hooks trigger an `aten::view(Tensor, int)`
  tracer assert** during ONNX export. Switched to per-block
  instantiation, peak export RAM went from "won't fit" to ~2 GB.
- **`torch.onnx.export(dynamo=True)` hangs on Windows** due to Unicode
  handling in the path resolver. Used `dynamo=False`.
- **`pixel_shuffle` operates on 4D, not 5D** tensors, despite the VAE
  using 5D throughout. Reshape around the call-site.

### Browser / dev-server quirks

- **Vite's `/@vite/client` HMR websocket still runs with
  `server.hmr: false`**, fails to connect, and leaves `this.ws`
  undefined. Its `sendError` then throws TypeErrors that swallow every
  real page error. Fix: inline HMR ws-stub at the top of `<head>` in
  tool pages.
- **Vite dev proxy `async fs import` inside middleware loses a race**
  with the SPA fallback. Fix: top-level `createReadStream` and
  `statSync`.
- **Clearing `node_modules/.vite`** is required after patching
  `node_modules/onnxruntime-web`. Vite's dep-optimizer cache survives
  dev server restarts.

### WebGPU device lost / hung (flaky, driver-side)

- Long runs occasionally fail mid-generation with
  `AbortError: Failed to execute 'mapAsync' on 'GPUBuffer': [Device]
  is lost` or `DXGI_ERROR_DEVICE_HUNG`. With 4-step UniPC at 30 blocks
  per step you churn through 120 WebGPU session create/release cycles
  per run; cumulative state pressure eventually trips the driver or
  the Windows TDR watchdog. Not a code bug, and retrying usually
  succeeds.
- Mitigations, in order:
  1. Close other GPU-using tabs and apps, restart the browser, retry.
  2. Bump the Windows TDR timeout. `regedit` →
     `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`, add
     DWORD `TdrDelay` = `10` (seconds; default 2 is aggressive for
     the long attention dispatches we do). Reboot after.
  3. If recurrent, reduce WebGPU session churn by keeping more blocks
     resident (see "perf win inventory" in the worklog).

### The PCIe AER rabbit hole (not our bug)

- On one development laptop, ten-minute generations started BSODing
  the machine with a WHEA fatal. Turned out to be PCIe AER corrected
  errors on the dGPU link escalating to uncorrected under sustained
  PCIe traffic (not a browser or code bug). Browsers are supposed to
  make this physically impossible. Mentioned here in case anyone
  else sees the same pattern on Arrow Lake + Blackwell laptop
  hardware: check Event Viewer for WHEA-Logger Event 17.
- What fixed it on the affected machine: **disable PCIe ASPM**
  (Control Panel → Power Options → Change plan settings → Change
  advanced power settings → PCI Express → Link State Power Management
  → Off, both On battery and Plugged in). That stops the L0s/L1
  power-state transitions that are the #1 source of AER corrected
  errors on the link.

## TODO

- Upload the quantized shards to `cretz/FastWan2.2-TI2V-5B-ONNX` and
  switch `FASTWAN_BASE` off the local dev proxy
- Mobile run: verify block residency + `maxBufferSize` on flagship
  Android Chrome (desktop confirmed)
- Perf: io-binding on block sessions and GPU-resident constants
  (estimated 25-35% combined win, not shipping-critical)
- First-frame I2V: the transformer supports it natively via latent
  injection; UI doesn't expose it yet
- Custom resolution / frame count (currently locked to 480x832, 81
  frames)
- Wake lock during long generation
- Intermediate frame download (save latent snapshot between steps)
- Hotkey to cancel mid-generation and free the GPU
