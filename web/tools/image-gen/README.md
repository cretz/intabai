# image-gen

In-browser text-to-image and image-to-image generation. Models download
once into your browser (OPFS) and run locally on WebGPU via onnxruntime-web.
Nothing leaves your tab.

## Shipped models

All four run on stock onnxruntime-web 1.24.3 + WebGPU and verified
end-to-end on desktop and mobile.

- [onnx-community/Janus-Pro-1B-ONNX](https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX)
  - Janus-Pro 1B, autoregressive multimodal LM. ~1.99 GB.
  - **Best model on both desktop and mobile.** Smallest total bundle,
    legitimately impressive quality for 1B params, and the only non-diffusion
    architecture in the lineup. Runs via @huggingface/transformers v4
    (dynamic-imported so SD1.5/SDXL users never pay the 432 KB bundled-ORT
    cost). ~30 s per image on Intel Xe-LPG iGPU at 384x384.
  - Mobile: the 698 MB monolithic language_model is split into graph +
    external-data sidecar during download via the ONNX external-data split
    infrastructure, verified working on flagship Android Chrome.
- [webnn/sdxl-turbo](https://huggingface.co/webnn/sdxl-turbo)
  - Microsoft WebNN team's q4f16 SDXL-Turbo. ~2.67 GB.
  - **Fastest model on desktop.** 1-step generation with no CFG needed
    (Euler scheduler, sigma=14.6146). ~6.6 s per image on Intel Xe-LPG
    iGPU at 512x512 - 14x faster than SD1.5.
  - Mobile: the 1.97 GB monolithic UNet is split into graph (~493 MB) +
    external-data sidecar (~1.47 GB) during download via the same ONNX
    split infrastructure as Janus. Verified working on flagship Android.
  - Tokenizer files shared with Segmind Vega (same CLIP BPE vocab).
- [gfodor/segmind-vega-fp16-onnx](https://huggingface.co/gfodor/segmind-vega-fp16-onnx)
  - Distilled SDXL (~0.74 B params). ~3.22 GB.
  - Sharper fine detail than SD1.5, better prompt adherence, native 1024x1024
    (we ship 768x768 desktop / 512x512 mobile to fit the WebGPU memory budget).
  - Confirmed working on flagship Android Chrome at 512x512 / 20 steps,
    ~11 s per UNet step (~3.7 min per generation).
- [nmkd/stable-diffusion-1.5-onnx-fp16](https://huggingface.co/nmkd/stable-diffusion-1.5-onnx-fp16)
  - Stable Diffusion 1.5, the universal fallback. ~2.13 GB.
  - Native 512x512. Wide topic coverage thanks to the huge SD1.5 finetune
    ecosystem the base model was trained against.

## Verified-runnable but not yet wired into the UI

- [lemonteaa/sdxs-onnx](https://huggingface.co/lemonteaa/sdxs-onnx)

Verified to load and run on desktop only (mobile blocked on GPU memory):

- [webnn/Z-Image-Turbo](https://huggingface.co/webnn/Z-Image-Turbo)
- [TensorStack/Nitro-E-onnx](https://huggingface.co/TensorStack/Nitro-E-onnx)

## Prompt tips

SD/SDXL models respond hugely to comma-separated keyword stuffing. A
conversational prompt like "a woman in a jungle" will give you a tepid
generic photo; the same scene loaded with the conventions below will give
you something dramatic and rendered. Pattern (5-8 of these in comma-
separated order works well):

1. **Subject + action + setting** - "wilderness woman hunting in jungle hiding behind leaves"
2. **Composition** - "closeup face portrait", "wide shot", "rule of thirds"
3. **Lighting** - "backlight", "golden hour", "rim light", "dramatic chiaroscuro"
4. **Medium / style** - "oil painting", "nature documentary", "35mm film", "anime", "octane render"
5. **Detail callouts** - "detailed eyes", "intricate jewelry", "fuzzy skin texture", "freckles"
6. **Lens / camera** - "85mm", "shallow depth of field", "lens flare", "bokeh"

Example that produces a dramatic result on Vega:

> backlight, wilderness woman hunting in jungle hiding behind leaves, face
> paintings closeup face portrait, detailed eyes, nature documentary, dry
> skin, fuzzy skin, lens flare

Tuning knobs:

- **CFG 7-9** is the SDXL sweet spot. Higher = stronger prompt adherence
  at the cost of weirdness.
- **Steps 20-30** for Vega. Diminishing returns above 30.
- **Resolution**: stay close to the model default. SDXL distortions
  multiply fast above 1024 and below 512.

## Reference image (img2img)

Optional. SD1.5, Vega, and SDXL-Turbo support strength-based init-image: encodes
the reference into latent space, partially re-noises it, lets the prompt
restyle the result. This **preserves layout, composition, and colors** -
it does not change subject, pose, or clothing. For real edits (clothing
swap, object removal, "fix the broken hand") you need inpainting, which
is on the roadmap.

Strength tuning: 0.5-0.6 for "same scene, different style", 0.75-0.85 for
"loose interpretation of the reference."
