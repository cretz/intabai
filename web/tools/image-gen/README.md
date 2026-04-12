# image-gen

Live at [intabai.dev/tools/image-gen/](https://intabai.dev/tools/image-gen/).

In-browser text-to-image and image-to-image generation. Models download
once into your browser (OPFS) and run locally on WebGPU via onnxruntime-web.
Nothing leaves your tab.

## Models

All run on desktop and mobile (flagship Android Chrome) via
onnxruntime-web + WebGPU.

- [webnn/Z-Image-Turbo](https://huggingface.co/webnn/Z-Image-Turbo) - S3-DiT 6B, q4f16. ~5.4 GB. Also available as a [sharded variant](https://huggingface.co/cretz/Z-Image-Turbo-ONNX-sharded) for mobile.
- [onnx-community/Janus-Pro-1B-ONNX](https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX) - Janus-Pro 1B, autoregressive multimodal. ~1.99 GB.
- [webnn/sdxl-turbo](https://huggingface.co/webnn/sdxl-turbo) - q4f16 SDXL-Turbo, 1-step generation. ~2.67 GB.
- [gfodor/segmind-vega-fp16-onnx](https://huggingface.co/gfodor/segmind-vega-fp16-onnx) - Distilled SDXL. ~3.22 GB.
- [nmkd/stable-diffusion-1.5-onnx-fp16](https://huggingface.co/nmkd/stable-diffusion-1.5-onnx-fp16) - Stable Diffusion 1.5. ~2.13 GB.
