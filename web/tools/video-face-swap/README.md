# video-face-swap

Live at [intabai.dev/tools/video-face-swap/](https://intabai.dev/tools/video-face-swap/).

Swap a face from a source image into every frame of a video, entirely in your browser.

## Features

- Runs fully in-tab: no upload, no server, no install
- Models cached to OPFS after first download
- Swap models: HyperSwap 1a/1b/1c (256x256), inswapper_128 (fp16/fp32)
- Face detectors: SCRFD 500m (default), YOLOFace 8n + 2DFAN4, MediaPipe
  FaceLandmarker (desktop), YuNet
- Optional face enhancers: GFPGAN 1.4, RestoreFormer++, CodeFormer
- XSeg occlusion mask for paste-back
- WebGPU accelerated (ONNX Runtime Web), with WASM fallback
- GPU compute paste-back for mobile
- MP4 output via WebCodecs + mp4-muxer, original audio passthrough
- Range selector with single-frame preview before full encode
- Three execution modes (off / per-frame worker / full worker) for
  comparing main-thread vs Web Worker pipelines
- Mobile-friendly defaults, persisted settings

## TODO

- Measure Android backgrounded behavior in full worker mode (tab hidden,
  screen off, app switched)
- Temporal coherence: detect every Nth frame, reuse + smooth landmarks
- Extend GPU compute to warp / normalize / mask combine (currently only
  paste-back is on GPU)
- Custom OpenCV.js WASM build as an alternative to GPU compute
- Streaming SHA-256 for patched-model verify
- PRelu patching for ArcFace
- CLAHE contrast on 2DFAN4 input (low-light quality)
- Face angle estimation before 2DFAN4 (rotate crop to upright)
- Source embedding refinement via 2DFAN4 regardless of detector
- Estimated output file size and processing time
- Pause/resume for long video processing
- Side-by-side before/after preview
- Face picker for multi-face source/target
- Output format options (WebM)
- MediaSource API for live preview during encoding
- Dark mode
- Face-specific upscalers (tghq_face_x8, face_dat_x4)
- Wake lock to prevent screen sleep during long processing
- Detect actual video fps instead of hardcoding 30
- Reduce frame-to-frame jitter (temporal smoothing of landmarks/affines)
- Improve paste-back edge blending
