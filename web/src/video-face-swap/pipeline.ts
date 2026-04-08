import * as ort from "onnxruntime-web";
import {
  estimateSimilarityTransform,
  warpAffine,
  pasteBack,
  rgbaToBgrFloat32,
  resizeImageData,
  cropImageData,
  gaussianBlur1f,
  createFeatheredMask,
} from "./cv";
import { GpuPasteBack } from "./gpu-paste";
import {
  findEnhancer,
  type DetectorId,
  type ModelSet,
  MEDIAPIPE_FACE_LANDMARKER,
  YUNET_FACE_DETECTOR,
  SCRFD_500M,
  allFiles,
} from "./models";
import { ModelCache } from "../shared/model-cache";
import { MediaPipeDetector } from "./mediapipe-detector";

// Warp templates from FaceFusion (normalized coordinates for 5-point alignment)
const TEMPLATE_ARCFACE_128 = [
  [0.36167656, 0.40387734],
  [0.63696719, 0.40235469],
  [0.50019687, 0.56044219],
  [0.38710391, 0.72160547],
  [0.61507734, 0.72034453],
];

const TEMPLATE_ARCFACE_112_V2 = [
  [0.34191607, 0.46157411],
  [0.65653393, 0.45983393],
  [0.500225, 0.64050536],
  [0.37097589, 0.82469196],
  [0.63151696, 0.82325089],
];

const TEMPLATE_FFHQ_512 = [
  [0.37691676, 0.46864664],
  [0.62285697, 0.46912813],
  [0.50123859, 0.61331904],
  [0.39308822, 0.725411],
  [0.61150205, 0.72490465],
];

interface Face {
  bbox: number[];
  score: number;
  landmark5: number[][];
}

export interface FrameTimings {
  detect: number;
  landmarks: number;
  swap: number;
  xseg: number;
  paste: number;
  enhance: number;
}

export interface FrameStats {
  frameIndex: number;
  totalFrames: number;
  fps: number;
  etaSeconds: number;
  timings: FrameTimings;
}

type SwapKind = "hyperswap" | "inswapper";

function classifySwapModel(id: string): SwapKind {
  return id.startsWith("inswapper") ? "inswapper" : "hyperswap";
}

/**
 * Print a marker line into the console so the WebGPU profiler's per-kernel
 * lines that follow can be attributed to the correct session/step. No-op
 * unless WebGPU profiling is currently enabled, so it costs nothing during
 * normal preview / full-video processing.
 */
function profileMark(label: string): void {
  const mode = (ort.env.webgpu as { profiling?: { mode?: string } }).profiling?.mode;
  if (mode && mode !== "off") {
    console.log(`[profile-mark] ${label}`);
  }
}

export class Pipeline {
  private sessions = new Map<string, ort.InferenceSession>();
  private cache: ModelCache;
  private aborted = false;
  private swapSessionId: string | null = null;
  private swapKind: SwapKind = "hyperswap";
  private inswapperEmap: Float32Array | null = null;
  private enhancerSessionId: string | null = null;
  private detectorId: DetectorId = "yoloface";
  private mediapipe: MediaPipeDetector | null = null;
  private gpuPaste: GpuPasteBack | null = null;
  private gpuDevice: GPUDevice | null = null;
  private gpuDevicePromise: Promise<GPUDevice> | null = null;
  /**
   * Resolved YuNet output names per (kind, stride). 4 kinds (cls/obj/bbox/kps)
   * x 3 strides (8/16/32) = 12 names. Computed once on first detect call.
   */
  private yunetOutputNames: string[] | null = null;

  private async ensureGpuDevice(): Promise<GPUDevice> {
    if (this.gpuDevice) return this.gpuDevice;
    if (!this.gpuDevicePromise) {
      this.gpuDevicePromise = GpuPasteBack.requestDevice();
    }
    this.gpuDevice = await this.gpuDevicePromise;
    return this.gpuDevice;
  }

  constructor(private readonly useGpuPaste: boolean = false) {
    this.cache = new ModelCache({
      opfsDirName: "intabai-video-face-swap",
      legacyDirNames: ["intabai-models"],
    });
  }

  private async pasteBackImpl(
    frame: ImageData,
    crop: ImageData,
    affineMatrix: number[],
    occlusionMask?: Float32Array,
  ): Promise<ImageData> {
    if (!this.useGpuPaste || !GpuPasteBack.isSupported()) {
      return pasteBack(frame, crop, affineMatrix, occlusionMask);
    }
    if (!this.gpuPaste) {
      const device = await this.ensureGpuDevice();
      this.gpuPaste = new GpuPasteBack(device);
    }
    // GPU shader takes one combined mask. Match the JS pasteBack semantics:
    // box-feathered mask, optionally min'd with the occlusion mask.
    const cw = crop.width;
    const ch = crop.height;
    const box = createFeatheredMask(cw, ch, 15, 15);
    let combined: Float32Array;
    if (occlusionMask) {
      combined = new Float32Array(cw * ch);
      for (let i = 0; i < combined.length; i++) {
        combined[i] = Math.min(occlusionMask[i], box[i]);
      }
    } else {
      combined = box;
    }
    try {
      return await this.gpuPaste.apply(frame, crop, affineMatrix, combined);
    } catch (err) {
      console.warn("[gpu-paste] failed, falling back to JS:", err);
      return pasteBack(frame, crop, affineMatrix, occlusionMask);
    }
  }

  abort(): void {
    this.aborted = true;
  }

  resetAbort(): void {
    this.aborted = false;
  }

  get isAborted(): boolean {
    return this.aborted;
  }

  async loadModels(
    set: ModelSet,
    enhancerId: string | null,
    detectorId: DetectorId,
  ): Promise<void> {
    const providers: string[] = [];
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      providers.push("webgpu");
    }
    providers.push("wasm");
    console.debug(`execution providers: ${providers.join(", ")}`);

    this.swapSessionId = set.primary.id;
    this.swapKind = classifySwapModel(set.primary.id);
    this.enhancerSessionId = null;
    this.detectorId = detectorId;
    this.inswapperEmap = null;

    const files = [...allFiles(set)];
    if (enhancerId) {
      const enhancer = findEnhancer(enhancerId);
      if (!enhancer) throw new Error(`unknown enhancer: ${enhancerId}`);
      files.push(enhancer);
      this.enhancerSessionId = enhancer.id;
    }

    for (const file of files) {
      // The MediaPipe .task file is bundled with the swap set on disk but is
      // not an ONNX model. Skip it for the ORT loader; it'll be loaded by
      // the MediaPipeDetector below if selected.
      if (file.id === MEDIAPIPE_FACE_LANDMARKER.id) continue;
      // The inswapper emap is a raw float32 matrix, not ONNX. Loaded below.
      if (file.id === "inswapper_emap") continue;
      // Alternate detectors are loaded only when selected.
      if (file.id === YUNET_FACE_DETECTOR.id && detectorId !== "yunet") continue;
      if (file.id === SCRFD_500M.id && detectorId !== "scrfd_500m") continue;
      // If the user picked any single-stage detector (MediaPipe / YuNet /
      // SCRFD), skip loading the now-unused YOLOFace and 2DFAN4 ONNX
      // sessions to save time and memory.
      const usingSingleStage =
        detectorId === "mediapipe" || detectorId === "yunet" || detectorId === "scrfd_500m";
      if (usingSingleStage && (file.id === "yoloface_8n" || file.id === "2dfan4")) {
        continue;
      }

      console.debug(`loading ${file.name}...`);
      const buffer = await this.cache.loadFile(file);
      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      this.sessions.set(file.id, session);
      console.debug(`loaded ${file.name}`);
    }

    if (detectorId === "mediapipe") {
      console.debug("loading MediaPipe FaceLandmarker...");
      const buffer = await this.cache.loadFile(MEDIAPIPE_FACE_LANDMARKER);
      this.mediapipe = new MediaPipeDetector();
      await this.mediapipe.load(new Uint8Array(buffer));
      console.debug("loaded MediaPipe FaceLandmarker");
    }

    if (this.swapKind === "inswapper") {
      console.debug("loading inswapper emap...");
      const emapFile = files.find((f) => f.id === "inswapper_emap");
      if (!emapFile) throw new Error("inswapper emap missing from set deps");
      const buffer = await this.cache.loadFile(emapFile);
      this.inswapperEmap = new Float32Array(buffer);
      if (this.inswapperEmap.length !== 512 * 512) {
        throw new Error(`unexpected inswapper emap size: ${this.inswapperEmap.length} floats`);
      }
      console.debug("loaded inswapper emap (512x512)");
    }
  }

  async releaseModels(): Promise<void> {
    for (const [id, session] of this.sessions) {
      await session.release();
      console.debug(`released ${id}`);
    }
    this.sessions.clear();
    if (this.mediapipe) {
      await this.mediapipe.release();
      this.mediapipe = null;
    }
    this.yunetOutputNames = null;
    if (this.gpuPaste) {
      this.gpuPaste.destroy();
      this.gpuPaste = null;
    }
    if (this.gpuDevice) {
      this.gpuDevice.destroy();
      this.gpuDevice = null;
    }
    this.gpuDevicePromise = null;
  }

  /** Extract source face embedding from ImageData */
  async extractSourceEmbeddingFromImageData(imageData: ImageData): Promise<Float32Array> {
    const sourceFaces = await this.detectFaces(imageData);
    if (sourceFaces.length === 0) {
      throw new Error("no face detected in source image");
    }
    console.debug(`source: detected ${sourceFaces.length} face(s), using first`);
    const embedding = await this.getEmbedding(imageData, sourceFaces[0]);
    if (this.swapKind === "inswapper") {
      return this.applyInswapperEmap(embedding);
    }
    return embedding;
  }

  /**
   * inswapper expects its source input to be the ArcFace embedding mapped
   * through a 512x512 matrix and re-normalized:
   *     mapped = embedding @ emap   (matrix multiply)
   *     source_input = mapped / norm(mapped)
   * This is mathematically equivalent whether the input embedding is raw or
   * already L2-normalized, so we feed the same normalized embedding our
   * existing getEmbedding produces.
   */
  private applyInswapperEmap(embedding: Float32Array): Float32Array {
    const emap = this.inswapperEmap;
    if (!emap) throw new Error("inswapper emap not loaded");
    const mapped = new Float32Array(512);
    // mapped[j] = sum_i embedding[i] * emap[i, j]    (row-major emap)
    for (let i = 0; i < 512; i++) {
      const e = embedding[i];
      const rowOffset = i * 512;
      for (let j = 0; j < 512; j++) {
        mapped[j] += e * emap[rowOffset + j];
      }
    }
    let norm = 0;
    for (let i = 0; i < 512; i++) norm += mapped[i] * mapped[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < 512; i++) mapped[i] /= norm;
    return mapped;
  }

  /** Process a single frame - public for preview */
  async swapFrame(
    frame: ImageData,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ): Promise<ImageData> {
    const { result } = await this.processFrame(frame, sourceEmbedding, useXseg);
    return result;
  }

  async processFrame(
    frame: ImageData,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ): Promise<{ result: ImageData; timings: FrameTimings }> {
    const timings: FrameTimings = {
      detect: 0,
      landmarks: 0,
      swap: 0,
      xseg: 0,
      paste: 0,
      enhance: 0,
    };

    let t = performance.now();
    const faces = await this.detectFaces(frame);
    timings.detect += performance.now() - t;
    if (faces.length === 0) return { result: frame, timings };

    let result = frame;
    for (const face of faces) {
      const r = await this.swapFace(result, face, sourceEmbedding, useXseg);
      result = r.result;
      timings.landmarks += r.landmarks;
      timings.swap += r.swap;
      timings.xseg += r.xseg;
      timings.paste += r.paste;
    }

    if (this.enhancerSessionId) {
      t = performance.now();
      const enhancedFaces = await this.detectFaces(result);
      timings.detect += performance.now() - t;
      for (const face of enhancedFaces) {
        const r = await this.enhanceFace(result, face);
        result = r.result;
        timings.landmarks += r.landmarks;
        timings.enhance += r.enhance;
        timings.paste += r.paste;
      }
    }

    return { result, timings };
  }

  // --- Face Detection (dispatches to YOLOFace, MediaPipe, or YuNet) ---

  private async detectFaces(imageData: ImageData): Promise<Face[]> {
    if (this.detectorId === "mediapipe") {
      const mp = this.mediapipe;
      if (!mp) throw new Error("mediapipe detector not loaded");
      // MediaPipe returns landmarks already in pixel space - no need for a
      // separate refinement pass via 2DFAN4 later.
      return mp.detect(imageData).map((f) => ({
        bbox: f.bbox,
        score: f.score,
        landmark5: f.landmark5,
      }));
    }
    if (this.detectorId === "yunet") {
      return this.detectFacesYunet(imageData);
    }
    if (this.detectorId === "scrfd_500m") {
      return this.detectFacesScrfd(imageData, SCRFD_500M.id);
    }
    return this.detectFacesYolo(imageData);
  }

  private async detectFacesYolo(imageData: ImageData): Promise<Face[]> {
    const session = this.sessions.get("yoloface_8n")!;
    const detectorSize = 640;

    // Resize to detector input size, preserving aspect ratio with padding
    const scale = Math.min(detectorSize / imageData.width, detectorSize / imageData.height);
    const newW = Math.round(imageData.width * scale);
    const newH = Math.round(imageData.height * scale);
    const resized = resizeImageData(imageData, newW, newH);

    // Pad to detectorSize x detectorSize with resized in the top-left corner
    // (zero-fill the rest). Pure JS so this runs in a worker too.
    const paddedData = new Uint8ClampedArray(detectorSize * detectorSize * 4);
    for (let y = 0; y < newH; y++) {
      const srcOff = y * newW * 4;
      const dstOff = y * detectorSize * 4;
      paddedData.set(resized.data.subarray(srcOff, srcOff + newW * 4), dstOff);
    }
    const padded = new ImageData(paddedData, detectorSize, detectorSize);

    // YOLOFace was trained on cv2-loaded BGR images (FaceFusion never
    // flips channels before feeding it). Convert RGBA -> BGR float32 NCHW
    // normalized to [0, 1] to match the training distribution.
    const float32 = rgbaToBgrFloat32(padded);

    const inputTensor = new ort.Tensor("float32", float32, [1, 3, detectorSize, detectorSize]);
    profileMark("yoloface");
    const results = await session.run({ [session.inputNames[0]]: inputTensor });
    const output = results[session.outputNames[0]];
    const outputData = output.data as Float32Array;

    // Parse YOLO output: [1, 20, N] where 20 = 4 bbox + 1 score + 15 landmarks (5x3)
    const dims = output.dims;
    const numDetections = Number(dims[2]);
    const faces: Face[] = [];
    const scoreThreshold = 0.5;

    for (let i = 0; i < numDetections; i++) {
      const cx = outputData[i];
      const cy = outputData[1 * numDetections + i];
      const w = outputData[2 * numDetections + i];
      const h = outputData[3 * numDetections + i];
      const score = outputData[4 * numDetections + i];

      if (score < scoreThreshold) continue;

      const ratioW = imageData.width / newW;
      const ratioH = imageData.height / newH;

      const bbox = [
        (cx - w / 2) * ratioW,
        (cy - h / 2) * ratioH,
        (cx + w / 2) * ratioW,
        (cy + h / 2) * ratioH,
      ];

      const landmark5: number[][] = [];
      for (let j = 0; j < 5; j++) {
        const lx = outputData[(5 + j * 3) * numDetections + i] * ratioW;
        const ly = outputData[(6 + j * 3) * numDetections + i] * ratioH;
        landmark5.push([lx, ly]);
      }

      faces.push({ bbox, score, landmark5 });
    }

    return this.nms(faces, 0.4);
  }

  /**
   * Resolve YuNet's 12 output tensor names into the canonical
   * [cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16,
   *  bbox_32, kps_8, kps_16, kps_32] order. Tries name-based matching
   * first (handles "cls_8", "loc_8", etc.) and falls back to positional
   * if names don't follow a recognizable pattern. Called once on first
   * detection then cached on this.yunetOutputNames.
   */
  private resolveYunetOutputNames(outputNames: readonly string[]): string[] {
    console.debug("[yunet] output names:", outputNames);
    if (outputNames.length !== 12) {
      throw new Error(`yunet: expected 12 outputs, got ${outputNames.length}`);
    }
    const strides = [8, 16, 32];
    const kinds = ["cls", "obj", "bbox", "kps"] as const;
    const resolved: (string | null)[] = Array.from({ length: 12 }, () => null);
    let allMatched = true;
    for (let ki = 0; ki < kinds.length; ki++) {
      const kind = kinds[ki];
      for (let si = 0; si < strides.length; si++) {
        const stride = strides[si];
        const re = new RegExp(`(^|[_-])${kind}[_-]?${stride}($|[._-])`);
        const match = outputNames.find((n) => re.test(n));
        if (match) {
          resolved[ki * 3 + si] = match;
        } else {
          allMatched = false;
        }
      }
    }
    if (!allMatched) {
      console.debug("[yunet] name-based resolution failed, using positional");
      this.yunetOutputNames = outputNames.slice();
    } else {
      this.yunetOutputNames = resolved as string[];
    }
    console.debug("[yunet] resolved order:", this.yunetOutputNames);
    return this.yunetOutputNames;
  }

  /**
   * YuNet 2023-03 face detector. Single-stage CNN with 12 output tensors
   * (cls/obj scores + bbox regression + 5-point landmark deltas, replicated
   * across feature strides 8/16/32). Outputs landmarks directly so we
   * don't need a separate refiner pass like 2DFAN4.
   *
   * Pipeline (matches FaceFusion's detect_with_yunet):
   *   1. Letterbox-pad input to 640x640 with origin in top-left.
   *   2. Build NCHW float32 tensor in pixel range [0, 255] (no /255 norm).
   *   3. Run model -> 12 outputs.
   *   4. For each stride S, walk the (W/S * H/S) anchor grid:
   *        anchor = (col*S, row*S)
   *        score  = cls_score[k] * obj_score[k]
   *        center = bbox[k, 0:2] * S + anchor
   *        size   = exp(bbox[k, 2:4]) * S
   *        landmarks[i] = kps[k, 2*i:2*i+2] * S + anchor   (5 of them)
   *   5. Threshold by score, scale back to original image space, NMS.
   */
  private async detectFacesYunet(imageData: ImageData): Promise<Face[]> {
    const session = this.sessions.get(YUNET_FACE_DETECTOR.id)!;
    // YuNet 2023-mar has a static 640x640 input - the model won't accept
    // any other shape. If we ever switch to a dynamic-axes export this
    // can become configurable.
    const detectorSize = 640;

    // Letterbox: scale preserving aspect ratio, pad with zeros at top-left.
    const scale = Math.min(detectorSize / imageData.width, detectorSize / imageData.height);
    const newW = Math.round(imageData.width * scale);
    const newH = Math.round(imageData.height * scale);
    const resized = resizeImageData(imageData, newW, newH);

    const paddedData = new Uint8ClampedArray(detectorSize * detectorSize * 4);
    for (let y = 0; y < newH; y++) {
      const srcOff = y * newW * 4;
      const dstOff = y * detectorSize * 4;
      paddedData.set(resized.data.subarray(srcOff, srcOff + newW * 4), dstOff);
    }
    const padded = new ImageData(paddedData, detectorSize, detectorSize);

    // YuNet wants pixel range [0, 255], NCHW, BGR order (cv2 native -
    // FaceFusion never flips channels before feeding it).
    const pixelCount = detectorSize * detectorSize;
    const float32 = new Float32Array(3 * pixelCount);
    const data = padded.data;
    for (let i = 0; i < pixelCount; i++) {
      float32[i] = data[i * 4 + 2]; // B
      float32[pixelCount + i] = data[i * 4 + 1]; // G
      float32[2 * pixelCount + i] = data[i * 4]; // R
    }

    const inputTensor = new ort.Tensor("float32", float32, [1, 3, detectorSize, detectorSize]);
    profileMark("yunet");
    const results = await session.run({ [session.inputNames[0]]: inputTensor });

    // YuNet has 12 outputs across strides 8/16/32 (cls/obj/bbox/kps).
    // Resolved to a flat array of 12 names once on first call - the index
    // layout is [cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8,
    // bbox_16, bbox_32, kps_8, kps_16, kps_32]. See resolveYunetOutputNames.
    const names = this.yunetOutputNames ?? this.resolveYunetOutputNames(session.outputNames);
    const strides = [8, 16, 32];
    const cls: Float32Array[] = [
      results[names[0]].data as Float32Array,
      results[names[1]].data as Float32Array,
      results[names[2]].data as Float32Array,
    ];
    const obj: Float32Array[] = [
      results[names[3]].data as Float32Array,
      results[names[4]].data as Float32Array,
      results[names[5]].data as Float32Array,
    ];
    const bbox: Float32Array[] = [
      results[names[6]].data as Float32Array,
      results[names[7]].data as Float32Array,
      results[names[8]].data as Float32Array,
    ];
    const kps: Float32Array[] = [
      results[names[9]].data as Float32Array,
      results[names[10]].data as Float32Array,
      results[names[11]].data as Float32Array,
    ];
    const scoreThreshold = 0.6;
    const ratioW = imageData.width / newW;
    const ratioH = imageData.height / newH;

    const faces: Face[] = [];
    for (let si = 0; si < strides.length; si++) {
      const stride = strides[si];
      const grid = detectorSize / stride; // square grid
      const clsS = cls[si];
      const objS = obj[si];
      const bboxS = bbox[si]; // shape [grid*grid, 4]
      const kpsS = kps[si]; // shape [grid*grid, 10]

      const numAnchors = grid * grid;
      for (let k = 0; k < numAnchors; k++) {
        const score = clsS[k] * objS[k];
        if (score < scoreThreshold) continue;

        // Anchor at top-left corner of cell k. Row-major flattening:
        // k = row*grid + col -> col = k % grid, row = k / grid.
        // (FaceFusion's mgrid trick collapses to the same indexing for
        // square grids.)
        const col = k % grid;
        const row = (k - col) / grid;
        const anchorX = col * stride;
        const anchorY = row * stride;

        const cx = bboxS[k * 4 + 0] * stride + anchorX;
        const cy = bboxS[k * 4 + 1] * stride + anchorY;
        const bw = Math.exp(bboxS[k * 4 + 2]) * stride;
        const bh = Math.exp(bboxS[k * 4 + 3]) * stride;

        const x1 = (cx - bw / 2) * ratioW;
        const y1 = (cy - bh / 2) * ratioH;
        const x2 = (cx + bw / 2) * ratioW;
        const y2 = (cy + bh / 2) * ratioH;

        const landmark5: number[][] = [];
        for (let i = 0; i < 5; i++) {
          const lx = (kpsS[k * 10 + i * 2 + 0] * stride + anchorX) * ratioW;
          const ly = (kpsS[k * 10 + i * 2 + 1] * stride + anchorY) * ratioH;
          landmark5.push([lx, ly]);
        }

        faces.push({ bbox: [x1, y1, x2, y2], score, landmark5 });
      }
    }

    return this.nms(faces, 0.4);
  }

  /**
   * SCRFD detector. Same parser for both 500m and 2.5g - they share the
   * architecture, only the backbone size differs. 9 outputs across strides
   * 8/16/32 (scores, bbox, kps), 2 anchors per grid cell, distance-encoded
   * bbox (left/top/right/bottom distances from anchor in stride units),
   * landmark deltas in (x, y) pairs from anchor in stride units.
   *
   * Pipeline (matches FaceFusion's detect_with_scrfd):
   *   1. Letterbox-pad input to 640x640 (top-left).
   *   2. Normalize to [-1, 1]: (pixel - 127.5) / 128.
   *   3. Run model -> 9 outputs (scores_8, scores_16, scores_32,
   *      bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32).
   *   4. For each stride S, walk anchors (2 per grid cell):
   *        anchor_idx_in_grid = k / 2
   *        col = anchor_idx_in_grid % grid
   *        row = anchor_idx_in_grid / grid
   *        anchor = (col*S, row*S)
   *        bbox: x1 = anchor.x - bbox[k,0]*S, y1 = anchor.y - bbox[k,1]*S,
   *              x2 = anchor.x + bbox[k,2]*S, y2 = anchor.y + bbox[k,3]*S
   *        landmark[i] = anchor + kps[k, 2i:2i+2] * S
   *   5. Threshold, scale to original image space, NMS.
   */
  private async detectFacesScrfd(imageData: ImageData, modelId: string): Promise<Face[]> {
    const session = this.sessions.get(modelId)!;
    const detectorSize = 640;

    // Letterbox: scale preserving aspect ratio, pad with zeros at top-left.
    const scale = Math.min(detectorSize / imageData.width, detectorSize / imageData.height);
    const newW = Math.round(imageData.width * scale);
    const newH = Math.round(imageData.height * scale);
    const resized = resizeImageData(imageData, newW, newH);

    const paddedData = new Uint8ClampedArray(detectorSize * detectorSize * 4);
    for (let y = 0; y < newH; y++) {
      const srcOff = y * newW * 4;
      const dstOff = y * detectorSize * 4;
      paddedData.set(resized.data.subarray(srcOff, srcOff + newW * 4), dstOff);
    }

    // Normalize: (pixel - 127.5) / 128, BGR NCHW float32 (cv2 native order
    // - FaceFusion never flips channels before feeding SCRFD).
    const pixelCount = detectorSize * detectorSize;
    const float32 = new Float32Array(3 * pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      float32[i] = (paddedData[i * 4 + 2] - 127.5) / 128;
      float32[pixelCount + i] = (paddedData[i * 4 + 1] - 127.5) / 128;
      float32[2 * pixelCount + i] = (paddedData[i * 4] - 127.5) / 128;
    }

    const inputTensor = new ort.Tensor("float32", float32, [1, 3, detectorSize, detectorSize]);
    profileMark(`scrfd-${modelId}`);
    const results = await session.run({ [session.inputNames[0]]: inputTensor });

    // Output order from FaceFusion / InsightFace exports:
    //   scores_8, scores_16, scores_32, bbox_8, bbox_16, bbox_32,
    //   kps_8,    kps_16,    kps_32
    const out = session.outputNames.map((n) => results[n].data as Float32Array);
    if (out.length !== 9) {
      throw new Error(`scrfd: expected 9 outputs, got ${out.length}`);
    }

    const strides = [8, 16, 32];
    const anchorTotal = 2;
    const scoreThreshold = 0.5;
    const ratioW = imageData.width / newW;
    const ratioH = imageData.height / newH;

    const faces: Face[] = [];
    for (let si = 0; si < strides.length; si++) {
      const stride = strides[si];
      const grid = detectorSize / stride;
      const scores = out[si];
      const bbox = out[si + 3];
      const kps = out[si + 6];
      const numAnchors = grid * grid * anchorTotal;
      for (let k = 0; k < numAnchors; k++) {
        const score = scores[k];
        if (score < scoreThreshold) continue;

        const cellIdx = (k - (k % anchorTotal)) / anchorTotal;
        const col = cellIdx % grid;
        const row = (cellIdx - col) / grid;
        const anchorX = col * stride;
        const anchorY = row * stride;

        const x1 = (anchorX - bbox[k * 4 + 0] * stride) * ratioW;
        const y1 = (anchorY - bbox[k * 4 + 1] * stride) * ratioH;
        const x2 = (anchorX + bbox[k * 4 + 2] * stride) * ratioW;
        const y2 = (anchorY + bbox[k * 4 + 3] * stride) * ratioH;

        const landmark5: number[][] = [];
        for (let i = 0; i < 5; i++) {
          const lx = (anchorX + kps[k * 10 + i * 2 + 0] * stride) * ratioW;
          const ly = (anchorY + kps[k * 10 + i * 2 + 1] * stride) * ratioH;
          landmark5.push([lx, ly]);
        }

        faces.push({ bbox: [x1, y1, x2, y2], score, landmark5 });
      }
    }

    return this.nms(faces, 0.4);
  }

  /** Non-maximum suppression: remove overlapping detections, keep highest score */
  private nms(faces: Face[], iouThreshold: number): Face[] {
    faces.sort((a, b) => b.score - a.score);
    const keep: Face[] = [];
    const suppressed = new Set<number>();

    for (let i = 0; i < faces.length; i++) {
      if (suppressed.has(i)) continue;
      keep.push(faces[i]);
      for (let j = i + 1; j < faces.length; j++) {
        if (suppressed.has(j)) continue;
        if (this.iou(faces[i].bbox, faces[j].bbox) > iouThreshold) {
          suppressed.add(j);
        }
      }
    }
    return keep;
  }

  private iou(a: number[], b: number[]): number {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (areaA + areaB - inter);
  }

  // --- Face Landmarks (2DFAN4) -> 68 points -> 5-point subset ---

  private async getLandmarks68(imageData: ImageData, face: Face): Promise<number[][]> {
    const session = this.sessions.get("2dfan4")!;

    // Crop face region with margin
    const [x1, y1, x2, y2] = face.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const size = Math.max(w, h) * 1.5;
    const cropX = Math.max(0, Math.round(cx - size / 2));
    const cropY = Math.max(0, Math.round(cy - size / 2));
    const cropW = Math.min(Math.round(size), imageData.width - cropX);
    const cropH = Math.min(Math.round(size), imageData.height - cropY);

    const cropped = cropImageData(imageData, cropX, cropY, cropW, cropH);
    const resized = resizeImageData(cropped, 256, 256);
    // 2DFAN4 was trained on cv2-loaded BGR images (FaceFusion does not
    // flip channels before feeding it).
    const float32 = rgbaToBgrFloat32(resized);

    const inputTensor = new ort.Tensor("float32", float32, [1, 3, 256, 256]);
    profileMark("2dfan4-landmarks");
    const results = await session.run({ [session.inputNames[0]]: inputTensor });

    // Output "landmarks" is [1, 68, 3] - (x, y, conf) in 64-unit internal space
    // Scale by 256/64 = 4 to get 256x256 crop-space coordinates
    const lmData = results["landmarks"].data as Float32Array;

    const landmarks: number[][] = [];
    for (let i = 0; i < 68; i++) {
      const lx = (lmData[i * 3] / 64) * 256;
      const ly = (lmData[i * 3 + 1] / 64) * 256;
      const origX = cropX + (lx / 256) * cropW;
      const origY = cropY + (ly / 256) * cropH;
      landmarks.push([origX, origY]);
    }

    return landmarks;
  }

  /**
   * Get the best 5-point landmarks for a detected face. For YOLOFace this
   * runs 2DFAN4 to refine, for MediaPipe the landmarks are already accurate
   * (taken straight from the 478-point face mesh).
   */
  private async getLandmarks5(imageData: ImageData, face: Face): Promise<number[][]> {
    // MediaPipe and YuNet both return 5-point landmarks directly from
    // the detector, so we use them as-is. YOLOFace gives us only a bbox,
    // so we run 2DFAN4 to get 68 points and reduce to 5.
    if (
      this.detectorId === "mediapipe" ||
      this.detectorId === "yunet" ||
      this.detectorId === "scrfd_500m"
    ) {
      return face.landmark5;
    }
    const lm68 = await this.getLandmarks68(imageData, face);
    return this.landmarks68to5(lm68);
  }

  /** Convert 68-point landmarks to 5-point (FaceFusion convention) */
  private landmarks68to5(lm68: number[][]): number[][] {
    const leftEye = [
      (lm68[36][0] + lm68[37][0] + lm68[38][0] + lm68[39][0] + lm68[40][0] + lm68[41][0]) / 6,
      (lm68[36][1] + lm68[37][1] + lm68[38][1] + lm68[39][1] + lm68[40][1] + lm68[41][1]) / 6,
    ];
    const rightEye = [
      (lm68[42][0] + lm68[43][0] + lm68[44][0] + lm68[45][0] + lm68[46][0] + lm68[47][0]) / 6,
      (lm68[42][1] + lm68[43][1] + lm68[44][1] + lm68[45][1] + lm68[46][1] + lm68[47][1]) / 6,
    ];
    return [leftEye, rightEye, lm68[30], lm68[48], lm68[54]];
  }

  // --- Face Embedding (ArcFace) ---

  private async getEmbedding(imageData: ImageData, face: Face): Promise<Float32Array> {
    const session = this.sessions.get("arcface_w600k_r50")!;
    const lm5 = await this.getLandmarks5(imageData, face);

    // Warp face to 112x112 using arcface_112_v2 template
    const crop = this.warpFace(imageData, lm5, TEMPLATE_ARCFACE_112_V2, 112);

    // Normalize: / 127.5 - 1, RGB order, NCHW
    const float32 = new Float32Array(3 * 112 * 112);
    const data = crop.data;
    const pixelCount = 112 * 112;
    for (let i = 0; i < pixelCount; i++) {
      float32[i] = data[i * 4] / 127.5 - 1; // R
      float32[pixelCount + i] = data[i * 4 + 1] / 127.5 - 1; // G
      float32[2 * pixelCount + i] = data[i * 4 + 2] / 127.5 - 1; // B
    }

    const inputTensor = new ort.Tensor("float32", float32, [1, 3, 112, 112]);
    profileMark("arcface-embedding");
    const results = await session.run({ [session.inputNames[0]]: inputTensor });
    const embedding = results[session.outputNames[0]].data as Float32Array;

    // L2 normalize
    let norm = 0;
    for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
    norm = Math.sqrt(norm);
    const normalized = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) normalized[i] = embedding[i] / norm;

    return normalized;
  }

  // --- XSeg Occlusion Mask ---

  private async getOcclusionMask(crop: ImageData): Promise<Float32Array> {
    const session = this.sessions.get("xseg_1")!;
    const size = 256;
    const resized = resizeImageData(crop, size, size);

    // XSeg input: /255.0, NHWC format, BGR order (cv2 native - FaceFusion
    // never flips channels before feeding it).
    const pixelCount = size * size;
    const float32 = new Float32Array(pixelCount * 3);
    for (let i = 0; i < pixelCount; i++) {
      float32[i * 3] = resized.data[i * 4 + 2] / 255.0; // B
      float32[i * 3 + 1] = resized.data[i * 4 + 1] / 255.0; // G
      float32[i * 3 + 2] = resized.data[i * 4] / 255.0; // R
    }

    const inputTensor = new ort.Tensor("float32", float32, [1, size, size, 3]);
    profileMark("xseg");
    const results = await session.run({ [session.inputNames[0]]: inputTensor });
    const rawMask = results[session.outputNames[0]].data as Float32Array;

    // Post-process the model output (still at model resolution): clip to
    // [0,1], gaussian-blur to soften jagged edges, then apply the
    // clip(0.5,1) -> -0.5 -> *2 contrast stretch from facefusion's
    // face_masker.create_occlusion_mask. The stretch sharpens the mask
    // so anything below 0.5 confidence becomes 0 while still preserving
    // a soft gradient near the boundary.
    const modelPixels = size * size;
    const clipped = new Float32Array(modelPixels);
    for (let i = 0; i < modelPixels; i++) {
      const v = rawMask[i];
      clipped[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    const blurred = gaussianBlur1f(clipped, size, size, 5);
    const processed = new Float32Array(modelPixels);
    for (let i = 0; i < modelPixels; i++) {
      const v = blurred[i];
      processed[i] = (v < 0.5 ? 0.5 : v > 1 ? 1 : v) * 2 - 1;
    }

    // Resize mask to crop dimensions if needed
    const cropPixels = crop.width * crop.height;
    if (crop.width === size && crop.height === size) {
      return processed;
    }

    // Need to resize - use nearest neighbor on the post-processed mask.
    // (The blur+stretch already happened at model resolution, so even
    // nearest-neighbor upscale produces visually smooth edges.)
    const mask = new Float32Array(cropPixels);
    const scaleX = size / crop.width;
    const scaleY = size / crop.height;
    for (let y = 0; y < crop.height; y++) {
      for (let x = 0; x < crop.width; x++) {
        const sx = Math.min(size - 1, Math.floor(x * scaleX));
        const sy = Math.min(size - 1, Math.floor(y * scaleY));
        mask[y * crop.width + x] = processed[sy * size + sx];
      }
    }
    return mask;
  }

  // --- Face Swap (HyperSwap) ---

  private async swapFace(
    frame: ImageData,
    face: Face,
    sourceEmbedding: Float32Array,
    useXseg: boolean,
  ): Promise<{ result: ImageData; landmarks: number; swap: number; xseg: number; paste: number }> {
    const swapSession = this.sessions.get(this.swapSessionId!)!;
    let t = performance.now();
    const lm5 = await this.getLandmarks5(frame, face);
    const tLandmarks = performance.now() - t;

    // HyperSwap: 256x256 crop with mean=0.5/std=0.5 normalization (output in
    // [-1, 1]). inswapper: 128x128 crop with /255 normalization (output in
    // [0, 1]). Both use the same arcface_128 template, just different sizes.
    const cropSize = this.swapKind === "inswapper" ? 128 : 256;

    t = performance.now();
    const crop = this.warpFace(frame, lm5, TEMPLATE_ARCFACE_128, cropSize);
    const affineMatrix = this.getAffineMatrix(lm5, TEMPLATE_ARCFACE_128, cropSize);

    const float32 = new Float32Array(3 * cropSize * cropSize);
    const data = crop.data;
    const pixelCount = cropSize * cropSize;
    if (this.swapKind === "inswapper") {
      // Normalize: pixel / 255, RGB order, NCHW
      for (let i = 0; i < pixelCount; i++) {
        float32[i] = data[i * 4] / 255.0;
        float32[pixelCount + i] = data[i * 4 + 1] / 255.0;
        float32[2 * pixelCount + i] = data[i * 4 + 2] / 255.0;
      }
    } else {
      // Normalize: (pixel/255 - 0.5) / 0.5, RGB order, NCHW
      for (let i = 0; i < pixelCount; i++) {
        float32[i] = (data[i * 4] / 255.0 - 0.5) / 0.5;
        float32[pixelCount + i] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5;
        float32[2 * pixelCount + i] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5;
      }
    }

    const targetTensor = new ort.Tensor("float32", float32, [1, 3, cropSize, cropSize]);
    const sourceTensor = new ort.Tensor("float32", sourceEmbedding, [1, 512]);

    // Robust input-name lookup: HyperSwap uses ("target", "source"); some
    // inswapper exports use ("target", "source"), others ("input", "source.1"),
    // etc. Pick by shape - the 4D tensor is the image, the 2D one is the
    // embedding.
    const feeds: Record<string, ort.Tensor> = {};
    const inputNames = swapSession.inputNames;
    if (inputNames.length !== 2) {
      throw new Error(`unexpected swap model input count: ${inputNames.length}`);
    }
    // Heuristic: name containing "source" (or "emb") gets the embedding,
    // everything else gets the image.
    const sourceName = inputNames.find((n) => /source|emb/i.test(n)) ?? inputNames[1];
    const targetName = inputNames.find((n) => n !== sourceName) ?? inputNames[0];
    feeds[targetName] = targetTensor;
    feeds[sourceName] = sourceTensor;

    profileMark(`swap-${this.swapKind}`);
    const results = await swapSession.run(feeds);
    const swappedData = results[Object.keys(results)[0]].data as Float32Array;

    const swappedCrop = new ImageData(cropSize, cropSize);
    if (this.swapKind === "inswapper") {
      // De-normalize: model output is RGB CHW in [0, 1] -> clip -> * 255
      for (let i = 0; i < pixelCount; i++) {
        const r = Math.max(0, Math.min(1, swappedData[i])) * 255;
        const g = Math.max(0, Math.min(1, swappedData[pixelCount + i])) * 255;
        const b = Math.max(0, Math.min(1, swappedData[2 * pixelCount + i])) * 255;
        swappedCrop.data[i * 4] = r;
        swappedCrop.data[i * 4 + 1] = g;
        swappedCrop.data[i * 4 + 2] = b;
        swappedCrop.data[i * 4 + 3] = 255;
      }
    } else {
      // De-normalize: model output is RGB CHW in [-1, 1] -> * 0.5 + 0.5 -> clip -> * 255
      for (let i = 0; i < pixelCount; i++) {
        const r = Math.max(0, Math.min(1, swappedData[i] * 0.5 + 0.5)) * 255;
        const g = Math.max(0, Math.min(1, swappedData[pixelCount + i] * 0.5 + 0.5)) * 255;
        const b = Math.max(0, Math.min(1, swappedData[2 * pixelCount + i] * 0.5 + 0.5)) * 255;
        swappedCrop.data[i * 4] = r;
        swappedCrop.data[i * 4 + 1] = g;
        swappedCrop.data[i * 4 + 2] = b;
        swappedCrop.data[i * 4 + 3] = 255;
      }
    }
    const tSwap = performance.now() - t;

    // Get XSeg occlusion mask for the swapped crop (optional - slow due to
    // unsupported Max op falling back to CPU)
    t = performance.now();
    const occlusionMask = useXseg ? await this.getOcclusionMask(swappedCrop) : undefined;
    const tXseg = performance.now() - t;

    t = performance.now();
    const result = await this.pasteBackImpl(frame, swappedCrop, affineMatrix, occlusionMask);
    const tPaste = performance.now() - t;

    return { result, landmarks: tLandmarks, swap: tSwap, xseg: tXseg, paste: tPaste };
  }

  // --- Face Enhancement (CodeFormer) ---

  private async enhanceFace(
    frame: ImageData,
    face: Face,
  ): Promise<{ result: ImageData; landmarks: number; enhance: number; paste: number }> {
    const session = this.sessions.get(this.enhancerSessionId!)!;
    let t = performance.now();
    const lm5 = await this.getLandmarks5(frame, face);
    const tLandmarks = performance.now() - t;
    const cropSize = 512;
    t = performance.now();

    const crop = this.warpFace(frame, lm5, TEMPLATE_FFHQ_512, cropSize);
    const affineMatrix = this.getAffineMatrix(lm5, TEMPLATE_FFHQ_512, cropSize);

    // Normalize: RGB order, (pixel/255 - 0.5) / 0.5
    const float32 = new Float32Array(3 * cropSize * cropSize);
    const data = crop.data;
    const pixelCount = cropSize * cropSize;
    for (let i = 0; i < pixelCount; i++) {
      float32[i] = (data[i * 4] / 255.0 - 0.5) / 0.5; // R
      float32[pixelCount + i] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5; // G
      float32[2 * pixelCount + i] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5; // B
    }

    const inputTensor = new ort.Tensor("float32", float32, [1, 3, cropSize, cropSize]);
    const weightTensor = new ort.Tensor("float64", new Float64Array([0.7]), [1]);

    const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]]: inputTensor };
    if (session.inputNames.includes("weight")) {
      feeds.weight = weightTensor;
    }

    profileMark(`enhance-${this.enhancerSessionId ?? "unknown"}`);
    const results = await session.run(feeds);
    const enhanced = results[session.outputNames[0]].data as Float32Array;

    // De-normalize: model output is RGB CHW. * 0.5 + 0.5 -> clip 0-1 -> * 255
    const enhancedCrop = new ImageData(cropSize, cropSize);
    for (let i = 0; i < pixelCount; i++) {
      const r = Math.max(0, Math.min(1, enhanced[i] * 0.5 + 0.5)) * 255;
      const g = Math.max(0, Math.min(1, enhanced[pixelCount + i] * 0.5 + 0.5)) * 255;
      const b = Math.max(0, Math.min(1, enhanced[2 * pixelCount + i] * 0.5 + 0.5)) * 255;
      enhancedCrop.data[i * 4] = r;
      enhancedCrop.data[i * 4 + 1] = g;
      enhancedCrop.data[i * 4 + 2] = b;
      enhancedCrop.data[i * 4 + 3] = 255;
    }
    const tEnhance = performance.now() - t;

    t = performance.now();
    const result = await this.pasteBackImpl(frame, enhancedCrop, affineMatrix);
    const tPaste = performance.now() - t;

    return { result, landmarks: tLandmarks, enhance: tEnhance, paste: tPaste };
  }

  // --- Geometry helpers ---

  private warpFace(
    imageData: ImageData,
    landmarks5: number[][],
    template: number[][],
    size: number,
  ): ImageData {
    const matrix = this.getAffineMatrix(landmarks5, template, size);
    return warpAffine(imageData, matrix, size, size);
  }

  private getAffineMatrix(landmarks5: number[][], template: number[][], size: number): number[] {
    const dst = template.map(([x, y]) => [x * size, y * size]);
    return estimateSimilarityTransform(landmarks5, dst);
  }
}
