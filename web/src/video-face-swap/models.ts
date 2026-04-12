import xsegPatch from "./patches/xseg_1.patch.json";
import { patchTransform, type Patch } from "../shared/model-patch";
import type { ModelFile } from "../shared/model-cache";

// Re-export so the rest of face-swap can keep importing ModelFile from here.
export type { ModelFile } from "../shared/model-cache";

const HF = "https://huggingface.co";

// HF repo roots. Each corresponds to the `<org>/<repo>` prefix of the file
// URLs below; keeping them as named constants so the model list UI can link
// the cache-row names to the source page without us having to string-munge a
// file URL at render time.
const HF_FF_300 = `${HF}/facefusion/models-3.0.0`;
const HF_FF_310 = `${HF}/facefusion/models-3.1.0`;
const HF_FF_330 = `${HF}/facefusion/models-3.3.0`;
const HF_FF_340 = `${HF}/facefusion/models-3.4.0`;
const HF_DEEPGHS_INSIGHTFACE = `${HF}/deepghs/insightface`;
const HF_DEEP_LIVE_CAM = `${HF}/hacksider/deep-live-cam`;

export interface ModelSet {
  id: string;
  name: string;
  description: string;
  primary: ModelFile;
  dependencies: ModelFile[];
}

// Shared pipeline models
const FACE_DETECTOR: ModelFile = {
  id: "yoloface_8n",
  name: "YOLOFace 8n (face detection)",
  url: `${HF_FF_300}/resolve/main/yoloface_8n.onnx`,
  sizeBytes: 6_400_000,
  hfRepoUrl: HF_FF_300,
};

const ARCFACE: ModelFile = {
  id: "arcface_w600k_r50",
  name: "ArcFace w600k (face recognition)",
  url: `${HF_FF_300}/resolve/main/arcface_w600k_r50.onnx`,
  sizeBytes: 174_000_000,
  hfRepoUrl: HF_FF_300,
};

const FACE_LANDMARKER: ModelFile = {
  id: "2dfan4",
  name: "2DFAN4 (face landmarks)",
  url: `${HF_FF_300}/resolve/main/2dfan4.onnx`,
  sizeBytes: 98_000_000,
  hfRepoUrl: HF_FF_300,
};

const XSEG: ModelFile = {
  id: "xseg_1",
  name: "XSeg (face segmentation)",
  url: `${HF_FF_310}/resolve/main/xseg_1.onnx`,
  sizeBytes: 70_000_000,
  hfRepoUrl: HF_FF_310,
  transform: patchTransform(xsegPatch as Patch),
};

// Alternate detector bundled with every swap set. MediaPipe FaceLandmarker is
// a single ~4MB GPU-native model that does face detection AND 478-point
// landmarks in one call, replacing both YOLOFace 8n and 2DFAN4. Much faster
// on mobile. The user picks at swap time which detector to actually use.
export const MEDIAPIPE_FACE_LANDMARKER: ModelFile = {
  id: "mediapipe_face_landmarker",
  name: "MediaPipe FaceLandmarker",
  url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  sizeBytes: 3_760_000,
};

// Second alternate detector. YuNet is a tiny single-stage detector that
// outputs bbox + 5 landmarks directly, so it replaces both YOLOFace 8n and
// 2DFAN4 (no separate landmark refiner pass). Runs through onnxruntime-web
// like every other ONNX model in this app, so it sidesteps the
// MediaPipe-on-Android-Chrome issues entirely.
export const YUNET_FACE_DETECTOR: ModelFile = {
  id: "yunet_2023_mar",
  name: "YuNet 2023-03 (face detection + 5 landmarks)",
  url: `${HF_FF_340}/resolve/main/yunet_2023_mar.onnx`,
  sizeBytes: 232_000,
  hfRepoUrl: HF_FF_340,
};

// SCRFD detectors. Same architecture, different backbone sizes. Both
// output bbox + 5 landmarks directly (no separate refiner needed),
// produced from FPN feature maps at strides 8/16/32 with 2 anchors per
// grid cell. 500m is the small/fast variant, 2.5g is the
// quality-balanced one that FaceFusion ships as default.
export const SCRFD_500M: ModelFile = {
  id: "scrfd_500m",
  name: "SCRFD 500m (face detection + 5 landmarks)",
  url: `${HF_DEEPGHS_INSIGHTFACE}/resolve/main/buffalo_s/det_500m.onnx`,
  sizeBytes: 2_525_000,
  hfRepoUrl: HF_DEEPGHS_INSIGHTFACE,
};

export const DETECTOR_OPTIONS = [
  {
    id: "yoloface",
    name: "YOLOFace + 2DFAN4 (slower)",
  },
  {
    id: "mediapipe",
    name: "MediaPipe FaceLandmarker (fast, desktop only - silently fails on mobile)",
  },
  {
    id: "yunet",
    name: "YuNet (faster)",
  },
  {
    id: "scrfd_500m",
    name: "SCRFD 500m (faster, better quality than YuNet)",
  },
] as const;

export type DetectorId = (typeof DETECTOR_OPTIONS)[number]["id"];

export const ENHANCERS: ModelFile[] = [
  {
    id: "gfpgan_1.4",
    name: "GFPGAN 1.4",
    url: `${HF_FF_300}/resolve/main/gfpgan_1.4.onnx`,
    sizeBytes: 340_000_000,
    hfRepoUrl: HF_FF_300,
  },
  {
    id: "restoreformer_plus_plus",
    name: "RestoreFormer++",
    url: `${HF_FF_300}/resolve/main/restoreformer_plus_plus.onnx`,
    sizeBytes: 294_000_000,
    hfRepoUrl: HF_FF_300,
  },
  {
    id: "codeformer",
    name: "CodeFormer (slower)",
    url: `${HF_FF_300}/resolve/main/codeformer.onnx`,
    sizeBytes: 377_000_000,
    hfRepoUrl: HF_FF_300,
  },
];

export function findEnhancer(id: string): ModelFile | undefined {
  return ENHANCERS.find((e) => e.id === id);
}

const HYPERSWAP_DEPS = [
  FACE_DETECTOR,
  FACE_LANDMARKER,
  ARCFACE,
  XSEG,
  MEDIAPIPE_FACE_LANDMARKER,
  YUNET_FACE_DETECTOR,
  SCRFD_500M,
];
const HYPERSWAP_DESC =
  "FaceFusion HyperSwap - high quality 256x256 face swap. 1a/1b/1c are different trained checkpoints with the same architecture and speed but different visual character - try them and pick what you like.";

// inswapper_128 source-embedding mapping matrix, pre-extracted at build time
// from the .onnx initializer (see notes/scripts/extract-inswapper-emap.py).
// Same matrix is used by both fp32 and fp16 variants. Hosted as a static
// asset under intabai/public/.
const INSWAPPER_EMAP: ModelFile = {
  id: "inswapper_emap",
  name: "inswapper embedding map",
  url: "/inswapper_emap.bin",
  sizeBytes: 1_048_576,
};

const INSWAPPER_DEPS = [
  FACE_DETECTOR,
  FACE_LANDMARKER,
  ARCFACE,
  XSEG,
  MEDIAPIPE_FACE_LANDMARKER,
  YUNET_FACE_DETECTOR,
  SCRFD_500M,
  INSWAPPER_EMAP,
];
const INSWAPPER_DESC =
  "Insightface inswapper - 128x128 face swap. Slower than HyperSwap in the browser despite the smaller crop, but some users prefer its visual character. Try it if HyperSwap output doesn't look right.";

export const MODEL_SETS: ModelSet[] = [
  {
    id: "hyperswap_1a",
    name: "HyperSwap 1a (faster)",
    description: HYPERSWAP_DESC,
    primary: {
      id: "hyperswap_1a_256",
      name: "HyperSwap 1a (face swap)",
      url: `${HF_FF_330}/resolve/main/hyperswap_1a_256.onnx`,
      sizeBytes: 403_000_000,
      hfRepoUrl: HF_FF_330,
    },
    dependencies: HYPERSWAP_DEPS,
  },
  {
    id: "hyperswap_1b",
    name: "HyperSwap 1b (faster)",
    description: HYPERSWAP_DESC,
    primary: {
      id: "hyperswap_1b_256",
      name: "HyperSwap 1b (face swap)",
      url: `${HF_FF_330}/resolve/main/hyperswap_1b_256.onnx`,
      sizeBytes: 403_000_000,
      hfRepoUrl: HF_FF_330,
    },
    dependencies: HYPERSWAP_DEPS,
  },
  {
    id: "hyperswap_1c",
    name: "HyperSwap 1c (faster)",
    description: HYPERSWAP_DESC,
    primary: {
      id: "hyperswap_1c_256",
      name: "HyperSwap 1c (face swap)",
      url: `${HF_FF_330}/resolve/main/hyperswap_1c_256.onnx`,
      sizeBytes: 403_000_000,
      hfRepoUrl: HF_FF_330,
    },
    dependencies: HYPERSWAP_DEPS,
  },
  {
    id: "inswapper_128_fp16",
    name: "inswapper 128 fp16",
    description: INSWAPPER_DESC,
    primary: {
      id: "inswapper_128_fp16",
      name: "inswapper 128 fp16 (face swap)",
      url: `${HF_DEEP_LIVE_CAM}/resolve/main/inswapper_128_fp16.onnx`,
      sizeBytes: 277_000_000,
      hfRepoUrl: HF_DEEP_LIVE_CAM,
    },
    dependencies: INSWAPPER_DEPS,
  },
  {
    id: "inswapper_128",
    name: "inswapper 128 fp32",
    description: INSWAPPER_DESC,
    primary: {
      id: "inswapper_128",
      name: "inswapper 128 fp32 (face swap)",
      url: `${HF_DEEP_LIVE_CAM}/resolve/main/inswapper_128.onnx`,
      sizeBytes: 554_000_000,
      hfRepoUrl: HF_DEEP_LIVE_CAM,
    },
    dependencies: INSWAPPER_DEPS,
  },
];

export function allFiles(set: ModelSet): ModelFile[] {
  return [set.primary, ...set.dependencies];
}

export function depsSize(set: ModelSet): number {
  return set.dependencies.reduce((sum, f) => sum + f.sizeBytes, 0);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${(bytes / 1_000_000).toFixed(0)} MB`;
}
