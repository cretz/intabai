// Wraps MediaPipe FaceLandmarker as a drop-in replacement for the
// YOLOFace + 2DFAN4 detect+landmarks pipeline. One ~4MB model that runs on
// the GPU and returns 478 face landmarks per face in a single call.
//
// Detection + landmarks combined is dramatically faster than YOLOFace 8n
// (~380ms) + 2DFAN4 (~1100ms) on mobile.

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

// Hosting note: the wasm runtime files come from a CDN at runtime. The .task
// model file is OPFS-cached separately by the existing model manager. Tab
// memory and the AI side stay zero-upload regardless.
const MEDIAPIPE_WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

// Indices into the 478-point MediaPipe face mesh used to derive the 5-point
// alignment that arcface_128 / arcface_112_v2 / ffhq_512 templates expect.
// Iris centers (468/473) are present in the float16/1 model and give much
// more stable eye centers than averaging the eye corners.
const IDX_LEFT_EYE = 468; // left iris center
const IDX_RIGHT_EYE = 473; // right iris center
const IDX_NOSE = 1;
const IDX_MOUTH_LEFT = 61;
const IDX_MOUTH_RIGHT = 291;

let cachedWasmFileset: WasmFileset | null = null;

async function getWasmFileset(): Promise<WasmFileset> {
  if (!cachedWasmFileset) {
    cachedWasmFileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);
  }
  return cachedWasmFileset;
}

export interface MediaPipeFace {
  bbox: number[]; // [x1, y1, x2, y2] in image pixel coords
  score: number;
  landmark5: number[][]; // [leftEye, rightEye, nose, leftMouth, rightMouth] in pixels
}

export class MediaPipeDetector {
  private landmarker: FaceLandmarker | null = null;

  async load(modelBuffer: Uint8Array): Promise<void> {
    const wasmFileset = await getWasmFileset();
    this.landmarker = await FaceLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numFaces: 5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  }

  async release(): Promise<void> {
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
    }
  }

  /** Run detection + landmarks on an ImageData. Returns one Face per detected face. */
  detect(imageData: ImageData): MediaPipeFace[] {
    if (!this.landmarker) throw new Error("MediaPipeDetector not loaded");

    // FaceLandmarker accepts ImageData directly as an ImageSource.
    const result = this.landmarker.detect(imageData);
    const w = imageData.width;
    const h = imageData.height;
    const faces: MediaPipeFace[] = [];

    for (const lm of result.faceLandmarks) {
      // bbox from extrema of all landmarks
      let x1 = Infinity,
        y1 = Infinity,
        x2 = -Infinity,
        y2 = -Infinity;
      for (const p of lm) {
        const px = p.x * w;
        const py = p.y * h;
        if (px < x1) x1 = px;
        if (py < y1) y1 = py;
        if (px > x2) x2 = px;
        if (py > y2) y2 = py;
      }

      const px = (i: number): number[] => [lm[i].x * w, lm[i].y * h];
      const landmark5 = [
        px(IDX_LEFT_EYE),
        px(IDX_RIGHT_EYE),
        px(IDX_NOSE),
        px(IDX_MOUTH_LEFT),
        px(IDX_MOUTH_RIGHT),
      ];

      faces.push({
        bbox: [x1, y1, x2, y2],
        score: 1.0, // mediapipe doesn't expose a per-face confidence here
        landmark5,
      });
    }

    return faces;
  }
}
