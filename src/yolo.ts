import type { TfliteModel } from 'react-native-fast-tflite';
import type { Image } from 'react-native-nitro-image';
import { COCO_LABELS } from './labels';

export const CONFIDENCE_THRESHOLD = 0.75;
const IOU_THRESHOLD = 0.45;

export type Detection = { label: string; confidence: number };

/** Resize to the model input size and convert to a float32 RGB [0..1] buffer. */
function toInputTensor(image: Image, size: number): ArrayBuffer {
  const raw = image.resize(size, size).toRawPixelData();
  const px = new Uint8Array(raw.buffer);
  const fmt = raw.pixelFormat;
  const bpp = fmt.length === 3 ? 3 : 4;
  // byte offsets of R, G, B within a pixel (works for RGBA/BGRA/ARGB/... and RGB/BGR)
  const r = fmt.indexOf('R');
  const g = fmt.indexOf('G');
  const b = fmt.indexOf('B');
  const out = new Float32Array(size * size * 3);
  for (let i = 0, o = 0; i < size * size; i++, o += 3) {
    const p = i * bpp;
    out[o] = px[p + r] / 255;
    out[o + 1] = px[p + g] / 255;
    out[o + 2] = px[p + b] / 255;
  }
  return out.buffer;
}

function iou(a: number[], b: number[]): number {
  // boxes are [cx, cy, w, h]
  const ax1 = a[0] - a[2] / 2, ay1 = a[1] - a[3] / 2, ax2 = a[0] + a[2] / 2, ay2 = a[1] + a[3] / 2;
  const bx1 = b[0] - b[2] / 2, by1 = b[1] - b[3] / 2, bx2 = b[0] + b[2] / 2, by2 = b[1] + b[3] / 2;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
}

export async function detect(model: TfliteModel, image: Image): Promise<Detection[]> {
  const size = model.inputs[0].shape[1]; // [1, H, W, 3]
  const [output] = await model.run([toInputTensor(image, size)]);
  const data = new Float32Array(output);

  // YOLO11 output is [1, 4 + numClasses, numBoxes] (or transposed).
  const [, d1, d2] = model.outputs[0].shape;
  const channelsFirst = d1 < d2;
  const numAttrs = channelsFirst ? d1 : d2;
  const numBoxes = channelsFirst ? d2 : d1;
  const at = (box: number, attr: number) =>
    channelsFirst ? data[attr * numBoxes + box] : data[box * numAttrs + attr];

  const candidates: { box: number[]; cls: number; score: number }[] = [];
  for (let i = 0; i < numBoxes; i++) {
    let cls = 0;
    let score = 0;
    for (let c = 4; c < numAttrs; c++) {
      const s = at(i, c);
      if (s > score) { score = s; cls = c - 4; }
    }
    if (score >= CONFIDENCE_THRESHOLD) {
      candidates.push({ box: [at(i, 0), at(i, 1), at(i, 2), at(i, 3)], cls, score });
    }
  }

  // Per-class non-max suppression
  candidates.sort((a, b) => b.score - a.score);
  const kept: typeof candidates = [];
  for (const c of candidates) {
    if (!kept.some((k) => k.cls === c.cls && iou(k.box, c.box) > IOU_THRESHOLD)) kept.push(c);
  }
  return kept.map((k) => ({ label: COCO_LABELS[k.cls] ?? `class ${k.cls}`, confidence: k.score }));
}
