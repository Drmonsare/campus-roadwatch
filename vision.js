export const DEFAULT_THRESHOLD = 0.35;
export const DEFAULT_IOU = 0.45;

export const MODELS = [
  { id: 'default', name: 'YOLOv8s RoadWatch (Built-in)', builtin: true },
];

let worker;
let workerAvailable =
  typeof Worker === 'function' &&
  typeof OffscreenCanvas === 'function' &&
  typeof createImageBitmap === 'function';
let nextId = 0;
const requests = new Map();
const localBuffers = new Map();

function disableWorker() {
  workerAvailable = false;
  worker?.terminate();
  worker = undefined;
  for (const request of requests.values()) request.resolve(null);
  requests.clear();
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./vision-worker.js', import.meta.url), { type: 'module' });
    worker.onerror = disableWorker;
    worker.onmessageerror = disableWorker;
    worker.onmessage = ({ data }) => {
      const request = requests.get(data.id);
      if (!request) return;
      requests.delete(data.id);
      if (data.error) request.reject(new Error(data.error));
      else request.resolve(data.result ?? data.registered);
    };
  }
  return worker;
}

export function overlapIoU(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const intersection = w * h;
  if (intersection <= 0) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

export function computeSeverity(box, imgWidth, imgHeight, score) {
  const frameArea = Math.max(1, imgWidth * imgHeight);
  const boxArea = Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
  const areaPct = (boxArea / frameArea) * 100;
  if (areaPct >= 6 || (areaPct >= 3.2 && score >= 0.7)) return { level: 'high', label: 'High', areaPct };
  if (areaPct >= 1.5 || score >= 0.65) return { level: 'medium', label: 'Medium', areaPct };
  return { level: 'low', label: 'Low', areaPct };
}

export function filterDetections(raw, threshold = DEFAULT_THRESHOLD, iouLimit = DEFAULT_IOU) {
  const filtered = raw.filter(item => item.score >= threshold);
  filtered.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const item of filtered) {
    const duplicate = kept.some(other => other.classId === item.classId && overlapIoU(other.box, item.box) > iouLimit);
    if (!duplicate) kept.push(item);
  }
  return kept;
}

export async function registerCustomModel(fileName, arrayBuffer) {
  const id = 'custom-' + Date.now();
  const cleanName = fileName.replace(/\.onnx$/i, '');
  const copy = arrayBuffer.slice(0);
  localBuffers.set(id, copy);

  if (workerAvailable) {
    try {
      const target = getWorker();
      const reqId = ++nextId;
      const transferCopy = arrayBuffer.slice(0);
      await new Promise((resolve, reject) => {
        requests.set(reqId, { resolve, reject });
        target.postMessage({ id: reqId, type: 'register', modelKey: id, buffer: transferCopy }, [transferCopy]);
      });
    } catch {
      disableWorker();
    }
  }

  const entry = { id, name: cleanName + ' (Custom ONNX)', builtin: false };
  MODELS.push(entry);
  return entry;
}

async function submitFrame(image, modelKey) {
  let bitmap;
  let target;
  try {
    bitmap = await createImageBitmap(image);
    if (!workerAvailable) {
      bitmap.close();
      return null;
    }
    target = getWorker();
  } catch {
    bitmap?.close();
    return null;
  }
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    requests.set(id, { resolve, reject });
    try {
      target.postMessage({ id, bitmap, modelKey }, [bitmap]);
    } catch {
      bitmap.close();
      disableWorker();
    }
  });
}

export async function infer(image, modelKey = 'default') {
  if (workerAvailable) {
    const result = await submitFrame(image, modelKey);
    if (result !== null) return result;
  }
  const mod = await import('./vision-worker.js');
  if (modelKey !== 'default' && localBuffers.has(modelKey)) {
    mod.registerModelBuffer(modelKey, localBuffers.get(modelKey));
  }
  return mod.runDetector(image, modelKey);
}
