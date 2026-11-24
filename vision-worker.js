import * as ort from './vendor/onnx/ort.wasm.min.mjs';

const SIZE = 640;
const RAW_FLOOR = 0.15;
const DEFAULT_MODEL_URL = new URL('./models/pothole.onnx', import.meta.url).href;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./vendor/onnx/', import.meta.url).href;

const sessions = new Map();
const customBuffers = new Map();

function loadSession(modelKey) {
  if (!sessions.has(modelKey)) {
    const source = modelKey === 'default' ? DEFAULT_MODEL_URL : customBuffers.get(modelKey);
    if (!source) {
      return Promise.reject(new Error('Selected model is not loaded.'));
    }
    const promise = ort.InferenceSession.create(source, { executionProviders: ['wasm'] }).catch(error => {
      sessions.delete(modelKey);
      throw error;
    });
    sessions.set(modelKey, promise);
  }
  return sessions.get(modelKey);
}

function preprocess(source) {
  const width = source.naturalWidth || source.videoWidth || source.width;
  const height = source.naturalHeight || source.videoHeight || source.height;
  if (!(width > 0 && height > 0)) throw new Error('Image dimensions could not be read.');
  const scale = Math.min(SIZE / width, SIZE / height);
  const scaledWidth = Math.max(1, Math.round(width * scale));
  const scaledHeight = Math.max(1, Math.round(height * scale));
  const left = Math.floor((SIZE - scaledWidth) / 2);
  const top = Math.floor((SIZE - scaledHeight) / 2);

  const canvas = typeof document === 'undefined' ? new OffscreenCanvas(SIZE, SIZE) : document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(source, left, top, scaledWidth, scaledHeight);

  const pixels = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    data[i] = pixels[4 * i] / 255;
    data[i + plane] = pixels[4 * i + 1] / 255;
    data[i + 2 * plane] = pixels[4 * i + 2] / 255;
  }
  return {
    tensor: new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]),
    width,
    height,
    left,
    top,
    scaledWidth,
    scaledHeight,
  };
}

function defaultClassLabel(classId, numClasses) {
  if (numClasses === 1) return 'Pothole';
  const rddLabels = ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack', 'Pothole'];
  if (numClasses === 4 && rddLabels[classId]) return rddLabels[classId];
  return 'Defect ' + (classId + 1);
}

function decodeRaw(output, frame) {
  const dims = output.dims;
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error('Unsupported ONNX output shape: ' + dims.join('x'));
  }

  // Standard YOLOv8/11 export is [1, 4+C, N]; also support [1, N, 4+C].
  const transposed = dims[1] > dims[2] && dims[2] >= 5;
  const channels = transposed ? dims[2] : dims[1];
  const count = transposed ? dims[1] : dims[2];
  const numClasses = channels - 4;
  if (numClasses < 1 || count < 1) {
    throw new Error('Unexpected YOLO output dimensions.');
  }

  const data = output.data;
  const toX = v => Math.max(0, Math.min(frame.width, (v - frame.left) * frame.width / frame.scaledWidth));
  const toY = v => Math.max(0, Math.min(frame.height, (v - frame.top) * frame.height / frame.scaledHeight));
  const raw = [];

  for (let i = 0; i < count; i++) {
    let bestScore = 0;
    let bestClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = transposed ? data[i * channels + 4 + c] : data[(4 + c) * count + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (!(bestScore >= RAW_FLOOR)) continue;

    const cx = transposed ? data[i * channels] : data[i];
    const cy = transposed ? data[i * channels + 1] : data[count + i];
    const bw = transposed ? data[i * channels + 2] : data[2 * count + i];
    const bh = transposed ? data[i * channels + 3] : data[3 * count + i];
    const box = [toX(cx - bw / 2), toY(cy - bh / 2), toX(cx + bw / 2), toY(cy + bh / 2)];
    if (box[2] > box[0] + 1 && box[3] > box[1] + 1) {
      raw.push({
        box,
        score: bestScore,
        classId: bestClass,
        label: defaultClassLabel(bestClass, numClasses),
      });
    }
  }

  raw.sort((a, b) => b.score - a.score);
  return raw.slice(0, 300);
}

export function registerModelBuffer(modelKey, buffer) {
  customBuffers.set(modelKey, new Uint8Array(buffer));
  sessions.delete(modelKey);
}

export async function runDetector(image, modelKey = 'default') {
  const start = performance.now();
  const detector = await loadSession(modelKey);
  const frame = preprocess(image);
  let outputs;
  try {
    outputs = await detector.run({ [detector.inputNames[0]]: frame.tensor });
    const raw = decodeRaw(outputs[detector.outputNames[0]], frame);
    return {
      raw,
      elapsedMs: Math.round(performance.now() - start),
    };
  } finally {
    frame.tensor.dispose();
    if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose());
  }
}

if (typeof document === 'undefined') {
  let queue = Promise.resolve();
  self.onmessage = ({ data }) => {
    if (data.type === 'register') {
      registerModelBuffer(data.modelKey, data.buffer);
      self.postMessage({ id: data.id, registered: true });
      return;
    }
    const { id, bitmap, modelKey } = data;
    queue = queue.then(async () => {
      try {
        const result = await runDetector(bitmap, modelKey);
        self.postMessage({ id, result });
      } catch (error) {
        self.postMessage({ id, error: error?.message || String(error) });
      } finally {
        bitmap.close();
      }
    });
  };
}
