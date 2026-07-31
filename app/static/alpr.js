// License plate detection + OCR, running entirely in the browser via ONNX
// Runtime Web. No image or plate text ever leaves the device.
//
// Two MIT-licensed models (see README for provenance):
//   detector: yolo-v9-t-384-license-plates-end2end  (open-image-models)
//   ocr:      cct_xs_v2_global                      (fast-plate-ocr)
//
// Every constant below was verified against the real .onnx files with
// scripts/verify_models.py — re-run that script if a model is ever swapped,
// since the two implementations have to agree exactly.

// --- Detector contract -----------------------------------------------------
// input  "images" : float32 NCHW [1,3,384,384], values /255, aspect NOT preserved
// output "output0": float32 ['batch', 7], one row per detection:
//                   [batch_idx, x1, y1, x2, y2, class_id, score] in 384-space px
// "end2end" means NMS is already baked into the graph, so there's no
// non-max-suppression to do here. It does NOT threshold internally though —
// it emits low-score junk rows that we must filter out ourselves.
const DET_SIZE = 384;
const DET_INPUT = "images";

// --- OCR contract ----------------------------------------------------------
// input  "input" : uint8 NHWC [N,64,128,3] — normalization is baked into the
//                  graph, so raw canvas pixel values go straight in.
// output "plate" : float32 [N,10,37] — 10 character slots over the alphabet,
//                  ALREADY softmaxed (each slot's row sums to 1), so argmax
//                  gives the character and the value at that index is the
//                  probability. This is a fixed-slot classification head, not
//                  CTC — no sequence decoding needed.
// output "region": float32 [N,66] — country classification, also softmaxed.
const OCR_W = 128;
const OCR_H = 64;
const OCR_INPUT = "input";
const ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
const PAD_CHAR = "_";

const REGIONS = [
  "Albania", "Andorra", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
  "Bahrain", "Belarus", "Belgium", "Bosnia and Herzegovina", "Brazil", "Bulgaria",
  "Cambodia", "Canada", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
  "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece", "Guernsey",
  "Hungary", "Iceland", "Indonesia", "Ireland", "Israel", "Italy", "Latvia",
  "Liechtenstein", "Lithuania", "Luxembourg", "Malaysia", "Malta", "Mexico", "Moldova",
  "Monaco", "Montenegro", "Netherlands", "New Zealand", "North Macedonia", "Norway",
  "Poland", "Portugal", "Qatar", "Romania", "San Marino", "Serbia", "Singapore",
  "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Thailand", "Turkey",
  "United States", "Ukraine", "United Kingdom", "Vietnam", "Unknown",
];

// Tuned against real photos with scripts/verify_models.py: genuine plates
// scored 0.87-0.998 while junk rows scored below 0.4, so 0.5 separates them
// with margin. OCR confidence is the *minimum* across slots — a plate too
// small to actually read came back at 0.21 while readable ones hit 1.000.
//
// ocrThreshold was raised from 0.4 to 0.6 once temporal voting landed: misread
// characters consistently carry low confidence, so this drops most of them
// before they ever get a vote. Lower it if distant plates stop registering.
// `allowedRegions` uses the region head as a misread detector rather than as
// information: a garbled reading of a US plate tends to be confidently
// classified as some other country (Brazil is a common one), so anything not
// classified United States is more likely a bad read than a foreign car.
// Set to null to accept every region. Add "Unknown" to the list if legitimate
// plates start getting dropped.
export const DEFAULTS = {
  detThreshold: 0.5,
  ocrThreshold: 0.6,
  minPlateLength: 4,
  allowedRegions: ["United States"],
};

const MODEL_BASE = "/static/models";
const DETECTOR_URL = `${MODEL_BASE}/yolo-v9-t-384-license-plates-end2end.onnx`;
const OCR_URL = `${MODEL_BASE}/cct_xs_v2_global.onnx`;

let detSession = null;
let ocrSession = null;
let backend = null;

// Scratch canvases, allocated once and reused every frame — allocating these
// per frame is a real cost in a loop that runs continuously.
const detCanvas = document.createElement("canvas");
detCanvas.width = DET_SIZE;
detCanvas.height = DET_SIZE;
const detCtx = detCanvas.getContext("2d", { willReadFrequently: true });

const ocrCanvas = document.createElement("canvas");
ocrCanvas.width = OCR_W;
ocrCanvas.height = OCR_H;
const ocrCtx = ocrCanvas.getContext("2d", { willReadFrequently: true });

// Reused input tensors — same reasoning as the canvases.
const detData = new Float32Array(1 * 3 * DET_SIZE * DET_SIZE);
const ocrData = new Uint8Array(1 * OCR_H * OCR_W * 3);

export function activeBackend() {
  return backend;
}

// Try WebGPU first, fall back to WASM. iOS Safari has shipped WebGPU since 17,
// but it isn't guaranteed, and some ops can silently fall back — the caller
// displays whichever backend actually loaded so slowness is explainable.
export async function loadModels(onProgress = () => {}) {
  const providers = [];
  if (navigator.gpu) providers.push("webgpu");
  providers.push("wasm");

  for (const provider of providers) {
    try {
      onProgress(`Loading models (${provider})...`);
      const opts = { executionProviders: [provider], graphOptimizationLevel: "all" };
      [detSession, ocrSession] = await Promise.all([
        ort.InferenceSession.create(DETECTOR_URL, opts),
        ort.InferenceSession.create(OCR_URL, opts),
      ]);
      backend = provider;
      return provider;
    } catch (err) {
      console.warn(`[alpr] ${provider} backend failed:`, err);
      detSession = ocrSession = null;
    }
  }
  throw new Error("Could not initialise ONNX Runtime on any backend");
}

// Source pixels -> detector input. The model was trained on a plain squash to
// 384x384 (keep_aspect_ratio is false), so deliberately no letterboxing here.
function fillDetectorInput(source) {
  detCtx.drawImage(source, 0, 0, DET_SIZE, DET_SIZE);
  const { data } = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE);
  const plane = DET_SIZE * DET_SIZE;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    detData[p] = data[i] / 255;
    detData[plane + p] = data[i + 1] / 255;
    detData[2 * plane + p] = data[i + 2] / 255;
  }
  return new ort.Tensor("float32", detData, [1, 3, DET_SIZE, DET_SIZE]);
}

// Crop region -> OCR input. Values stay raw uint8 (the graph normalizes).
function fillOcrInput(source, box) {
  const w = Math.max(1, box.x2 - box.x1);
  const h = Math.max(1, box.y2 - box.y1);
  ocrCtx.drawImage(source, box.x1, box.y1, w, h, 0, 0, OCR_W, OCR_H);
  const { data } = ocrCtx.getImageData(0, 0, OCR_W, OCR_H);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 3) {
    ocrData[p] = data[i];
    ocrData[p + 1] = data[i + 1];
    ocrData[p + 2] = data[i + 2];
  }
  return new ort.Tensor("uint8", ocrData, [1, OCR_H, OCR_W, 3]);
}

function decodeDetections(output, sourceWidth, sourceHeight, threshold) {
  const rows = output.dims[0];
  const stride = output.dims[1]; // 7
  const d = output.data;
  const scaleX = sourceWidth / DET_SIZE;
  const scaleY = sourceHeight / DET_SIZE;
  const boxes = [];

  for (let i = 0; i < rows; i++) {
    const o = i * stride;
    const score = d[o + 6];
    if (score < threshold) continue;
    boxes.push({
      x1: Math.max(0, Math.round(d[o + 1] * scaleX)),
      y1: Math.max(0, Math.round(d[o + 2] * scaleY)),
      x2: Math.min(sourceWidth, Math.round(d[o + 3] * scaleX)),
      y2: Math.min(sourceHeight, Math.round(d[o + 4] * scaleY)),
      score,
    });
  }
  return boxes;
}

// Fixed-slot argmax decode. `confidence` is the weakest slot rather than the
// mean, so one unreadable character drags the score down instead of being
// averaged away by nine confident ones.
function decodePlate(plateTensor, regionTensor) {
  const [, slots, classes] = plateTensor.dims;
  const p = plateTensor.data;
  let text = "";
  let confidence = 1;

  for (let s = 0; s < slots; s++) {
    let best = 0;
    let bestIdx = 0;
    for (let c = 0; c < classes; c++) {
      const v = p[s * classes + c];
      if (v > best) {
        best = v;
        bestIdx = c;
      }
    }
    const ch = ALPHABET[bestIdx];
    if (ch !== PAD_CHAR) {
      text += ch;
      confidence = Math.min(confidence, best);
    }
  }

  let region = null;
  if (regionTensor) {
    const r = regionTensor.data;
    let best = 0;
    let bestIdx = 0;
    for (let i = 0; i < r.length; i++) {
      if (r[i] > best) {
        best = r[i];
        bestIdx = i;
      }
    }
    region = { name: REGIONS[bestIdx] ?? "Unknown", confidence: best };
  }

  return { text, confidence, region };
}

// Junk boxes reliably OCR to short garbage, so length is the cheapest useful
// filter. Real plates are alphanumeric by construction (the alphabet has no
// punctuation), so length + confidence covers the rest — no format regex,
// which would wrongly reject vanity plates.
//
// The region check is a second misread signal, not a nationality check: see
// DEFAULTS.allowedRegions. A read rejected here never reaches the tracker, so
// it can't vote.
export function isPlausible(text, confidence, region, opts) {
  if (text.length < opts.minPlateLength) return false;
  if (confidence < opts.ocrThreshold) return false;
  if (opts.allowedRegions && region && !opts.allowedRegions.includes(region.name)) return false;
  return true;
}

/**
 * Detect plates in a frame and read them.
 * @param source canvas/video/image to read pixels from
 * @returns {Promise<Array<{text, confidence, region, box, detScore}>>}
 */
export async function readPlates(source, options = {}) {
  if (!detSession || !ocrSession) return [];
  const opts = { ...DEFAULTS, ...options };

  const width = source.videoWidth || source.width;
  const height = source.videoHeight || source.height;
  if (!width || !height) return [];

  const detOut = await detSession.run({ [DET_INPUT]: fillDetectorInput(source) });
  const boxes = decodeDetections(detOut.output0, width, height, opts.detThreshold);

  const results = [];
  for (const box of boxes) {
    if (box.x2 <= box.x1 || box.y2 <= box.y1) continue;
    const ocrOut = await ocrSession.run({ [OCR_INPUT]: fillOcrInput(source, box) });
    const { text, confidence, region } = decodePlate(ocrOut.plate, ocrOut.region);
    if (!isPlausible(text, confidence, region, opts)) continue;
    results.push({ text, confidence, region, box, detScore: box.score });
  }
  return results;
}
