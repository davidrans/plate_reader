"""Verify the ONNX models' I/O contract that app/static/alpr.js reimplements in JS.

The browser does all real inference; this script exists so the contract can be
re-checked from Python whenever a model is swapped or upgraded. If the shapes,
dtypes, or output layouts printed here stop matching the constants in alpr.js,
the JS needs updating to match.

Run against the checked-in test images:

    uv run python scripts/verify_models.py tests/assets/*.jpg
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS = Path(__file__).parent.parent / "app" / "static" / "models"
DETECTOR = MODELS / "yolo-v9-t-384-license-plates-end2end.onnx"
OCR = MODELS / "cct_xs_v2_global.onnx"

DET_SIZE = 384
OCR_W, OCR_H = 128, 64
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
DET_THRESHOLD = 0.5

REGIONS = [
    "Albania",
    "Andorra",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahrain",
    "Belarus",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Bulgaria",
    "Cambodia",
    "Canada",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Georgia",
    "Germany",
    "Gibraltar",
    "Greece",
    "Guernsey",
    "Hungary",
    "Iceland",
    "Indonesia",
    "Ireland",
    "Israel",
    "Italy",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malaysia",
    "Malta",
    "Mexico",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "New Zealand",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "San Marino",
    "Serbia",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "United States",
    "Ukraine",
    "United Kingdom",
    "Vietnam",
    "Unknown",
]


def print_contract(det: ort.InferenceSession, ocr: ort.InferenceSession) -> None:
    print("=== ONNX I/O contract (must match app/static/alpr.js) ===")
    for label, sess in (("detector", det), ("ocr", ocr)):
        for t in sess.get_inputs():
            print(f"  {label:8s} IN   {t.name:8s} {t.shape} {t.type}")
        for t in sess.get_outputs():
            print(f"  {label:8s} OUT  {t.name:8s} {t.shape} {t.type}")
    print(f"  regions: {len(REGIONS)} (must equal the region output's last dim)")
    print()


def run(image_path: Path, det: ort.InferenceSession, ocr: ort.InferenceSession) -> None:
    det_in = det.get_inputs()[0].name
    ocr_in = ocr.get_inputs()[0].name

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Detector: float32 NCHW, /255, fixed 384x384 (aspect ratio not preserved).
    resized = img.resize((DET_SIZE, DET_SIZE), Image.BILINEAR)
    blob = np.array(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    detections = det.run(None, {det_in: blob})[0]

    print(f"--- {image_path.name} ({width}x{height}) raw rows: {len(detections)}")

    scale_x, scale_y = width / DET_SIZE, height / DET_SIZE
    for row in detections:
        # Row layout: [batch_idx, x1, y1, x2, y2, class_id, score]
        score = float(row[6])
        # The end2end graph does NOT filter internally — it emits low-score
        # junk rows that must be thresholded here (and in alpr.js).
        if score < DET_THRESHOLD:
            continue

        box = (
            max(0, int(row[1] * scale_x)),
            max(0, int(row[2] * scale_y)),
            min(width, int(row[3] * scale_x)),
            min(height, int(row[4] * scale_y)),
        )

        # OCR: uint8 NHWC 64x128x3 — normalization is baked into the graph.
        crop = img.crop(box).resize((OCR_W, OCR_H), Image.BILINEAR)
        plate, region = ocr.run(None, {ocr_in: np.array(crop, dtype=np.uint8)[None]})

        # Outputs are already softmaxed (rows sum to 1) — argmax + read prob directly.
        text = "".join(ALPHABET[i] for i in plate[0].argmax(-1)).rstrip("_")
        confidence = float(plate[0].max(-1).min())
        print(
            f"    det={score:.3f} box={box} -> {text!r} "
            f"ocr_conf={confidence:.3f} region={REGIONS[int(region[0].argmax())]}"
        )


def main() -> None:
    det = ort.InferenceSession(str(DETECTOR))
    ocr = ort.InferenceSession(str(OCR))
    print_contract(det, ocr)

    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: verify_models.py IMAGE [IMAGE ...]")
        raise SystemExit(1)
    for p in paths:
        run(p, det, ocr)


if __name__ == "__main__":
    main()
