# plate_reader

An installable PWA that reads license plates with the phone camera. Plate
**detection and OCR both run on-device in the browser** via ONNX Runtime Web —
no image and no plate text is ever uploaded. The FastAPI backend only serves
static files; it never loads a model.

## Why a PWA rather than a native app

iOS Safari's engine is used by every browser on iOS (Apple requires it), so an
installed PWA and a plain web page have identical camera capabilities. What
that buys and costs:

- ✅ `getUserMedia` live camera works, installed or not.
- ✅ Good enough for continuous scanning **while the phone is held up and
  pointed at the street** — the intended use.
- ❌ **No background camera access.** Must be foreground with the screen on.
- ❌ No torch/flash or manual focus control from the web APIs.
- ❌ **No CoreML/Neural Engine access.** Inference is WebGPU or WASM only,
  slower than a native app could achieve.
- ❌ **HTTPS required.** A plain `http://` LAN address silently fails to get
  camera permission on a real device — see [Local development](#local-development).

## Models

Both models are **MIT licensed** and bundled in `app/static/models/`
(~10.6 MB total, cached by the service worker after first load).

| Role | File | Size | Source |
|---|---|---|---|
| Detection | `yolo-v9-t-384-license-plates-end2end.onnx` | 7.4 MB | [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models) |
| OCR | `cct_xs_v2_global.onnx` | 3.2 MB | [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) |

Larger/more accurate variants exist (e.g. `yolo-v9-s-608` at 27 MB, mAP50
0.966 vs. the bundled `t-384`'s 0.920). Swapping one means updating the
constants in `app/static/alpr.js`, the `ASSETS` list and `CACHE` version in
`app/static/sw.js`, and re-running `scripts/verify_models.py`.

**`openalpr/openalpr` was evaluated and rejected**: it's AGPL-3.0 (copyleft,
which matters for a hosted service), was last pushed January 2024, and its
classical-CV pipeline is tuned for fixed traffic cameras rather than handheld
off-angle shots. Recording that here so the decision isn't relitigated.

### Model I/O contract

`app/static/alpr.js` reimplements in JS what `scripts/verify_models.py` does in
Python. **The two must agree**, so the non-obvious parts are worth stating:

- **Detector** — `float32` NCHW `[1,3,384,384]`, values `/255`, aspect ratio
  deliberately *not* preserved (the model was trained on a plain squash).
  Output is `[N,7]` rows of `[batch_idx, x1, y1, x2, y2, class_id, score]` in
  384-space pixels.
  - "end2end" means **NMS is baked into the graph** — there's no non-max
    suppression to write.
  - But it does **not** threshold internally: it emits low-score junk rows
    that *must* be filtered by score, or you get phantom plates.
- **OCR** — `uint8` **NHWC** `[N,64,128,3]` (note: channels-*last*, unlike the
  detector). Normalization is baked into the graph, so raw canvas pixels go
  straight in.
  - Output `plate` is `[N,10,37]`: 10 fixed character slots over the alphabet
    `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_` (`_` = padding). It's a
    **classification head, not CTC**, so decoding is a per-slot argmax — no
    sequence decoder needed.
  - Values are **already softmaxed** (each slot's row sums to 1), so the value
    at the argmax *is* the probability. Applying softmax again would flatten
    the confidences to near-uniform.
  - Output `region` is `[N,66]` — country classification over `REGIONS` in
    `alpr.js`, which must stay 66 entries long and in the model's order.

### Thresholds

Defaults in `alpr.js` (`DEFAULTS`), tuned against the real photos in
`tests/assets/` rather than guessed:

| Setting | Value | Why |
|---|---|---|
| `detThreshold` | 0.5 | Real plates scored 0.87–0.998; junk rows scored <0.4. |
| `ocrThreshold` | 0.4 | A plate too small to read scored 0.21; readable ones 1.000. |
| `minPlateLength` | 4 | Junk boxes OCR to short garbage. |

OCR confidence is the **minimum** across character slots, not the mean — one
unreadable character should drag the score down rather than be averaged away
by nine confident ones. There's deliberately no plate-format regex: it would
wrongly reject vanity and non-US plates, and the model's alphabet already
excludes punctuation.

## Commands

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000

# Re-check the ONNX contract from Python (dev deps only; the app never does this)
uv run python scripts/verify_models.py tests/assets/zaz.jpg tests/assets/sgp.jpg
```

### Test harness

`tests/harness.html` runs the real `alpr.js` pipeline against the checked-in
sample images and compares against the Python reference — no camera needed.
With the server running, open <http://localhost:8000/tests/harness.html>.

Current result on desktop Chrome (WASM fallback, no WebGPU): **2/2 exact
matches**, 156–214 ms/frame.

```
/tests/assets/zaz.jpg  (250x187)  214 ms
  PASS text=AE6133CT det=0.916 ocr=0.999 region=Ukraine
/tests/assets/sgp.jpg  (899x226)  156 ms
  PASS text=SDN7484U det=0.998 ocr=1.000 region=Singapore
```

The `/tests` mount only exists when the directory is present; `tests/` and
`scripts/` are in `.dockerignore`, so it's absent from a deployed container.

## Local development

Camera access needs a secure context. `localhost` counts, so desktop Chrome
works directly — but **testing on a real iPhone needs HTTPS**:

```bash
cloudflared tunnel --url http://localhost:8000   # or: ngrok http 8000
```

Then open the generated `https://…` URL on the phone. Without this, the Start
button reports a camera error (the app detects `isSecureContext` and says so
explicitly rather than showing a bare `NotAllowedError`).

## Docker / Coolify

```bash
docker compose up --build
```

No environment variables and no ML dependencies in the image — inference is
entirely client-side, so the container is just a static file server.

## Notes and limitations

- **On-device speed is the open question.** Desktop WASM measures 156–214
  ms/frame; iPhone WebGPU should beat that, but it hasn't been measured on
  real hardware yet. The UI shows live ms/frame and which backend is active,
  so this is visible at a glance. If WebGPU is unavailable it silently falls
  back to WASM — the status line is how you tell.
- The OCR model is trained on **Latin-alphabet** plates across 66 regions.
  Reported accuracy for US plates is 92.9% full-plate exact match / 98.7%
  per-character.
- Small or distant plates fail: the detector squashes the frame to 384×384, so
  a plate occupying a few dozen source pixels won't survive. The app requests
  1920×1080 capture to mitigate this.
- Recent plates live in `localStorage` only (capped at 50, consecutive
  duplicates debounced for 10s). Nothing is stored server-side, so the list is
  per-device and vanishes if site data is cleared.
- No Python test suite — the logic that matters runs in the browser, and
  `tests/harness.html` plus `scripts/verify_models.py` cover it from both sides.
