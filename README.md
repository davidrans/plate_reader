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

### There is no US-state identification — the photo covers it instead

The region head classifies **country**, not state: `"United States"` is a
single one of its 66 classes. Nothing in this model stack can tell Colorado
from Wyoming, and the whole upstream model zoo is country-level (Argentine /
European / global). `"United States"` is hidden from rows because — with the
US-only filter always applied, it is true of every row and carries no information.

Adding real state classification would mean a **separate model trained on US
plate designs** (colours, logos, fonts) rather than their text: a dataset and
training project, not a configuration change, and no off-the-shelf
permissively licensed model was found for it.

**So each row stores a cropped photo of the plate instead**, and you read the
state off it directly. This is strictly more useful than a classifier would
be: it also makes a wrong OCR result obvious at a glance. Tap the thumbnail to
enlarge it (`image-rendering: pixelated`, since these crops get upscaled and
smoothing makes small text worse).

- `captureCrop()` in `alpr.js` preserves aspect ratio, unlike the OCR input
  which is squashed to 128×64, and **never upscales** — a distant plate stays
  small rather than becoming blur. Capped at 220 px wide.
- The crop kept is the one from the **highest-confidence** frame, not the most
  recent — the pipeline is already voting across frames, so it may as well keep
  the clearest view. The quality bar lives on the row (`imageConfidence`), not
  just the track: a track's own bar resets each sighting, so without it a later
  blurry pass would overwrite a good photo.
- Encoding is deferred: the tracker holds the winning crop as a canvas and
  `takeCropIfChanged()` hands it over only when it improves, so the JPEG encode
  doesn't run every frame.
- ~5 KB per photo at quality 0.7, so a full 50-row list is ~250 KB — well
  inside `localStorage`'s ~5 MB. If the quota is hit anyway, `saveRecent()`
  drops the **oldest photos** (keeping their text) until it fits, rather than
  losing the list.

### Thresholds

Defaults in `alpr.js` (`DEFAULTS`), tuned against the real photos in
`tests/assets/` rather than guessed:

| Setting | Value | Why |
|---|---|---|
| `detThreshold` | 0.5 | Real plates scored 0.87–0.998; junk rows scored <0.4. |
| `ocrThreshold` | 0.6 | A plate too small to read scored 0.21; readable ones 1.000. Raised from 0.4 alongside voting — misread characters carry low confidence, so this drops most of them before they can vote. **Lower this first if distant plates stop registering.** |
| `minPlateLength` | 4 | Junk boxes OCR to short garbage. |
| `allowedRegions` | `["United States"]` | Used as a **misread detector**, not a nationality check — a garbled read of a US plate is usually classified confidently as another country. Set to `null` to accept every region, or add `"Unknown"` if real plates get skipped. |

OCR confidence is the **minimum** across character slots, not the mean — one
unreadable character should drag the score down rather than be averaged away
by nine confident ones. There's deliberately no plate-format regex: it would
wrongly reject vanity and non-US plates, and the model's alphabet already
excludes punctuation.

## Temporal voting (`app/static/tracker.js`, `app/static/recent.js`)

A single frame's OCR is unreliable. Held on one plate the pipeline produces
10–20 reads per couple of seconds — mostly correct, occasionally off by a
character. Recording each frame directly filled the list with near-duplicates,
so reads are pooled and voted on instead.

- A **track** is one physical plate across frames. A detection joins a track if
  its box overlaps the track's last box (IoU ≥ 0.3) **or** its text is within
  ~2 edits. Each signal covers the other's failure: overlap survives a misread
  that changes the text, text matching survives fast panning that breaks the
  overlap.
- Votes are **weighted by confidence**, so a shaky read can't outvote confident
  ones. The winner is the highest total weight.
- A track is committed to the list after `minVotes` (3) reads and **corrects
  itself in place** if the leader changes while the plate is still in frame.
  It's finalised once unmatched for `trackTimeoutMs` (1 s).
- **One detection per track per frame.** Without this, two different cars whose
  plates read alike would collapse into one track.
- Character confusions OCR actually makes (`0OQD`, `1IL`, `5S`, `8B`, `2Z`,
  `6G`) cost half a normal edit, so `ABCI23` merges into `ABC123` while a real
  neighbour like `ABD123` stays separate.

The list keeps **one row per plate** with a sighting count; re-seeing a plate
bumps `count` and `lastSeen` instead of adding a row. The count increments once
per *track*, not per frame.

**Known trade-off:** two genuinely different plates differing only by a
confusable character can merge into one row. Box overlap prevents this while
both are on screen together, but not across separate sightings. Widen
`MERGE_DISTANCE` in `recent.js` to `0` to disable cross-sighting merging.

## Caching (`app/static/sw.js`)

Two caches, because the assets have opposite lifetimes:

| Cache | Contents | Strategy | Bump when |
|---|---|---|---|
| `plate-reader-models-v1` | the two `.onnx` files (~10 MB) | cache-first | the model files change |
| `plate-reader-shell-v3` | HTML, JS, manifest (a few KB) | **network-first** | never needs bumping |

They were originally one cache-first cache, which had two bugs:

1. Shipping new code required remembering to bump the version, or browsers kept
   serving the old copy. The plate-photo release went out stale this way.
2. `cache.add()` always hits the network, so **every service worker update
   re-downloaded both models** — 10 MB to ship a few KB of changed JS.

Network-first for the shell means a deploy can't go stale (the cache is only an
offline fallback), and the separate model cache survives code deploys. The
install step skips assets already cached, so an update re-fetches nothing it
already has.

The status line reports `from cache` vs `downloaded` plus a load time, so a
suspected re-download can be confirmed instead of inferred.

### iOS caveats

- **Camera permission is re-prompted on most launches** of an installed PWA.
  Each launch is a new browsing session as far as WebKit is concerned, and
  there is no web-side fix — it's one of the native-vs-PWA gaps listed at the
  top.
- iOS can evict Cache Storage under storage pressure, so a re-download is
  possible even when everything here is correct. The status line is how you
  tell the difference.

## Commands

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000

# Re-check the ONNX contract from Python (dev deps only; the app never does this)
uv run python scripts/verify_models.py tests/assets/zaz.jpg tests/assets/sgp.jpg
```

### Test harness

`tests/harness.html` covers both halves — no camera needed. With the server
running, open <http://localhost:8000/tests/harness.html>.

1. **Unit checks** for `tracker.js` and `recent.js`, driven by synthetic frame
   sequences with injected timestamps (fully deterministic, no models loaded).
   These encode the duplicate bug directly: 15 frames of one plate with a
   misread in the middle must yield exactly one row with `count === 1`.
2. **Image cases** running the real `alpr.js` pipeline against the checked-in
   photos, compared against `scripts/verify_models.py`'s output.

Current result on desktop Chrome (WASM fallback, no WebGPU):

```
--- tracker: 20/20 checks passed ---
--- recent:  43/43 cumulative checks passed ---
/tests/assets/zaz.jpg  (250x187)  213 ms
  PASS text=AE6133CT det=0.916 ocr=0.999 region=Ukraine
/tests/assets/sgp.jpg  (899x226)  157 ms
  PASS text=SDN7484U det=0.998 ocr=1.000 region=Singapore
=== images: 2/2 matched the Python reference ===
=== unit checks: 43/43 passed ===
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
