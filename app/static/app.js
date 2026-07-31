import { DEFAULTS, activeBackend, loadModels, readPlates } from "./alpr.js";
import { PlateTracker } from "./tracker.js";
import { RecentPlates, normalizeStored } from "./recent.js";

const RECENT_KEY = "plate-reader:recent";

// Individual frames are never recorded directly: tracker.js pools each plate's
// reads and votes on the text, and recent.js keeps one row per plate. A misread
// in one frame therefore loses to the correct reading in the others.
const tracker = new PlateTracker();
const recent = new RecentPlates(loadRecent());

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const startButton = document.getElementById("start-button");
const recentList = document.getElementById("recent-list");
const emptyHint = document.getElementById("empty-hint");
const clearButton = document.getElementById("clear-button");
const fpsEl = document.getElementById("fps");
const usOnlyToggle = document.getElementById("opt-us-only");

// Persisted so the choice survives a reload — it's a tuning knob you'd want to
// leave where you set it, same as the theme.
const US_ONLY_KEY = "plate-reader:us-only";
usOnlyToggle.checked = localStorage.getItem(US_ONLY_KEY) !== "false";
usOnlyToggle.addEventListener("change", () => {
  localStorage.setItem(US_ONLY_KEY, String(usOnlyToggle.checked));
  // Reads already banked under the old setting shouldn't keep voting.
  tracker.reset();
});

function currentOptions() {
  return { ...DEFAULTS, allowedRegions: usOnlyToggle.checked ? DEFAULTS.allowedRegions : null };
}

let running = false;
let stream = null;

// --- recent plates (client-side only; nothing is sent anywhere) ------------

function loadRecent() {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    return normalizeStored(raw ? JSON.parse(raw) : []);
  } catch {
    return [];
  }
}

function saveRecent() {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(recent.entries));
  } catch (err) {
    console.warn("[app] could not persist recent plates:", err);
  }
}

function relativeTime(ts) {
  const secs = Math.round((Date.now() - ts) / 1000);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

function renderRecent() {
  emptyHint.hidden = recent.entries.length > 0;
  clearButton.hidden = recent.entries.length === 0;
  recentList.replaceChildren(
    ...recent.entries.map((entry) => {
      const li = document.createElement("li");
      li.className =
        "list-group-item d-flex justify-content-between align-items-center gap-2";

      const left = document.createElement("div");
      const plate = document.createElement("div");
      plate.className = "font-monospace fs-5 fw-semibold";
      plate.textContent = entry.text;
      const meta = document.createElement("div");
      meta.className = "text-body-secondary small";
      meta.textContent = [
        entry.count > 1 ? `seen ${entry.count}×` : "",
        relativeTime(entry.lastSeen),
        entry.region,
        `${Math.round(entry.confidence * 100)}%`,
      ]
        .filter(Boolean)
        .join(" · ");
      left.append(plate, meta);

      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "btn btn-sm btn-outline-secondary";
      copy.textContent = "Copy";
      copy.addEventListener("click", () => copyPlate(entry.text));

      li.append(left, copy);
      return li;
    }),
  );
}

function copyPlate(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => showToast(`Copied ${text}`))
    .catch(() => showToast("Couldn't copy — select it manually"));
}

function showToast(message) {
  document.getElementById("copy-toast-body").textContent = message;
  bootstrap.Toast.getOrCreateInstance(document.getElementById("copy-toast"), {
    delay: 1800,
  }).show();
}

// The list logic itself lives in recent.js (DOM-free, unit tested in the
// harness) — this just persists and repaints after each write.
function upsertPlate(confirmedTrack) {
  recent.upsert(confirmedTrack);
  saveRecent();
  renderRecent();
}

clearButton.addEventListener("click", () => {
  recent.clear();
  // Release row ownership too, or a plate still in frame would keep updating
  // the row it had before the list was cleared.
  tracker.reset();
  saveRecent();
  renderRecent();
});

// --- camera + capture loop -------------------------------------------------

// Draws the live tracks rather than the raw per-frame reads: the voted leader
// is stable, whereas a single frame's text flickers through misreads. Tracks
// still gathering votes are drawn dimmer, so it's visible that a plate has
// been spotted before it's confident enough to record.
function drawOverlay(live) {
  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.lineWidth = Math.max(2, overlay.width / 250);
  overlayCtx.font = `${Math.max(16, overlay.width / 28)}px system-ui, sans-serif`;
  overlayCtx.textBaseline = "bottom";

  for (const { track, text, votes } of live) {
    const confirmed = votes >= tracker.opts.minVotes;
    overlayCtx.strokeStyle = confirmed ? "#20c997" : "#ffc107";
    overlayCtx.fillStyle = confirmed ? "#20c997" : "#ffc107";
    const { box } = track;
    overlayCtx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    overlayCtx.fillText(text, box.x1, Math.max(24, box.y1 - 4));
  }
}

async function startCamera() {
  // Rear camera, and a request for a reasonably high resolution — plates are
  // small in frame, and the detector squashes to 384px, so starting from a
  // larger source meaningfully helps readability at distance.
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  // Drop in-flight tracks so restarting doesn't resume voting on plates that
  // are no longer in view — and so the next sighting counts as a new one.
  tracker.reset();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  startButton.textContent = "Start scanning";
  startButton.classList.replace("btn-danger", "btn-primary");
  fpsEl.textContent = "";
}

// Self-scheduling rather than setInterval: each pass waits for the previous
// inference to finish, so the loop naturally settles at whatever rate the
// device sustains instead of queueing work faster than it can be processed.
async function loop() {
  let lastFrame = performance.now();
  let smoothed = 0;

  while (running) {
    try {
      const results = await readPlates(video, currentOptions());
      const { live, confirmed } = tracker.update(results);
      drawOverlay(live);
      for (const t of confirmed) upsertPlate(t);
    } catch (err) {
      console.error("[app] inference failed:", err);
      setStatus(`Inference error: ${err.message}`, "danger");
      stopCamera();
      return;
    }

    const now = performance.now();
    const ms = now - lastFrame;
    lastFrame = now;
    smoothed = smoothed ? smoothed * 0.8 + ms * 0.2 : ms;
    fpsEl.textContent = `${Math.round(smoothed)} ms/frame · ${(1000 / smoothed).toFixed(1)} fps · ${activeBackend()}`;

    // Yield to the event loop so the UI stays responsive between frames.
    await new Promise((r) => setTimeout(r, 0));
  }
}

function setStatus(message, variant = "secondary") {
  statusEl.className = `alert alert-${variant} py-2`;
  statusEl.textContent = message;
  statusEl.hidden = false;
}

startButton.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    return;
  }

  startButton.disabled = true;
  try {
    if (!activeBackend()) {
      setStatus("Loading models (~10 MB, cached after first load)...");
      const backend = await loadModels(setStatus);
      setStatus(`Models ready — running on ${backend}.`, "success");
    }
    await startCamera();
    running = true;
    startButton.textContent = "Stop";
    startButton.classList.replace("btn-primary", "btn-danger");
    statusEl.hidden = true;
    loop();
  } catch (err) {
    console.error(err);
    // The overwhelmingly common cause on iOS is a non-HTTPS origin, which
    // silently denies camera access — call that out rather than showing a
    // bare NotAllowedError.
    const hint = window.isSecureContext
      ? "Check that camera permission is granted."
      : "This page must be served over HTTPS for camera access.";
    setStatus(`${err.message}. ${hint}`, "danger");
  } finally {
    startButton.disabled = false;
  }
});

renderRecent();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch((err) => {
    console.warn("[app] service worker registration failed:", err);
  });
}
