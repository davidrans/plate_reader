import { DEFAULTS, activeBackend, loadModels, readPlates } from "./alpr.js";

const RECENT_KEY = "plate-reader:recent";
const RECENT_LIMIT = 50;
// Holding the camera on one plate re-reads it many times a second. Suppress a
// repeat of the same text within this window so the list doesn't fill up with
// one plate — but still allow it later, since seeing the same car twice on
// different occasions is legitimately worth recording.
const DUPLICATE_WINDOW_MS = 10000;

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const startButton = document.getElementById("start-button");
const recentList = document.getElementById("recent-list");
const emptyHint = document.getElementById("empty-hint");
const clearButton = document.getElementById("clear-button");
const fpsEl = document.getElementById("fps");

let running = false;
let stream = null;
let recent = loadRecent();

// --- recent plates (client-side only; nothing is sent anywhere) ------------

function loadRecent() {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveRecent() {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(recent));
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
  emptyHint.hidden = recent.length > 0;
  clearButton.hidden = recent.length === 0;
  recentList.replaceChildren(
    ...recent.map((entry) => {
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
        relativeTime(entry.ts),
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

function recordPlate({ text, confidence, region }) {
  const now = Date.now();
  const last = recent[0];
  if (last && last.text === text && now - last.ts < DUPLICATE_WINDOW_MS) {
    last.ts = now;
    last.confidence = Math.max(last.confidence, confidence);
    saveRecent();
    renderRecent();
    return;
  }
  recent.unshift({
    text,
    confidence,
    region: region?.name && region.name !== "Unknown" ? region.name : "",
    ts: now,
  });
  recent = recent.slice(0, RECENT_LIMIT);
  saveRecent();
  renderRecent();
}

clearButton.addEventListener("click", () => {
  recent = [];
  saveRecent();
  renderRecent();
});

// --- camera + capture loop -------------------------------------------------

function drawOverlay(results) {
  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.lineWidth = Math.max(2, overlay.width / 250);
  overlayCtx.strokeStyle = "#20c997";
  overlayCtx.fillStyle = "#20c997";
  overlayCtx.font = `${Math.max(16, overlay.width / 28)}px system-ui, sans-serif`;
  overlayCtx.textBaseline = "bottom";

  for (const { text, box } of results) {
    overlayCtx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    overlayCtx.fillText(text, box.x1, Math.max(overlayCtx.font.length, box.y1 - 4));
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
      const results = await readPlates(video, DEFAULTS);
      drawOverlay(results);
      for (const r of results) recordPlate(r);
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
