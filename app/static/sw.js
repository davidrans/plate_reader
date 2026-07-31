// Two caches, because the models and the app shell have opposite lifetimes and
// sharing one cache made them fight:
//
//   MODEL_CACHE — ~10 MB, effectively immutable (the model version is part of
//     the filename, so a different model is a different URL). Must survive
//     every code deploy: re-downloading 10 MB on a phone to ship a few KB of
//     changed JS is unacceptable. Only bump this if the model files change.
//   SHELL_CACHE — a few KB, changes every deploy. Served network-first, so it
//     is only an offline fallback and cannot go stale.
//
// This replaces a single cache-first cache, which had two bugs: shipping new
// code required remembering to bump the version (the plate-photo release went
// out stale because I forgot), and because `cache.add()` always hits the
// network, *every* service worker update re-downloaded both models.
const MODEL_CACHE = "plate-reader-models-v1";
const SHELL_CACHE = "plate-reader-shell-v3";

const MODELS = [
  "/static/models/yolo-v9-t-384-license-plates-end2end.onnx",
  "/static/models/cct_xs_v2_global.onnx",
];

const SHELL = [
  "/",
  "/manifest.json",
  "/static/app.js",
  "/static/alpr.js",
  "/static/tracker.js",
  "/static/recent.js",
  "/static/theme.js",
];

// `cache.add()` always fetches, so anything already stored must be skipped or
// each update would re-download it.
async function precache(cacheName, urls) {
  const cache = await caches.open(cacheName);
  await Promise.all(
    urls.map(async (url) => {
      if (await cache.match(url)) return;
      await cache.add(url).catch((err) => console.warn("[sw] skipped", url, err));
    }),
  );
}

self.addEventListener("install", (event) => {
  // Individually, not addAll: one failed asset shouldn't reject the whole
  // install and leave the app uncached.
  event.waitUntil(
    Promise.all([precache(MODEL_CACHE, MODELS), precache(SHELL_CACHE, SHELL)]).then(() =>
      self.skipWaiting(),
    ),
  );
});

self.addEventListener("activate", (event) => {
  const keep = new Set([MODEL_CACHE, SHELL_CACHE]);
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => !keep.has(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function cachePut(cacheName, request, response) {
  if (!response.ok) return response;
  const copy = response.clone();
  caches.open(cacheName).then((cache) => cache.put(request, copy));
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  // Only handle our own origin — the Bootstrap/ORT CDN requests have their own
  // caching and opaque responses aren't worth storing here.
  if (url.origin !== self.location.origin) return;

  // Models: cache-first, and never re-fetched once stored.
  if (url.pathname.endsWith(".onnx")) {
    event.respondWith(
      caches
        .match(request)
        .then((hit) => hit || fetch(request).then((r) => cachePut(MODEL_CACHE, request, r))),
    );
    return;
  }

  // Everything else: network-first. Cache-first here meant a deploy could ship
  // new code that browsers never ran because the old copy was still cached.
  // These files are a few KB, so the network round-trip is cheap, and the
  // cache remains as the offline fallback.
  event.respondWith(
    fetch(request)
      .then((r) => cachePut(SHELL_CACHE, request, r))
      .catch(() => caches.match(request)),
  );
});
