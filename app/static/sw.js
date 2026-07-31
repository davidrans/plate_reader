// Caches the app shell and — the part that actually matters — the two ~10 MB
// ONNX models, so they're downloaded once instead of on every launch.
//
// Bump CACHE when any cached asset changes; the activate handler deletes older
// caches. The models are content-addressed by filename, so swapping a model
// means a new filename and a cache bump.
const CACHE = "plate-reader-v2";

const ASSETS = [
  "/",
  "/manifest.json",
  "/static/app.js",
  "/static/alpr.js",
  "/static/tracker.js",
  "/static/recent.js",
  "/static/theme.js",
  "/static/models/yolo-v9-t-384-license-plates-end2end.onnx",
  "/static/models/cct_xs_v2_global.onnx",
];

self.addEventListener("install", (event) => {
  // Not addAll: one failed asset would reject the whole install and leave the
  // app uncached. Cache what we can and let the rest fall through to network.
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) =>
        Promise.all(
          ASSETS.map((url) =>
            cache.add(url).catch((err) => console.warn("[sw] skipped", url, err)),
          ),
        ),
      )
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  // Only handle our own origin — the Bootstrap/ORT CDN requests have their own
  // caching and opaque responses aren't worth storing here.
  if (url.origin !== self.location.origin) return;

  // Cache-first: the models are large and immutable, and the shell is small.
  event.respondWith(
    caches.match(request).then(
      (hit) =>
        hit ||
        fetch(request).then((response) => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(CACHE).then((cache) => cache.put(request, copy));
          }
          return response;
        }),
    ),
  );
});
