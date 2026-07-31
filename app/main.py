from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).parent
STATIC = BASE / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# Same convention as court_date_api/crs_search_api. Cheaper for Coolify's
# container health check than rendering the full page at "/".
@app.get("/health")
def health():
    return {"status": "ok"}


# A service worker can only control pages at or below its own path, so it has
# to be served from the root rather than /static. Same for the manifest, which
# is fetched relative to the document.
@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC / "sw.js", media_type="application/javascript")


@app.get("/manifest.json")
def manifest():
    return FileResponse(STATIC / "manifest.json", media_type="application/manifest+json")


# Dev only: serves tests/harness.html and its sample images so the JS pipeline
# can be checked against scripts/verify_models.py without a camera. The tests
# directory is excluded from the Docker image, so this mount simply doesn't
# exist in a deployed container.
TESTS = BASE.parent / "tests"
if TESTS.is_dir():
    app.mount("/tests", StaticFiles(directory=TESTS, html=True), name="tests")
