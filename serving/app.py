"""
FastAPI wrapper around a model bundle.

Loads the bundle once at process startup (lifespan) and serves it from memory.
Two backends, chosen automatically from the file at MODEL_BUNDLE_PATH:
  * early-fusion RATF ensemble  (file has model_class == "RATF_M6_Granular_V2")
  * late-fusion bundle          (everything else; the original serving.bundle.LoadedBundle)
Rolling back is therefore just pointing MODEL_BUNDLE_PATH at the old bundle.

Run locally:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

import io
import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serving")

BUNDLE_PATH = os.environ.get("MODEL_BUNDLE_PATH", "models/bundles/latest_clip_b32_clip.pt")
RATF_MODEL_CLASS = "RATF_M6_Granular_V2"

_bundle_holder = {}


def _is_ratf_file(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return isinstance(blob, dict) and blob.get("model_class") == RATF_MODEL_CLASS
    finally:
        del blob


def load_backend(path):
    if _is_ratf_file(path):
        from serving.ratf_bundle import LoadedRatfBundle
        return LoadedRatfBundle(path)
    from serving.bundle import LoadedBundle
    return LoadedBundle(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model bundle from %s", BUNDLE_PATH)
    _bundle_holder["bundle"] = load_backend(BUNDLE_PATH)
    logger.info(
        "Bundle loaded: version=%s train_end=%s",
        _bundle_holder["bundle"].bundle["version"],
        _bundle_holder["bundle"].bundle["train_end"],
    )
    yield
    _bundle_holder.clear()


app = FastAPI(title="yt-performance-predictor", lifespan=lifespan)


def get_bundle():
    bundle = _bundle_holder.get("bundle")
    if bundle is None:
        # Shouldn't happen outside of tests that bypass lifespan.
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


@app.get("/health")
def health():
    bundle = get_bundle()
    return {
        "status": "ok",
        "model_version": bundle.bundle["version"],
        "train_end": bundle.bundle["train_end"],
    }


@app.post("/predict")
async def predict(
    thumbnail: UploadFile = File(...),
    title: str = Form(...),
    trailing_avg_views: float = Form(...),
    duration_seconds: float = Form(...),
    genre: str = Form(...),
    # Required only by the late-fusion bundle; the early-fusion model does not use it.
    subscriber_count_at_upload: float | None = Form(None),
):
    bundle = get_bundle()

    if getattr(bundle, "uses_subscribers", True) and subscriber_count_at_upload is None:
        raise HTTPException(status_code=422, detail="subscriber_count_at_upload is required by this model")

    raw = await thumbnail.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now, not lazily inside bundle.predict()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="thumbnail is not a readable image")

    try:
        result = bundle.predict(
            thumbnail=image,
            title=title,
            subscriber_count_at_upload=subscriber_count_at_upload,
            trailing_avg_views=trailing_avg_views,
            duration_seconds=duration_seconds,
            genre=genre,
        )
    except ValueError as e:
        # build_tabular_row raises ValueError for bundle/request mismatches
        raise HTTPException(status_code=400, detail=str(e))

    return result
