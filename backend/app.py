"""FastAPI service for the demo frontend.

Run locally:
    uvicorn backend.app:app --reload --port 8000

The API takes **skeletons, never pixels**. Pose estimation happens in the
browser, so video never leaves the user's device and the server has nothing
identifiable to store even in principle. That is the privacy claim in Section 19
expressed as an architecture rather than a policy - there is no endpoint that
accepts a frame.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from . import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("api")

DISCLAIMER = (
    "Research prototype, NOT a medical device. The bundled checkpoint is trained on a "
    "synthetic fixture, not on real fall data, and its outputs carry no clinical meaning."
)

app = FastAPI(
    title="Pre-Impact Fall Anticipation API",
    description=(
        "Per-frame fall-imminence scoring with joint-level evidence and a "
        "deletion/insertion faithfulness test. Accepts skeletons only - never video.\n\n"
        + DISCLAIMER
    ),
    version="0.1.0",
)

# The Vercel deployment plus local development. Set ALLOWED_ORIGINS on Render to
# a comma-separated list to lock this down to the real frontend domain.
_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()] or ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app" if not _origins else None,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Score streams and keypoint payloads are long lists of numbers and compress well.
app.add_middleware(GZipMiddleware, minimum_size=1000)


# -- schemas -------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    keypoints: list[list[list[float]]] = Field(
        ..., description="[T, 17, 3] of (x, y, confidence) in pixels"
    )
    fps: float = Field(30.0, gt=0, le=240, description="frame rate; sets the units of lead time")
    impact_frame: int | None = Field(
        None, description="t*, if known. Lead time is undefined without it."
    )
    threshold: float | None = Field(None, gt=0, lt=1, description="overrides the checkpoint's tau")
    persistence: int | None = Field(None, ge=1, le=30, description="overrides k")
    explain: bool = True
    top_k: int = Field(3, ge=1, le=17)
    clip_id: str = "upload"


class FaithfulnessRequest(BaseModel):
    keypoints: list[list[list[float]]]
    frame: int = Field(..., ge=0, description="the warning frame to test")
    fps: float = Field(30.0, gt=0, le=240)
    impact_frame: int | None = None
    num_random: int = Field(3, ge=1, le=8, description="random control orderings to average")
    baseline: str = Field("zero", pattern="^(zero|neighbour)$")
    clip_id: str = "upload"


# -- routes --------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "pre-impact fall anticipation",
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health() -> dict:
    """Liveness plus the model's identity, so the client can show what it is talking to."""
    try:
        loaded = inference.load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model": loaded.info,
        "decision": {
            "threshold": loaded.decision.threshold,
            "persistence": loaded.decision.persistence,
            "refractory_frames": loaded.decision.refractory_frames,
        },
        "limits": {"max_frames": inference.MAX_FRAMES},
        "disclaimer": DISCLAIMER,
    }


@app.get("/skeleton")
def skeleton() -> dict:
    """Joint names and bones, so the client draws the graph the model actually uses."""
    return inference.skeleton_spec()


@app.get("/clips")
def clips() -> dict:
    """Bundled demo clips - falls and hard-negative ADLs."""
    return {"clips": inference.demo_clip_summaries()}


@app.get("/clips/{clip_id}")
def clip(clip_id: str) -> dict:
    try:
        return inference.demo_clip_payload(clip_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no demo clip {clip_id!r}") from exc


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Score a skeleton sequence and describe every warning it produces."""
    try:
        return inference.analyze(
            keypoints=np.asarray(request.keypoints, dtype=np.float32),
            fps=request.fps,
            impact_frame=request.impact_frame,
            threshold=request.threshold,
            persistence=request.persistence,
            explain=request.explain,
            top_k=request.top_k,
            clip_id=request.clip_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/faithfulness")
def faithfulness(request: FaithfulnessRequest) -> dict:
    """Deletion/insertion test for one warning (RQ2). Slow by nature - see inference.py."""
    try:
        return inference.faithfulness(
            keypoints=np.asarray(request.keypoints, dtype=np.float32),
            frame=request.frame,
            fps=request.fps,
            impact_frame=request.impact_frame,
            num_random=request.num_random,
            baseline=request.baseline,
            clip_id=request.clip_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.on_event("startup")
def warm_up() -> None:
    """Load the checkpoint at boot.

    Render's free tier spins a container down after inactivity, so the next
    request pays a cold start. Doing the load here moves that cost off the first
    user request and into the platform's own start-up window.
    """
    try:
        loaded = inference.load_model()
        logger.info(
            "loaded %s (%d params, window %d) on %s",
            loaded.info["backbone"], loaded.info["parameters"],
            loaded.info["window"], loaded.device,
        )
        logger.info("%d demo clips bundled", len(inference.demo_clips()))
    except FileNotFoundError as exc:
        logger.error("no checkpoint: %s", exc)
