"""FastAPI service for the demo frontend.

Run locally:
    uvicorn backend.app:app --reload --port 8000

The API takes **skeletons, never pixels**. Pose estimation happens in the browser,
so video never leaves the user's device and the server has nothing identifiable
to store even in principle. That is the privacy claim in Section 19 expressed as
an architecture rather than a policy - there is no endpoint that accepts a frame.

Every error has one shape, `{"error": {"code", "message", "request_id", ...}}`,
with a stable machine-readable code. Tracebacks stay in the server log; the
client gets a request id it can quote.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import inference
from .validation import ApiError, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("api")

DISCLAIMER = (
    "Research prototype, NOT a medical device. The bundled checkpoint is trained on UR Fall, "
    "a public dataset of falls ACTED by young volunteers - not recorded elderly falls. Across "
    "5-fold cross-validation this variant warned on 24 of 30 falls but produced 19 false alarms "
    "in 5.8 minutes of normal activity; that is far above what any care setting would tolerate, "
    "and too little negative data to put a reliable per-hour figure on. Outputs carry no "
    "clinical meaning."
)

# 600 frames x 17 joints x 3 numbers is a few hundred kB of JSON, so 2 MB is
# generous for a legitimate request and small enough that a hostile one cannot
# make the 512 MB container allocate its way to an OOM before parsing finishes.
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(2 * 1024 * 1024)))

app = FastAPI(
    title="Pre-Impact Fall Anticipation API",
    description=(
        "Per-frame fall-imminence scoring with joint-level evidence and a "
        "deletion/insertion faithfulness test. Accepts skeletons only - never video.\n\n"
        + DISCLAIMER
    ),
    version="0.2.0",
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
    expose_headers=["X-Request-ID", "Retry-After"],
)
# Score streams and keypoint payloads are long lists of numbers and compress well.
app.add_middleware(GZipMiddleware, minimum_size=1000)

# One analysis at a time. The 512 MB container holds the torch runtime (403 MB)
# plus one request's activations; two concurrent requests would exceed it and the
# platform would kill the process, taking the in-flight request down with it. A
# fast, honest 503 is better than that.
_analysis_slot = threading.Lock()


# -- request plumbing ----------------------------------------------------------

@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign a request id and enforce the body-size limit before parsing."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        error = ApiError(413, "too_large",
                         f"request body is {int(declared)} bytes; the limit is {MAX_BODY_BYTES}",
                         limit=MAX_BODY_BYTES)
        response = JSONResponse(error.body(request_id), status_code=413)
    else:
        response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    return response


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    headers = {"Retry-After": str(exc.extra["retry_after"])} if "retry_after" in exc.extra else None
    return JSONResponse(exc.body(_request_id(request)), status_code=exc.status, headers=headers)


@app.exception_handler(RequestValidationError)
async def schema_error_handler(request: Request, exc: RequestValidationError):
    """Pydantic's own errors, reshaped so a client parses exactly one error format."""
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    error = ApiError(422, "bad_request", f"{where or 'request'}: {first.get('msg', 'invalid')}")
    return JSONResponse(error.body(_request_id(request)), status_code=422)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Anything unforeseen: log it in full, tell the client nothing internal."""
    rid = _request_id(request)
    if isinstance(exc, ValueError):
        # `inference` keeps its own defensive checks. `backend.validation` should
        # have caught anything reaching them, so this is a belt-and-braces mapping
        # to the documented shape rather than a path real traffic takes.
        logger.warning("late validation error, request %s: %s", rid, exc)
        return JSONResponse(ApiError(422, "bad_request", str(exc)).body(rid), status_code=422)
    logger.exception("unhandled error, request %s", rid)
    if isinstance(exc, MemoryError):
        error = ApiError(503, "out_of_memory", "the server ran out of memory; try a shorter clip")
        return JSONResponse(error.body(rid), status_code=503)
    error = ApiError(500, "internal_error", "an unexpected error occurred")
    return JSONResponse(error.body(rid), status_code=500)


# -- schemas -------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    # Deliberately untyped. A typed `list[list[list[float]]]` makes pydantic reject
    # `null`, which is exactly how a pose estimator reports a person it lost for a
    # few frames - the event this service exists for. Validation is done by
    # `backend.validation`, which has defined behaviour for every case.
    keypoints: list = Field(..., description="[T, 17, 3] of (x, y, confidence) in pixels; null = missing")
    fps: float = Field(30.0, description="frame rate; sets the units of lead time")
    impact_frame: int | None = Field(
        None, description="t*, if known. Lead time is undefined without it."
    )
    threshold: float | None = Field(None, gt=0, lt=1, description="overrides the checkpoint's tau")
    persistence: int | None = Field(None, ge=1, le=30, description="overrides k")
    explain: bool = True
    top_k: int = Field(3, ge=1, le=17)
    clip_id: str = Field("upload", max_length=120)


class FaithfulnessRequest(BaseModel):
    keypoints: list
    frame: int = Field(..., ge=0, description="the warning frame to test")
    fps: float = Field(30.0)
    impact_frame: int | None = None
    num_random: int = Field(3, ge=1, le=8, description="random control orderings to average")
    baseline: str = Field("zero", pattern="^(zero|neighbour)$")
    clip_id: str = Field("upload", max_length=120)


def _load():
    try:
        return inference.load_model()
    except FileNotFoundError as exc:
        raise ApiError(503, "model_unavailable", str(exc)) from exc


class _Slot:
    """Context manager for the single analysis slot; refuses instead of queueing."""

    def __enter__(self):
        if not _analysis_slot.acquire(blocking=False):
            raise ApiError(503, "busy", "another analysis is running; retry shortly", retry_after=3)
        return self

    def __exit__(self, *exc):
        _analysis_slot.release()
        return False


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
    loaded = _load()
    return {
        "status": "ok",
        "schema_version": inference.SCHEMA_VERSION,
        "model": loaded.info,
        "decision": {
            "threshold": loaded.decision.threshold,
            "persistence": loaded.decision.persistence,
            "refractory_frames": loaded.decision.refractory_frames,
        },
        "limits": {"max_frames": inference.MAX_FRAMES, "max_body_bytes": MAX_BODY_BYTES},
        "disclaimer": DISCLAIMER,
    }


@app.get("/ready")
def ready() -> JSONResponse:
    """Whether the model is loaded, without triggering a load.

    Separate from `/health` so a client can tell "the container is up" from "the
    container is up and warm", which on a free tier that sleeps are minutes apart.
    """
    loaded = inference.load_model.cache_info().currsize > 0
    return JSONResponse({"ready": loaded}, status_code=200 if loaded else 503)


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
        raise ApiError(404, "no_such_clip", f"no demo clip {clip_id!r}") from exc


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Score a skeleton sequence and describe every warning it produces."""
    loaded = _load()
    prepared = validate(
        request.keypoints, request.fps,
        min_frames=loaded.features.window, max_frames=inference.MAX_FRAMES,
        impact_frame=request.impact_frame,
    )

    base = {"degraded": prepared.degraded, "interpolated_frames": prepared.interpolated_frames,
            "confidence_clipped": prepared.clipped_confidence}

    if prepared.no_person:
        # Nobody in shot is a legitimate outcome, not a low score and not an error.
        # Returning scores here would present "no data" as "no fall".
        return {
            "schema_version": inference.SCHEMA_VERSION, "clip_id": request.clip_id,
            "no_person": True, "frames": int(prepared.keypoints.shape[0]), "fps": request.fps,
            "scores": [], "phases": None, "warnings": [], "impact_frame": request.impact_frame,
            **base,
        }

    with _Slot():
        result = inference.analyze(
            keypoints=prepared.keypoints,
            fps=request.fps,
            impact_frame=request.impact_frame,
            threshold=request.threshold,
            persistence=request.persistence,
            explain=request.explain,
            top_k=request.top_k,
            clip_id=request.clip_id,
        )
    return {**result, "no_person": False, **base}


@app.post("/faithfulness")
def faithfulness(request: FaithfulnessRequest) -> dict:
    """Deletion/insertion test for one warning (RQ2). Slow by nature - see inference.py."""
    loaded = _load()
    prepared = validate(
        request.keypoints, request.fps,
        min_frames=loaded.features.window, max_frames=inference.MAX_FRAMES,
        impact_frame=request.impact_frame,
    )
    if prepared.no_person:
        raise ApiError(422, "no_person", "no person was detected in this clip, so there is no warning to test")
    if request.frame >= prepared.keypoints.shape[0]:
        raise ApiError(422, "bad_frame", f"frame {request.frame} is outside a "
                                         f"{prepared.keypoints.shape[0]}-frame clip")

    with _Slot():
        return inference.faithfulness(
            keypoints=prepared.keypoints,
            frame=request.frame,
            fps=request.fps,
            impact_frame=request.impact_frame,
            num_random=request.num_random,
            baseline=request.baseline,
            clip_id=request.clip_id,
        )


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
            "loaded %s (%d params, window %d, phase head: %s) on %s",
            loaded.info["backbone"], loaded.info["parameters"],
            loaded.info["window"], loaded.info["has_phase_head"], loaded.device,
        )
        logger.info("%d demo clips bundled", len(inference.demo_clips()))
    except FileNotFoundError as exc:
        logger.error("no checkpoint: %s", exc)
