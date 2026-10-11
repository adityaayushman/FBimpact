"""The HTTP layer, against the real bundled checkpoint.

Imports torch (through `backend.inference`), so unlike `test_api_validation.py`
this file must NOT be run alongside a training job on a memory-constrained
machine. It is the acceptance check for the response contract; the 39 validation
cases cover the input rules without needing the model.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module  # noqa: E402
from backend import inference  # noqa: E402

pytestmark = pytest.mark.skipif(
    not inference.DEFAULT_CHECKPOINT.exists(), reason="no bundled checkpoint to serve"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def demo(client, kind="fall"):
    listing = client.get("/clips").json()["clips"]
    chosen = next(c for c in listing if c["label"] == kind)
    return client.get(f"/clips/{chosen['clip_id']}").json()


def body(payload, **extra):
    return {"keypoints": payload["keypoints"], "fps": payload["fps"],
            "impact_frame": payload.get("impact_frame"), "explain": False, **extra}


# -- response contract ------------------------------------------------------

def test_health_reports_the_phase_head_flag_and_schema_version(client):
    h = client.get("/health").json()
    assert h["schema_version"] == inference.SCHEMA_VERSION
    assert isinstance(h["model"]["has_phase_head"], bool)


def test_analyze_returns_a_versioned_response_with_phases_matching_the_checkpoint(client):
    clip = demo(client)
    r = client.post("/analyze", json=body(clip)).json()
    assert r["schema_version"] == inference.SCHEMA_VERSION
    has_head = client.get("/health").json()["model"]["has_phase_head"]
    if has_head:
        phases = r["phases"]
        assert phases["names"] == ["normal", "imminent", "falling", "grounded"]
        probs = np.array(phases["probs"])
        assert probs.shape == (r["frames"], 4)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
    else:
        assert r["phases"] is None, "no phase head must mean null, never zeros or a derived guess"


def test_scores_cover_every_frame(client):
    clip = demo(client)
    r = client.post("/analyze", json=body(clip)).json()
    assert len(r["scores"]) == r["frames"] == len(clip["keypoints"])


# -- the one error shape ----------------------------------------------------

def assert_error(response, status, code):
    assert response.status_code == status, response.text
    err = response.json()["error"]
    assert err["code"] == code
    assert err["request_id"] and response.headers["x-request-id"] == err["request_id"]
    assert "Traceback" not in response.text
    return err


def test_bad_shape_uses_the_standard_error_body(client):
    assert_error(client.post("/analyze", json={"keypoints": [[[0, 0, 1]] * 5] * 40}), 422, "bad_shape")


def test_too_short_is_refused_not_padded(client):
    clip = demo(client)
    r = client.post("/analyze", json=body({**clip, "keypoints": clip["keypoints"][:5]}))
    assert_error(r, 422, "too_short")


def test_schema_errors_are_reshaped_into_the_same_format(client):
    r = client.post("/analyze", json={"keypoints": demo(client)["keypoints"], "top_k": 99})
    assert_error(r, 422, "bad_request")


def test_oversized_body_is_413_before_parsing(client):
    huge = "x" * (app_module.MAX_BODY_BYTES + 10)
    r = client.post("/analyze", content=huge, headers={"content-type": "application/json"})
    assert_error(r, 413, "too_large")


def test_unknown_demo_clip_is_a_standard_404(client):
    assert_error(client.get("/clips/does-not-exist"), 404, "no_such_clip")


# -- dropout ----------------------------------------------------------------

def test_brief_dropout_is_bridged_flagged_and_still_scored(client):
    clip = demo(client)
    kp = [list(map(list, f)) for f in clip["keypoints"]]
    for f in range(40, 45):
        kp[f] = [[None, None, None] for _ in range(17)]
    r = client.post("/analyze", json=body({**clip, "keypoints": kp})).json()
    assert r["degraded"] is True
    assert r["interpolated_frames"] == [40, 41, 42, 43, 44]
    assert len(r["scores"]) == len(kp)


def test_no_person_is_a_success_with_no_scores_not_a_low_score(client):
    clip = demo(client)
    kp = [[[None, None, None] for _ in range(17)] for _ in clip["keypoints"]]
    r = client.post("/analyze", json=body({**clip, "keypoints": kp}))
    assert r.status_code == 200
    out = r.json()
    assert out["no_person"] is True and out["scores"] == [] and out["warnings"] == []


def test_excess_dropout_is_refused_with_the_numbers(client):
    clip = demo(client)
    kp = [list(map(list, f)) for f in clip["keypoints"]]
    for f in range(20, 40):
        kp[f] = [[None, None, None] for _ in range(17)]
    err = assert_error(client.post("/analyze", json=body({**clip, "keypoints": kp})), 422, "too_much_missing")
    assert err["longest_gap"] == 20


# -- concurrency ------------------------------------------------------------

def test_a_busy_slot_returns_503_with_retry_after_and_recovers(client):
    clip = demo(client)
    assert app_module._analysis_slot.acquire(blocking=False)
    try:
        r = client.post("/analyze", json=body(clip))
        assert_error(r, 503, "busy")
        assert r.headers["retry-after"] == "3"
    finally:
        app_module._analysis_slot.release()
    assert client.post("/analyze", json=body(clip)).status_code == 200, (
        "the next valid request must succeed once the slot is free"
    )


def test_ready_reflects_a_loaded_model(client):
    assert client.get("/ready").json()["ready"] is True


def test_the_slot_is_released_after_an_error(client):
    """A failed request must not leave the service permanently busy."""
    client.post("/analyze", json={"keypoints": []})
    assert app_module._analysis_slot.acquire(blocking=False)
    app_module._analysis_slot.release()
