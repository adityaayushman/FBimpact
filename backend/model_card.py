"""A model card built from the loaded checkpoint, not typed in by hand.

Everything that describes *this* model (what it was trained on, which features it
reads, whether it has a phase head) is derived from the checkpoint's own recorded
config, so replacing the checkpoint cannot leave a stale description behind. The
card deliberately contains no measured results: those belong to the published
protocol, and a number copied into server code is wrong the day the model changes.

Pure functions over a plain dict - no torch - so they are unit-testable anywhere.
"""

from __future__ import annotations

# A checkpoint is "current" when it matches what the project's best configuration
# reads: the full feature set and the four-phase head. Anything else is still a
# valid model; it is labelled so a visitor does not mistake it for the latest one.
CURRENT_FEATURE_SET = "full"

INTENDED_USE = (
    "Research demonstration of pre-impact fall anticipation from 2D skeletons: "
    "scoring, per-frame imminence, joint-level evidence and a faithfulness test."
)

OUT_OF_SCOPE = (
    "Any clinical, diagnostic, monitoring or safety use. Making decisions about a "
    "real person's care. Use on children, on several people in one frame, or on "
    "footage whose frame rate or camera geometry differs greatly from the "
    "training data.",
)

LIMITS = (
    "Trained and evaluated on public datasets of falls acted by young volunteers; "
    "no recorded falls by older adults were used.",
    "Public datasets here do not label individual actors, so evaluation separates "
    "clips, not people.",
    "False alarms on normal activity are a central weakness; see the published "
    "evaluation for measured counts and the protocol that produced them.",
    "Input is 2D keypoints from a pose estimator; errors in pose (occlusion, "
    "low resolution) pass straight into the score.",
)


def _generation(info: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if info.get("feature_set") != CURRENT_FEATURE_SET:
        reasons.append(
            f"reads the '{info.get('feature_set', 'unknown')}' feature set "
            f"({info.get('in_channels', '?')} channels), not the full set"
        )
    if not info.get("has_phase_head"):
        reasons.append("has no four-phase head (binary imminence only)")
    return ("current" if not reasons else "earlier", reasons)


def build_card(info: dict, schema_version: int, disclaimer: str) -> dict:
    """The card for a loaded checkpoint's `info` dict."""
    generation, reasons = _generation(info)
    return {
        "schema_version": schema_version,
        "model": {
            "backbone": info.get("backbone"),
            "parameters": info.get("parameters"),
            "causal": info.get("causal"),
            "window_frames": info.get("window"),
            "feature_set": info.get("feature_set"),
            "input_channels": info.get("in_channels"),
            "phase_head": bool(info.get("has_phase_head")),
            "explanation": info.get("explain_method"),
            "trained_on": info.get("trained_on"),
            "epoch": info.get("epoch"),
        },
        "generation": generation,
        "generation_reasons": reasons,
        "intended_use": INTENDED_USE,
        "out_of_scope": list(OUT_OF_SCOPE),
        "limits": list(LIMITS),
        "evaluation": (
            "Measured results are published with their protocol on the project "
            "site and are not repeated here, so they cannot drift from the model."
        ),
        "disclaimer": disclaimer,
    }
