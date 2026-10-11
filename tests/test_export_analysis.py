"""The verdict -> site-wording table is what stops 'not shown' becoming 'improves'."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_arms as aa  # noqa: E402
import export_analysis as ea  # noqa: E402


def test_unknown_verdicts_fall_to_the_most_conservative_wording():
    level, text = ea.WORDING.get("something the analysis never emits", ea.FALLBACK)
    assert level == "not_shown" and "improve" not in text.lower()


def test_only_the_fully_met_verdict_is_called_supported():
    supported = [v for v, (lvl, _) in ea.WORDING.items() if lvl == "supported"]
    assert supported == ["reduces false alarms (all three conditions met)"]


def test_partial_and_pending_wording_never_claims_a_general_reduction():
    for verdict, (level, text) in ea.WORDING.items():
        if level in ("partial", "pending", "not_shown"):
            assert "reduces false alarms" not in text.lower(), verdict
            assert "improves" not in text.lower(), verdict


def test_every_verdict_the_analysis_can_emit_has_wording():
    """If analyze_arms adds a verdict, the site must not silently use the fallback."""
    emitted = {
        "not shown",
        "reduces false alarms (all three conditions met)",
        "fewer false alarms at the default threshold; not shown at matched recall",
        "conditions 1-2 met; condition 3 (matched recall) not computed",
    }
    assert emitted <= set(ea.WORDING)


def test_the_emitted_verdicts_really_are_the_ones_in_analyze_arms():
    """Guard against the two files drifting: the strings must appear in the source."""
    source = Path(aa.__file__).read_text(encoding="utf-8")
    for verdict in ea.WORDING:
        assert verdict in source, f"analyze_arms no longer emits {verdict!r}"


def test_num_treats_blank_and_nan_as_missing():
    assert ea.num("") is None and ea.num("nan") is None and ea.num(None) is None
    assert ea.num("1.5") == 1.5
