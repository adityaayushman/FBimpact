from backend.model_card import build_card

CURRENT = {"backbone": "stgcn", "parameters": 400000, "in_channels": 14, "window": 25,
           "feature_set": "full", "has_phase_head": True, "causal": True,
           "explain_method": "attention", "trained_on": "combined", "epoch": 12}
EARLIER = {**CURRENT, "in_channels": 4, "feature_set": "xyv", "has_phase_head": False,
           "trained_on": "urfd"}


def test_a_checkpoint_matching_the_best_configuration_is_current():
    card = build_card(CURRENT, 2, "d")
    assert card["generation"] == "current"
    assert card["generation_reasons"] == []


def test_an_older_checkpoint_says_why_it_is_older():
    card = build_card(EARLIER, 2, "d")
    assert card["generation"] == "earlier"
    text = " ".join(card["generation_reasons"])
    assert "xyv" in text and "4 channels" in text and "no four-phase head" in text


def test_a_missing_field_does_not_crash_and_counts_as_earlier():
    card = build_card({}, 2, "d")
    assert card["generation"] == "earlier"
    assert card["model"]["trained_on"] is None


def test_the_card_carries_no_measured_results():
    card = build_card(CURRENT, 2, "d")
    blob = str(card).lower()
    for token in ("recall", "false alarms per", "auc", "%"):
        assert token not in blob


def test_provenance_comes_from_the_checkpoint_not_from_constants():
    assert build_card(CURRENT, 2, "d")["model"]["trained_on"] == "combined"
    assert build_card(EARLIER, 2, "d")["model"]["trained_on"] == "urfd"
    assert build_card(CURRENT, 2, "d")["schema_version"] == 2
