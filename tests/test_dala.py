from __future__ import annotations

from dfm_evals.tasks.dala import (
    _extract_label,
    _macro_f1_from_pairs,
    _mcc_from_pairs,
    dala_scorer,
)


def test_extract_label_supports_english_and_danish_variants() -> None:
    assert _extract_label("ja") == "correct"
    assert _extract_label("nej") == "incorrect"
    assert _extract_label("correct") == "correct"
    assert _extract_label("Det er korrekt.") == "correct"
    assert _extract_label("incorrect") == "incorrect"
    assert _extract_label("Ukorrekt.") == "incorrect"
    assert _extract_label("forkert") == "incorrect"
    assert _extract_label("maybe") is None


def test_macro_f1_from_pairs_matches_balanced_binary_case() -> None:
    pairs = [
        ("correct", "correct"),
        ("incorrect", "correct"),
        ("correct", "incorrect"),
        ("incorrect", "incorrect"),
    ]

    assert _macro_f1_from_pairs(pairs) == 0.5


def test_mcc_from_pairs_handles_perfect_and_inverted_predictions() -> None:
    perfect_pairs = [
        ("correct", "correct"),
        ("incorrect", "incorrect"),
    ]
    inverted_pairs = [
        ("correct", "incorrect"),
        ("incorrect", "correct"),
    ]

    assert _mcc_from_pairs(perfect_pairs) == 1.0
    assert _mcc_from_pairs(inverted_pairs) == -1.0


def test_dala_scorer_uses_scalar_metric_registration() -> None:
    registry_info = dala_scorer().__registry_info__
    metrics = registry_info.metadata["metrics"]

    assert isinstance(metrics, list)
    assert len(metrics) == 2
