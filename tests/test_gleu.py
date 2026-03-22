import pytest

from dfm_evals.scorers.gleu import compute_gleu, max_gleu_score


def test_compute_gleu_is_case_insensitive_by_default() -> None:
    assert compute_gleu("Hello world", "hello world") == 1.0


def test_max_gleu_score_picks_best_reference() -> None:
    assert max_gleu_score("red fox", ["blue whale", "red fox"]) == 1.0


def test_compute_gleu_supports_custom_tokenizer() -> None:
    score = compute_gleu(
        "a,b,c",
        "a,b,d",
        tokenizer=lambda text: text.split(","),
    )

    assert 0.0 < score < 1.0


def test_max_gleu_score_returns_zero_for_no_references() -> None:
    assert max_gleu_score("hello", []) == 0.0


def test_compute_gleu_returns_zero_when_range_yields_no_ngrams() -> None:
    assert compute_gleu("hello", "hello", min_n=2, max_n=4) == 0.0


def test_compute_gleu_returns_zero_for_empty_strings() -> None:
    assert compute_gleu("", "") == 0.0


def test_compute_gleu_rejects_invalid_ngram_range() -> None:
    with pytest.raises(ValueError, match="max_n"):
        compute_gleu("hello", "hello", min_n=2, max_n=1)
