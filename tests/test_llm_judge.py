import pytest

from dfm_evals.scorers.llm_judge import (
    _build_prompt_variables,
    llm_judge,
    parse_judge_score,
)


def test_parse_judge_score_valid_json() -> None:
    parsed = parse_judge_score('{"score": 3, "reason": "Overvejende korrekt."}')

    assert parsed.raw_score == 3
    assert parsed.normalized_score == 0.75
    assert parsed.parse_error is None


def test_parse_judge_score_extracts_json_from_wrapped_text() -> None:
    parsed = parse_judge_score(
        'Her er vurderingen:\n```json\n{"score": 4, "reason": "Samme betydning."}\n```'
    )

    assert parsed.raw_score == 4
    assert parsed.normalized_score == 1.0
    assert parsed.parse_error is None


def test_parse_judge_score_invalid_json_returns_zero() -> None:
    parsed = parse_judge_score("ikke-json")

    assert parsed.raw_score == 0
    assert parsed.normalized_score == 0.0
    assert parsed.parse_error == "invalid_json"


def test_parse_judge_score_invalid_score_returns_zero() -> None:
    parsed = parse_judge_score('{"score": 7, "reason": "for hojt"}')

    assert parsed.raw_score == 0
    assert parsed.normalized_score == 0.0
    assert parsed.parse_error == "invalid_score"


def test_parse_judge_score_without_reason_is_accepted() -> None:
    parsed = parse_judge_score('{"score": 2}')

    assert parsed.raw_score == 2
    assert parsed.normalized_score == 0.5
    assert parsed.parse_error is None
    assert parsed.reason == "No reason provided by judge."


def test_parse_judge_score_supports_custom_range() -> None:
    parsed = parse_judge_score(
        '{"score": 7, "reason": "ok"}', min_score=1, max_score=10
    )

    assert parsed.raw_score == 7
    assert parsed.normalized_score == pytest.approx((7 - 1) / (10 - 1))
    assert parsed.parse_error is None


def test_llm_judge_requires_prompt_template() -> None:
    with pytest.raises(ValueError, match="prompt_template"):
        llm_judge(prompt_template=None)


def test_llm_judge_rejects_invalid_score_range() -> None:
    with pytest.raises(ValueError, match="max_score"):
        llm_judge(
            prompt_template="{reference} {prediction}",
            min_score=4,
            max_score=4,
        )


def test_build_prompt_variables_supports_custom_metadata_mapping() -> None:
    variables = _build_prompt_variables(
        reference="facit",
        prediction="svar",
        min_score=1,
        max_score=10,
        metadata={"idiom": "laegge lag paa"},
        prompt_fields={"input_term": "idiom"},
    )

    assert variables["reference"] == "facit"
    assert variables["prediction"] == "svar"
    assert variables["min_score"] == "1"
    assert variables["max_score"] == "10"
    assert variables["input_term"] == "laegge lag paa"


def test_llm_judge_rejects_empty_prompt_field_mapping() -> None:
    with pytest.raises(ValueError, match="prompt_fields"):
        llm_judge(
            prompt_template="{reference} {prediction}",
            prompt_fields={"": "idiom"},
        )
