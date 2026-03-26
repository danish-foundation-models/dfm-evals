import pytest

from dfm_evals.tasks.danske_talemaader import (
    _build_prompt,
    danske_talemaader,
    record_to_sample,
)


def test_record_to_sample_maps_expected_fields() -> None:
    sample = record_to_sample(
        {
            "id": "42",
            "talemaade_udtryk": "lægge låg på noget",
            "ddo_definition": "afslutte, skjule eller lægge en dæmper på noget",
        }
    )

    assert sample.id == "42"
    assert sample.target == ["afslutte, skjule eller lægge en dæmper på noget"]
    assert sample.metadata["talemaade_udtryk"] == "lægge låg på noget"
    assert (
        sample.metadata["meaning"] == "afslutte, skjule eller lægge en dæmper på noget"
    )
    assert 'Talemåde: "lægge låg på noget"' in sample.input


def test_record_to_sample_accepts_udtryk_id_metadata() -> None:
    sample = record_to_sample(
        {
            "id": "dtm_0",
            "udtryk_id": 59010875,
            "talemaade_udtryk": "lægge låg på noget",
            "ddo_definition": "afslutte, skjule eller lægge en dæmper på noget",
        }
    )

    assert sample.id == "dtm_0"
    assert sample.target == ["afslutte, skjule eller lægge en dæmper på noget"]


def test_record_to_sample_rejects_missing_talemaade() -> None:
    with pytest.raises(ValueError, match="talemaade_udtryk"):
        record_to_sample({"ddo_definition": "x"})


def test_record_to_sample_rejects_missing_meaning() -> None:
    with pytest.raises(ValueError, match="ddo_definition"):
        record_to_sample({"talemaade_udtryk": "lægge låg på noget"})


def test_build_prompt_includes_instruction_and_idiom() -> None:
    prompt = _build_prompt("snyde nogen så vandet driver")
    assert "Forklar kort, hvad talemåden betyder" in prompt
    assert 'Talemåde: "snyde nogen så vandet driver"' in prompt


def test_task_rejects_missing_judge_configuration() -> None:
    with pytest.raises(ValueError, match="judge_model"):
        danske_talemaader(judge_model=None, judge_model_role=None)


def test_task_rejects_invalid_judge_score_range() -> None:
    with pytest.raises(ValueError, match="judge_max_score"):
        danske_talemaader(judge_min_score=2, judge_max_score=2)
