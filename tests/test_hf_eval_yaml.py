import pytest
from inspect_ai._util.error import PrerequisiteError

from dfm_evals.hf_eval_yaml import _record_matches_filter


def test_record_matches_filter_raises_on_invalid_between_comparison() -> None:
    with pytest.raises(PrerequisiteError, match="between"):
        _record_matches_filter(
            {"score": "high"},
            {"column": "score", "op": "between", "value": [1, 5]},
        )


def test_record_matches_filter_raises_on_invalid_relational_comparison() -> None:
    with pytest.raises(PrerequisiteError, match="gt"):
        _record_matches_filter(
            {"score": "high"},
            {"column": "score", "op": "gt", "value": 5},
        )


def test_record_matches_filter_supports_logical_conditions() -> None:
    record = {"lang": "da", "score": 7, "tags": ["qa", "demo"]}

    assert _record_matches_filter(
        record,
        {
            "all": [
                {"column": "lang", "op": "eq", "value": "da"},
                {
                    "any": [
                        {"column": "score", "op": "gte", "value": 10},
                        {"column": "tags", "op": "contains", "value": "qa"},
                    ]
                },
            ]
        },
    )


def test_record_matches_filter_supports_not_contains() -> None:
    assert _record_matches_filter(
        {"tags": ["qa", "demo"]},
        {"column": "tags", "op": "not_contains", "value": "prod"},
    )
