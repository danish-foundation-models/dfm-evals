import json

import pytest

from dfm_evals.tasks.piqa import _extract_choice, _load_records


def test_extract_choice_maps_letter_a_to_a() -> None:
    assert _extract_choice("A", "Use solution zero.", "Use solution one.") == "A"


def test_extract_choice_maps_letter_b_to_b() -> None:
    assert _extract_choice("B", "Use solution zero.", "Use solution one.") == "B"


def test_extract_choice_rejects_numeric_zero() -> None:
    assert _extract_choice("0", "Use solution zero.", "Use solution one.") is None


def test_extract_choice_rejects_numeric_one() -> None:
    assert _extract_choice("1", "Use solution zero.", "Use solution one.") is None


def test_load_records_normalizes_text_fields(tmp_path) -> None:
    dataset_path = tmp_path / "piqa.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "row-1",
                    "prompt": "  Prompt  ",
                    "solution0": "  First solution  ",
                    "solution1": "\tSecond solution\t",
                    "label": 0,
                }
            ]
        ),
        encoding="utf-8",
    )

    records = _load_records(dataset_path)

    assert records == [
        {
            "id": "row-1",
            "prompt": "Prompt",
            "solution0": "First solution",
            "solution1": "Second solution",
            "label": 0,
        }
    ]


def test_load_records_reports_multiple_malformed_rows(tmp_path) -> None:
    dataset_path = tmp_path / "piqa.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "bad-1",
                    "prompt": "",
                    "solution0": "valid",
                    "solution1": "valid",
                    "label": 0,
                },
                "bad row",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed PIQA rows"):
        _load_records(dataset_path)
