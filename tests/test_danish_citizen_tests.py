from __future__ import annotations

from dfm_evals.tasks.danish_citizen_tests import (
    _extract_choice,
    _mcc_from_pairs,
    _normalize_answer_label,
    _select_split_records,
)


def test_extract_choice_supports_letter_and_option_text() -> None:
    options = {
        "a": "Mindst to procent",
        "b": "Mindst fire procent",
        "c": "Mindst fem procent",
    }

    assert _extract_choice("A", options) == "a"
    assert _extract_choice("Jeg vælger B.", options) == "b"
    assert _extract_choice("Mindst fire procent", options) == "b"
    assert _extract_choice("ved ikke", options) is None


def test_select_split_records_matches_expected_partitioning() -> None:
    citizenship = [
        {
            "question": f"cit-{idx}",
            "options": ["Ja", "Nej"],
            "answer": "a",
            "test_type": "indfødsretsprøven",
            "year": 2024,
            "version": "vinter",
        }
        for idx in range(2)
    ]
    permanent = [
        {
            "question": f"perm-{idx}",
            "options": ["Ja", "Nej"],
            "answer": "a",
            "test_type": "medborgerskabsprøven",
            "year": year,
            "version": "vinter",
        }
        for idx, year in enumerate([2024, 2023, 2022, 2021, 2020, 2019], start=1)
    ]
    records = citizenship + permanent

    test_records = _select_split_records(records=records, split="test")
    val_records = _select_split_records(records=records, split="val")
    train_records = _select_split_records(records=records, split="train")

    test_questions = {record["question"] for record in test_records}
    val_questions = {record["question"] for record in val_records}
    train_questions = {record["question"] for record in train_records}

    assert {"cit-0", "cit-1"} <= test_questions
    assert test_questions.isdisjoint(val_questions)
    assert test_questions.isdisjoint(train_questions)
    assert val_questions.isdisjoint(train_questions)
    assert test_questions | val_questions | train_questions == {
        record["question"] for record in records
    }


def test_mcc_from_pairs_handles_perfect_predictions() -> None:
    pairs = [
        (_normalize_answer_label("a"), "a"),
        (_normalize_answer_label("b"), "b"),
        (_normalize_answer_label("c"), "c"),
        (_normalize_answer_label("a"), "a"),
    ]

    assert _mcc_from_pairs(pairs, labels=["a", "b", "c"]) == 1.0
