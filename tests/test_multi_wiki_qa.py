import sys
from types import SimpleNamespace

from dfm_evals.tasks.multi_wiki_qa import (
    PUBLIC_SOURCE_DATASET_ID,
    _extract_answer_texts,
    _is_valid_public_record,
    _load_public_records,
    _normalize_split_name,
    _records_with_unique_ids,
)


def test_extract_answer_texts_strips_and_dedupes() -> None:
    record = {
        "answers": {
            "text": [
                "  Copenhagen  ",
                "Copenhagen",
                "",
                "  Aarhus  ",
                None,
            ]
        }
    }

    assert _extract_answer_texts(record) == ["Copenhagen", "Aarhus"]


def test_is_valid_public_record_requires_cleaned_answer_texts() -> None:
    record = {
        "context": "x" * 40,
        "question": "y" * 20,
        "answers": {"text": [" ", ""]},
    }

    assert _is_valid_public_record(record) is False


def test_records_with_unique_ids_add_suffixes_and_fallback_ids() -> None:
    records = [
        {"id": "shared", "context": "a", "question": "b", "answers": {"text": ["c"]}},
        {"id": "shared", "context": "d", "question": "e", "answers": {"text": ["f"]}},
        {"id": " ", "context": "g", "question": "h", "answers": {"text": ["i"]}},
    ]

    unique_records = _records_with_unique_ids(records, language="da", split="test")

    assert [record["id"] for record in unique_records] == [
        "shared",
        "shared__1",
        "da_test_2",
    ]


def test_normalize_split_name_accepts_aliases() -> None:
    assert _normalize_split_name("training") == "train"
    assert _normalize_split_name("dev") == "val"


def test_public_source_dataset_id_uses_high_quality_subset() -> None:
    assert PUBLIC_SOURCE_DATASET_ID == "oliverkinch/multi-wiki-qa-high-quality-subset"


def test_load_public_records_falls_back_to_default_builder_config(
    monkeypatch,
) -> None:
    rows = [{"id": "x", "context": "c", "question": "q", "answers": {"text": ["a"]}}]
    calls: list[tuple[str, str | None, str]] = []

    def fake_load_dataset(path: str, name: str | None = None, split: str = "train"):
        calls.append((path, name, split))
        if name == "da":
            raise ValueError("BuilderConfig 'da' not found. Available: ['default']")
        return rows

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    assert _load_public_records(PUBLIC_SOURCE_DATASET_ID, "da") == rows
    assert calls == [
        (PUBLIC_SOURCE_DATASET_ID, "da", "train"),
        (PUBLIC_SOURCE_DATASET_ID, None, "train"),
    ]
