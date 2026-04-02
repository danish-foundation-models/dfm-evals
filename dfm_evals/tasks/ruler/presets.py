from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .qa_data import QADatasetName

Family = Literal[
    "niah",
    "variable_tracking",
    "common_words_extraction",
    "freq_words_extraction",
    "qa",
]
HaystackKind = Literal["noise", "essay", "needle"]
ValueKind = Literal["numbers", "words", "uuids"]
MatchMode = Literal["all", "any"]


@dataclass(frozen=True)
class RulerPreset:
    name: str
    family: Family
    completion_tokens: int
    match_mode: MatchMode
    haystack_kind: HaystackKind = "noise"
    key_kind: ValueKind = "words"
    value_kind: ValueKind = "numbers"
    num_keys: int = 1
    num_values_per_key: int = 1
    num_queries: int = 1
    num_chains: int = 1
    num_hops: int = 4
    freq_cw: int = 30
    freq_ucw: int = 3
    num_cw: int = 10
    alpha: float = 2.0
    qa_dataset: QADatasetName | None = None


PRESETS: dict[str, RulerPreset] = {
    "niah_single_1": RulerPreset(
        name="niah_single_1",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="noise",
    ),
    "niah_single_2": RulerPreset(
        name="niah_single_2",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="essay",
    ),
    "niah_single_3": RulerPreset(
        name="niah_single_3",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="essay",
        value_kind="uuids",
    ),
    "niah_multikey_1": RulerPreset(
        name="niah_multikey_1",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="essay",
        num_keys=4,
    ),
    "niah_multikey_2": RulerPreset(
        name="niah_multikey_2",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="needle",
    ),
    "niah_multikey_3": RulerPreset(
        name="niah_multikey_3",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="needle",
        key_kind="uuids",
        value_kind="uuids",
    ),
    "niah_multivalue": RulerPreset(
        name="niah_multivalue",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="essay",
        num_values_per_key=4,
    ),
    "niah_multiquery": RulerPreset(
        name="niah_multiquery",
        family="niah",
        completion_tokens=128,
        match_mode="all",
        haystack_kind="essay",
        num_keys=4,
        num_queries=4,
    ),
    "vt": RulerPreset(
        name="vt",
        family="variable_tracking",
        completion_tokens=30,
        match_mode="all",
        haystack_kind="noise",
        num_chains=1,
        num_hops=4,
    ),
    "cwe": RulerPreset(
        name="cwe",
        family="common_words_extraction",
        completion_tokens=120,
        match_mode="all",
        freq_cw=30,
        freq_ucw=3,
        num_cw=10,
    ),
    "fwe": RulerPreset(
        name="fwe",
        family="freq_words_extraction",
        completion_tokens=50,
        match_mode="all",
        alpha=2.0,
    ),
    "qa_1": RulerPreset(
        name="qa_1",
        family="qa",
        completion_tokens=32,
        match_mode="any",
        qa_dataset="squad",
    ),
    "qa_2": RulerPreset(
        name="qa_2",
        family="qa",
        completion_tokens=32,
        match_mode="any",
        qa_dataset="hotpotqa",
    ),
}


def get_preset(name: str) -> RulerPreset:
    normalized = name.strip()
    if not normalized:
        raise ValueError("`variant` must be a non-empty string.")

    try:
        return PRESETS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown RULER variant {name!r}. Supported variants: {supported}"
        ) from exc
