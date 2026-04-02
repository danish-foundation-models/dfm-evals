from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from typing import Sequence

from inspect_ai.dataset import Sample

from .corpora import ADJECTIVES, ESSAY_SENTENCES, NOISE_SENTENCES, NOUNS
from .presets import RulerPreset
from .qa_data import QABundle, QADocument, QAExample, load_qa_bundle
from .tokenizers import LengthEstimator

_FACT_TEMPLATE = "The special value for {key} is {value}."
_CASE_SEED_STEP = 104_729


@dataclass(frozen=True)
class GeneratedCase:
    prompt: str
    targets: list[str]
    metadata: dict[str, object]


def generate_samples(
    *,
    preset: RulerPreset,
    estimator: LengthEstimator,
    max_seq_length: int,
    reserved_output_tokens: int,
    context_buffer_tokens: int,
    num_samples: int,
    seed: int,
    remove_newline_tab: bool,
) -> list[Sample]:
    haystack_units = _fit_haystack_units(
        preset=preset,
        estimator=estimator,
        max_seq_length=max_seq_length,
        reserved_output_tokens=reserved_output_tokens,
        context_buffer_tokens=context_buffer_tokens,
        seed=seed,
    )

    samples: list[Sample] = []
    for index in range(num_samples):
        case, estimated_tokens, used_haystack_units = _generate_case_with_budget(
            preset=preset,
            estimator=estimator,
            max_seq_length=max_seq_length,
            reserved_output_tokens=reserved_output_tokens,
            context_buffer_tokens=context_buffer_tokens,
            haystack_units=haystack_units,
            seed=seed,
            index=index,
            remove_newline_tab=remove_newline_tab,
        )
        metadata = {
            **case.metadata,
            "variant": preset.name,
            "family": preset.family,
            "match_mode": preset.match_mode,
            "completion_tokens": reserved_output_tokens,
            "input_tokens": estimated_tokens,
            "max_seq_length": max_seq_length,
            "context_buffer_tokens": context_buffer_tokens,
            "haystack_units": used_haystack_units,
        }
        samples.append(
            Sample(
                id=f"{preset.name}-{index}",
                input=case.prompt,
                target=case.targets,
                metadata=metadata,
            )
        )

    return samples


def _fit_haystack_units(
    *,
    preset: RulerPreset,
    estimator: LengthEstimator,
    max_seq_length: int,
    reserved_output_tokens: int,
    context_buffer_tokens: int,
    seed: int,
) -> int:
    minimum_units = _minimum_units_for_preset(preset)

    def fits(units: int) -> bool:
        case = _generate_case(
            preset=preset,
            haystack_units=max(units, minimum_units),
            seed=seed,
            index=0,
            remove_newline_tab=False,
        )
        total = estimator.count_tokens(case.prompt) + reserved_output_tokens + context_buffer_tokens
        return total <= max_seq_length

    if not fits(minimum_units):
        return minimum_units

    lower = minimum_units
    upper = max(minimum_units, 1)
    while fits(upper) and upper < 1_000_000:
        lower = upper
        upper *= 2

    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if fits(middle):
            lower = middle
        else:
            upper = middle

    return lower


def _generate_case_with_budget(
    *,
    preset: RulerPreset,
    estimator: LengthEstimator,
    max_seq_length: int,
    reserved_output_tokens: int,
    context_buffer_tokens: int,
    haystack_units: int,
    seed: int,
    index: int,
    remove_newline_tab: bool,
) -> tuple[GeneratedCase, int, int]:
    minimum_units = _minimum_units_for_preset(preset)
    units = haystack_units
    while True:
        case = _generate_case(
            preset=preset,
            haystack_units=units,
            seed=seed,
            index=index,
            remove_newline_tab=remove_newline_tab,
        )
        estimated_tokens = estimator.count_tokens(case.prompt)
        total = estimated_tokens + reserved_output_tokens + context_buffer_tokens
        if total <= max_seq_length or units <= minimum_units:
            return case, estimated_tokens, units
        units -= 1


def _generate_case(
    *,
    preset: RulerPreset,
    haystack_units: int,
    seed: int,
    index: int,
    remove_newline_tab: bool,
) -> GeneratedCase:
    if preset.family == "niah":
        case = _generate_niah_case(
            preset=preset,
            haystack_units=haystack_units,
            seed=seed,
            index=index,
        )
    elif preset.family == "variable_tracking":
        case = _generate_variable_tracking_case(
            preset=preset,
            haystack_units=haystack_units,
            seed=seed,
            index=index,
        )
    elif preset.family == "common_words_extraction":
        case = _generate_common_words_case(
            preset=preset,
            haystack_units=haystack_units,
            seed=seed,
            index=index,
        )
    elif preset.family == "freq_words_extraction":
        case = _generate_freq_words_case(
            preset=preset,
            haystack_units=haystack_units,
            seed=seed,
            index=index,
        )
    elif preset.family == "qa":
        case = _generate_qa_case(
            preset=preset,
            haystack_units=haystack_units,
            seed=seed,
            index=index,
        )
    else:
        raise ValueError(f"Unsupported RULER family {preset.family!r}")

    if not remove_newline_tab:
        return case

    normalized_prompt = " ".join(
        case.prompt.replace("\n", " ").replace("\t", " ").split()
    )
    return GeneratedCase(
        prompt=normalized_prompt,
        targets=case.targets,
        metadata=case.metadata,
    )


def _generate_niah_case(
    *, preset: RulerPreset, haystack_units: int, seed: int, index: int
) -> GeneratedCase:
    rng = _rng_for_case(seed, index)

    num_keys = max(preset.num_keys, preset.num_queries)
    keys = _sample_unique_values(preset.key_kind, num_keys, rng)
    values_by_key: list[list[str]] = []
    needles: list[str] = []

    for key in keys:
        values = _sample_unique_values(preset.value_kind, preset.num_values_per_key, rng)
        values_by_key.append(values)
        for value in values:
            needles.append(_FACT_TEMPLATE.format(key=key, value=value))

    rng.shuffle(needles)

    filler = _make_haystack_sentences(
        haystack_kind=preset.haystack_kind,
        haystack_units=haystack_units,
        rng=rng,
        key_kind=preset.key_kind,
        value_kind=preset.value_kind,
    )
    context_lines = _merge_insertions(filler, needles, rng)

    query_indices = rng.sample(range(len(keys)), preset.num_queries)
    queried_keys = [keys[i] for i in query_indices]
    targets = [value for i in query_indices for value in values_by_key[i]]
    query = _join_series(queried_keys)

    prompt = (
        "Read the text below carefully. Hidden inside are factual statements in "
        'the form "The special value for KEY is VALUE." '
        "Answer with only the requested value or a comma-separated list of values.\n\n"
        f"{chr(10).join(context_lines)}\n\n"
        f"Question: What special value or values are associated with {query}?"
    )

    return GeneratedCase(
        prompt=prompt,
        targets=targets,
        metadata={"query_keys": queried_keys},
    )


def _generate_variable_tracking_case(
    *, preset: RulerPreset, haystack_units: int, seed: int, index: int
) -> GeneratedCase:
    rng = _rng_for_case(seed, index)

    chains = [
        _generate_variable_chain(num_hops=preset.num_hops, rng=rng)
        for _ in range(preset.num_chains)
    ]
    query_value = chains[0].value
    targets = list(chains[0].variables)
    chain_statements = _interleave_sequences([chain.statements for chain in chains], rng)

    filler = _make_haystack_sentences(
        haystack_kind=preset.haystack_kind,
        haystack_units=haystack_units,
        rng=rng,
        key_kind="words",
        value_kind="numbers",
    )
    context_lines = _merge_insertions(filler, chain_statements, rng)

    prompt = (
        "Track the variable assignments in the text below. "
        "Answer with only the variable names, separated by commas, that are "
        f"assigned the value {query_value}.\n\n"
        f"{chr(10).join(context_lines)}\n\n"
        f"Question: Which variables are assigned the value {query_value}?"
    )

    return GeneratedCase(
        prompt=prompt,
        targets=targets,
        metadata={"query_value": query_value},
    )


def _generate_common_words_case(
    *, preset: RulerPreset, haystack_units: int, seed: int, index: int
) -> GeneratedCase:
    rng = _rng_for_case(seed, index)
    total_unique_words = max(haystack_units, preset.num_cw + 8)
    vocabulary = _sample_unique_context_words(total_unique_words, rng)
    common_words = vocabulary[: preset.num_cw]
    uncommon_words = vocabulary[preset.num_cw :]

    numbered_words = [
        *common_words * preset.freq_cw,
        *uncommon_words * preset.freq_ucw,
    ]
    rng.shuffle(numbered_words)

    context = " ".join(
        f"{position + 1}. {word}" for position, word in enumerate(numbered_words)
    )
    prompt = (
        "Below is a numbered list of words. In these words, some appear more often "
        "than others. Memorize the ones that appear most often.\n"
        f"{context}\n"
        f"Question: What are the {preset.num_cw} most common words in the above list?"
    )

    return GeneratedCase(
        prompt=prompt,
        targets=common_words,
        metadata={
            "num_common_words": preset.num_cw,
            "total_unique_words": total_unique_words,
        },
    )


def _generate_freq_words_case(
    *, preset: RulerPreset, haystack_units: int, seed: int, index: int
) -> GeneratedCase:
    rng = _rng_for_case(seed, index)
    sampled_word_count = max(haystack_units, 96)
    vocab_size = max(16, min(256, sampled_word_count // 6))
    vocabulary = _generate_coded_vocabulary(vocab_size, rng)
    counts = _zipf_counts(
        sample_count=sampled_word_count,
        vocab_size=len(vocabulary),
        alpha=preset.alpha,
    )
    counts[0] = max(counts[0], counts[1] + 2)

    sampled_words: list[str] = []
    for word, count in zip(vocabulary, counts):
        sampled_words.extend([word] * count)
    rng.shuffle(sampled_words)

    prompt = (
        "Read the following coded text and track the frequency of each coded word. "
        "Find the three most frequently appeared coded words. "
        f"{' '.join(sampled_words)}\n"
        "Question: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?"
    )

    return GeneratedCase(
        prompt=prompt,
        targets=vocabulary[1:4],
        metadata={
            "alpha": preset.alpha,
            "sampled_word_count": sampled_word_count,
            "vocab_size": vocab_size,
        },
    )


def _generate_qa_case(
    *, preset: RulerPreset, haystack_units: int, seed: int, index: int
) -> GeneratedCase:
    if preset.qa_dataset is None:
        raise ValueError("QA presets must define `qa_dataset`.")

    rng = _rng_for_case(seed, index)
    bundle = load_qa_bundle(preset.qa_dataset)
    example = _sample_qa_example(bundle, rng)
    distractors = _sample_distractor_documents(
        bundle=bundle,
        example=example,
        max_distractors=max(haystack_units, 0),
        rng=rng,
    )
    documents = [*example.documents, *distractors]
    rng.shuffle(documents)

    prompt = (
        "Answer the question based on the given documents. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given documents.\n\n"
        f"{_format_qa_documents(documents)}\n\n"
        "Answer the question based on the given documents. Only give me the answer "
        "and do not output any other words.\n\n"
        f"Question: {example.question}"
    )

    return GeneratedCase(
        prompt=prompt,
        targets=example.answers,
        metadata={
            "qa_dataset": preset.qa_dataset,
            "document_count": len(documents),
            "support_document_count": len(example.documents),
        },
    )


@dataclass(frozen=True)
class VariableChain:
    value: str
    variables: list[str]
    statements: list[str]


def _generate_variable_chain(num_hops: int, rng: random.Random) -> VariableChain:
    variables = _sample_unique_variable_names(num_hops + 1, rng)
    value = _generate_random_value("numbers", rng)
    statements = [f"VAR {variables[0]} = {value}."]
    for previous, current in zip(variables, variables[1:]):
        statements.append(f"VAR {current} = VAR {previous}.")
    return VariableChain(value=value, variables=variables, statements=statements)


def _interleave_sequences(
    sequences: Sequence[Sequence[str]], rng: random.Random
) -> list[str]:
    positions = [0 for _ in sequences]
    active = [index for index, sequence in enumerate(sequences) if sequence]
    output: list[str] = []

    while active:
        sequence_index = rng.choice(active)
        sequence = sequences[sequence_index]
        position = positions[sequence_index]
        output.append(sequence[position])
        positions[sequence_index] += 1
        if positions[sequence_index] >= len(sequence):
            active.remove(sequence_index)

    return output


def _make_haystack_sentences(
    *,
    haystack_kind: str,
    haystack_units: int,
    rng: random.Random,
    key_kind: str,
    value_kind: str,
) -> list[str]:
    units = max(haystack_units, 1)
    if haystack_kind == "noise":
        return _cycled_slice(NOISE_SENTENCES, units, rng)
    if haystack_kind == "essay":
        return _cycled_slice(ESSAY_SENTENCES, units, rng)
    if haystack_kind == "needle":
        return [
            _FACT_TEMPLATE.format(
                key=_generate_random_value(key_kind, rng),
                value=_generate_random_value(value_kind, rng),
            )
            for _ in range(units)
        ]
    raise ValueError(f"Unsupported haystack kind {haystack_kind!r}")


def _merge_insertions(
    filler: Sequence[str], insertions: Sequence[str], rng: random.Random
) -> list[str]:
    total = len(filler) + len(insertions)
    insertion_positions = set(rng.sample(range(total), len(insertions)))
    merged: list[str] = []
    filler_index = 0
    insertion_index = 0

    for position in range(total):
        if position in insertion_positions:
            merged.append(insertions[insertion_index])
            insertion_index += 1
        else:
            merged.append(filler[filler_index])
            filler_index += 1

    return merged


def _cycled_slice(
    base: Sequence[str], count: int, rng: random.Random
) -> list[str]:
    offset = rng.randrange(len(base))
    return [base[(offset + index) % len(base)] for index in range(count)]


def _sample_unique_values(
    value_kind: str, count: int, rng: random.Random
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < count:
        value = _generate_random_value(value_kind, rng)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _sample_unique_variable_names(count: int, rng: random.Random) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < count:
        candidate = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        if candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)
    return values


def _generate_random_value(value_kind: str, rng: random.Random) -> str:
    if value_kind == "numbers":
        return str(rng.randint(1_000_000, 9_999_999))
    if value_kind == "words":
        return f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}"
    if value_kind == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise ValueError(f"Unsupported value kind {value_kind!r}")


def _join_series(items: Sequence[str]) -> str:
    values = list(items)
    if not values:
        raise ValueError("Cannot join an empty series.")
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _rng_for_case(seed: int, index: int) -> random.Random:
    return random.Random(seed + (index * _CASE_SEED_STEP))


def _minimum_units_for_preset(preset: RulerPreset) -> int:
    if preset.family == "qa":
        return 0
    return 1


def _sample_unique_context_words(count: int, rng: random.Random) -> list[str]:
    pool = _context_word_pool()
    if count > len(pool):
        raise ValueError(f"Requested {count} unique context words, but only {len(pool)} are available.")
    return rng.sample(pool, count)


def _context_word_pool() -> list[str]:
    words = [*ADJECTIVES, *NOUNS]
    words.extend(f"{adjective}-{noun}" for adjective in ADJECTIVES for noun in NOUNS)
    return list(dict.fromkeys(words))


def _generate_coded_vocabulary(vocab_size: int, rng: random.Random) -> list[str]:
    values: list[str] = ["...."]
    seen: set[str] = set(values)
    while len(values) < vocab_size:
        candidate = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        if candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)
    return values


def _zipf_counts(sample_count: int, vocab_size: int, alpha: float) -> list[int]:
    weights = [1.0 / math.pow(rank + 1, alpha) for rank in range(vocab_size)]
    total = sum(weights)
    raw_counts = [sample_count * weight / total for weight in weights]
    counts = [int(value) for value in raw_counts]
    remainder = max(sample_count - sum(counts), 0)

    fractions = sorted(
        range(vocab_size),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in fractions[:remainder]:
        counts[index] += 1

    for index in range(min(4, len(counts))):
        if counts[index] == 0:
            counts[index] = 1

    return counts


def _sample_qa_example(bundle: QABundle, rng: random.Random) -> QAExample:
    if not bundle.examples:
        raise ValueError("QA bundle does not contain any examples.")
    return bundle.examples[rng.randrange(len(bundle.examples))]


def _sample_distractor_documents(
    *,
    bundle: QABundle,
    example: QAExample,
    max_distractors: int,
    rng: random.Random,
) -> list[QADocument]:
    if max_distractors <= 0:
        return []

    support_ids = {document.id for document in example.documents}
    candidates = [
        document
        for document in bundle.distractor_documents
        if document.id not in support_ids
    ]
    if not candidates:
        return []

    sample_size = min(max_distractors, len(candidates))
    return rng.sample(candidates, sample_size)


def _format_qa_documents(documents: Sequence[QADocument]) -> str:
    formatted: list[str] = []
    for index, document in enumerate(documents, start=1):
        formatted.append(
            f"Document {index}\nTitle: {document.title}\n{document.text}"
        )
    return "\n\n".join(formatted)
