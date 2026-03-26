from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
DEFAULT_MIN_SCORE = 0
DEFAULT_MAX_SCORE = 4


@dataclass(frozen=True)
class ParsedJudgeScore:
    raw_score: int
    normalized_score: float
    reason: str
    parse_error: str | None = None


@scorer(metrics=[mean(), stderr()])
def llm_judge(
    model: str | Model | None = None,
    model_role: str | None = "grader",
    temperature: float = 0.0,
    max_tokens: int = 512,
    prompt_template: str | None = None,
    prompt_fields: dict[str, str] | None = None,
    min_score: int = DEFAULT_MIN_SCORE,
    max_score: int = DEFAULT_MAX_SCORE,
) -> Scorer:
    if model is None and model_role is None:
        raise ValueError("Either `model` or `model_role` must be provided.")

    resolved_model: Model | None = model if isinstance(model, Model) else None
    generate_config = GenerateConfig(temperature=temperature, max_tokens=max_tokens)

    if prompt_template is None:
        raise ValueError("`prompt_template` must be provided.")
    if max_score <= min_score:
        raise ValueError("`max_score` must be greater than `min_score`.")
    if prompt_fields is not None:
        for template_field, metadata_field in prompt_fields.items():
            if not template_field.strip() or not metadata_field.strip():
                raise ValueError(
                    "`prompt_fields` must map non-empty template fields "
                    "to non-empty metadata fields."
                )

    async def score(state: TaskState, target: Target) -> Score:
        nonlocal resolved_model
        if resolved_model is None:
            if model is not None:
                resolved_model = model if isinstance(model, Model) else get_model(model)
            elif model_role is not None:
                resolved_model = get_model(role=model_role)
            else:
                resolved_model = get_model()

        reference = _first_reference(target)
        prediction = state.output.completion.strip()
        prompt_variables = _build_prompt_variables(
            reference=reference,
            prediction=prediction,
            min_score=min_score,
            max_score=max_score,
            metadata=state.metadata,
            prompt_fields=prompt_fields,
        )
        judge_prompt = prompt_template.format(**prompt_variables)
        judge_output = await resolved_model.generate(
            judge_prompt, config=generate_config
        )
        parsed = parse_judge_score(
            judge_output.completion,
            min_score=min_score,
            max_score=max_score,
        )

        return Score(
            value=parsed.normalized_score,
            answer=prediction,
            explanation=parsed.reason,
            metadata={
                "raw_score": parsed.raw_score,
                "parse_error": parsed.parse_error,
                "judge_model": str(resolved_model),
                "judge_completion": judge_output.completion,
            },
        )

    return score


def _first_reference(target: Target) -> str:
    references = [
        ref.strip() for ref in target.target if isinstance(ref, str) and ref.strip()
    ]
    if not references:
        raise ValueError("LLM judge requires at least one non-empty target reference.")
    return references[0]


def _build_prompt_variables(
    *,
    reference: str,
    prediction: str,
    min_score: int,
    max_score: int,
    metadata: dict[str, Any] | None,
    prompt_fields: dict[str, str] | None,
) -> dict[str, str]:
    variables = {
        "reference": reference,
        "prediction": prediction,
        "min_score": str(min_score),
        "max_score": str(max_score),
    }
    if prompt_fields is None:
        return variables

    metadata_obj = metadata if isinstance(metadata, dict) else {}
    for template_field, metadata_field in prompt_fields.items():
        value = metadata_obj.get(metadata_field, "")
        variables[template_field] = str(value).strip()
    return variables


def parse_judge_score(
    text: str,
    *,
    min_score: int = DEFAULT_MIN_SCORE,
    max_score: int = DEFAULT_MAX_SCORE,
) -> ParsedJudgeScore:
    if max_score <= min_score:
        raise ValueError("`max_score` must be greater than `min_score`.")

    payload = _parse_json_payload(text)
    if payload is None:
        return ParsedJudgeScore(
            raw_score=0,
            normalized_score=0.0,
            reason="Judge response could not be parsed as JSON.",
            parse_error="invalid_json",
        )

    raw_score = payload.get("score")
    if isinstance(raw_score, bool):
        raw_score = None
    if not isinstance(raw_score, int) or not min_score <= raw_score <= max_score:
        return ParsedJudgeScore(
            raw_score=0,
            normalized_score=0.0,
            reason="Judge response contained invalid score.",
            parse_error="invalid_score",
        )

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No reason provided by judge."

    return ParsedJudgeScore(
        raw_score=raw_score,
        normalized_score=(raw_score - min_score) / (max_score - min_score),
        reason=reason.strip(),
        parse_error=None,
    )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    parsed_direct = _load_json_object(stripped)
    if parsed_direct is not None:
        return parsed_direct

    match = _JSON_OBJECT_PATTERN.search(stripped)
    if match is None:
        return None
    return _load_json_object(match.group(0))


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
