from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Any

import yaml
from inspect_ai._eval.loader import scorer_from_spec, solver_from_spec
from inspect_ai._eval.task import Task
from inspect_ai._eval.task.epochs import Epochs
from inspect_ai._eval.task.hf import HFFieldSpec
from inspect_ai._eval.task.util import split_spec
from inspect_ai._util.error import PrerequisiteError, pip_dependency_error
from inspect_ai._util.registry import (
    registry_find,
    registry_info,
    registry_lookup,
    registry_unqualified_name,
)
from inspect_ai._util.version import verify_required_version
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.dataset._dataset import DatasetRecord
from inspect_ai.scorer._scorer import Scorer, ScorerSpec
from inspect_ai.solver._solver import Solver, SolverSpec


def task_create_from_hf_extended(task_name: str, **kwargs: Any) -> list[Task]:
    """Load hf/<repo> tasks with dfm-evals extensions (custom scorers + filters)."""
    try:
        from huggingface_hub import errors as hf_errors
        from huggingface_hub import hf_hub_download

        verify_required_version("HuggingFace Tasks", "huggingface_hub", "1.0.0")
    except ImportError:
        raise pip_dependency_error(
            "HuggingFace Dataset Tasks (hf/)", ["huggingface_hub"]
        ) from None

    task_spec, revision = split_spec(task_name.replace("hf/", ""))
    if revision is None:
        revision = kwargs.get("revision", "main")

    repo_id, requested_task_id = _parse_task_spec(task_spec)

    try:
        yaml_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="eval.yaml",
                repo_type="dataset",
                revision=revision,
            )
        )
    except hf_errors.EntryNotFoundError:
        raise PrerequisiteError(
            f"No 'eval.yaml' file found for Hugging Face Dataset '{repo_id}'"
        ) from None

    global_config = yaml.safe_load(yaml_path.read_text())
    if not isinstance(global_config, dict):
        raise PrerequisiteError("eval.yaml root must be a mapping.")

    task_configs = global_config.get("tasks")
    if not isinstance(task_configs, list) or len(task_configs) == 0:
        raise PrerequisiteError("eval.yaml does not include non-empty 'tasks' list.")

    tasks: list[Task] = []
    for idx, task_config in enumerate(task_configs):
        if not isinstance(task_config, dict):
            raise PrerequisiteError(f"Task config #{idx} in eval.yaml must be a mapping.")

        task_id = task_config.get("id")
        if len(task_configs) > 1 and not isinstance(task_id, str):
            raise PrerequisiteError(
                "Task 'id' field is required if there are more than 1 tasks in 'eval.yaml'"
            )

        if requested_task_id is not None and task_id != requested_task_id:
            continue

        raw_field_spec = task_config.get("field_spec")
        if not isinstance(raw_field_spec, dict):
            raise PrerequisiteError(
                f"Task '{task_id or idx}' missing required mapping field 'field_spec'."
            )
        field_spec = HFFieldSpec(**raw_field_spec)

        filters = task_config.get("filters")

        def record_to_sample_hf(
            record: DatasetRecord,
            field_spec: HFFieldSpec = field_spec,
            filters: Any = filters,
        ) -> Sample | list[Sample]:
            if filters is not None and not _record_matches_filter(record, filters):
                return []
            return _record_to_sample_hf(record, field_spec)

        dataset = hf_dataset(
            path=repo_id,
            revision=revision,
            name=str(task_config.get("config", "default")),
            split=str(task_config.get("split", "test")),
            sample_fields=record_to_sample_hf,
        )

        if bool(task_config.get("shuffle_choices")):
            dataset.shuffle_choices()

        solvers = _build_solvers(task_config)
        scorers = _build_scorers(task_config)

        epochs_raw = task_config.get("epochs", 1)
        if not isinstance(epochs_raw, int) or epochs_raw < 1:
            raise PrerequisiteError(
                f"Task '{task_id or idx}' has invalid 'epochs' value: {epochs_raw}"
            )
        epoch_reducer = task_config.get("epoch_reducer")
        if epoch_reducer is not None and not isinstance(epoch_reducer, str):
            raise PrerequisiteError(
                f"Task '{task_id or idx}' has invalid 'epoch_reducer' value."
            )

        task = Task(
            name=f"{task_name}/{task_id}" if len(task_configs) > 1 else task_name,
            dataset=dataset,
            solver=solvers,
            scorer=scorers,
            epochs=Epochs(epochs_raw, epoch_reducer),
        )
        tasks.append(task)

    if len(tasks) == 0:
        raise PrerequisiteError(f"No tasks matching '{task_name}' were found.")

    return tasks


def _build_solvers(task_config: dict[str, Any]) -> list[Solver]:
    raw_solvers = task_config.get("solvers")
    if not isinstance(raw_solvers, list) or len(raw_solvers) == 0:
        raise PrerequisiteError("Task config requires non-empty 'solvers' list.")

    solvers: list[Solver] = []
    for idx, raw_solver in enumerate(raw_solvers):
        if not isinstance(raw_solver, dict):
            raise PrerequisiteError(f"Invalid solver entry #{idx}: expected mapping.")
        name = raw_solver.get("name")
        if not isinstance(name, str) or name.strip() == "":
            raise PrerequisiteError(f"Invalid solver entry #{idx}: missing 'name'.")
        args = raw_solver.get("args", {})
        if not isinstance(args, dict):
            raise PrerequisiteError(f"Invalid solver entry #{idx}: 'args' must be mapping.")

        solvers.append(
            solver_from_spec(
                SolverSpec(solver=name.strip(), args=args, args_passed=args)
            )
        )

    return solvers


def _build_scorers(task_config: dict[str, Any]) -> list[Scorer]:
    raw_scorers = task_config.get("scorers")
    if not isinstance(raw_scorers, list) or len(raw_scorers) == 0:
        raise PrerequisiteError("Task config requires non-empty 'scorers' list.")

    # Ensure local scorer decorators execute before name resolution.
    import dfm_evals.scorers  # noqa: F401

    scorers: list[Scorer] = []
    for idx, raw_scorer in enumerate(raw_scorers):
        if not isinstance(raw_scorer, dict):
            raise PrerequisiteError(f"Invalid scorer entry #{idx}: expected mapping.")
        name = raw_scorer.get("name")
        if not isinstance(name, str) or name.strip() == "":
            raise PrerequisiteError(f"Invalid scorer entry #{idx}: missing 'name'.")
        args = raw_scorer.get("args", {})
        if not isinstance(args, dict):
            raise PrerequisiteError(f"Invalid scorer entry #{idx}: 'args' must be mapping.")

        scorer_name = _resolve_scorer_name(name.strip())
        scorers.append(
            scorer_from_spec(
                ScorerSpec(scorer=scorer_name),
                task_path=None,
                **args,
            )
        )

    return scorers


def _parse_task_spec(task_spec: str) -> tuple[str, str | None]:
    parts = task_spec.split("/")
    if len(parts) == 2:
        return task_spec, None
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}", parts[2]
    raise ValueError(f"Expected 2 or 3 components in task spec, got {len(parts)}.")


def _sanitize_target(record: DatasetRecord, target: str, is_choices: bool) -> str:
    if target.startswith("literal:"):
        return target.split(":", 1)[1]

    target_value = record[target]
    if isinstance(target_value, int) and is_choices:
        target_value = ascii_uppercase[target_value]
    return str(target_value)


def _sanitize_choices(
    record: DatasetRecord, choices: str | list[str] | None
) -> Any | None:
    if choices is None:
        return None
    if isinstance(choices, list):
        return [record[choice] for choice in choices]
    return record[choices]


_MISSING = object()


def _record_matches_filter(record: DatasetRecord, filter: Any) -> bool:
    if not isinstance(filter, dict):
        raise PrerequisiteError("Task `filters` must be a mapping.")

    if "all" in filter:
        children = filter["all"]
        if not isinstance(children, list) or len(children) == 0:
            raise PrerequisiteError("'all' filter must contain at least one condition.")
        return all(_record_matches_filter(record, child) for child in children)

    if "any" in filter:
        children = filter["any"]
        if not isinstance(children, list) or len(children) == 0:
            raise PrerequisiteError("'any' filter must contain at least one condition.")
        return any(_record_matches_filter(record, child) for child in children)

    if "not" in filter:
        return not _record_matches_filter(record, filter["not"])

    column = filter.get("column")
    op = filter.get("op")
    value = filter.get("value")
    if not isinstance(column, str) or not isinstance(op, str):
        raise PrerequisiteError("Filter condition must include string 'column' and 'op'.")

    record_value = record.get(column, _MISSING)

    if op == "exists":
        return record_value is not _MISSING
    if op == "is_null":
        return record_value is None
    if op == "not_null":
        return record_value is not _MISSING and record_value is not None

    if record_value is _MISSING:
        return False

    if op == "eq":
        return record_value == value
    if op == "ne":
        return record_value != value
    if op == "in":
        if not isinstance(value, (list, tuple, set)):
            raise PrerequisiteError("Filter op 'in' requires list/tuple/set value.")
        return record_value in value
    if op == "not_in":
        if not isinstance(value, (list, tuple, set)):
            raise PrerequisiteError("Filter op 'not_in' requires list/tuple/set value.")
        return record_value not in value
    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise PrerequisiteError("Filter op 'between' requires [low, high].")
        low, high = value
        try:
            return low <= record_value <= high
        except TypeError:
            return False
    if op == "contains":
        if isinstance(record_value, str):
            return isinstance(value, str) and value in record_value
        if isinstance(record_value, (list, tuple, set)):
            return value in record_value
        if isinstance(record_value, dict):
            return value in record_value
        return False
    if op == "not_contains":
        if isinstance(record_value, str):
            return isinstance(value, str) and value not in record_value
        if isinstance(record_value, (list, tuple, set)):
            return value not in record_value
        if isinstance(record_value, dict):
            return value not in record_value
        return False

    try:
        if op == "gt":
            return record_value > value
        if op == "gte":
            return record_value >= value
        if op == "lt":
            return record_value < value
        if op == "lte":
            return record_value <= value
    except TypeError:
        return False

    raise PrerequisiteError(f"Unsupported filter operation: {op}")


def _record_to_sample_hf(record: DatasetRecord, field_spec: HFFieldSpec) -> Sample:
    # We support direct literal target and multi-column choices from eval.yaml.
    choices = _sanitize_choices(record, field_spec.choices)
    return Sample(
        input=record[field_spec.input],
        target=_sanitize_target(record, field_spec.target, is_choices=choices is not None),
        choices=choices,
        id=record.get(field_spec.id, None),
        metadata=record if field_spec.metadata is None else {m: record[m] for m in field_spec.metadata},
        sandbox=record.get(field_spec.sandbox),
        files=record.get(field_spec.files),
        setup=record.get(field_spec.setup),
    )


def _resolve_scorer_name(name: str) -> str:
    """Resolve scorer names to the canonical registry key.

    For unqualified names, use them as-is when already resolvable. Otherwise, map
    local dfm_evals scorers by matching their unqualified name.
    """
    if "/" in name:
        return name

    if registry_lookup("scorer", name) is not None:
        return name

    matches = [
        registry_info(scorer_obj).name
        for scorer_obj in registry_find(lambda info: info.type == "scorer")
        if registry_unqualified_name(registry_info(scorer_obj)) == name
        and registry_info(scorer_obj).name.startswith("dfm_evals/")
    ]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise PrerequisiteError(
            f"Ambiguous scorer name '{name}'. Use a fully-qualified scorer name."
        )

    return name


def install_hf_eval_yaml_extensions() -> None:
    """Install the extended hf/eval.yaml loader into inspect_ai.

    inspect_ai binds `task_create_from_hf` into both the hf task module and
    eval loader namespace. Patch both references so CLI task resolution uses
    the extended loader that supports custom scorers and row filters.
    """
    import inspect_ai._eval.loader as inspect_loader
    import inspect_ai._eval.task.hf as inspect_hf_task

    inspect_hf_task.task_create_from_hf = task_create_from_hf_extended
    inspect_loader.task_create_from_hf = task_create_from_hf_extended
