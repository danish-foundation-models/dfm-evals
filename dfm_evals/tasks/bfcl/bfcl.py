import ast
import json
import logging
import math
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import (
    TaskState,
)

from dfm_evals.tasks.bfcl.data import (
    BFCLRecord,
    download_github_directory,
    load_records_by_category,
)
from dfm_evals.tasks.bfcl.solver import bfcl_solver
from dfm_evals.tasks.bfcl.utils import (
    CATEGORIES,
    V1_CATEGORIES,
    V2_CATEGORIES,
    V3_CATEGORIES,
    _validate_categories,
)

logger = logging.getLogger(__name__)

GITHUB_REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
GITHUB_COMMIT = "dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a"
GITHUB_DATA_PATH = "berkeley-function-call-leaderboard/bfcl_eval/data"

# Default to all implemented BFCL V1 categories.
DEFAULT_CATEGORIES = sorted(V1_CATEGORIES)
DATASET_LOCATION = (
    Path(
        os.environ.get(
            "DFM_EVALS_CACHE_DIR",
            os.environ.get("DFM_EVAL_CACHE_DIR", Path.home() / ".cache" / "dfm_evals"),
        )
    )
    / "BFCL"
)
TRANSLATION_DIR = Path(__file__).parent / "translations" / "exec_simple"
EXEC_SIMPLE_DA_TRANSLATIONS_PATH = TRANSLATION_DIR / "exec_simple_prompts_da.json"

PYTHON_TYPE_MAPPING: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

PYTHON_NESTED_TYPE_CHECK_LIST = {"array", "tuple"}
NESTED_CONVERSION_TYPE_LIST = {"Array", "ArrayList", "array"}

JAVA_TYPE_CONVERSION: dict[str, type[Any]] = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION: dict[str, type[Any]] = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}


@task(name="bfcl-v1")
def bfcl(
    category: str | list[str] = DEFAULT_CATEGORIES,
    shuffle: bool = True,
    prompt_language: str = "en",
) -> Task:
    # 1. Validate user input for category names.
    category_names = [category] if isinstance(category, str) else list(category)
    _validate_categories(
        category_names=category_names,
        valid_categories=V1_CATEGORIES,
        invalid_category_errors={
            frozenset(V2_CATEGORIES): "V2 (live) categories are not yet implemented",
            frozenset(
                V3_CATEGORIES
            ): "V3 (multi-turn) categories are not yet implemented",
        },
    )

    prompt_translations = _resolve_prompt_translations(
        category_names=category_names,
        prompt_language=prompt_language,
    )

    # 2. Download the dataset files from GitHub
    if not DATASET_LOCATION.exists():
        DATASET_LOCATION.mkdir(parents=True, exist_ok=True)
        try:
            download_github_directory(
                repo_url=GITHUB_REPO_URL,
                commit=GITHUB_COMMIT,
                sparse_path=GITHUB_DATA_PATH,
                local_dir=DATASET_LOCATION,
            )
        except Exception:
            shutil.rmtree(DATASET_LOCATION, True)
            raise

    # 3. Load in the dataset files as a dict of records.
    records_by_id: dict[str, BFCLRecord] = {}
    for name in category_names:
        category_records_by_id = load_records_by_category(
            category_name=name, cache_dir=DATASET_LOCATION
        )
        records_by_id.update(category_records_by_id)

    # 4. Concatenate all subsets into a single MemoryDataset.
    samples = [
        record_to_sample(records_by_id[k], prompt_translations=prompt_translations)
        for k in sorted(records_by_id.keys())
    ]
    dataset = MemoryDataset(
        samples=samples,
        name="BFCL",
        location=str(DATASET_LOCATION),
    )

    if shuffle:
        logger.info("Shuffling dataset")
        dataset.shuffle()

    return Task(
        dataset=dataset,
        solver=bfcl_solver(),
        scorer=bfcl_scorer(),
    )


@task(name="bfcl-v1-da")
def bfcl_da(shuffle: bool = True) -> Task:
    """BFCL exec_simple with prompts manually translated to Danish."""
    return bfcl(category="exec_simple", shuffle=shuffle, prompt_language="da")


@scorer([accuracy()])
def bfcl_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        scorer_name = state.metadata.get("scorer")
        match scorer_name:
            case "execution":
                return _score_execution(state)
            case "ast":
                return _score_ast(state)
            case "irrelevance":
                return _score_irrelevance(state)
            case _:
                raise ValueError(f"Unknown BFCL scorer: {scorer_name}")

    return score


def record_to_sample(
    record: BFCLRecord,
    prompt_translations: dict[str, str] | None = None,
) -> Sample:
    """
    Convertdataset record to Inspect Sample for single-turn tasks.

    Args:
        record: Raw dataset record with question, function, and ground_truth
        config: Additional informatin related to the record's category/ subset.

    Returns:
        Sample with input, target, and metadata (tools, target_obj)
    """
    config = CATEGORIES[record.category_name]
    question = _prepare_question_messages(record, prompt_translations)
    metadata: dict[str, Any] = {
        "tools": record.function or [],
        "category_name": config.name,
        "scorer": config.matching_function,
    }
    match config.matching_function:
        case "execution":
            formatted_target = _build_execution_target(record, metadata)
        case "ast":
            expected_ast_calls = _build_ast_targets(record)
            expected_ast_calls = _finalize_expected_ast_calls(record, expected_ast_calls)
            metadata["expected_ast_calls"] = expected_ast_calls
            metadata["target_obj"] = {"calls": expected_ast_calls}
            formatted_target = repr(expected_ast_calls)
        case "irrelevance":
            metadata["target_obj"] = {"no_tool_call": True}
            formatted_target = "NO_TOOL_CALL"
        case _:
            raise NotImplementedError(f"Not yet implemented: {config.name}")

    return Sample(
        id=record.id,
        input=question,
        target=formatted_target,
        metadata=metadata,
    )


def tool_call_to_string(function_name: str, arguments: dict[str, Any]) -> str:
    args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
    return f"{function_name}({args_str})"


def parse_target(target: str) -> dict[str, Any]:
    parsed = ast.parse(target, mode="single")
    if len(parsed.body) != 1:
        raise ValueError(
            f"Expected exactly one statement in target, got {len(parsed.body)}: {target}"
        )
    body = parsed.body[0]
    if not isinstance(body, ast.Expr):
        raise TypeError(
            f"Expected an expression statement, got {type(body).__name__}: {target}"
        )
    if not isinstance(body.value, ast.Call):
        raise TypeError(
            f"Expected a function call, got {type(body.value).__name__}: {target}"
        )
    function_name = _resolve_function_name(body.value.func)
    if body.value.keywords is None:
        raise ValueError(f"Expected keyword arguments in function call: {target}")

    arguments: dict[str, Any] = {}
    for kw in body.value.keywords:
        if kw.arg is None:
            raise ValueError(f"Var keyword arguments are not supported: {target}")
        arguments[kw.arg] = _safe_eval_ast_value(kw.value)

    parsed_target: dict[str, Any] = {
        "function": function_name,
        "arguments": arguments,
    }
    if body.value.args:
        parsed_target["positional_arguments"] = [
            _safe_eval_ast_value(arg) for arg in body.value.args
        ]

    return parsed_target


def _prepare_question_messages(
    record: BFCLRecord,
    prompt_translations: dict[str, str] | None,
) -> list[ChatMessage]:
    if not record.question:
        raise ValueError(f"Missing question turns for record '{record.id}'")

    raw_question = [dict(msg) for msg in record.question[0]]
    if prompt_translations is not None:
        translated_prompt = prompt_translations.get(record.id)
        if translated_prompt is None:
            raise ValueError(
                f"Missing Danish prompt translation for record id '{record.id}'"
            )

        user_messages = [msg for msg in raw_question if msg.get("role") == "user"]
        if len(user_messages) != 1:
            raise ValueError(
                f"Expected exactly one user message for record '{record.id}'"
            )
        user_messages[0]["content"] = translated_prompt

    return _convert_to_chat_messages(raw_question)


def _build_execution_target(record: BFCLRecord, metadata: dict[str, Any]) -> str:
    string_targets = [item for item in record.ground_truth if isinstance(item, str)]
    if len(string_targets) != len(record.ground_truth):
        raise TypeError(
            f"Execution category '{record.category_name}' expected string ground truth entries"
        )
    if len(string_targets) == 0:
        raise ValueError(
            f"Execution category '{record.category_name}' has empty ground truth"
        )

    # `rest` stores execution outputs, not explicit function-call targets.
    if record.category_name == "rest":
        tool_schema = (record.function or [None])[0]
        metadata["execution_mode"] = "rest_schema"
        metadata["rest_tool_schema"] = tool_schema
        metadata["rest_ground_truth"] = string_targets[0]
        metadata["target_obj"] = {"response": string_targets[0]}
        return string_targets[0]

    expected_calls: list[dict[str, Any]] = []
    for target in string_targets:
        parsed_target = parse_target(target)
        parsed_target = _bind_positional_arguments(parsed_target, record.function or [])
        expected_calls.append(_normalize_json_value(parsed_target))

    metadata["execution_mode"] = "call_match"
    metadata["expected_calls"] = expected_calls
    metadata["target_obj"] = (
        expected_calls[0] if len(expected_calls) == 1 else {"calls": expected_calls}
    )
    return repr(expected_calls)


def _build_ast_targets(record: BFCLRecord) -> list[dict[str, Any]]:
    expected_calls: list[dict[str, Any]] = []
    for item in record.ground_truth:
        if not isinstance(item, dict):
            raise TypeError(
                f"AST category '{record.category_name}' expected dict ground truth entries"
            )
        if len(item) != 1:
            raise ValueError(
                f"AST ground truth calls must contain exactly one function, got keys={list(item.keys())}"
            )

        function_name, raw_arguments = next(iter(item.items()))
        if not isinstance(raw_arguments, dict):
            raise TypeError(
                f"AST ground truth arguments for '{function_name}' must be a dict"
            )

        argument_candidates: dict[str, list[Any]] = {}
        for arg_name, raw_candidates in raw_arguments.items():
            if isinstance(raw_candidates, list):
                candidates = raw_candidates
            else:
                candidates = [raw_candidates]
            argument_candidates[arg_name] = [
                _normalize_json_value(candidate) for candidate in candidates
            ]

        expected_calls.append(
            {"function": function_name, "arguments": argument_candidates}
        )

    return expected_calls


def _finalize_expected_ast_calls(
    record: BFCLRecord,
    expected_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = CATEGORIES[record.category_name]
    if (not config.is_parallel and not config.is_multiple) and len(expected_calls) > 1:
        return [expected_calls[0]]
    return expected_calls


def _bind_positional_arguments(
    parsed_target: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    positional_arguments = parsed_target.pop("positional_arguments", [])
    if not positional_arguments:
        return parsed_target

    function_name = parsed_target["function"]
    matching_tool = next(
        (tool for tool in tools if tool.get("name") == function_name),
        None,
    )
    if matching_tool is None:
        raise ValueError(
            f"Unable to bind positional arguments for '{function_name}' because no matching tool definition was found"
        )

    properties = matching_tool.get("parameters", {}).get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(
            f"Malformed tool parameters for '{function_name}' while binding positional arguments"
        )

    parameter_names = list(properties.keys())
    if len(positional_arguments) > len(parameter_names):
        raise ValueError(
            f"Too many positional arguments for '{function_name}': expected at most {len(parameter_names)}, got {len(positional_arguments)}"
        )

    bound_arguments = dict(parsed_target["arguments"])
    for index, value in enumerate(positional_arguments):
        parameter_name = parameter_names[index]
        if parameter_name in bound_arguments:
            raise ValueError(
                f"Cannot bind positional argument for '{function_name}.{parameter_name}' because it already has a keyword value"
            )
        bound_arguments[parameter_name] = value
    parsed_target["arguments"] = bound_arguments
    return parsed_target


def _score_execution(state: TaskState) -> Score:
    execution_mode = state.metadata.get("execution_mode", "call_match")
    if execution_mode == "rest_schema":
        return _score_rest_execution(state)

    expected_calls = state.metadata.get("expected_calls", [])
    if not isinstance(expected_calls, list):
        return Score(value=INCORRECT, answer="Malformed metadata: expected_calls")

    assistant_messages, actual_calls = _collect_assistant_calls(state)
    if len(assistant_messages) == 0:
        return Score(value=INCORRECT, answer="No assistant message")

    if len(actual_calls) != len(expected_calls):
        return Score(
            value=INCORRECT,
            answer=f"Expected {len(expected_calls)} tool call(s), got {len(actual_calls)}",
        )

    matches = _match_calls_unordered(
        actual_calls=actual_calls,
        expected_calls=expected_calls,
        call_matcher=_execution_call_matches,
    )
    return (
        Score(value=CORRECT, answer=repr([_call_to_string(c) for c in actual_calls]))
        if matches
        else Score(
            value=INCORRECT,
            answer=(
                f"expected={repr([_call_to_string(c) for c in expected_calls])}, "
                f"actual={repr([_call_to_string(c) for c in actual_calls])}"
            ),
        )
    )


def _score_rest_execution(state: TaskState) -> Score:
    assistant_messages, actual_calls = _collect_assistant_calls(state)
    if len(assistant_messages) == 0:
        return Score(value=INCORRECT, answer="No assistant message")

    if len(actual_calls) != 1:
        return Score(
            value=INCORRECT,
            answer=f"Expected exactly one tool call for rest category, got {len(actual_calls)}",
        )

    tool_schema = state.metadata.get("rest_tool_schema")
    if not isinstance(tool_schema, dict):
        return Score(value=INCORRECT, answer="Malformed metadata: rest_tool_schema")

    function_name = tool_schema.get("name")
    if actual_calls[0]["function"] != function_name:
        return Score(
            value=INCORRECT,
            answer=(
                f"Expected function '{function_name}', got '{actual_calls[0]['function']}'"
            ),
        )

    parameter_schema = tool_schema.get("parameters", {})
    required_error = _validate_required_parameters(
        arguments=actual_calls[0]["arguments"],
        parameter_schema=parameter_schema,
    )
    if required_error is not None:
        return Score(value=INCORRECT, answer=required_error)

    return Score(value=CORRECT, answer=repr(_call_to_string(actual_calls[0])))


def _score_ast(state: TaskState) -> Score:
    expected_calls = state.metadata.get("expected_ast_calls", [])
    if not isinstance(expected_calls, list):
        return Score(value=INCORRECT, answer="Malformed metadata: expected_ast_calls")

    assistant_messages, actual_calls = _collect_assistant_calls(state)
    if len(assistant_messages) == 0:
        return Score(value=INCORRECT, answer="No assistant message")

    if len(actual_calls) != len(expected_calls):
        return Score(
            value=INCORRECT,
            answer=f"Expected {len(expected_calls)} tool call(s), got {len(actual_calls)}",
        )

    category_name = state.metadata.get("category_name")
    tools = state.metadata.get("tools", [])

    def _match(actual_call: dict[str, Any], expected_call: dict[str, Any]) -> bool:
        return _ast_call_matches(
            actual_call,
            expected_call,
            tools=tools if isinstance(tools, list) else [],
            category_name=category_name if isinstance(category_name, str) else None,
        )

    matches = _match_calls_unordered(
        actual_calls=actual_calls,
        expected_calls=expected_calls,
        call_matcher=_match,
    )
    return (
        Score(value=CORRECT, answer=repr([_call_to_string(c) for c in actual_calls]))
        if matches
        else Score(
            value=INCORRECT,
            answer=(
                f"expected={repr(expected_calls)}, "
                f"actual={repr([_call_to_string(c) for c in actual_calls])}"
            ),
        )
    )


def _score_irrelevance(state: TaskState) -> Score:
    assistant_messages, actual_calls = _collect_assistant_calls(state)
    if len(assistant_messages) == 0:
        return Score(value=INCORRECT, answer="No assistant message")
    if len(actual_calls) != 0:
        return Score(
            value=INCORRECT,
            answer=f"Expected no tool calls, got {repr([_call_to_string(c) for c in actual_calls])}",
        )
    return Score(value=CORRECT, answer="No tool calls")


def _collect_assistant_calls(
    state: TaskState,
) -> tuple[list[ChatMessageAssistant], list[dict[str, Any]]]:
    assistant_messages = [
        m for m in state.messages if isinstance(m, ChatMessageAssistant)
    ]

    calls: list[dict[str, Any]] = []
    for message in assistant_messages:
        for tool_call in message.tool_calls or []:
            calls.append(
                {
                    "function": tool_call.function,
                    "arguments": _normalize_json_value(tool_call.arguments or {}),
                }
            )

    return assistant_messages, calls


def _match_calls_unordered(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
    call_matcher: Any,
) -> bool:
    if len(actual_calls) != len(expected_calls):
        return False

    used_actual_indices: set[int] = set()

    def dfs(expected_index: int) -> bool:
        if expected_index == len(expected_calls):
            return True

        expected_call = expected_calls[expected_index]
        for actual_index, actual_call in enumerate(actual_calls):
            if actual_index in used_actual_indices:
                continue
            if not call_matcher(actual_call, expected_call):
                continue
            used_actual_indices.add(actual_index)
            if dfs(expected_index + 1):
                return True
            used_actual_indices.remove(actual_index)

        return False

    return dfs(0)


def _execution_call_matches(actual_call: dict[str, Any], expected_call: dict[str, Any]) -> bool:
    if actual_call.get("function") != expected_call.get("function"):
        return False
    actual_arguments = actual_call.get("arguments", {})
    expected_arguments = expected_call.get("arguments", {})
    if not isinstance(actual_arguments, dict) or not isinstance(expected_arguments, dict):
        return False
    if set(actual_arguments.keys()) != set(expected_arguments.keys()):
        return False

    return all(
        _values_equal(actual_arguments[key], expected_arguments[key])
        for key in expected_arguments
    )


def _get_possible_answer_type(possible_answer: list[Any]) -> type[Any] | None:
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def _standardize_string(input_string: str) -> str:
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def _string_checker(model_output: str, possible_answer: list[Any]) -> bool:
    standardized_model_output = _standardize_string(model_output)
    standardized_possible_answers = [
        _standardize_string(item)
        for item in possible_answer
        if isinstance(item, str)
    ]
    return standardized_model_output in standardized_possible_answers


def _list_checker(model_output: list[Any], possible_answer: list[Any]) -> bool:
    standardized_model_output = list(model_output)
    for idx, item in enumerate(standardized_model_output):
        if isinstance(item, str):
            standardized_model_output[idx] = _standardize_string(item)

    standardized_possible_answers: list[list[Any]] = []
    for item in possible_answer:
        if not isinstance(item, list):
            continue
        standardized_item: list[Any] = []
        for nested_item in item:
            if isinstance(nested_item, str):
                standardized_item.append(_standardize_string(nested_item))
            else:
                standardized_item.append(nested_item)
        standardized_possible_answers.append(standardized_item)

    return standardized_model_output in standardized_possible_answers


def _dict_checker(model_output: dict[str, Any], possible_answers: list[Any]) -> bool:
    for answer in possible_answers:
        if answer == "":
            continue
        if not isinstance(answer, dict):
            continue

        keys_match = True
        for key, value in model_output.items():
            if key not in answer:
                keys_match = False
                break

            standardized_value = (
                _standardize_string(value) if isinstance(value, str) else value
            )
            standardized_candidates = [
                _standardize_string(candidate) if isinstance(candidate, str) else candidate
                for candidate in answer[key]
            ]
            if standardized_value not in standardized_candidates:
                keys_match = False
                break

        if not keys_match:
            continue

        missing_required = any(
            key not in model_output and "" not in value_candidates
            for key, value_candidates in answer.items()
        )
        if not missing_required:
            return True

    return False


def _list_dict_checker(model_output: list[Any], possible_answers: list[Any]) -> bool:
    for answer in possible_answers:
        if not isinstance(answer, list):
            continue
        if len(model_output) != len(answer):
            continue
        if all(
            isinstance(item, dict)
            and _dict_checker(item, [expected_dict])
            for item, expected_dict in zip(model_output, answer, strict=True)
        ):
            return True
    return False


def _type_checker(
    value: Any,
    possible_answer: list[Any],
    expected_type_converted: type[Any],
    nested_type_converted: type[Any] | None,
) -> tuple[bool, bool]:
    is_variable = False

    possible_answer_type = _get_possible_answer_type(possible_answer)
    if (
        possible_answer_type is not None
        and possible_answer_type != expected_type_converted
    ):
        is_variable = True

    if type(value) is expected_type_converted:
        if nested_type_converted is None:
            return True, is_variable

        for possible_answer_item in possible_answer:
            flag = True
            if isinstance(possible_answer_item, list):
                for value_item in value:
                    nested_valid, _ = _type_checker(
                        value_item,
                        possible_answer_item,
                        nested_type_converted,
                        None,
                    )
                    if not nested_valid:
                        flag = False
                        break
            if flag:
                return True, is_variable

        return False, is_variable

    if possible_answer_type is not None and type(value) is possible_answer_type:
        return True, True

    return False, is_variable


def _parse_java_value(value_str: str) -> Any:
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    if re.match(r"^-?\d+[lL]$", value_str):
        return int(value_str[:-1])
    if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
        return float(re.sub(r"[fF]$", "", value_str))
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str


def _parse_java_arraylist(input_str: str, nested_type: str | None = None) -> Any:
    match_as_list = re.search(
        r"new\s+ArrayList<\w*>\(Arrays\.asList\((.+?)\)\)", input_str
    )
    if match_as_list:
        elements = []
        for element_str in match_as_list.group(1).split(","):
            element_str = element_str.strip()
            if nested_type == "char":
                elements.append(element_str[1:-1])
            elif nested_type == "String":
                elements.append(element_str[1:-1])
            else:
                elements.append(
                    _java_type_converter(element_str, nested_type)
                    if nested_type is not None
                    else _parse_java_value(element_str)
                )
        return elements

    match_add = re.search(
        r"new\s+ArrayList<\w*>\(\)\s*\{\{\s*(.+?)\s*\}\}",
        input_str,
        re.DOTALL,
    )
    if match_add:
        elements = []
        for match in re.findall(r"add\((.+?)\)", match_add.group(1)):
            value_str = match.strip()
            if nested_type == "char":
                elements.append(value_str[1:-1])
            elif nested_type == "String":
                elements.append(value_str[1:-1])
            else:
                elements.append(
                    _java_type_converter(value_str, nested_type)
                    if nested_type is not None
                    else _parse_java_value(value_str)
                )
        return elements

    if re.search(r"new\s+ArrayList<\w*>\(\)", input_str):
        return []

    return input_str


def _parse_java_array(input_str: str, nested_type: str | None = None) -> Any:
    match = re.search(r"new\s+\w+\[\]\s*\{(.*?)\}", input_str)
    if not match:
        return input_str
    items = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if nested_type is None:
        return [_parse_java_value(item) for item in items]
    return [_java_type_converter(item, nested_type) for item in items]


def _parse_java_hashmap(input_str: str) -> Any:
    match = re.search(
        r"new\s+HashMap<.*?>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}",
        input_str,
        re.DOTALL,
    )
    if match:
        result: dict[str, Any] = {}
        for key, value in re.findall(r'put\("(.*?)",\s*(.*?)\)', match.group(1)):
            result[key] = _parse_java_value(value.strip())
        return result

    if re.search(r"new\s+HashMap<.*?>\s*\(\)", input_str):
        return {}

    return input_str


def _java_type_converter(
    value: Any, expected_type: str | None, nested_type: str | None = None
) -> Any:
    if expected_type is None:
        return str(value)

    value_str = str(value)
    if expected_type in {"byte", "short", "integer"}:
        if not re.match(r"^-?\d+$", value_str):
            return value_str
        return int(value_str)
    if expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
            return value_str
        return float(re.sub(r"[fF]$", "", value_str))
    if expected_type == "double":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value_str):
            return value_str
        return float(value_str)
    if expected_type == "long":
        if not re.match(r"^-?\d+[lL]$", value_str):
            return value_str
        return int(re.sub(r"[lL]$", "", value_str))
    if expected_type == "boolean":
        if value_str not in {"true", "false"}:
            return value_str
        return value_str == "true"
    if expected_type == "char":
        if not re.match(r"^\'.$\'", value_str):
            return value_str
        return value_str
    if expected_type == "ArrayList":
        return _parse_java_arraylist(value_str, nested_type)
    if expected_type == "Array":
        return _parse_java_array(value_str, nested_type)
    if expected_type == "HashMap":
        return _parse_java_hashmap(value_str)
    if expected_type in {"String", "any"}:
        return value_str
    return value_str


def _parse_js_value(value_str: str) -> Any:
    cleaned = value_str.strip()
    if cleaned == "true":
        return True
    if cleaned == "false":
        return False
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        return cleaned[1:-1]
    try:
        return int(cleaned)
    except ValueError:
        try:
            return float(cleaned)
        except ValueError:
            return cleaned


def _parse_js_collection(
    code: str, collection_type: str, nested_type: str | None = None
) -> Any:
    code = code.strip()
    if collection_type == "array":
        array_2d_pattern = (
            r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]|\bnew\s+Array\(\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\)"
        )
        array_pattern = r"\[(.*?)\]|\bnew\s+Array\((.*?)\)"

        if re.match(array_2d_pattern, code):
            inner_arrays = re.findall(r"\[(.*?)\]", code)
            result: list[list[Any]] = []
            for idx, inner_array in enumerate(inner_arrays):
                inner = inner_array.strip()
                if idx == 0 and inner.startswith("["):
                    inner = inner[1:]
                elements = [e.strip() for e in inner.split(",")]
                result.append([_parse_js_value(e) for e in elements])
            return result

        array_match = re.match(array_pattern, code)
        if array_match:
            if array_match.group(1) is not None:
                content = array_match.group(1).strip()
            elif array_match.group(2) is not None:
                content = array_match.group(2).strip()
            else:
                content = ""
            elements = content.split(",") if content else []
            if nested_type is not None:
                converted: list[Any] = []
                for element in elements:
                    element = element.strip()
                    if element.startswith(("'", '"')):
                        converted.append(
                            _js_type_converter(element, nested_type, "String")
                        )
                    else:
                        converted.append(_js_type_converter(element, nested_type))
                return converted
            return [_parse_js_value(element.strip()) for element in elements]

        return code

    if collection_type == "dict":
        if code == "{}":
            return {}
        dict_match = re.match(r"\{(.*?)\}", code)
        if not dict_match:
            return code
        content = dict_match.group(1)
        pairs = re.findall(r"([^:]+):\s*(.*?)(?:,\s*(?=[^,]+:)|$)", content)
        result: dict[str, Any] = {}
        for key, value in pairs:
            key = key.strip().strip("'\"")
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                result[key] = _parse_js_collection(value, "array")
            elif value.startswith("{") and value.endswith("}"):
                result[key] = _parse_js_collection(value, "dict")
            else:
                result[key] = _parse_js_value(value.strip("'\""))
        return result

    return code


def _js_type_converter(
    value: Any, expected_type: str | None, nested_type: str | None = None
) -> Any:
    if expected_type is None:
        return str(value)

    value_str = str(value)
    if expected_type == "String":
        if not (
            (value_str.startswith('"') and value_str.endswith('"'))
            or (value_str.startswith("'") and value_str.endswith("'"))
        ):
            return value_str
        return value_str[1:-1]
    if expected_type == "integer":
        if not re.match(r"^-?\d+$", value_str):
            return value_str
        return int(value_str)
    if expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?$", value_str):
            return value_str
        return float(value_str)
    if expected_type == "Bigint":
        if not re.match(r"^-?\d+n$", value_str):
            return value_str
        return int(value_str[:-1])
    if expected_type == "Boolean":
        if value_str not in {"true", "false"}:
            return value_str
        return value_str == "true"
    if expected_type == "dict":
        return _parse_js_collection(value_str, "dict", nested_type)
    if expected_type == "array":
        return _parse_js_collection(value_str, "array", nested_type)
    if expected_type == "any":
        return value_str
    return value_str


def _infer_java_type_from_candidates(candidates: list[Any]) -> str:
    for candidate in candidates:
        if candidate == "":
            continue
        if isinstance(candidate, bool):
            return "boolean"
        if isinstance(candidate, int):
            return "integer"
        if isinstance(candidate, float):
            return "double"
        if isinstance(candidate, dict):
            return "HashMap"
        if isinstance(candidate, list):
            return "ArrayList"
        return "String"
    return "String"


def _infer_js_type_from_candidates(candidates: list[Any]) -> str:
    for candidate in candidates:
        if candidate == "":
            continue
        if isinstance(candidate, bool):
            return "Boolean"
        if isinstance(candidate, int):
            return "integer"
        if isinstance(candidate, float):
            return "float"
        if isinstance(candidate, dict):
            return "dict"
        if isinstance(candidate, list):
            return "array"
        return "String"
    return "String"


def _infer_nested_type_from_candidates(candidates: list[Any], language: str) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, list) and len(candidate) > 0:
            sample = candidate[0]
            if language == "java":
                return _infer_java_type_from_candidates([sample])
            if language == "javascript":
                return _infer_js_type_from_candidates([sample])
            return None
    return None


def _find_tool_for_function(
    tools: list[dict[str, Any]],
    function_name: str,
) -> dict[str, Any] | None:
    for tool in tools:
        if tool.get("name") == function_name:
            return tool
    if "." in function_name:
        underscored = function_name.replace(".", "_")
        for tool in tools:
            if tool.get("name") == underscored:
                return tool
    if "_" in function_name:
        dotted = function_name.replace("_", ".")
        for tool in tools:
            if tool.get("name") == dotted:
                return tool
    return None


def _ast_call_matches_strict(
    actual_call: dict[str, Any],
    expected_call: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    category_name: str,
) -> bool:
    expected_function = expected_call.get("function")
    actual_function = actual_call.get("function")
    if not isinstance(expected_function, str) or not isinstance(actual_function, str):
        return False
    if actual_function != expected_function:
        if actual_function.replace("_", ".") != expected_function and actual_function != expected_function.replace(
            ".", "_"
        ):
            return False

    actual_arguments = actual_call.get("arguments", {})
    expected_argument_candidates = expected_call.get("arguments", {})
    if not isinstance(actual_arguments, dict) or not isinstance(
        expected_argument_candidates, dict
    ):
        return False

    function_schema = _find_tool_for_function(tools, expected_function)
    if function_schema is None:
        return False

    parameter_schema = function_schema.get("parameters", {})
    if not isinstance(parameter_schema, dict):
        return False

    param_details = parameter_schema.get("properties", {})
    required_params = parameter_schema.get("required", [])
    if not isinstance(param_details, dict) or not isinstance(required_params, list):
        return False

    for required_param in required_params:
        if required_param not in actual_arguments:
            return False

    language = "python"
    if "javascript" in category_name:
        language = "javascript"
    elif "java" in category_name and "javascript" not in category_name:
        language = "java"

    for param_name, value in actual_arguments.items():
        if param_name not in param_details or param_name not in expected_argument_candidates:
            return False

        candidates = expected_argument_candidates[param_name]
        if not isinstance(candidates, list):
            return False

        detail = param_details[param_name]
        if not isinstance(detail, dict):
            return False

        expected_type_description = detail.get("type")
        nested_type_description: str | None = None
        items = detail.get("items")
        if isinstance(items, dict):
            nested_raw = items.get("type")
            if isinstance(nested_raw, str):
                nested_type_description = nested_raw

        if language == "java":
            if expected_type_description == "string":
                expected_type_description = _infer_java_type_from_candidates(candidates)
            if (
                expected_type_description in NESTED_CONVERSION_TYPE_LIST
                and nested_type_description is None
            ):
                nested_type_description = _infer_nested_type_from_candidates(
                    candidates, language
                )
            expected_type_converted = JAVA_TYPE_CONVERSION.get(
                str(expected_type_description), str
            )
            nested_type_converted = (
                JAVA_TYPE_CONVERSION.get(nested_type_description)
                if nested_type_description is not None
                else None
            )
            if not isinstance(value, str):
                return False
            value = _java_type_converter(value, str(expected_type_description), nested_type_description)
        elif language == "javascript":
            if expected_type_description == "string":
                expected_type_description = _infer_js_type_from_candidates(candidates)
            if (
                expected_type_description in NESTED_CONVERSION_TYPE_LIST
                and nested_type_description is None
            ):
                nested_type_description = _infer_nested_type_from_candidates(
                    candidates, language
                )
            expected_type_converted = JS_TYPE_CONVERSION.get(
                str(expected_type_description), str
            )
            nested_type_converted = (
                JS_TYPE_CONVERSION.get(nested_type_description)
                if nested_type_description is not None
                else None
            )
            if not isinstance(value, str):
                return False
            value = _js_type_converter(value, str(expected_type_description), nested_type_description)
        else:
            expected_type_converted = PYTHON_TYPE_MAPPING.get(
                str(expected_type_description), str
            )
            nested_type_converted = None
            if str(expected_type_description) in PYTHON_NESTED_TYPE_CHECK_LIST:
                if nested_type_description is None:
                    nested_type_description = "any"
                nested_type_converted = PYTHON_TYPE_MAPPING.get(
                    nested_type_description, str
                )

            if expected_type_description == "tuple" and isinstance(value, tuple):
                value = list(value)
            if expected_type_description == "float" and isinstance(value, int):
                value = float(value)

        type_valid, is_variable = _type_checker(
            value=value,
            possible_answer=candidates,
            expected_type_converted=expected_type_converted,
            nested_type_converted=nested_type_converted,
        )
        if not type_valid:
            return False

        if not is_variable:
            if expected_type_converted is dict:
                if not isinstance(value, dict):
                    return False
                if not _dict_checker(value, candidates):
                    return False
                continue
            if expected_type_converted is list and nested_type_converted is dict:
                if not isinstance(value, list):
                    return False
                if not _list_dict_checker(value, candidates):
                    return False
                continue
            if expected_type_converted is str:
                if not isinstance(value, str):
                    return False
                if not _string_checker(value, candidates):
                    return False
                continue
            if expected_type_converted is list:
                if not isinstance(value, list):
                    return False
                if not _list_checker(value, candidates):
                    return False
                continue

        if value not in candidates:
            return False

    for param_name, candidates in expected_argument_candidates.items():
        if not isinstance(candidates, list):
            return False
        if param_name not in actual_arguments and "" not in candidates:
            return False

    return True


def _ast_call_matches(
    actual_call: dict[str, Any],
    expected_call: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    category_name: str | None = None,
) -> bool:
    if tools is not None and category_name is not None:
        return _ast_call_matches_strict(
            actual_call,
            expected_call,
            tools=tools,
            category_name=category_name,
        )

    if actual_call.get("function") != expected_call.get("function"):
        return False
    actual_arguments = actual_call.get("arguments", {})
    expected_argument_candidates = expected_call.get("arguments", {})
    if not isinstance(actual_arguments, dict) or not isinstance(
        expected_argument_candidates, dict
    ):
        return False

    if set(actual_arguments.keys()) - set(expected_argument_candidates.keys()):
        return False

    for arg_name, candidates in expected_argument_candidates.items():
        if not isinstance(candidates, list):
            return False

        if arg_name not in actual_arguments:
            if any(candidate == "" for candidate in candidates):
                continue
            return False

        actual_value = actual_arguments[arg_name]
        if not any(_ast_candidate_matches(actual_value, candidate) for candidate in candidates):
            return False

    return True


def _ast_candidate_matches(actual_value: Any, candidate: Any) -> bool:
    if candidate == "":
        return actual_value == "" or actual_value is None
    return _values_equal(actual_value, candidate)


def _values_equal(actual: Any, expected: Any) -> bool:
    actual_norm = _normalize_json_value(actual)
    expected_norm = _normalize_json_value(expected)

    if isinstance(actual_norm, bool) or isinstance(expected_norm, bool):
        return actual_norm == expected_norm

    if isinstance(actual_norm, (int, float)) and isinstance(expected_norm, (int, float)):
        return math.isclose(
            float(actual_norm),
            float(expected_norm),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    if isinstance(actual_norm, list) and isinstance(expected_norm, list):
        if len(actual_norm) != len(expected_norm):
            return False
        return all(
            _values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual_norm, expected_norm, strict=True)
        )

    if isinstance(actual_norm, dict) and isinstance(expected_norm, dict):
        if set(actual_norm.keys()) != set(expected_norm.keys()):
            return False
        return all(
            _values_equal(actual_norm[key], expected_norm[key])
            for key in expected_norm
        )

    return actual_norm == expected_norm


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_json_value(val) for key, val in value.items()}
    return value


def _call_to_string(call_obj: dict[str, Any]) -> str:
    function_name = call_obj.get("function", "<unknown>")
    arguments = call_obj.get("arguments", {})
    if not isinstance(arguments, dict):
        return f"{function_name}(<non-dict args>)"
    return tool_call_to_string(function_name, arguments)


def _validate_required_parameters(
    arguments: dict[str, Any],
    parameter_schema: dict[str, Any],
) -> str | None:
    if not isinstance(arguments, dict):
        return "Tool arguments must be an object"
    if not isinstance(parameter_schema, dict):
        return "Malformed parameter schema for rest category"

    properties = parameter_schema.get("properties", {})
    required = parameter_schema.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []

    for required_key in required:
        if required_key not in arguments:
            return f"Missing required parameter '{required_key}'"
        value = arguments[required_key]
        if value == "" or value is None:
            return f"Required parameter '{required_key}' is empty"

        child_schema = properties.get(required_key)
        if isinstance(child_schema, dict):
            nested_required_error = _validate_required_parameters(
                arguments=value if isinstance(value, dict) else {},
                parameter_schema=child_schema,
            )
            if nested_required_error is not None:
                return f"{required_key}.{nested_required_error}"

    return None


def _resolve_function_name(function_node: ast.expr) -> str:
    if isinstance(function_node, ast.Name):
        return function_node.id

    if isinstance(function_node, ast.Attribute):
        parts: list[str] = []
        current: ast.expr = function_node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            raise TypeError(
                f"Unsupported function target node: {type(function_node).__name__}"
            )
        parts.append(current.id)
        return ".".join(reversed(parts))

    raise TypeError(f"Unsupported function target node: {type(function_node).__name__}")


def _safe_eval_ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval_ast_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_ast_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _safe_eval_ast_value(key): _safe_eval_ast_value(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Set):
        return {_safe_eval_ast_value(item) for item in node.elts}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _safe_eval_ast_value(node.operand)
        if not isinstance(operand, (int, float)):
            raise ValueError(f"Unary operation expected number, got {type(operand).__name__}")
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _safe_eval_ast_value(node.left)
        right = _safe_eval_ast_value(node.right)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError(
                f"Binary operation expected numeric operands, got {type(left).__name__} and {type(right).__name__}"
            )
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    raise ValueError(f"Unsupported AST value node: {type(node).__name__}")


def _resolve_prompt_translations(
    category_names: list[str], prompt_language: str
) -> dict[str, str] | None:
    language = prompt_language.strip().lower()
    if language in {"", "en", "none"}:
        return None

    if language != "da":
        raise ValueError(
            f"Unsupported prompt_language '{prompt_language}'. Supported values: 'en', 'da'."
        )

    if set(category_names) != {"exec_simple"}:
        raise ValueError(
            "prompt_language='da' is currently only supported for category='exec_simple'."
        )

    return _load_exec_simple_danish_prompt_map()


@lru_cache(maxsize=1)
def _load_exec_simple_danish_prompt_map() -> dict[str, str]:
    if not EXEC_SIMPLE_DA_TRANSLATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Danish prompt file not found: {EXEC_SIMPLE_DA_TRANSLATIONS_PATH}"
        )

    data = json.loads(EXEC_SIMPLE_DA_TRANSLATIONS_PATH.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"Expected list in {EXEC_SIMPLE_DA_TRANSLATIONS_PATH}, got {type(data).__name__}"
        )

    prompt_map: dict[str, str] = {}
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Row {idx} in {EXEC_SIMPLE_DA_TRANSLATIONS_PATH} is not an object"
            )

        sample_id = row.get("id")
        prompt_da = row.get("prompt_da")
        if not isinstance(sample_id, str) or not isinstance(prompt_da, str):
            raise ValueError(
                f"Row {idx} in {EXEC_SIMPLE_DA_TRANSLATIONS_PATH} must have string keys 'id' and 'prompt_da'"
            )
        if sample_id in prompt_map:
            raise ValueError(
                f"Duplicate sample id '{sample_id}' in {EXEC_SIMPLE_DA_TRANSLATIONS_PATH}"
            )

        prompt_map[sample_id] = prompt_da

    expected_exec_simple_samples = 100
    if len(prompt_map) != expected_exec_simple_samples:
        raise ValueError(
            "Danish prompt map has "
            f"{len(prompt_map)} entries, expected {expected_exec_simple_samples}"
        )

    return prompt_map


def _convert_to_chat_messages(messages: list[dict[str, str]]) -> list[ChatMessage]:
    """Convert raw message dicts to Inspect ChatMessage objects."""
    chat_messages: list[ChatMessage] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            chat_messages.append(ChatMessageUser(content=content))
        elif role == "system":
            chat_messages.append(ChatMessageSystem(content=content))
        elif role == "assistant":
            chat_messages.append(ChatMessageAssistant(content=content))
        else:
            raise ValueError(f"Unexpected role in input messages: {role}")
    return chat_messages
