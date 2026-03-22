"""
Code adapted from the model handler to generate responses:

https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py
"""

from __future__ import annotations

from typing import Any, TypedDict, cast, get_args

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.tool import ToolInfo, ToolParam, ToolParams
from inspect_ai.util import JSONType  # type: ignore


class BFCLParamSchema(TypedDict, total=False):
    type: str
    description: str
    default: Any
    enum: list[Any]
    items: BFCLParamSchema
    properties: dict[str, BFCLParamSchema]
    additionalProperties: Any
    required: list[str]


class BFCLToolSchema(TypedDict):
    name: str
    description: str
    parameters: BFCLParamSchema


@solver
def bfcl_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tool_specs = state.metadata["tools"]
        assert isinstance(tool_specs, list)
        tool_infos = [
            create_tool_info_from_dict(tool_spec) for tool_spec in tool_specs
        ]
        state.tools.extend(tool_infos)  # type: ignore
        return await generate(state, tool_calls="none")

    return solve


def create_tool_info_from_dict(tool_dict: BFCLToolSchema) -> ToolInfo:
    """
    Create a ToolInfo instance from a dictionary.

    Args:
        tool_dict: Dictionary containing tool information

    Returns:
        ToolInfo instance
    """
    name = tool_dict.get("name")
    description = tool_dict.get("description")
    parameters_dict = tool_dict.get("parameters")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError("Tool schema requires string `name` and `description`.")
    if not isinstance(parameters_dict, dict):
        raise ValueError("Tool schema requires a `parameters` mapping.")

    parameters = create_tool_param(parameters_dict)
    if parameters.properties is None:
        raise ValueError(
            f"Malformed parameter field in tool_dict - no parameter properties: {parameters}"
        )
    if "additionalProperties" in parameters.properties:
        raise ValueError(
            f"Malformed parameter field in tool_dict - unexpected additional properties: {parameters}"  # TODO: investiate why this check occurs
        )

    tool_params = ToolParams(
        properties=parameters.properties,
        required=parameters.required or [],
    )
    # Create and return the ToolInfo instance
    return ToolInfo(
        name=name,
        description=description,
        parameters=tool_params,
    )


def create_tool_param(param_dict: BFCLParamSchema) -> ToolParam:
    properties_dict = param_dict.get("properties")
    if properties_dict is not None and not isinstance(properties_dict, dict):
        raise ValueError("Tool schema `properties` must be a mapping.")

    items_dict = param_dict.get("items")
    if items_dict is not None and not isinstance(items_dict, dict):
        raise ValueError("Tool schema `items` must be a mapping.")

    properties = None
    if properties_dict is not None:
        properties = {
            key: create_tool_param(value) for key, value in properties_dict.items()
        }

    items = None if items_dict is None else create_tool_param(items_dict)

    return ToolParam(
        type=get_type(param_dict.get("type")),
        description=param_dict.get("description"),
        default=param_dict.get("default"),
        enum=param_dict.get("enum"),
        items=items,
        properties=properties,  # type: ignore
        additionalProperties=param_dict.get("additionalProperties"),
        required=param_dict.get("required"),
    )


def get_type(bfcl_type: str | None) -> JSONType | None:
    if bfcl_type is None:
        return None
    if not isinstance(bfcl_type, str):
        raise ValueError(f"Invalid type: {bfcl_type}")

    normalized_type = bfcl_type.lower()
    match normalized_type:
        case "dict":
            json_type = "object"
        case "float":
            json_type = "number"
        case "tuple":
            json_type = "array"
        case "bool":
            json_type = "boolean"
        case "any":
            return None
        case _:
            json_type = normalized_type

    if json_type not in get_args(JSONType):
        raise ValueError(f"Invalid type: {json_type}")

    return cast(JSONType, json_type)
