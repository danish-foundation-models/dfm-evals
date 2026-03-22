import pytest

from dfm_evals.tasks.bfcl.solver import create_tool_info_from_dict, get_type


def test_create_tool_info_from_dict_requires_parameters_mapping() -> None:
    with pytest.raises(ValueError, match="parameters"):
        create_tool_info_from_dict(  # type: ignore[arg-type]
            {
                "name": "search",
                "description": "Look things up.",
            }
        )


def test_create_tool_info_from_dict_builds_required_properties() -> None:
    tool = create_tool_info_from_dict(
        {
            "name": "search",
            "description": "Look things up.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    }
                },
                "required": ["query"],
            },
        }
    )

    assert tool.name == "search"
    assert tool.parameters.required == ["query"]
    assert tool.parameters.properties["query"].type == "string"


def test_get_type_rejects_unknown_types() -> None:
    with pytest.raises(ValueError, match="Invalid type"):
        get_type("mystery")
