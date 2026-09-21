import pytest

from strine.custom_tools import CustomToolError, CustomToolSpec, generate_custom_tool
from strine.providers.base import NormalizedResponse, ToolCall
from tests.fakes import FakeProvider


def _tool_response(name, description, input_schema, code):
    return NormalizedResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="create_custom_tool",
                input={
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                    "code": code,
                },
            )
        ],
    )


def test_generate_custom_tool_returns_spec_for_valid_code():
    provider = FakeProvider(
        responses=[
            _tool_response(
                "parse_invoice_total",
                "Extrai o valor total de um texto de nota fiscal.",
                {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                "def execute(text):\n    return 'total: 0'\n",
            )
        ]
    )

    spec = generate_custom_tool("extrai o total de uma nota fiscal", provider)

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "parse_invoice_total"
    assert "def execute" in spec.code


def test_generate_custom_tool_forces_the_create_custom_tool_tool():
    provider = FakeProvider(
        responses=[
            _tool_response("valid_name", "desc", {"type": "object", "properties": {}}, "def execute():\n    return 'x'\n")
        ]
    )

    generate_custom_tool("qualquer coisa", provider)

    assert provider.calls[0]["force_tool"] == "create_custom_tool"


def test_generate_custom_tool_rejects_code_with_syntax_error():
    provider = FakeProvider(
        responses=[
            _tool_response("broken_tool", "Tool quebrada.", {"type": "object", "properties": {}}, "def execute(:\n    pass")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_code_without_execute_function():
    provider = FakeProvider(
        responses=[
            _tool_response("no_execute_tool", "Tool sem execute.", {"type": "object", "properties": {}}, "def other_function():\n    pass")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_execute_as_class_method():
    provider = FakeProvider(
        responses=[
            _tool_response(
                "bad_scope",
                "desc",
                {"type": "object", "properties": {}},
                "class Foo:\n    def execute(self, **kwargs):\n        return 'x'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_name_with_path_separator():
    provider = FakeProvider(
        responses=[
            _tool_response("../../evil", "desc", {"type": "object", "properties": {}}, "def execute():\n    return 'x'\n")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_input_schema_not_object_type():
    provider = FakeProvider(
        responses=[
            _tool_response("valid_name", "desc", {"type": "array"}, "def execute():\n    return 'x'\n")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_wraps_provider_error():
    provider = FakeProvider(raise_error="boom")

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_raises_when_no_tool_call_returned():
    provider = FakeProvider(
        responses=[NormalizedResponse(stop_reason="end_turn", text="oi", tool_calls=[])]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_execute_as_nested_function():
    """Test that execute defined as a nested function is rejected."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "nested_func_tool",
                "Tool com execute aninhada.",
                {"type": "object", "properties": {}},
                "def helper():\n    def execute(**kwargs):\n        return 'nested'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_async_execute():
    """Test that async def execute is rejected (only sync functions allowed)."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "async_tool",
                "Tool com async execute.",
                {"type": "object", "properties": {}},
                "async def execute(**kwargs):\n    return 'async'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_accepts_valid_module_level_execute():
    """Test that a valid module-level execute function is correctly accepted."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "valid_tool",
                "Tool válida.",
                {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
                "def execute(x):\n    return f'result: {x}'\n",
            )
        ]
    )

    spec = generate_custom_tool("qualquer coisa", provider)

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "valid_tool"
    assert spec.code == "def execute(x):\n    return f'result: {x}'\n"


def test_generate_custom_tool_rejects_name_with_forward_slash():
    """Test that a name containing forward slash is rejected."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "foo/bar",
                "Tool with slash.",
                {"type": "object", "properties": {}},
                "def execute():\n    return 'result'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_name_with_space():
    """Test that a name containing spaces is rejected."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "tool with spaces",
                "Tool with spaces.",
                {"type": "object", "properties": {}},
                "def execute():\n    return 'result'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_name_with_uppercase():
    """Test that a name containing uppercase letters is rejected (per slug convention)."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "ParseInvoice",
                "Tool with uppercase.",
                {"type": "object", "properties": {}},
                "def execute():\n    return 'result'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_accepts_valid_slug_with_underscores_and_hyphens():
    """Test that valid slug names with underscores and hyphens are accepted."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "parse_invoice-total",
                "Valid slug tool.",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "def execute(text):\n    return 'result'\n",
            )
        ]
    )

    spec = generate_custom_tool("qualquer coisa", provider)

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "parse_invoice-total"


def test_generate_custom_tool_rejects_input_schema_not_a_dict():
    """Test that a non-dict input_schema is rejected."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "bad_schema_tool",
                "Tool with malformed schema.",
                "not a dict",
                "def execute():\n    return 'result'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_input_schema_without_properties():
    """Test that an input_schema missing a dict 'properties' is rejected."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "bad_schema_tool",
                "Tool with malformed schema.",
                {"type": "object"},
                "def execute():\n    return 'result'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_empty_name():
    """Test that an empty name is rejected."""
    provider = FakeProvider(
        responses=[
            _tool_response(
                "",
                "Tool with empty name.",
                {"type": "object", "properties": {}},
                "def execute():\n    return 'result'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)
