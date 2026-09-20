from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.custom_tools import CustomToolError, CustomToolSpec, generate_custom_tool


def _fake_response(input_dict):
    tool_use_block = SimpleNamespace(type="tool_use", input=input_dict)
    return SimpleNamespace(content=[tool_use_block])


def test_generate_custom_tool_returns_spec_for_valid_code():
    fake = _fake_response(
        {
            "name": "parse_invoice_total",
            "description": "Extrai o valor total de um texto de nota fiscal.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "code": "def execute(text):\n    return 'total: 0'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        spec = generate_custom_tool("extrai o total de uma nota fiscal", api_key="sk-ant-fake")

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "parse_invoice_total"
    assert "def execute" in spec.code


def test_generate_custom_tool_rejects_code_with_syntax_error():
    fake = _fake_response(
        {
            "name": "broken_tool",
            "description": "Tool quebrada.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute(:\n    pass",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_code_without_execute_function():
    fake = _fake_response(
        {
            "name": "no_execute_tool",
            "description": "Tool sem execute.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def other_function():\n    pass",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_wraps_api_error():
    import anthropic

    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = anthropic.APIError(
            "boom", request=SimpleNamespace(), body=None
        )
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_execute_as_class_method():
    """Test that execute defined as a method inside a class is rejected."""
    fake = _fake_response(
        {
            "name": "class_method_tool",
            "description": "Tool com execute como método de classe.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "class MyTool:\n    def execute(self, **kwargs):\n        return 'hi'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_execute_as_nested_function():
    """Test that execute defined as a nested function is rejected."""
    fake = _fake_response(
        {
            "name": "nested_func_tool",
            "description": "Tool com execute aninhada.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def helper():\n    def execute(**kwargs):\n        return 'nested'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_async_execute():
    """Test that async def execute is rejected (only sync functions allowed)."""
    fake = _fake_response(
        {
            "name": "async_tool",
            "description": "Tool com async execute.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "async def execute(**kwargs):\n    return 'async'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_accepts_valid_module_level_execute():
    """Test that a valid module-level execute function is correctly accepted."""
    fake = _fake_response(
        {
            "name": "valid_tool",
            "description": "Tool válida.",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "code": "def execute(x):\n    return f'result: {x}'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        spec = generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "valid_tool"
    assert spec.code == "def execute(x):\n    return f'result: {x}'\n"


def test_generate_custom_tool_rejects_name_with_path_separator():
    """Test that a name containing path separators is rejected."""
    fake = _fake_response(
        {
            "name": "../../evil",
            "description": "Malicious tool.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute():\n    return 'evil'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_name_with_forward_slash():
    """Test that a name containing forward slash is rejected."""
    fake = _fake_response(
        {
            "name": "foo/bar",
            "description": "Tool with slash.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_name_with_space():
    """Test that a name containing spaces is rejected."""
    fake = _fake_response(
        {
            "name": "tool with spaces",
            "description": "Tool with spaces.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_name_with_uppercase():
    """Test that a name containing uppercase letters is rejected (per slug convention)."""
    fake = _fake_response(
        {
            "name": "ParseInvoice",
            "description": "Tool with uppercase.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_accepts_valid_slug_with_underscores_and_hyphens():
    """Test that valid slug names with underscores and hyphens are accepted."""
    fake = _fake_response(
        {
            "name": "parse_invoice-total",
            "description": "Valid slug tool.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "code": "def execute(text):\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        spec = generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "parse_invoice-total"


def test_generate_custom_tool_rejects_input_schema_not_a_dict():
    """Test that a non-dict input_schema is rejected."""
    fake = _fake_response(
        {
            "name": "bad_schema_tool",
            "description": "Tool with malformed schema.",
            "input_schema": "not a dict",
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_input_schema_with_wrong_type():
    """Test that an input_schema whose 'type' isn't 'object' is rejected."""
    fake = _fake_response(
        {
            "name": "bad_schema_tool",
            "description": "Tool with malformed schema.",
            "input_schema": {"type": "string", "properties": {}},
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_input_schema_without_properties():
    """Test that an input_schema missing a dict 'properties' is rejected."""
    fake = _fake_response(
        {
            "name": "bad_schema_tool",
            "description": "Tool with malformed schema.",
            "input_schema": {"type": "object"},
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_empty_name():
    """Test that an empty name is rejected."""
    fake = _fake_response(
        {
            "name": "",
            "description": "Tool with empty name.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute():\n    return 'result'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")
