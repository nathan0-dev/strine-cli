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
