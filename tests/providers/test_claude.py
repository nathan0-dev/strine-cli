from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.providers.base import ProviderError
from strine.providers.claude import ClaudeProvider


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id, name, input_dict):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_dict)


def _fake_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def test_build_user_message_returns_anthropic_shape():
    with patch("strine.providers.claude.anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="sk-ant-fake")

    assert provider.build_user_message("oi") == {"role": "user", "content": "oi"}


def test_create_message_returns_text_when_no_tool_call():
    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _fake_response(
            _text_block("Olá!")
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        response = provider.create_message("system", [provider.build_user_message("oi")])

    assert response.stop_reason == "end_turn"
    assert response.text == "Olá!"
    assert response.tool_calls == []


def test_create_message_returns_tool_calls():
    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _fake_response(
            _tool_use_block("toolu_1", "query_database", {"query": "SELECT 1"})
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "toolu_1"
    assert response.tool_calls[0].name == "query_database"
    assert response.tool_calls[0].input == {"query": "SELECT 1"}


def test_create_message_with_force_tool_passes_tool_choice():
    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _fake_response(
            _tool_use_block("toolu_1", "create_agent_plan", {})
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "create_agent_plan", "description": "...", "input_schema": {}}],
            force_tool="create_agent_plan",
        )

    call_kwargs = MockAnthropic.return_value.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "create_agent_plan"}


def test_create_message_wraps_api_error():
    import anthropic

    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = anthropic.APIError(
            "boom", request=SimpleNamespace(), body=None
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_build_assistant_message_includes_text_and_tool_calls():
    from strine.providers.base import NormalizedResponse, ToolCall

    with patch("strine.providers.claude.anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="sk-ant-fake")

    response = NormalizedResponse(
        stop_reason="tool_use",
        text="deixa eu checar",
        tool_calls=[ToolCall(id="toolu_1", name="query_database", input={"query": "SELECT 1"})],
    )
    message = provider.build_assistant_message(response)

    assert message["role"] == "assistant"
    assert {"type": "text", "text": "deixa eu checar"} in message["content"]
    assert {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "query_database",
        "input": {"query": "SELECT 1"},
    } in message["content"]


def test_build_tool_result_messages_groups_all_results_into_one_turn():
    """Claude agrupa todos os resultados de tool num único turno 'user' —
    diferente do formato OpenAI, que exige uma mensagem por resultado. Daí
    o contrato retornar uma lista: cada provider decide quantas mensagens
    seu formato nativo precisa."""
    with patch("strine.providers.claude.anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="sk-ant-fake")

    messages = provider.build_tool_result_messages(
        [
            {"tool_call_id": "toolu_1", "name": "query_database", "content": "42"},
            {"tool_call_id": "toolu_2", "name": "web_search", "content": "resultado"},
        ]
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "resultado"},
            ],
        }
    ]
