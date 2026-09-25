import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.providers.base import NormalizedResponse, ProviderError, ToolCall
from strine.providers.openai_compatible import (
    GroqProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


def _tool_call(call_id, name, arguments):
    """arguments chega da API como STRING JSON, não como dict."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _fake_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_provider(cls=OpenAIProvider, model=None):
    with patch("strine.providers.openai_compatible.openai.OpenAI"):
        return cls(api_key="sk-fake", model=model)


# --- formato das mensagens ---


def test_build_user_message_returns_openai_shape():
    provider = _make_provider()

    assert provider.build_user_message("oi") == {"role": "user", "content": "oi"}


def test_build_assistant_message_serializes_tool_arguments_as_json_string():
    provider = _make_provider()

    message = provider.build_assistant_message(
        NormalizedResponse(
            stop_reason="tool_use",
            text="deixa eu checar",
            tool_calls=[
                ToolCall(id="call_1", name="query_database", input={"query": "SELECT 1"})
            ],
        )
    )

    assert message["role"] == "assistant"
    assert message["content"] == "deixa eu checar"
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "query_database",
                "arguments": json.dumps({"query": "SELECT 1"}),
            },
        }
    ]


def test_build_tool_result_messages_returns_one_message_per_result():
    """Diferença central do formato OpenAI: cada resultado é uma mensagem
    {"role": "tool"} separada, ligada pelo tool_call_id — Claude e Gemini
    agrupam tudo num turno só."""
    provider = _make_provider()

    messages = provider.build_tool_result_messages(
        [
            {"tool_call_id": "call_1", "name": "query_database", "content": "42"},
            {"tool_call_id": "call_2", "name": "web_search", "content": "resultado"},
        ]
    )

    assert messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        {"role": "tool", "tool_call_id": "call_2", "content": "resultado"},
    ]


# --- create_message ---


def test_create_message_returns_text_when_no_tool_call():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            content="Olá!"
        )
        provider = OpenAIProvider(api_key="sk-fake")
        response = provider.create_message("system", [provider.build_user_message("oi")])

    assert response.stop_reason == "end_turn"
    assert response.text == "Olá!"
    assert response.tool_calls == []


def test_create_message_parses_tool_calls_and_decodes_json_arguments():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            tool_calls=[
                _tool_call("call_1", "query_database", '{"query": "SELECT 1"}')
            ]
        )
        provider = OpenAIProvider(api_key="sk-fake")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "query_database"
    assert response.tool_calls[0].input == {"query": "SELECT 1"}


def test_create_message_translates_generic_tool_schema_to_openai_format():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            content="ok"
        )
        provider = OpenAIProvider(api_key="sk-fake")
        provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[
                {
                    "name": "query_database",
                    "description": "Consulta o banco.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
        )

    kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Consulta o banco.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]


def test_create_message_prepends_system_prompt_without_mutating_caller_messages():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            content="ok"
        )
        provider = OpenAIProvider(api_key="sk-fake")
        history = [provider.build_user_message("oi")]
        provider.create_message("Você é útil.", history)

    kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "Você é útil."}
    assert kwargs["messages"][1] == {"role": "user", "content": "oi"}
    # o histórico do chamador não pode ganhar a mensagem de system
    assert history == [{"role": "user", "content": "oi"}]


def test_create_message_with_force_tool_sets_tool_choice():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            tool_calls=[_tool_call("call_1", "create_agent_plan", "{}")]
        )
        provider = OpenAIProvider(api_key="sk-fake")
        provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "create_agent_plan", "description": "...", "input_schema": {}}],
            force_tool="create_agent_plan",
        )

    kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "create_agent_plan"},
    }


def test_create_message_wraps_api_error():
    import openai

    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.side_effect = openai.APIError(
            "boom", request=SimpleNamespace(), body=None
        )
        provider = OpenAIProvider(api_key="sk-fake")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_create_message_raises_provider_error_on_empty_choices():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[]
        )
        provider = OpenAIProvider(api_key="sk-fake")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_create_message_raises_provider_error_on_malformed_tool_arguments():
    """arguments malformado não pode deixar um JSONDecodeError cru escapar."""
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            tool_calls=[_tool_call("call_1", "query_database", "{nao eh json")]
        )
        provider = OpenAIProvider(api_key="sk-fake")
        with pytest.raises(ProviderError):
            provider.create_message(
                "system",
                [provider.build_user_message("oi")],
                tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
            )


def test_create_message_handles_tool_call_with_empty_arguments():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            tool_calls=[_tool_call("call_1", "sem_args", "")]
        )
        provider = OpenAIProvider(api_key="sk-fake")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "sem_args", "description": "...", "input_schema": {}}],
        )

    assert response.tool_calls[0].input == {}


# --- as três subclasses ---


def test_each_provider_has_its_own_name_and_base_url():
    assert OpenAIProvider.name == "gpt"
    assert OpenAIProvider.base_url is None  # usa o default da própria OpenAI

    assert GroqProvider.name == "groq"
    assert GroqProvider.base_url == "https://api.groq.com/openai/v1"

    assert OpenRouterProvider.name == "openrouter"
    assert OpenRouterProvider.base_url == "https://openrouter.ai/api/v1"


def test_provider_passes_its_base_url_to_the_sdk_client():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        GroqProvider(api_key="gsk-fake")

    MockOpenAI.assert_called_once_with(
        api_key="gsk-fake", base_url="https://api.groq.com/openai/v1"
    )


def test_each_provider_uses_its_own_default_model():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        for cls in (OpenAIProvider, GroqProvider, OpenRouterProvider):
            MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
                content="ok"
            )
            provider = cls(api_key="fake")
            provider.create_message("system", [provider.build_user_message("oi")])

            kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == cls.default_model
            assert cls.default_model, f"{cls.name} precisa de um modelo padrão"


def test_explicit_model_overrides_the_provider_default():
    with patch("strine.providers.openai_compatible.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _fake_response(
            content="ok"
        )
        provider = OpenRouterProvider(api_key="fake", model="meta-llama/llama-3.3-70b")
        provider.create_message("system", [provider.build_user_message("oi")])

    kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "meta-llama/llama-3.3-70b"
