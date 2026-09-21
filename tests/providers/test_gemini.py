from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.providers.base import ProviderError
from strine.providers.gemini import GeminiProvider


def _text_part(text):
    return SimpleNamespace(text=text, function_call=None)


def _function_call_part(name, args, call_id=None):
    call = SimpleNamespace(id=call_id, name=name, args=args)
    return SimpleNamespace(text=None, function_call=call)


def _fake_response(*parts):
    candidate = SimpleNamespace(content=SimpleNamespace(parts=list(parts)))
    return SimpleNamespace(candidates=[candidate])


def test_create_message_returns_text_when_no_function_call():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _text_part("Olá!")
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message("system", [provider.build_user_message("oi")])

    assert response.stop_reason == "end_turn"
    assert response.text == "Olá!"
    assert response.tool_calls == []


def test_create_message_returns_tool_calls_with_native_id():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("query_database", {"query": "SELECT 1"}, call_id="call_abc")
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {"type": "object", "properties": {}}}],
        )

    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_abc"
    assert response.tool_calls[0].name == "query_database"
    assert response.tool_calls[0].input == {"query": "SELECT 1"}


def test_create_message_generates_synthetic_id_when_missing():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("query_database", {"query": "SELECT 1"}, call_id=None)
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    assert response.tool_calls[0].id == "call_0"


def test_create_message_with_force_tool_sets_tool_config():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("create_agent_plan", {})
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "create_agent_plan", "description": "...", "input_schema": {}}],
            force_tool="create_agent_plan",
        )

    call_kwargs = MockClient.return_value.models.generate_content.call_args.kwargs
    tool_config = call_kwargs["config"].tool_config
    assert tool_config.function_calling_config.mode == "ANY"
    assert tool_config.function_calling_config.allowed_function_names == ["create_agent_plan"]


def test_create_message_wraps_api_error():
    from google.genai import errors as genai_errors

    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.side_effect = genai_errors.ClientError(
            code=401, response_json={"error": {"message": "boom"}}, response=SimpleNamespace()
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_build_tool_result_message_wraps_content_in_output_dict():
    with patch("strine.providers.gemini.genai.Client"):
        provider = GeminiProvider(api_key="fake-gemini-key")

    message = provider.build_tool_result_message(
        [{"tool_call_id": "call_abc", "name": "query_database", "content": "42"}]
    )

    part = message.parts[0]
    assert part.function_response.id == "call_abc"
    assert part.function_response.name == "query_database"
    assert part.function_response.response == {"output": "42"}


def test_native_tool_call_id_round_trips_into_wire_messages():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("query_database", {"query": "SELECT 1"}, call_id="call_abc")
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    assistant_message = provider.build_assistant_message(response)
    assert assistant_message.parts[0].function_call.id == "call_abc"

    tool_result_message = provider.build_tool_result_message(
        [{"tool_call_id": "call_abc", "name": "query_database", "content": "42"}]
    )
    assert tool_result_message.parts[0].function_response.id == "call_abc"


def test_synthetic_tool_call_id_is_not_echoed_back_to_the_wire():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("query_database", {"query": "SELECT 1"}, call_id=None)
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    # Internal bookkeeping still uses the synthetic id.
    assert response.tool_calls[0].id == "call_0"

    assistant_message = provider.build_assistant_message(response)
    assert assistant_message.parts[0].function_call.id is None

    tool_result_message = provider.build_tool_result_message(
        [{"tool_call_id": "call_0", "name": "query_database", "content": "42"}]
    )
    assert tool_result_message.parts[0].function_response.id is None


def test_create_message_raises_provider_error_on_empty_candidates():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = SimpleNamespace(
            candidates=[]
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_create_message_raises_provider_error_when_content_is_none():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        blocked_candidate = SimpleNamespace(content=None)
        MockClient.return_value.models.generate_content.return_value = SimpleNamespace(
            candidates=[blocked_candidate]
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])
