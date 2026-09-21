from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.runtime import AgentRuntimeError, prepare_agent_tools, run_agent


# --- prepare_agent_tools ---


def test_prepare_agent_tools_builds_schemas_and_executors_for_builtin_tools():
    agent_config = {"tools": ["sql", "web_search"], "custom_tools": []}

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)

    schema_names = {schema["name"] for schema in tool_schemas}
    assert schema_names == {"query_database", "web_search"}
    assert set(executors.keys()) == {"query_database", "web_search"}
    assert callable(executors["query_database"])
    assert callable(executors["web_search"])
    assert warnings == []


def test_prepare_agent_tools_warns_and_skips_unknown_builtin_tool():
    agent_config = {"tools": ["sql", "not-a-real-tool"], "custom_tools": []}

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)

    schema_names = {schema["name"] for schema in tool_schemas}
    assert schema_names == {"query_database"}
    assert len(warnings) == 1
    assert "not-a-real-tool" in warnings[0]


def test_prepare_agent_tools_loads_valid_custom_tool(tmp_path):
    module_path = tmp_path / "parse_thing.py"
    module_path.write_text("def execute(x):\n    return f'got {x}'\n")

    agent_config = {
        "tools": [],
        "custom_tools": [
            {
                "name": "parse_thing",
                "description": "Parses a thing.",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
                "module_path": str(module_path),
            }
        ],
    }

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)

    assert warnings == []
    assert len(tool_schemas) == 1
    assert tool_schemas[0]["name"] == "parse_thing"
    assert executors["parse_thing"](x="hi") == "got hi"


def test_prepare_agent_tools_warns_and_skips_missing_custom_tool_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.py"

    agent_config = {
        "tools": ["sql"],
        "custom_tools": [
            {
                "name": "ghost_tool",
                "description": "Does not exist on disk.",
                "input_schema": {"type": "object", "properties": {}},
                "module_path": str(missing_path),
            }
        ],
    }

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)

    schema_names = {schema["name"] for schema in tool_schemas}
    assert schema_names == {"query_database"}
    assert "ghost_tool" not in executors
    assert len(warnings) == 1
    assert "ghost_tool" in warnings[0]


def test_prepare_agent_tools_handles_missing_custom_tools_key():
    agent_config = {"tools": ["sql"]}

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)

    assert {schema["name"] for schema in tool_schemas} == {"query_database"}
    assert warnings == []


# --- run_agent ---


def _text_response(text):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
    )


def _tool_use_response(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use", id=block_id, name=name, input=tool_input
            )
        ],
    )


def test_run_agent_returns_text_when_no_tool_needed():
    with patch("strine.runtime.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _text_response(
            "Olá! Como posso ajudar?"
        )
        result = run_agent(
            system_prompt="You are helpful.",
            tool_schemas=[],
            executors={},
            user_input="oi",
            api_key="sk-ant-fake",
        )

    assert result == "Olá! Como posso ajudar?"


def test_run_agent_executes_tool_and_returns_final_text():
    executed = {}

    def fake_execute(query):
        executed["query"] = query
        return "42"

    responses = [
        _tool_use_response("query_database", {"query": "SELECT 1"}),
        _text_response("O resultado é 42."),
    ]

    with patch("strine.runtime.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = responses
        result = run_agent(
            system_prompt="You are helpful.",
            tool_schemas=[{"name": "query_database", "description": "...", "input_schema": {}}],
            executors={"query_database": fake_execute},
            user_input="quantos usuários temos?",
            api_key="sk-ant-fake",
        )

    assert executed["query"] == "SELECT 1"
    assert result == "O resultado é 42."


def test_run_agent_handles_tool_execution_error_gracefully():
    def failing_execute(**kwargs):
        raise ValueError("boom")

    responses = [
        _tool_use_response("broken_tool", {}),
        _text_response("Tive um problema, mas seguimos."),
    ]

    with patch("strine.runtime.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = responses
        result = run_agent(
            system_prompt="You are helpful.",
            tool_schemas=[{"name": "broken_tool", "description": "...", "input_schema": {}}],
            executors={"broken_tool": failing_execute},
            user_input="usa a tool quebrada",
            api_key="sk-ant-fake",
        )

    assert result == "Tive um problema, mas seguimos."


def test_run_agent_stops_after_max_tool_rounds():
    call_count = {"n": 0}

    def always_tool_use(*args, **kwargs):
        call_count["n"] += 1
        return _tool_use_response("loopy_tool", {})

    with patch("strine.runtime.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = always_tool_use
        result = run_agent(
            system_prompt="You are helpful.",
            tool_schemas=[{"name": "loopy_tool", "description": "...", "input_schema": {}}],
            executors={"loopy_tool": lambda **kwargs: "still going"},
            user_input="loop forever",
            api_key="sk-ant-fake",
        )

    # initial call + 5 tool rounds = 6 calls total, never unbounded
    assert call_count["n"] == 6
    assert "limite" in result.lower()


def test_run_agent_raises_runtime_error_on_api_error():
    import anthropic

    with patch("strine.runtime.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = anthropic.APIError(
            "boom", request=SimpleNamespace(), body=None
        )
        with pytest.raises(AgentRuntimeError):
            run_agent(
                system_prompt="You are helpful.",
                tool_schemas=[],
                executors={},
                user_input="oi",
                api_key="sk-ant-fake",
            )
