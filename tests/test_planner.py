from types import SimpleNamespace
from unittest.mock import patch

from strine.planner import VALID_TOOLS, AgentConfig, plan_agent


def test_valid_tools_includes_all_eight():
    assert VALID_TOOLS == {
        "sql",
        "slack",
        "webhook",
        "http_request",
        "web_search",
        "send_email",
        "file_read",
        "file_write",
    }


def test_agent_config_to_dict_includes_custom_tools_default_empty():
    config = AgentConfig(name="foo", prompt="bar", tools=["sql"])
    assert config.to_dict() == {
        "name": "foo",
        "prompt": "bar",
        "tools": ["sql"],
        "custom_tools": [],
    }


def _fake_response(input_dict):
    tool_use_block = SimpleNamespace(type="tool_use", input=input_dict)
    return SimpleNamespace(content=[tool_use_block])


def test_plan_agent_filters_invalid_tool_names():
    fake = _fake_response(
        {"name": "my-agent", "prompt": "You are helpful.", "tools": ["sql", "not-a-real-tool"]}
    )
    with patch("strine.planner.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        config = plan_agent("descrição qualquer", api_key="sk-ant-fake")

    assert config.tools == ["sql"]
    assert config.custom_tools == []


def test_plan_agent_accepts_new_tool_names():
    fake = _fake_response(
        {
            "name": "researcher",
            "prompt": "You research things.",
            "tools": ["web_search", "http_request"],
        }
    )
    with patch("strine.planner.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        config = plan_agent("descrição qualquer", api_key="sk-ant-fake")

    assert set(config.tools) == {"web_search", "http_request"}
