import pytest

from strine.planner import VALID_TOOLS, AgentConfig, PlannerError, plan_agent
from strine.providers.base import NormalizedResponse, ToolCall
from tests.fakes import FakeProvider


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


def test_agent_config_to_dict_includes_provider_model_and_custom_tools_defaults():
    config = AgentConfig(name="foo", prompt="bar", tools=["sql"])
    assert config.to_dict() == {
        "name": "foo",
        "prompt": "bar",
        "tools": ["sql"],
        "custom_tools": [],
        "provider": "claude",
        "model": None,
    }


def test_plan_agent_records_the_resolved_model_from_the_provider():
    """O agent.json grava o modelo que foi de fato usado, não None — assim
    o agent continua reproduzível se o padrão do provider mudar depois."""
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])
    provider.model = "llama-3.3-70b-versatile"

    config = plan_agent("descrição qualquer", provider)

    assert config.model == "llama-3.3-70b-versatile"


def _plan_response(name, tools, prompt="You are helpful."):
    return NormalizedResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="create_agent_plan",
                input={"name": name, "prompt": prompt, "tools": tools},
            )
        ],
    )


def test_plan_agent_filters_invalid_tool_names():
    provider = FakeProvider(responses=[_plan_response("my-agent", ["sql", "not-a-real-tool"])])

    config = plan_agent("descrição qualquer", provider)

    assert config.tools == ["sql"]
    assert config.custom_tools == []


def test_plan_agent_stores_provider_name():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    config = plan_agent("descrição qualquer", provider)

    assert config.provider == "fake"


def test_plan_agent_accepts_new_tool_names():
    provider = FakeProvider(
        responses=[_plan_response("researcher", ["web_search", "http_request"])]
    )

    config = plan_agent("descrição qualquer", provider)

    assert set(config.tools) == {"web_search", "http_request"}


def test_plan_agent_rejects_unsafe_name_with_path_separator():
    provider = FakeProvider(responses=[_plan_response("../../evil", [])])

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_rejects_name_with_space():
    provider = FakeProvider(responses=[_plan_response("my agent", [])])

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_rejects_empty_name():
    provider = FakeProvider(responses=[_plan_response("", [])])

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_accepts_valid_slug_name():
    provider = FakeProvider(responses=[_plan_response("sales-analyzer", [])])

    config = plan_agent("descrição qualquer", provider)

    assert config.name == "sales-analyzer"


def test_plan_agent_raises_planner_error_when_no_tool_call_returned():
    provider = FakeProvider(
        responses=[NormalizedResponse(stop_reason="end_turn", text="oi", tool_calls=[])]
    )

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_wraps_provider_error():
    provider = FakeProvider(raise_error="API fora do ar")

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_forces_the_create_agent_plan_tool():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    plan_agent("descrição qualquer", provider)

    assert provider.calls[0]["force_tool"] == "create_agent_plan"
