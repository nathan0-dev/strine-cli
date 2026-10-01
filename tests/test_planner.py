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


# --- fixed_tools: o usuário decidiu as tools, o prompt precisa ser coerente ---


def test_plan_agent_with_fixed_tools_overrides_what_the_model_chose():
    provider = FakeProvider(responses=[_plan_response("my-agent", ["sql"])])

    config = plan_agent(
        "descrição qualquer", provider, fixed_tools=["http_request", "file_read"]
    )

    assert config.tools == ["http_request", "file_read"]


def test_plan_agent_with_empty_fixed_tools_means_no_tools():
    provider = FakeProvider(responses=[_plan_response("my-agent", ["sql"])])

    config = plan_agent("descrição qualquer", provider, fixed_tools=[])

    assert config.tools == []


def test_plan_agent_fixed_tools_ignores_names_outside_the_catalog():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    config = plan_agent("descrição qualquer", provider, fixed_tools=["sql", "bogus"])

    assert config.tools == ["sql"]


def test_plan_agent_tells_the_model_which_tools_the_user_chose():
    """Sem isso o prompt gerado pode continuar dizendo "não use ferramentas"
    mesmo depois do usuário ter adicionado tools."""
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    plan_agent("um agent de pesquisa", provider, fixed_tools=["web_search"])

    sent = provider.calls[0]["messages"][0]["text"]
    assert "um agent de pesquisa" in sent
    assert "web_search" in sent


def test_plan_agent_with_empty_fixed_tools_tells_the_model_there_are_none():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    plan_agent("um agent qualquer", provider, fixed_tools=[])

    assert "none" in provider.calls[0]["messages"][0]["text"].lower()


def test_plan_agent_without_fixed_tools_sends_the_description_untouched():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    plan_agent("descrição qualquer", provider)

    assert provider.calls[0]["messages"][0]["text"] == "descrição qualquer"
