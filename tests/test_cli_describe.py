import contextlib
import json
import os
import tempfile
from unittest.mock import patch

from typer.testing import CliRunner

from strine.cli import app
from strine.custom_tools import CustomToolSpec
from strine.planner import AgentConfig
from tests.fakes import FakeProvider

runner = CliRunner()


@contextlib.contextmanager
def isolated_filesystem():
    """Drop-in replacement for CliRunner.isolated_filesystem().

    This project's pinned typer (0.27.x) vendors its own click
    implementation and no longer exposes isolated_filesystem() on
    CliRunner, so we recreate the same behavior: run the block inside a
    fresh temp directory and restore cwd afterward.
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            yield tmp_dir
        finally:
            os.chdir(old_cwd)


def _patch_common():
    return (
        patch("strine.cli.get_provider", return_value=FakeProvider(name_override="claude")),
        patch(
            "strine.cli.plan_agent",
            return_value=AgentConfig(name="test-agent", prompt="You help.", tools=["sql"]),
        ),
    )


def test_describe_skips_custom_tool_when_user_presses_enter():
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan:
            result = runner.invoke(app, ["describe", "um", "agent", "qualquer"], input="\n\n")

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []


def test_describe_saves_custom_tool_when_user_confirms():
    custom_spec = CustomToolSpec(
        name="parse_thing",
        description="Faz algo customizado.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        code="def execute(x):\n    return x\n",
    )
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan, patch("strine.cli.generate_custom_tool", return_value=custom_spec):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="\nparseia uma coisa\ny\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert len(saved["custom_tools"]) == 1
        assert saved["custom_tools"][0]["name"] == "parse_thing"
        assert saved["custom_tools"][0]["module_path"] == "test-agent_tools/parse_thing.py"

        with open("test-agent_tools/parse_thing.py") as f:
            assert "def execute(x):" in f.read()


def test_describe_discards_custom_tool_when_user_declines():
    custom_spec = CustomToolSpec(
        name="parse_thing",
        description="Faz algo customizado.",
        input_schema={"type": "object", "properties": {}},
        code="def execute():\n    return 'x'\n",
    )
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan, patch("strine.cli.generate_custom_tool", return_value=custom_spec):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="\nparseia uma coisa\nn\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []
        assert not os.path.exists("test-agent_tools")


def test_describe_passes_selected_provider_to_get_provider():
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key as mock_get_provider, patch_plan:
            result = runner.invoke(
                app, ["describe", "--provider", "gemini", "um", "agent", "qualquer"], input="\n\n"
            )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("gemini", model=None)


def test_describe_unknown_provider_shows_friendly_error():
    from strine.providers import UnknownProviderError

    with isolated_filesystem():
        with patch(
            "strine.cli.get_provider",
            side_effect=UnknownProviderError("Provider 'bogus' não é reconhecido."),
        ):
            result = runner.invoke(
                app, ["describe", "--provider", "bogus", "um", "agent", "qualquer"]
            )

        assert result.exit_code == 1
        assert "bogus" in result.output


def test_describe_skips_custom_tool_colliding_with_builtin_tool_name():
    custom_spec = CustomToolSpec(
        name="web_search",
        description="Colide com a tool nativa web_search.",
        input_schema={"type": "object", "properties": {}},
        code="def execute():\n    return 'x'\n",
    )
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan, patch("strine.cli.generate_custom_tool", return_value=custom_spec):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="\nparseia uma coisa\n",
            )

        assert result.exit_code == 0
        assert "web_search" in result.output
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []
        assert not os.path.exists("test-agent_tools")


def test_describe_handles_custom_tool_generation_failure_gracefully():
    from strine.custom_tools import CustomToolError

    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with (
            patch_key,
            patch_plan,
            patch("strine.cli.generate_custom_tool", side_effect=CustomToolError("falhou")),
        ):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="\nparseia uma coisa\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []


def test_describe_passes_model_flag_to_get_provider():
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key as mock_get_provider, patch_plan:
            result = runner.invoke(
                app,
                [
                    "describe",
                    "--provider",
                    "openrouter",
                    "--model",
                    "meta-llama/llama-3.3-70b",
                    "um",
                    "agent",
                ],
                input="\n\n",
            )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with(
            "openrouter", model="meta-llama/llama-3.3-70b"
        )


def test_describe_without_model_flag_passes_none():
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key as mock_get_provider, patch_plan:
            result = runner.invoke(app, ["describe", "um", "agent"], input="\n\n")

        assert result.exit_code == 0
        assert mock_get_provider.call_args.kwargs["model"] is None


# --- catálogo de tools + seleção interativa ---

_ALL_TOOL_KEYS = [
    "sql",
    "slack",
    "webhook",
    "http_request",
    "web_search",
    "send_email",
    "file_read",
    "file_write",
]


def _plan_that_honors_fixed_tools(suggested):
    """Simula o planner de verdade: sem fixed_tools devolve a sugestão; com
    fixed_tools, devolve exatamente o que o usuário escolheu."""

    def fake_plan(description, provider, fixed_tools=None):
        tools = list(suggested) if fixed_tools is None else list(fixed_tools)
        return AgentConfig(name="test-agent", prompt="You help.", tools=tools)

    return fake_plan


def _invoke_describe(input_text, suggested=("sql",)):
    fake_plan = _plan_that_honors_fixed_tools(suggested)
    with isolated_filesystem():
        with (
            patch("strine.cli.get_provider", return_value=FakeProvider(name_override="claude")),
            patch("strine.cli.plan_agent", side_effect=fake_plan) as mock_plan,
        ):
            result = runner.invoke(app, ["describe", "um", "agent"], input=input_text)
        saved = None
        if os.path.exists("test-agent.json"):
            saved = json.loads(open("test-agent.json").read())
    return result, saved, mock_plan


def test_describe_shows_the_whole_tool_catalog_before_asking_about_custom_tools():
    result, _, _ = _invoke_describe("\n\n")

    assert result.exit_code == 0
    for key in _ALL_TOOL_KEYS:
        assert key in result.output
    # the catalog appears before the custom-tool question
    assert result.output.index("file_write") < result.output.index("None of these fit")


def test_describe_enter_keeps_the_planners_suggestion_without_replanning():
    result, saved, mock_plan = _invoke_describe("\n\n", suggested=("sql",))

    assert result.exit_code == 0
    assert saved["tools"] == ["sql"]
    assert mock_plan.call_count == 1


def test_describe_lets_the_user_add_a_tool():
    result, saved, _ = _invoke_describe("sql, http_request\n\n", suggested=("sql",))

    assert result.exit_code == 0
    assert saved["tools"] == ["sql", "http_request"]


def test_describe_lets_the_user_remove_a_tool():
    result, saved, _ = _invoke_describe("http_request\n\n", suggested=("sql", "web_search"))

    assert result.exit_code == 0
    assert saved["tools"] == ["http_request"]


def test_describe_none_clears_every_tool():
    result, saved, _ = _invoke_describe("none\n\n", suggested=("sql",))

    assert result.exit_code == 0
    assert saved["tools"] == []


def test_describe_replans_with_the_users_tools_when_the_selection_changed():
    """Senão o prompt gerado pode dizer 'não use ferramentas' mesmo depois do
    usuário ter adicionado uma."""
    result, saved, mock_plan = _invoke_describe("http_request\n\n", suggested=())

    assert result.exit_code == 0
    assert mock_plan.call_count == 2
    assert mock_plan.call_args_list[0].kwargs.get("fixed_tools") is None
    assert mock_plan.call_args_list[1].kwargs["fixed_tools"] == ["http_request"]
    assert saved["tools"] == ["http_request"]


def test_describe_does_not_replan_when_only_the_order_changed():
    result, _, mock_plan = _invoke_describe("web_search, sql\n\n", suggested=("sql", "web_search"))

    assert result.exit_code == 0
    assert mock_plan.call_count == 1


def test_describe_rejects_an_unknown_tool_name_and_asks_again():
    result, saved, _ = _invoke_describe("bogus\nsql\n\n", suggested=("web_search",))

    assert result.exit_code == 0
    assert "bogus" in result.output
    assert saved["tools"] == ["sql"]


def test_describe_warns_when_a_chosen_tool_is_missing_its_credential(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result, _, _ = _invoke_describe("\n\n", suggested=("web_search",))

    assert result.exit_code == 0
    assert "TAVILY_API_KEY" in result.output
    assert "⚠" in result.output


def test_describe_does_not_warn_when_the_credential_is_configured(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    result, _, _ = _invoke_describe("\n\n", suggested=("web_search",))

    assert result.exit_code == 0
    assert "TAVILY_API_KEY" not in result.output
    assert "⚠" not in result.output


def test_describe_does_not_warn_about_tools_that_need_no_credential(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result, _, _ = _invoke_describe("\n\n", suggested=("http_request",))

    assert result.exit_code == 0
    assert "⚠" not in result.output


# --- comando `strine tools` ---


def test_tools_command_lists_the_whole_catalog_and_creates_nothing():
    with isolated_filesystem():
        result = runner.invoke(app, ["tools"])
        created = os.listdir(".")

    assert result.exit_code == 0
    for key in _ALL_TOOL_KEYS:
        assert key in result.output
    assert created == []


def test_tools_command_shows_which_credentials_are_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with isolated_filesystem():
        result = runner.invoke(app, ["tools"])

    assert "TAVILY_API_KEY" in result.output


def test_tools_command_shows_configured_credentials_as_ok(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    with isolated_filesystem():
        result = runner.invoke(app, ["tools"])

    assert "TAVILY_API_KEY" not in result.output


def test_tools_is_a_reserved_subcommand_not_the_start_of_a_description():
    """main() reescreve o argv pra tratar qualquer primeiro token não
    reservado como descrição — 'tools' precisa estar na lista."""
    from strine.cli import _RESERVED_FIRST_TOKENS

    assert "tools" in _RESERVED_FIRST_TOKENS
