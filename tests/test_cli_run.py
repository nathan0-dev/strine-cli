import json
from unittest.mock import patch

from typer.testing import CliRunner

from strine.cli import app
from strine.runtime import AgentRuntimeError
from tests.fakes import FakeProvider
from tests.test_cli_describe import isolated_filesystem

runner = CliRunner()

_AGENT_JSON = {
    "name": "test-agent",
    "prompt": "You are helpful.",
    "tools": ["sql"],
    "custom_tools": [],
    "provider": "claude",
}


def _write_agent_json(path="test-agent.json"):
    with open(path, "w") as f:
        json.dump(_AGENT_JSON, f)
    return path


def test_run_missing_file_shows_friendly_error_and_exits_1():
    with isolated_filesystem():
        result = runner.invoke(app, ["run", "./does-not-exist.json"])

    assert result.exit_code == 1
    assert "não encontrado" in result.output.lower() or "not found" in result.output.lower()


def test_run_invalid_json_shows_friendly_error_and_exits_1():
    with isolated_filesystem():
        with open("broken.json", "w") as f:
            f.write("{not valid json")
        result = runner.invoke(app, ["run", "./broken.json"])

    assert result.exit_code == 1
    assert "inválido" in result.output.lower() or "invalid" in result.output.lower()


def test_run_missing_api_key_shows_friendly_error_and_exits_1():
    from strine.config import MissingAPIKeyError

    with isolated_filesystem():
        _write_agent_json()
        with patch("strine.cli.get_provider", side_effect=MissingAPIKeyError("sem key")):
            result = runner.invoke(app, ["run", "./test-agent.json"])

    assert result.exit_code == 1
    assert "sem key" in result.output


def test_run_executes_turn_and_exits_on_sair():
    with isolated_filesystem():
        _write_agent_json()
        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()),
            patch(
                "strine.cli.prepare_agent_tools",
                return_value=([], {}, []),
            ),
            patch("strine.cli.run_agent", return_value="Olá! Tudo certo.") as mock_run_agent,
        ):
            result = runner.invoke(
                app, ["run", "./test-agent.json"], input="oi\nsair\n"
            )

    assert result.exit_code == 0
    assert "Olá! Tudo certo." in result.output
    mock_run_agent.assert_called_once()
    _, kwargs = mock_run_agent.call_args
    assert kwargs.get("user_input") == "oi" or "oi" in mock_run_agent.call_args.args


def test_run_prints_warnings_from_prepare_agent_tools():
    with isolated_filesystem():
        _write_agent_json()
        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()),
            patch(
                "strine.cli.prepare_agent_tools",
                return_value=([], {}, ["Tool 'ghost' não pôde ser carregada."]),
            ),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

    assert result.exit_code == 0
    assert "ghost" in result.output


def test_run_handles_agent_runtime_error_without_crashing_repl():
    with isolated_filesystem():
        _write_agent_json()
        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()),
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch(
                "strine.cli.run_agent",
                side_effect=AgentRuntimeError("API fora do ar"),
            ),
        ):
            result = runner.invoke(
                app, ["run", "./test-agent.json"], input="oi\nsair\n"
            )

    assert result.exit_code == 0
    assert "API fora do ar" in result.output


def test_run_uses_provider_stored_in_agent_json():
    agent_with_gemini = dict(_AGENT_JSON, provider="gemini")
    with isolated_filesystem():
        with open("test-agent.json", "w") as f:
            json.dump(agent_with_gemini, f)

        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()) as mock_get_provider,
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch("strine.cli.run_agent", return_value="oi"),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("gemini", model=None)


def test_run_defaults_to_claude_when_agent_json_has_no_provider_key():
    agent_without_provider = {k: v for k, v in _AGENT_JSON.items() if k != "provider"}
    with isolated_filesystem():
        with open("test-agent.json", "w") as f:
            json.dump(agent_without_provider, f)

        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()) as mock_get_provider,
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch("strine.cli.run_agent", return_value="oi"),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("claude", model=None)


def test_run_passes_model_from_agent_json_to_get_provider():
    agent_with_model = dict(_AGENT_JSON, provider="groq", model="llama-3.3-70b-versatile")
    with isolated_filesystem():
        with open("test-agent.json", "w") as f:
            json.dump(agent_with_model, f)

        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()) as mock_get_provider,
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch("strine.cli.run_agent", return_value="oi"),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("groq", model="llama-3.3-70b-versatile")


def test_run_handles_agent_json_without_model_key():
    """agent.json criado antes da flag --model existir continua rodando."""
    legacy = {k: v for k, v in _AGENT_JSON.items() if k != "model"}
    with isolated_filesystem():
        with open("test-agent.json", "w") as f:
            json.dump(legacy, f)

        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()) as mock_get_provider,
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch("strine.cli.run_agent", return_value="oi"),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("claude", model=None)
