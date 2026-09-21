import json
from unittest.mock import patch

from typer.testing import CliRunner

from strine.cli import app
from strine.runtime import AgentRuntimeError
from tests.test_cli_describe import isolated_filesystem

runner = CliRunner()

_AGENT_JSON = {
    "name": "test-agent",
    "prompt": "You are helpful.",
    "tools": ["sql"],
    "custom_tools": [],
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
    from unittest.mock import MagicMock
    with isolated_filesystem():
        _write_agent_json()
        mock_provider = MagicMock()
        mock_provider.api_key = "sk-ant-fake"
        with (
            patch("strine.cli.get_provider", return_value=mock_provider),
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
    from unittest.mock import MagicMock
    with isolated_filesystem():
        _write_agent_json()
        mock_provider = MagicMock()
        mock_provider.api_key = "sk-ant-fake"
        with (
            patch("strine.cli.get_provider", return_value=mock_provider),
            patch(
                "strine.cli.prepare_agent_tools",
                return_value=([], {}, ["Tool 'ghost' não pôde ser carregada."]),
            ),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

    assert result.exit_code == 0
    assert "ghost" in result.output


def test_run_handles_agent_runtime_error_without_crashing_repl():
    from unittest.mock import MagicMock
    with isolated_filesystem():
        _write_agent_json()
        mock_provider = MagicMock()
        mock_provider.api_key = "sk-ant-fake"
        with (
            patch("strine.cli.get_provider", return_value=mock_provider),
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
