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
            result = runner.invoke(app, ["describe", "um", "agent", "qualquer"], input="\n")

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
                input="parseia uma coisa\ny\n",
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
                input="parseia uma coisa\nn\n",
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
                app, ["describe", "--provider", "gemini", "um", "agent", "qualquer"], input="\n"
            )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("gemini")


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
                input="parseia uma coisa\n",
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
                input="parseia uma coisa\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []
