from unittest.mock import patch

import pytest

from strine.config import MissingAPIKeyError
from strine.providers import PROVIDERS, UnknownProviderError, get_provider
from strine.providers.claude import ClaudeProvider
from strine.providers.gemini import GeminiProvider


def test_providers_registry_has_claude_and_gemini():
    assert PROVIDERS == {"claude": ClaudeProvider, "gemini": GeminiProvider}


def test_get_provider_unknown_name_raises_friendly_error():
    with pytest.raises(UnknownProviderError) as exc_info:
        get_provider("bogus")

    assert "claude" in str(exc_info.value)
    assert "gemini" in str(exc_info.value)


def test_get_provider_claude_uses_anthropic_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")

    provider = get_provider("claude")

    assert isinstance(provider, ClaudeProvider)
    assert provider.name == "claude"


def test_get_provider_claude_missing_key_raises_missing_api_key_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_provider("claude")

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_get_provider_gemini_missing_key_raises_missing_api_key_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_provider("gemini")

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_get_provider_prefers_dotenv_in_current_working_directory(tmp_path, monkeypatch):
    """Mesma proteção que já existia em load_api_key(): resolve o .env a
    partir do cwd de quem roda `strine`, não de onde o pacote está
    instalado. Mocka o construtor da SDK da Anthropic pra capturar
    exatamente qual api_key foi usada — sem isso o teste não prova nada
    além de "não levantou exceção"."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-project-dotenv\n")

    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        get_provider("claude")

    MockAnthropic.assert_called_once_with(api_key="sk-ant-from-project-dotenv")
