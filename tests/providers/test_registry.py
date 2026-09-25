from unittest.mock import patch

import pytest

from strine.config import MissingAPIKeyError
from strine.providers import PROVIDERS, UnknownProviderError, get_provider
from strine.providers.claude import ClaudeProvider
from strine.providers.gemini import GeminiProvider
from strine.providers.openai_compatible import (
    GroqProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


def test_providers_registry_has_all_five_providers():
    assert PROVIDERS == {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "gpt": OpenAIProvider,
        "groq": GroqProvider,
        "openrouter": OpenRouterProvider,
    }


def test_get_provider_unknown_name_lists_every_valid_provider():
    with pytest.raises(UnknownProviderError) as exc_info:
        get_provider("bogus")

    message = str(exc_info.value)
    for name in ("claude", "gemini", "gpt", "groq", "openrouter"):
        assert name in message


@pytest.mark.parametrize(
    "provider_name,env_var",
    [
        ("gpt", "OPENAI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_openai_compatible_providers_use_their_own_env_var(
    provider_name, env_var, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_provider(provider_name)

    assert env_var in str(exc_info.value)


def test_get_provider_passes_model_through_to_the_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")

    with patch("strine.providers.openai_compatible.openai.OpenAI"):
        provider = get_provider("openrouter", model="meta-llama/llama-3.3-70b")

    assert provider.model == "meta-llama/llama-3.3-70b"


def test_get_provider_without_model_uses_the_provider_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")

    with patch("strine.providers.openai_compatible.openai.OpenAI"):
        provider = get_provider("groq")

    assert provider.model == GroqProvider.default_model


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
