import os
from typing import Optional

from strine.config import MissingAPIKeyError, load_project_env
from strine.providers.base import Provider
from strine.providers.claude import ClaudeProvider
from strine.providers.gemini import GeminiProvider
from strine.providers.openai_compatible import (
    GroqProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

PROVIDERS = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "gpt": OpenAIProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
}

_PROVIDER_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

DEFAULT_PROVIDER = os.getenv("STRINE_DEFAULT_PROVIDER", "claude")


class UnknownProviderError(RuntimeError):
    pass


def get_provider(name: str, model: Optional[str] = None) -> Provider:
    """Instantiates the right provider, already loaded with the API key from .env.

    model, if passed, overrides that provider's default model — essential
    for OpenRouter, which routes to hundreds of models.
    """
    if name not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"Provider '{name}' is not recognized. Available providers: {valid}."
        )

    load_project_env()

    env_var = _PROVIDER_ENV_VARS[name]
    api_key = os.getenv(env_var)
    if not api_key:
        raise MissingAPIKeyError(
            f"{env_var} is not configured.\n\n"
            f"Configure your API key before using the '{name}' provider:\n"
            f"  1. Copy .env.example to .env:  cp .env.example .env\n"
            f"  2. Edit .env and add:          {env_var}=...\n"
            f"  3. Or export it directly:      export {env_var}=...\n"
        )

    provider_cls = PROVIDERS[name]
    return provider_cls(api_key=api_key, model=model)
