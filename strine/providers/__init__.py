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
    """Instancia o provider certo, já com a API key carregada do .env.

    model, se passado, sobrescreve o modelo padrão daquele provider —
    essencial pro OpenRouter, que roteia pra centenas de modelos.

    """
    if name not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"Provider '{name}' não é reconhecido. Providers disponíveis: {valid}."
        )

    load_project_env()

    env_var = _PROVIDER_ENV_VARS[name]
    api_key = os.getenv(env_var)
    if not api_key:
        raise MissingAPIKeyError(
            f"{env_var} não configurada.\n\n"
            f"Configure sua chave da API antes de usar o provider '{name}':\n"
            f"  1. Copie .env.example para .env:  cp .env.example .env\n"
            f"  2. Edite .env e adicione:          {env_var}=...\n"
            f"  3. Ou exporte diretamente:          export {env_var}=...\n"
        )

    provider_cls = PROVIDERS[name]
    return provider_cls(api_key=api_key, model=model)
