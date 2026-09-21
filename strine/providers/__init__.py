import os

from dotenv import find_dotenv, load_dotenv

from strine.config import MissingAPIKeyError
from strine.providers.base import Provider
from strine.providers.claude import ClaudeProvider
from strine.providers.gemini import GeminiProvider

PROVIDERS = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
}

_PROVIDER_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

DEFAULT_PROVIDER = os.getenv("STRINE_DEFAULT_PROVIDER", "claude")


class UnknownProviderError(RuntimeError):
    pass


def get_provider(name: str) -> Provider:
    """Instancia o provider certo, já com a API key carregada do .env.

    usecwd=True: resolve o .env a partir do diretório onde o usuário RODOU
    o strine, não de onde o pacote está instalado (mesma proteção que
    load_api_key() já tinha). override=True: o .env do projeto tem
    prioridade sobre uma variável de ambiente já exportada no shell.
    """
    if name not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"Provider '{name}' não é reconhecido. Providers disponíveis: {valid}."
        )

    load_dotenv(find_dotenv(usecwd=True), override=True)

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
    return provider_cls(api_key=api_key)
