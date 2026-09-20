import os

from dotenv import load_dotenv


class MissingAPIKeyError(RuntimeError):
    pass


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY não configurada.\n\n"
            "Configure sua chave da API Anthropic antes de usar o Strine:\n"
            "  1. Copie .env.example para .env:  cp .env.example .env\n"
            "  2. Edite .env e adicione:          ANTHROPIC_API_KEY=sk-ant-...\n"
            "  3. Ou exporte diretamente:          export ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Gere uma chave em: https://console.anthropic.com/settings/keys"
        )
    return api_key
