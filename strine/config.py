import os

from dotenv import load_dotenv


class MissingAPIKeyError(RuntimeError):
    pass


def load_api_key() -> str:
    # override=True: o .env do projeto tem prioridade sobre uma variável de
    # ambiente já exportada no shell (evita usar uma key antiga/errada
    # silenciosamente quando o usuário configura uma nova no .env).
    load_dotenv(override=True)
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
