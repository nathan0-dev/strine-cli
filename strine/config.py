import os

from dotenv import find_dotenv, load_dotenv

DEFAULT_MODEL = "claude-sonnet-5"


class MissingAPIKeyError(RuntimeError):
    pass


def load_api_key() -> str:
    # usecwd=True: procura o .env a partir do diretório onde o usuário RODOU
    # o strine, não a partir de onde o pacote strine está instalado — sem
    # isso, um .env que exista por acaso perto do código-fonte (ex: clone de
    # desenvolvimento) vazaria pra qualquer execução do strine, em qualquer
    # diretório.
    # override=True: o .env do projeto tem prioridade sobre uma variável de
    # ambiente já exportada no shell (evita usar uma key antiga/errada
    # silenciosamente quando o usuário configura uma nova no .env).
    load_dotenv(find_dotenv(usecwd=True), override=True)
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
