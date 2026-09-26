from dotenv import find_dotenv, load_dotenv

DEFAULT_MODEL = "claude-sonnet-5"


class MissingAPIKeyError(RuntimeError):
    pass


def load_project_env() -> None:
    """Carrega o .env do diretório onde o usuário RODOU o strine.

    usecwd=True: resolve o .env a partir do cwd, não de onde o pacote está
    instalado — sem isso, um .env perto do código-fonte (ex: clone de
    desenvolvimento) vazaria pra qualquer execução, em qualquer diretório.
    override=True: o .env do projeto tem prioridade sobre uma variável já
    exportada no shell (evita usar uma key antiga/errada silenciosamente).
    """
    load_dotenv(find_dotenv(usecwd=True), override=True)
