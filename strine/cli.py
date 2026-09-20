import sys
from typing import List

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Nomes que devem ser tratados como subcomando explícito, nunca como início
# de uma descrição em linguagem natural.
_RESERVED_FIRST_TOKENS = {"run", "describe", "--help", "-h"}


@app.command(name="describe", hidden=True)
def describe_agent(
    description: List[str] = typer.Argument(
        ...,
        help="Descrição em linguagem natural do agent que você quer criar.",
    ),
) -> None:
    """Cria um novo agent a partir de uma descrição em linguagem natural."""
    text = " ".join(description)
    typer.echo("not implemented yet")


@app.command()
def run(
    config_path: str = typer.Argument(
        ...,
        help="Caminho pro arquivo JSON de config do agent (ex: ./sales-analyzer.json).",
    ),
) -> None:
    """Roda um agent a partir de um arquivo de config gerado anteriormente."""
    typer.echo("not implemented yet")


def main() -> None:
    """Entry point real do pacote.

    O Typer/Click não lida bem com um argumento posicional livre convivendo
    com subcomandos nomeados (ex: "strine run x.json" vs "strine run pra
    ganhar dinheiro" — o "run" seria ambíguo). Pra permitir
    "strine <descrição>" como comando implícito, reescrevemos o argv pra
    despachar explicitamente pro comando "describe" quando o primeiro token
    não é um subcomando reservado.
    """
    args = sys.argv[1:]
    if args and args[0] not in _RESERVED_FIRST_TOKENS:
        sys.argv = [sys.argv[0], "describe", *args]
    app()


if __name__ == "__main__":
    main()
