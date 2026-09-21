import json
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from strine.config import MissingAPIKeyError
from strine.providers import get_provider
from strine.custom_tools import CustomToolError, generate_custom_tool
from strine.planner import PlannerError, plan_agent
from strine.runtime import AgentRuntimeError, prepare_agent_tools, run_agent
from strine.tools import TOOLS

_EXIT_WORDS = {"sair", "exit"}

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
    console = Console()
    text = " ".join(description)

    try:
        provider = get_provider("claude")
        api_key = provider.api_key
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    try:
        with console.status("[bold cyan]Planejando o agent...[/bold cyan]"):
            agent_config = plan_agent(text, api_key)
    except PlannerError as exc:
        console.print(f"[red]Erro ao planejar o agent: {exc}[/red]")
        raise typer.Exit(code=1)

    tools_label = ", ".join(agent_config.tools) or "nenhuma"
    console.print(f"[bold green]✓[/bold green] Agent: [bold]{agent_config.name}[/bold]")
    console.print(f"[bold green]✓[/bold green] Tools escolhidas: [cyan]{tools_label}[/cyan]")
    console.print(Panel(agent_config.prompt, title="Prompt gerado", border_style="cyan"))

    custom_description = typer.prompt(
        "\nQuer adicionar uma tool customizada? Descreva o que ela precisa "
        "fazer (Enter pra pular)",
        default="",
        show_default=False,
    )

    if custom_description.strip():
        custom_spec = None
        try:
            with console.status("[bold cyan]Gerando tool customizada...[/bold cyan]"):
                custom_spec = generate_custom_tool(custom_description, api_key)
        except CustomToolError as exc:
            console.print(f"[red]Não foi possível gerar a tool customizada: {exc}[/red]")

        if custom_spec is not None and custom_spec.name in TOOLS:
            console.print(
                f"[red]Não foi possível usar a tool customizada: o nome "
                f"'{custom_spec.name}' colide com uma tool nativa já "
                "disponível. Pulei a tool customizada.[/red]"
            )
            custom_spec = None

        if custom_spec is not None:
            console.print("\n[bold]Código gerado para a tool customizada:[/bold]\n")
            console.print(Syntax(custom_spec.code, "python"))
            tools_dir = Path(f"{agent_config.name}_tools")
            module_path = tools_dir / f"{custom_spec.name}.py"
            if typer.confirm(f"\nSalvar em {module_path}? (Y/n)", default=True):
                tools_dir.mkdir(parents=True, exist_ok=True)
                module_path.write_text(custom_spec.code)

                agent_config.custom_tools.append(
                    {
                        "name": custom_spec.name,
                        "description": custom_spec.description,
                        "input_schema": custom_spec.input_schema,
                        "module_path": str(module_path),
                    }
                )
                console.print(f"[bold green]✓[/bold green] Tool customizada salva em {module_path}")
            else:
                console.print("Ok, seguindo sem essa tool.")

    output_path = Path(f"{agent_config.name}.json")
    output_path.write_text(
        json.dumps(agent_config.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )

    console.print(
        f"\n[bold green]Agent salvo em[/bold green] ./{output_path.name}. "
        f"Rode com: [bold]strine run ./{output_path.name}[/bold]"
    )


@app.command()
def run(
    config_path: str = typer.Argument(
        ...,
        help="Caminho pro arquivo JSON de config do agent (ex: ./sales-analyzer.json).",
    ),
) -> None:
    """Roda um agent a partir de um arquivo de config gerado anteriormente."""
    console = Console()
    path = Path(config_path)

    try:
        raw = path.read_text()
    except FileNotFoundError:
        console.print(f"[red]Arquivo não encontrado: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        agent_config = json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Arquivo de config inválido ({config_path}): {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        provider = get_provider("claude")
        api_key = provider.api_key
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)
    for warning in warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    name = agent_config.get("name", "agent")
    tools_label = ", ".join(agent_config.get("tools", [])) or "nenhuma"
    system_prompt = agent_config.get("prompt", "")

    console.print(
        Panel(f"[bold]{name}[/bold]\nTools: [cyan]{tools_label}[/cyan]", title="Agent")
    )
    console.print("Digite sua pergunta, ou 'sair' pra encerrar.\n")

    def on_tool_call(tool_name: str) -> None:
        console.print(f"[dim]🔧 Usando {tool_name}...[/dim]")

    while True:
        try:
            user_input = typer.prompt("Você")
        except (KeyboardInterrupt, EOFError, typer.Abort):
            console.print("\nAté mais!")
            break

        if user_input.strip().lower() in _EXIT_WORDS:
            console.print("Até mais!")
            break

        if not user_input.strip():
            continue

        try:
            with console.status("[bold cyan]Pensando...[/bold cyan]"):
                response_text = run_agent(
                    system_prompt=system_prompt,
                    tool_schemas=tool_schemas,
                    executors=executors,
                    user_input=user_input,
                    api_key=api_key,
                    on_tool_call=on_tool_call,
                )
        except AgentRuntimeError as exc:
            console.print(f"[red]Erro: {exc}[/red]")
            continue

        console.print(f"\n[bold cyan]Agent:[/bold cyan] {response_text}\n")


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
