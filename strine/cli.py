import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from strine.config import MissingAPIKeyError, load_project_env
from strine.custom_tools import CustomToolError, generate_custom_tool
from strine.planner import PlannerError, plan_agent
from strine.providers import DEFAULT_PROVIDER, PROVIDERS, UnknownProviderError, get_provider
from strine.runtime import AgentRuntimeError, prepare_agent_tools, run_agent
from strine.theme import GREEN, ORANGE, YELLOW, render_banner
from strine.tool_catalog import missing_env_vars, parse_tool_selection, required_env_vars
from strine.tools import TOOLS

_EXIT_WORDS = {"exit", "quit"}

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Tokens that must be treated as an explicit subcommand, never as the start
# of a natural-language description.
_RESERVED_FIRST_TOKENS = {"run", "describe", "tools", "--help", "-h"}


def _render_catalog(console: Console, selected: Optional[List[str]] = None) -> None:
    """Shows every built-in tool. With `selected`, marks (✓) the chosen ones."""
    show_marks = selected is not None
    table = Table(title="Available tools", title_justify="left")
    if show_marks:
        table.add_column("", no_wrap=True)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("What it does", overflow="fold")
    table.add_column("Credential", no_wrap=True)

    for key, entry in TOOLS.items():
        missing = missing_env_vars(key)
        if not required_env_vars(key):
            credential = "[dim]—[/dim]"
        elif missing:
            credential = f"[yellow]missing {', '.join(missing)}[/yellow]"
        else:
            credential = "[green]ok[/green]"

        row = [key, entry["schema"]["description"], credential]
        if show_marks:
            row.insert(0, "[bold green]✓[/bold green]" if key in selected else "")
        table.add_row(*row)

    console.print(table)


def _prompt_tool_selection(console: Console, suggested: List[str]) -> List[str]:
    """Asks which tools the agent should use. Enter keeps the suggestion."""
    while True:
        raw = typer.prompt(
            "\nAgent tools — Enter keeps the ones marked (✓); or type the "
            "names you want, separated by commas ('none' for none)",
            default="",
            show_default=False,
        )
        if not raw.strip():
            return list(suggested)

        selected, unknown = parse_tool_selection(raw, TOOLS.keys())
        if unknown:
            console.print(
                f"[red]Unknown tool: {', '.join(unknown)}. "
                f"Options: {', '.join(TOOLS)}[/red]"
            )
            continue
        return selected


@app.command(name="tools")
def list_tools() -> None:
    """Lists the tools that are ready to use and which credentials are missing."""
    load_project_env()
    console = Console()

    missing_tools = [key for key in TOOLS if missing_env_vars(key)]
    ready_count = len(TOOLS) - len(missing_tools)
    providers_label = " · ".join(PROVIDERS)

    render_banner(
        console,
        stats=[
            ("tools", f"{len(TOOLS)} built-in"),
            ("ready", f"[{GREEN}]{ready_count}[/{GREEN}] configured"),
            ("missing", f"[{YELLOW}]{len(missing_tools)}[/{YELLOW}] credentials"),
            ("providers", providers_label),
        ],
    )

    _render_catalog(console)
    console.print(
        "\n[dim]When creating an agent, you pick from these or describe a "
        "new one.[/dim]"
    )


@app.command(name="describe", hidden=True)
def describe_agent(
    description: List[str] = typer.Argument(
        ...,
        help="Natural-language description of the agent you want to create.",
    ),
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER,
        "--provider",
        help="AI provider to use (claude, gemini, gpt, groq, openrouter).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Specific model to use. Without this, uses the provider's default.",
    ),
) -> None:
    """Creates a new agent from a natural-language description."""
    console = Console()
    text = " ".join(description)

    # The catalog shows what's missing from .env — needs to be loaded
    # before this, regardless of how the provider gets resolved.
    load_project_env()

    try:
        provider = get_provider(provider_name, model=model)
    except UnknownProviderError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    try:
        with console.status("[bold cyan]Planning the agent...[/bold cyan]"):
            agent_config = plan_agent(text, provider)
    except PlannerError as exc:
        console.print(f"[red]Error planning the agent: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]✓[/bold green] Agent: [bold]{agent_config.name}[/bold]\n")
    _render_catalog(console, selected=agent_config.tools)

    selected_tools = _prompt_tool_selection(console, agent_config.tools)

    if set(selected_tools) != set(agent_config.tools):
        # The generated prompt was written for the suggested tools; if the
        # user changed the list, the model rewrites it already knowing the
        # final tools.
        try:
            with console.status(
                "[bold cyan]Adjusting the agent to the chosen tools...[/bold cyan]"
            ):
                agent_config = plan_agent(text, provider, fixed_tools=selected_tools)
        except PlannerError as exc:
            console.print(f"[red]Error adjusting the agent: {exc}[/red]")
            raise typer.Exit(code=1)

    tools_label = ", ".join(agent_config.tools) or "none"
    console.print(f"[bold green]✓[/bold green] Agent tools: [cyan]{tools_label}[/cyan]")
    for key in agent_config.tools:
        missing = missing_env_vars(key)
        if missing:
            console.print(
                f"[yellow]⚠ {key} is missing {', '.join(missing)} in .env — "
                "the agent won't be able to use this tool until you "
                "configure it.[/yellow]"
            )
    console.print(Panel(agent_config.prompt, title="Generated prompt", border_style="cyan"))

    custom_description = typer.prompt(
        "\nNone of these fit? Describe a new tool the agent needs "
        "(Enter to skip)",
        default="",
        show_default=False,
    )

    if custom_description.strip():
        custom_spec = None
        try:
            with console.status("[bold cyan]Generating custom tool...[/bold cyan]"):
                custom_spec = generate_custom_tool(custom_description, provider)
        except CustomToolError as exc:
            console.print(f"[red]Could not generate the custom tool: {exc}[/red]")

        if custom_spec is not None and custom_spec.name in TOOLS:
            console.print(
                f"[red]Could not use the custom tool: the name "
                f"'{custom_spec.name}' collides with a built-in tool "
                "that's already available. Skipped the custom tool.[/red]"
            )
            custom_spec = None

        if custom_spec is not None:
            console.print("\n[bold]Generated code for the custom tool:[/bold]\n")
            console.print(Syntax(custom_spec.code, "python"))
            tools_dir = Path(f"{agent_config.name}_tools")
            module_path = tools_dir / f"{custom_spec.name}.py"
            if typer.confirm(f"\nSave to {module_path}? (Y/n)", default=True):
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
                console.print(f"[bold green]✓[/bold green] Custom tool saved to {module_path}")
            else:
                console.print("Ok, continuing without that tool.")

    output_path = Path(f"{agent_config.name}.json")
    output_path.write_text(
        json.dumps(agent_config.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )

    console.print(
        f"\n[bold green]Agent saved to[/bold green] ./{output_path.name}. "
        f"Run it with: [bold]strine run ./{output_path.name}[/bold]"
    )


@app.command()
def run(
    config_path: str = typer.Argument(
        ...,
        help="Path to the agent's JSON config file (e.g. ./sales-analyzer.json).",
    ),
) -> None:
    """Runs an agent from a previously generated config file."""
    console = Console()
    path = Path(config_path)

    try:
        raw = path.read_text()
    except FileNotFoundError:
        console.print(f"[red]File not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        agent_config = json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid config file ({config_path}): {exc}[/red]")
        raise typer.Exit(code=1)

    provider_name = agent_config.get("provider", "claude")
    try:
        provider = get_provider(provider_name, model=agent_config.get("model"))
    except (UnknownProviderError, MissingAPIKeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    tool_schemas, executors, warnings = prepare_agent_tools(agent_config)
    for warning in warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    name = agent_config.get("name", "agent")
    tool_names = agent_config.get("tools", [])
    tools_label = ", ".join(tool_names) or "none"
    system_prompt = agent_config.get("prompt", "")

    render_banner(
        console,
        stats=[
            ("agent", name),
            ("provider", f"{provider.name} · {provider.model}"),
            ("tools", f"[{GREEN}]{len(tool_names)}[/{GREEN}] selected"),
            ("status", f"[{ORANGE}]●[/{ORANGE}] configured"),
        ],
    )
    console.print(f"Tools: [cyan]{tools_label}[/cyan]\n")
    console.print("Type your question, or 'exit' to quit.\n")

    def on_tool_call(tool_name: str) -> None:
        console.print(f"[{GREEN}]◆ using {tool_name}[/{GREEN}]")

    while True:
        try:
            user_input = typer.prompt("you")
        except (KeyboardInterrupt, EOFError, typer.Abort):
            console.print("\nBye!")
            break

        if user_input.strip().lower() in _EXIT_WORDS:
            console.print("Bye!")
            break

        if not user_input.strip():
            continue

        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                response_text = run_agent(
                    system_prompt=system_prompt,
                    tool_schemas=tool_schemas,
                    executors=executors,
                    user_input=user_input,
                    provider=provider,
                    on_tool_call=on_tool_call,
                )
        except AgentRuntimeError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            continue

        console.print(f"\n[bold cyan]agent[/bold cyan] {response_text}\n")


def main() -> None:
    """Real entry point of the package.

    Typer/Click doesn't handle a free positional argument coexisting with
    named subcommands well (e.g. "strine run x.json" vs "strine run to
    make money" — "run" would be ambiguous). To allow "strine
    <description>" as an implicit command, we rewrite argv to dispatch
    explicitly to the "describe" command when the first token isn't a
    reserved subcommand.
    """
    args = sys.argv[1:]
    if args and args[0] not in _RESERVED_FIRST_TOKENS:
        sys.argv = [sys.argv[0], "describe", *args]
    app()


if __name__ == "__main__":
    main()
