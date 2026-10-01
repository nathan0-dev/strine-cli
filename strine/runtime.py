import importlib.util
from pathlib import Path
from typing import Callable, Optional

from strine.providers.base import Provider, ProviderError
from strine.tools import TOOLS

MAX_TOOL_ROUNDS = 5


class AgentRuntimeError(RuntimeError):
    pass


def prepare_agent_tools(agent_config: dict):
    """Builds the tool schemas + name->execute map for an agent.

    Combines the built-in tools (via the central registry) with the
    agent's custom tools (dynamic import of the .py file saved at
    creation time). Never fails the whole call because of one broken
    tool — it just logs a warning and continues without it.
    """
    tool_schemas = []
    executors = {}
    warnings = []

    for name in agent_config.get("tools", []):
        entry = TOOLS.get(name)
        if entry is None:
            warnings.append(f"Tool '{name}' is not recognized, skipping.")
            continue
        schema = entry["schema"]
        tool_schemas.append(schema)
        executors[schema["name"]] = entry["execute"]

    for custom in agent_config.get("custom_tools", []):
        name = custom.get("name", "?")
        module_path = custom.get("module_path")

        try:
            if not module_path or not Path(module_path).is_file():
                raise FileNotFoundError(f"file not found: {module_path}")

            spec = importlib.util.spec_from_file_location(f"custom_tool_{name}", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load the module at {module_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            execute_fn = module.execute
        except Exception as exc:
            warnings.append(
                f"Custom tool '{name}' could not be loaded ({exc}). Skipping."
            )
            continue

        tool_schemas.append(
            {
                "name": custom["name"],
                "description": custom.get("description", ""),
                "input_schema": custom.get("input_schema", {"type": "object", "properties": {}}),
            }
        )
        executors[name] = execute_fn

    return tool_schemas, executors, warnings


def run_agent(
    system_prompt: str,
    tool_schemas: list,
    executors: dict,
    user_input: str,
    provider: Provider,
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> str:
    """Runs a user's question against the agent, executing real tools.

    Each call is a fresh conversation (no memory across REPL questions).
    If the model requests a tool, it gets executed and the result goes
    back to it, in a loop capped at MAX_TOOL_ROUNDS round-trips so it
    never runs indefinitely.

    on_tool_call, if passed, is called with the tool's name right before
    it runs — lets the caller (cli.py) show visual feedback without this
    module doing I/O directly.
    """
    messages = [provider.build_user_message(user_input)]

    try:
        response = provider.create_message(system_prompt, messages, tools=tool_schemas)
    except ProviderError as exc:
        raise AgentRuntimeError(f"Error calling the API: {exc}") from exc

    rounds = 0
    while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
        messages.append(provider.build_assistant_message(response))

        tool_results = []
        for call in response.tool_calls:
            executor = executors.get(call.name)
            if executor is None:
                result_text = f"Tool '{call.name}' is not available."
            else:
                if on_tool_call is not None:
                    on_tool_call(call.name)
                try:
                    result_text = executor(**call.input)
                except Exception as exc:
                    result_text = f"Error executing tool '{call.name}': {exc}"

            tool_results.append(
                {"tool_call_id": call.id, "name": call.name, "content": result_text}
            )

        messages.extend(provider.build_tool_result_messages(tool_results))
        rounds += 1

        try:
            response = provider.create_message(system_prompt, messages, tools=tool_schemas)
        except ProviderError as exc:
            raise AgentRuntimeError(f"Error calling the API: {exc}") from exc

    final_text = response.text

    if response.stop_reason == "tool_use" and rounds >= MAX_TOOL_ROUNDS:
        note = (
            "(The agent hit the tool-call limit for this question; the "
            "response may be incomplete.)"
        )
        final_text = f"{final_text}\n\n{note}" if final_text else note

    return final_text
