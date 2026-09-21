import importlib.util
from pathlib import Path
from typing import Callable, Optional

import anthropic

from strine.config import DEFAULT_MODEL
from strine.tools import TOOLS

MODEL = DEFAULT_MODEL
MAX_TOOL_ROUNDS = 5


class AgentRuntimeError(RuntimeError):
    pass


def prepare_agent_tools(agent_config: dict):
    """Monta os schemas de tools + mapa nome->execute pra um agent.

    Combina as tools pré-construídas (via registry central) com as tools
    customizadas do agent (import dinâmico do arquivo .py salvo na criação).
    Nunca falha a chamada inteira por causa de uma tool problemática — só
    registra um warning e segue sem ela.
    """
    tool_schemas = []
    executors = {}
    warnings = []

    for name in agent_config.get("tools", []):
        entry = TOOLS.get(name)
        if entry is None:
            warnings.append(f"Tool '{name}' não é reconhecida, ignorando.")
            continue
        schema = entry["schema"]
        tool_schemas.append(schema)
        executors[schema["name"]] = entry["execute"]

    for custom in agent_config.get("custom_tools", []):
        name = custom.get("name", "?")
        module_path = custom.get("module_path")

        try:
            if not module_path or not Path(module_path).is_file():
                raise FileNotFoundError(f"arquivo não encontrado: {module_path}")

            spec = importlib.util.spec_from_file_location(f"custom_tool_{name}", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"não foi possível carregar o módulo em {module_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            execute_fn = module.execute
        except Exception as exc:
            warnings.append(
                f"Tool customizada '{name}' não pôde ser carregada ({exc}). Ignorando."
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


def _call_claude(client, system_prompt: str, tool_schemas: list, messages: list):
    kwargs = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": messages,
    }
    if tool_schemas:
        kwargs["tools"] = tool_schemas

    try:
        return client.messages.create(**kwargs)
    except anthropic.APIError as exc:
        raise AgentRuntimeError(f"Erro ao chamar a API da Anthropic: {exc}") from exc


def run_agent(
    system_prompt: str,
    tool_schemas: list,
    executors: dict,
    user_input: str,
    api_key: str,
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> str:
    """Roda uma pergunta do usuário contra o agent, executando tools de verdade.

    Cada chamada é uma conversa nova (sem memória entre perguntas do REPL).
    Se o Claude pedir uma tool, ela é executada e o resultado volta pra ele,
    num loop limitado a MAX_TOOL_ROUNDS idas-e-voltas, pra nunca rodar
    indefinidamente.

    on_tool_call, se passado, é chamado com o nome da tool logo antes dela
    ser executada — permite ao chamador (cli.py) mostrar feedback visual
    sem que esse módulo precise fazer I/O diretamente.
    """
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_input}]

    response = _call_claude(client, system_prompt, tool_schemas, messages)

    rounds = 0
    while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            executor = executors.get(block.name)
            if executor is None:
                result_text = f"Tool '{block.name}' não está disponível."
            else:
                if on_tool_call is not None:
                    on_tool_call(block.name)
                try:
                    result_text = executor(**block.input)
                except Exception as exc:
                    result_text = f"Erro ao executar a tool '{block.name}': {exc}"

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
            )

        messages.append({"role": "user", "content": tool_results})
        rounds += 1
        response = _call_claude(client, system_prompt, tool_schemas, messages)

    final_text = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    if response.stop_reason == "tool_use" and rounds >= MAX_TOOL_ROUNDS:
        note = (
            "(O agent atingiu o limite de chamadas de tools nesta pergunta; "
            "a resposta pode estar incompleta.)"
        )
        final_text = f"{final_text}\n\n{note}" if final_text else note

    return final_text
