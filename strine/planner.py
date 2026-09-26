import re
from dataclasses import dataclass, field
from typing import List, Optional

from strine.providers.base import Provider, ProviderError

VALID_TOOLS = {
    "sql",
    "slack",
    "webhook",
    "http_request",
    "web_search",
    "send_email",
    "file_read",
    "file_write",
}

SYSTEM_PROMPT = """Você é um "agent architect". Dado o pedido de um usuário em
linguagem natural, decida como montar um agent de IA:

1. Quais tools esse agent precisa, escolhendo apenas entre: "sql", "slack",
   "webhook", "http_request", "web_search", "send_email", "file_read",
   "file_write". Se nenhuma for necessária, retorne uma lista vazia.
2. Um system prompt claro e específico para o agent que vai ser criado,
   descrevendo seu papel, escopo e como deve se comportar.
3. Um nome curto e descritivo em formato slug (minúsculas, hífens, sem
   espaços), ex: "sales-analyzer", "support-bot".

Escolha uma tool apenas quando o pedido do usuário claramente precisar dela:
- "sql": o agent precisa consultar um banco de dados.
- "slack": o agent precisa postar mensagens no Slack.
- "webhook": o agent precisa disparar uma chamada HTTP simples (POST com payload fixo).
- "http_request": o agent precisa chamar uma API HTTP externa de forma mais flexível (qualquer método, headers, auth).
- "web_search": o agent precisa pesquisar informação atual na internet.
- "send_email": o agent precisa enviar emails.
- "file_read": o agent precisa ler arquivos locais.
- "file_write": o agent precisa escrever/salvar arquivos locais.

Prefira "http_request" a "webhook" para integrações novas, a menos que o
pedido seja literalmente só um POST simples. Não invente necessidade de
tools que o usuário não pediu."""

AGENT_PLAN_TOOL = {
    "name": "create_agent_plan",
    "description": "Registra o plano decidido para o agent: nome, system prompt e tools necessárias.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Nome curto em formato slug (minúsculas, hífens), ex: sales-analyzer.",
            },
            "prompt": {
                "type": "string",
                "description": "System prompt completo para o agent que será criado.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(VALID_TOOLS)},
                "description": "Tools necessárias para esse agent. Lista vazia se nenhuma for necessária.",
            },
        },
        "required": ["name", "prompt", "tools"],
    },
}


@dataclass
class AgentConfig:
    name: str
    prompt: str
    tools: List[str] = field(default_factory=list)
    custom_tools: List[dict] = field(default_factory=list)
    provider: str = "claude"
    model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "tools": self.tools,
            "custom_tools": self.custom_tools,
            "provider": self.provider,
            "model": self.model,
        }


class PlannerError(RuntimeError):
    pass


def _validate_name(name: str) -> None:
    """Validate that the agent name is a safe slug.

    Only lowercase letters, digits, underscores, and hyphens are allowed.
    This mirrors custom_tools.py's _validate_name — the agent name is used
    to build filesystem paths (e.g. "{name}_tools/", "{name}.json"), so it
    needs the same protection against path traversal / unsafe characters.
    """
    if not name:
        raise PlannerError("O nome do agent não pode estar vazio.")

    if not re.match(r"^[a-z0-9_-]+$", name):
        raise PlannerError(
            f"O nome do agent '{name}' é inválido. "
            "Use apenas letras minúsculas, dígitos, underscores e hífens."
        )


def _describe_user_tool_choice(fixed_tools: List[str]) -> str:
    listed = ", ".join(fixed_tools) if fixed_tools else "nenhuma"
    return (
        "\n\n[Decisão do usuário: o agent deve usar EXATAMENTE estas tools: "
        f"{listed}. Escreva o system prompt coerente com essa lista — se for "
        "'nenhuma', o agent não tem ferramentas; se houver tools, o prompt "
        "deve dizer que ele as usa quando precisar. Não mencione tools fora "
        "da lista.]"
    )


def plan_agent(
    description: str,
    provider: Provider,
    fixed_tools: Optional[List[str]] = None,
) -> AgentConfig:
    """Chama o provider pra decidir tools + system prompt + nome do agent.

    Usa tool use forçado (force_tool) em vez de pedir "responda em JSON" e
    fazer parsing manual — o schema garante o formato da resposta.

    fixed_tools, se passado, é a decisão final do usuário (ex: depois de
    ver o catálogo e trocar a sugestão): o modelo escreve o prompt já sabendo
    dessas tools, e elas — não o que o modelo escolheria — vão pro resultado.
    Sem isso o prompt poderia continuar dizendo "não use ferramentas" mesmo
    depois do usuário ter adicionado uma.
    """
    user_text = description
    if fixed_tools is not None:
        user_text += _describe_user_tool_choice(fixed_tools)

    try:
        response = provider.create_message(
            system_prompt=SYSTEM_PROMPT,
            messages=[provider.build_user_message(user_text)],
            tools=[AGENT_PLAN_TOOL],
            force_tool="create_agent_plan",
        )
    except ProviderError as exc:
        raise PlannerError(f"Erro ao chamar a API: {exc}") from exc

    if not response.tool_calls:
        raise PlannerError("O modelo não retornou um plano estruturado.")

    plan = response.tool_calls[0].input
    chosen = plan.get("tools", []) if fixed_tools is None else fixed_tools
    tools = [t for t in chosen if t in VALID_TOOLS]

    _validate_name(plan["name"])

    return AgentConfig(
        name=plan["name"],
        prompt=plan["prompt"],
        tools=tools,
        provider=provider.name,
        model=getattr(provider, "model", None),
    )
