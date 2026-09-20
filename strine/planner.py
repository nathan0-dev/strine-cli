from dataclasses import dataclass, field
from typing import List

import anthropic

MODEL = "claude-sonnet-5"

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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "tools": self.tools,
            "custom_tools": self.custom_tools,
        }


class PlannerError(RuntimeError):
    pass


def plan_agent(description: str, api_key: str) -> AgentConfig:
    """Chama o Claude pra decidir tools + system prompt + nome do agent.

    Usa tool use forçado (tool_choice) em vez de pedir "responda em JSON" e
    fazer parsing manual — o schema garante o formato da resposta.
    """
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[AGENT_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "create_agent_plan"},
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.APIError as exc:
        raise PlannerError(f"Erro ao chamar a API da Anthropic: {exc}") from exc

    tool_use = next(
        (block for block in response.content if block.type == "tool_use"), None
    )
    if tool_use is None:
        raise PlannerError("O modelo não retornou um plano estruturado.")

    plan = tool_use.input
    tools = [t for t in plan.get("tools", []) if t in VALID_TOOLS]

    return AgentConfig(name=plan["name"], prompt=plan["prompt"], tools=tools)
