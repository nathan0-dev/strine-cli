import ast
import re
from dataclasses import dataclass

import anthropic

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """Você escreve uma tool Python customizada pra um agent de IA,
a partir da descrição de um usuário.

Regras obrigatórias do código gerado:
- Defina exatamente uma função chamada `execute` que recebe os parâmetros
  descritos no seu próprio input_schema como keyword arguments e retorna
  uma string.
- Só pode importar a biblioteca padrão do Python ou o pacote `requests`.
  Nenhum outro pacote de terceiros está disponível.
- Nunca deixe uma exceção não tratada escapar de `execute` — capture erros
  internamente (try/except) e retorne uma mensagem de erro como string,
  igual o resto do código faria.
- O código deve ser completo e correto, sem placeholders."""

CREATE_CUSTOM_TOOL_TOOL = {
    "name": "create_custom_tool",
    "description": "Registra a tool customizada gerada: nome, descrição, schema de input e código Python.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Nome curto em formato slug (minúsculas, underscores), ex: parse_invoice_total.",
            },
            "description": {
                "type": "string",
                "description": "Descrição curta do que a tool faz, pro agent entender quando usá-la.",
            },
            "input_schema": {
                "type": "object",
                "description": "JSON schema (formato input_schema da Anthropic) dos parâmetros que a tool recebe.",
            },
            "code": {
                "type": "string",
                "description": "Código Python completo definindo a função execute(**kwargs) -> str.",
            },
        },
        "required": ["name", "description", "input_schema", "code"],
    },
}


@dataclass
class CustomToolSpec:
    name: str
    description: str
    input_schema: dict
    code: str


class CustomToolError(RuntimeError):
    pass


def _validate_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CustomToolError(f"O código gerado tem um erro de sintaxe: {exc}") from exc

    has_execute = any(
        isinstance(node, ast.FunctionDef) and node.name == "execute"
        for node in tree.body
    )
    if not has_execute:
        raise CustomToolError(
            "O código gerado não define uma função 'execute'."
        )


def _validate_name(name: str) -> None:
    """Validate that the tool name is a safe slug.

    Only lowercase letters, digits, underscores, and hyphens are allowed.
    Rejects names that could be used for path traversal attacks.
    """
    if not name:
        raise CustomToolError("O nome da tool não pode estar vazio.")

    if not re.match(r"^[a-z0-9_-]+$", name):
        raise CustomToolError(
            f"O nome da tool '{name}' é inválido. "
            "Use apenas letras minúsculas, dígitos, underscores e hífens."
        )


def generate_custom_tool(description: str, api_key: str) -> CustomToolSpec:
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[CREATE_CUSTOM_TOOL_TOOL],
            tool_choice={"type": "tool", "name": "create_custom_tool"},
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.APIError as exc:
        raise CustomToolError(f"Erro ao chamar a API da Anthropic: {exc}") from exc

    tool_use = next(
        (block for block in response.content if block.type == "tool_use"), None
    )
    if tool_use is None:
        raise CustomToolError("O modelo não retornou uma tool customizada estruturada.")

    plan = tool_use.input
    _validate_code(plan["code"])
    _validate_name(plan["name"])

    return CustomToolSpec(
        name=plan["name"],
        description=plan["description"],
        input_schema=plan["input_schema"],
        code=plan["code"],
    )
