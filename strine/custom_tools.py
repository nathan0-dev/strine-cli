import ast
import re
from dataclasses import dataclass

from strine.providers.base import Provider, ProviderError

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


def _validate_input_schema(input_schema: dict) -> None:
    """Validate that input_schema is a plausible JSON schema.

    Must be a dict, with "type" == "object" and a "properties" dict —
    the minimal shape the Anthropic tool-use input_schema format requires.
    """
    if not isinstance(input_schema, dict):
        raise CustomToolError(
            "O input_schema gerado é inválido: deve ser um objeto JSON."
        )

    if input_schema.get("type") != "object":
        raise CustomToolError(
            "O input_schema gerado é inválido: 'type' deve ser 'object'."
        )

    if not isinstance(input_schema.get("properties"), dict):
        raise CustomToolError(
            "O input_schema gerado é inválido: 'properties' deve ser um objeto."
        )


def generate_custom_tool(description: str, provider: Provider) -> CustomToolSpec:
    try:
        response = provider.create_message(
            system_prompt=SYSTEM_PROMPT,
            messages=[provider.build_user_message(description)],
            tools=[CREATE_CUSTOM_TOOL_TOOL],
            force_tool="create_custom_tool",
        )
    except ProviderError as exc:
        raise CustomToolError(f"Erro ao chamar a API: {exc}") from exc

    if not response.tool_calls:
        raise CustomToolError("O modelo não retornou uma tool customizada estruturada.")

    plan = response.tool_calls[0].input
    _validate_code(plan["code"])
    _validate_name(plan["name"])
    _validate_input_schema(plan["input_schema"])

    return CustomToolSpec(
        name=plan["name"],
        description=plan["description"],
        input_schema=plan["input_schema"],
        code=plan["code"],
    )
