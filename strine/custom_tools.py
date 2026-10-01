import ast
import re
from dataclasses import dataclass

from strine.providers.base import Provider, ProviderError

SYSTEM_PROMPT = """You write a custom Python tool for an AI agent, based
on a user's description.

Mandatory rules for the generated code:
- Define exactly one function called `execute` that receives the
  parameters described in its own input_schema as keyword arguments and
  returns a string.
- You may only import Python's standard library or the `requests`
  package. No other third-party package is available.
- Never let an unhandled exception escape `execute` — catch errors
  internally (try/except) and return an error message as a string, the
  same way the rest of the code does.
- The code must be complete and correct, with no placeholders."""

CREATE_CUSTOM_TOOL_TOOL = {
    "name": "create_custom_tool",
    "description": "Registers the generated custom tool: name, description, input schema, and Python code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name in slug format (lowercase, underscores), e.g. parse_invoice_total.",
            },
            "description": {
                "type": "string",
                "description": "Short description of what the tool does, so the agent knows when to use it.",
            },
            "input_schema": {
                "type": "object",
                "description": "JSON schema (Anthropic input_schema format) of the parameters the tool receives.",
            },
            "code": {
                "type": "string",
                "description": "Complete Python code defining the execute(**kwargs) -> str function.",
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
        raise CustomToolError(f"The generated code has a syntax error: {exc}") from exc

    has_execute = any(
        isinstance(node, ast.FunctionDef) and node.name == "execute"
        for node in tree.body
    )
    if not has_execute:
        raise CustomToolError(
            "The generated code does not define an 'execute' function."
        )


def _validate_name(name: str) -> None:
    """Validate that the tool name is a safe slug.

    Only lowercase letters, digits, underscores, and hyphens are allowed.
    Rejects names that could be used for path traversal attacks.
    """
    if not name:
        raise CustomToolError("The tool name cannot be empty.")

    if not re.match(r"^[a-z0-9_-]+$", name):
        raise CustomToolError(
            f"The tool name '{name}' is invalid. "
            "Use only lowercase letters, digits, underscores, and hyphens."
        )


def _validate_input_schema(input_schema: dict) -> None:
    """Validate that input_schema is a plausible JSON schema.

    Must be a dict, with "type" == "object" and a "properties" dict —
    the minimal shape the Anthropic tool-use input_schema format requires.
    """
    if not isinstance(input_schema, dict):
        raise CustomToolError(
            "The generated input_schema is invalid: it must be a JSON object."
        )

    if input_schema.get("type") != "object":
        raise CustomToolError(
            "The generated input_schema is invalid: 'type' must be 'object'."
        )

    if not isinstance(input_schema.get("properties"), dict):
        raise CustomToolError(
            "The generated input_schema is invalid: 'properties' must be an object."
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
        raise CustomToolError(f"Error calling the API: {exc}") from exc

    if not response.tool_calls:
        raise CustomToolError("The model did not return a structured custom tool.")

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
