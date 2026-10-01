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

SYSTEM_PROMPT = """You are an "agent architect". Given a user's request in
natural language, decide how to assemble an AI agent:

1. Which tools this agent needs, choosing only from: "sql", "slack",
   "webhook", "http_request", "web_search", "send_email", "file_read",
   "file_write". Return an empty list if none are necessary.
2. A clear, specific system prompt for the agent that will be created,
   describing its role, scope, and how it should behave.
3. A short, descriptive name in slug format (lowercase, hyphens, no
   spaces), e.g. "sales-analyzer", "support-bot".

Only choose a tool when the user's request clearly needs it:
- "sql": the agent needs to query a database.
- "slack": the agent needs to post messages to Slack.
- "webhook": the agent needs to fire a simple HTTP call (POST with a fixed payload).
- "http_request": the agent needs to call an external HTTP API more flexibly (any method, headers, auth).
- "web_search": the agent needs to search for current information on the internet.
- "send_email": the agent needs to send emails.
- "file_read": the agent needs to read local files.
- "file_write": the agent needs to write/save local files.

Prefer "http_request" over "webhook" for new integrations, unless the
request is literally just a simple POST. Don't invent a need for tools
the user didn't ask for."""

AGENT_PLAN_TOOL = {
    "name": "create_agent_plan",
    "description": "Registers the decided plan for the agent: name, system prompt, and required tools.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name in slug format (lowercase, hyphens), e.g. sales-analyzer.",
            },
            "prompt": {
                "type": "string",
                "description": "Complete system prompt for the agent that will be created.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(VALID_TOOLS)},
                "description": "Tools required for this agent. Empty list if none are necessary.",
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
        raise PlannerError("The agent name cannot be empty.")

    if not re.match(r"^[a-z0-9_-]+$", name):
        raise PlannerError(
            f"The agent name '{name}' is invalid. "
            "Use only lowercase letters, digits, underscores, and hyphens."
        )


def _describe_user_tool_choice(fixed_tools: List[str]) -> str:
    listed = ", ".join(fixed_tools) if fixed_tools else "none"
    return (
        "\n\n[User decision: the agent must use EXACTLY these tools: "
        f"{listed}. Write the system prompt to match this list — if it's "
        "'none', the agent has no tools; if there are tools, the prompt "
        "should say it uses them when needed. Don't mention tools outside "
        "this list.]"
    )


def plan_agent(
    description: str,
    provider: Provider,
    fixed_tools: Optional[List[str]] = None,
) -> AgentConfig:
    """Calls the provider to decide tools + system prompt + agent name.

    Uses forced tool use (force_tool) instead of asking for "respond in
    JSON" and parsing manually — the schema guarantees the response
    format.

    fixed_tools, if passed, is the user's final decision (e.g. after
    seeing the catalog and changing the suggestion): the model writes the
    prompt already knowing these tools, and they — not whatever the model
    would have chosen — go into the result. Without this the prompt could
    keep saying "don't use tools" even after the user added one.
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
        raise PlannerError(f"Error calling the API: {exc}") from exc

    if not response.tool_calls:
        raise PlannerError("The model did not return a structured plan.")

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
