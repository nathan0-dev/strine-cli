import json
from typing import Optional

import openai

from strine.providers.base import NormalizedResponse, Provider, ProviderError, ToolCall


class OpenAICompatibleProvider(Provider):
    """Base for providers that speak OpenAI's function-calling format.

    GPT, Groq, and OpenRouter all use exactly the same API (Chat
    Completions) — only the base_url and API key differ. So one adapter
    serves all three: each subclass just sets `name`, `base_url`, and
    `default_model`.

    base_url None means "use OpenAI's own default endpoint".
    """

    base_url: Optional[str] = None
    default_model: str = ""

    def __init__(self, api_key: str, model: Optional[str] = None):
        self._client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
        self.model = model or self.default_model

    def build_user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def create_message(
        self,
        system_prompt: str,
        messages: list,
        tools: Optional[list] = None,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        # In the OpenAI format the system prompt is the first message in
        # the history (unlike Anthropic/Gemini, which have a dedicated
        # field). Build a new list so we don't pollute the caller's history.
        full_messages = [{"role": "system", "content": system_prompt}, *messages]

        kwargs = {"model": self.model, "messages": full_messages}

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
                for tool in tools
            ]

        if force_tool:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": force_tool},
            }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise ProviderError(f"Error calling the {self.name} API: {exc}") from exc

        if not response.choices:
            raise ProviderError(
                f"The {self.name} API did not return any response "
                "(the request may have been filtered)."
            )

        message = response.choices[0].message

        tool_calls = []
        for call in message.tool_calls or []:
            raw_arguments = call.function.arguments
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"The {self.name} API returned invalid arguments for "
                    f"tool '{call.function.name}': {exc}"
                ) from exc
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, input=arguments)
            )

        return NormalizedResponse(
            stop_reason="tool_use" if tool_calls else "end_turn",
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
        )

    def build_assistant_message(self, response: NormalizedResponse) -> dict:
        message = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # the API expects arguments as a JSON string
                        "arguments": json.dumps(call.input),
                    },
                }
                for call in response.tool_calls
            ]
        return message

    def build_tool_result_messages(self, tool_results: list) -> list:
        # The OpenAI format requires a separate message per result, linked
        # to the original call via tool_call_id.
        return [
            {
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"],
            }
            for result in tool_results
        ]


class OpenAIProvider(OpenAICompatibleProvider):
    name = "gpt"
    base_url = None
    default_model = "gpt-6-sol"


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
    default_model = "openai/gpt-oss-120b"


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    # OpenRouter routes to hundreds of models — this default is just a
    # sensible starting point; use --model to pick any other.
    default_model = "openai/gpt-6-sol"
