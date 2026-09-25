from typing import Optional

import anthropic

from strine.config import DEFAULT_MODEL
from strine.providers.base import NormalizedResponse, Provider, ProviderError, ToolCall


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def build_user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def create_message(
        self,
        system_prompt: str,
        messages: list,
        tools: Optional[list] = None,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", {}),
                }
                for tool in tools
            ]
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise ProviderError(f"Erro ao chamar a API da Anthropic: {exc}") from exc

        tool_calls = [
            ToolCall(id=block.id, name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        stop_reason = "tool_use" if tool_calls else "end_turn"

        return NormalizedResponse(stop_reason=stop_reason, text=text, tool_calls=tool_calls)

    def build_assistant_message(self, response: NormalizedResponse) -> dict:
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for call in response.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            )
        return {"role": "assistant", "content": content}

    def build_tool_result_messages(self, tool_results: list) -> list:
        # Claude agrupa todos os resultados num único turno 'user'.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result["tool_call_id"],
                        "content": result["content"],
                    }
                    for result in tool_results
                ],
            }
        ]
