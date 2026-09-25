import json
from typing import Optional

import openai

from strine.providers.base import NormalizedResponse, Provider, ProviderError, ToolCall


class OpenAICompatibleProvider(Provider):
    """Base pros providers que falam o formato de function calling da OpenAI.

    GPT, Groq e OpenRouter usam exatamente a mesma API (Chat Completions) —
    só mudam a base_url e a API key. Por isso um adapter só atende os três:
    cada subclasse define apenas `name`, `base_url` e `default_model`.

    base_url None significa "usa o endpoint padrão da própria OpenAI".
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
        # No formato OpenAI o system prompt é a primeira mensagem do
        # histórico (diferente de Anthropic/Gemini, que têm campo próprio).
        # Monta uma lista nova pra não poluir o histórico do chamador.
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
            raise ProviderError(f"Erro ao chamar a API de {self.name}: {exc}") from exc

        if not response.choices:
            raise ProviderError(
                f"A API de {self.name} não retornou nenhuma resposta "
                "(a requisição pode ter sido filtrada)."
            )

        message = response.choices[0].message

        tool_calls = []
        for call in message.tool_calls or []:
            raw_arguments = call.function.arguments
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"A API de {self.name} retornou argumentos inválidos para a "
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
                        # a API espera os argumentos como string JSON
                        "arguments": json.dumps(call.input),
                    },
                }
                for call in response.tool_calls
            ]
        return message

    def build_tool_result_messages(self, tool_results: list) -> list:
        # O formato OpenAI exige uma mensagem separada por resultado,
        # ligada à chamada original pelo tool_call_id.
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
    default_model = "llama-3.3-70b-versatile"


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    # O OpenRouter roteia pra centenas de modelos — esse default é só um
    # ponto de partida sensato; use --model pra escolher qualquer outro.
    default_model = "openai/gpt-6-sol"
