from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from strine.providers.base import NormalizedResponse, Provider, ProviderError, ToolCall

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self._client = genai.Client(api_key=api_key)
        self._model = model or DEFAULT_GEMINI_MODEL

    def build_user_message(self, text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    def create_message(
        self,
        system_prompt: str,
        messages: list,
        tools: Optional[list] = None,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        config_kwargs = {"system_instruction": system_prompt}

        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters_json_schema=tool.get("input_schema", {}),
                )
                for tool in tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]

        if force_tool:
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[force_tool],
                )
            )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=messages,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as exc:
            raise ProviderError(f"Erro ao chamar a API do Gemini: {exc}") from exc

        parts = response.candidates[0].content.parts or []

        tool_calls = []
        text_chunks = []
        for index, part in enumerate(parts):
            if part.function_call is not None:
                call = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=call.id or f"call_{index}",
                        name=call.name,
                        input=call.args or {},
                    )
                )
            elif part.text:
                text_chunks.append(part.text)

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return NormalizedResponse(
            stop_reason=stop_reason,
            text="\n".join(text_chunks).strip(),
            tool_calls=tool_calls,
        )

    def build_assistant_message(self, response: NormalizedResponse) -> types.Content:
        parts = []
        if response.text:
            parts.append(types.Part(text=response.text))
        for call in response.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(id=call.id, name=call.name, args=call.input)
                )
            )
        return types.Content(role="model", parts=parts)

    def build_tool_result_message(self, tool_results: list) -> types.Content:
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=result["tool_call_id"],
                    name=result["name"],
                    response={"output": result["content"]},
                )
            )
            for result in tool_results
        ]
        return types.Content(role="user", parts=parts)
