from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from strine.providers.base import NormalizedResponse, Provider, ProviderError, ToolCall

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self._client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_GEMINI_MODEL
        # Ids the Gemini API did not populate on FunctionCall.id, which we
        # invent only for internal correlation (runtime.py). These must
        # never be sent back to the API as if they were its own native ids.
        self._synthetic_tool_call_ids: set = set()

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
                model=self.model,
                contents=messages,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as exc:
            raise ProviderError(f"Error calling the Gemini API: {exc}") from exc

        if not response.candidates or response.candidates[0].content is None:
            raise ProviderError(
                "Gemini did not return any content — the response may have "
                "been blocked by safety filters."
            )

        parts = response.candidates[0].content.parts or []

        self._synthetic_tool_call_ids = set()
        tool_calls = []
        text_chunks = []
        for index, part in enumerate(parts):
            if part.function_call is not None:
                call = part.function_call
                native_id = call.id
                call_id = native_id or f"call_{index}"
                if not native_id:
                    self._synthetic_tool_call_ids.add(call_id)
                tool_calls.append(
                    ToolCall(
                        id=call_id,
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
            wire_id = None if call.id in self._synthetic_tool_call_ids else call.id
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(id=wire_id, name=call.name, args=call.input)
                )
            )
        return types.Content(role="model", parts=parts)

    def build_tool_result_messages(self, tool_results: list) -> list:
        # Gemini groups all function_response entries into a single Content.
        parts = []
        for result in tool_results:
            call_id = result["tool_call_id"]
            wire_id = None if call_id in self._synthetic_tool_call_ids else call_id
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=wire_id,
                        name=result["name"],
                        response={"output": result["content"]},
                    )
                )
            )
        return [types.Content(role="user", parts=parts)]
