from strine.providers.base import NormalizedResponse, Provider, ProviderError


class FakeProvider(Provider):
    """Provider de teste: devolve respostas programadas em sequência.

    Uso:
        provider = FakeProvider(responses=[
            NormalizedResponse(stop_reason="end_turn", text="oi", tool_calls=[]),
        ])
        provider.create_message(...)  # devolve a primeira resposta da lista

    Se `raise_error` for passado, create_message levanta ProviderError em
    vez de devolver uma resposta (pra testar o caminho de erro).
    """

    name = "fake"

    def __init__(self, responses=None, raise_error=None, name_override=None):
        self._responses = list(responses or [])
        self._raise_error = raise_error
        self.calls = []  # cada chamada a create_message fica registrada aqui
        if name_override is not None:
            self.name = name_override

    def build_user_message(self, text: str) -> dict:
        return {"role": "user", "text": text}

    def create_message(self, system_prompt, messages, tools=None, force_tool=None) -> NormalizedResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": list(messages),
                "tools": tools,
                "force_tool": force_tool,
            }
        )
        if self._raise_error is not None:
            raise ProviderError(self._raise_error)
        if not self._responses:
            raise AssertionError("FakeProvider ficou sem respostas programadas")
        return self._responses.pop(0)

    def build_assistant_message(self, response: NormalizedResponse) -> dict:
        return {"role": "assistant", "tool_calls": list(response.tool_calls), "text": response.text}

    def build_tool_result_messages(self, tool_results: list) -> list:
        return [{"role": "tool_result", "results": list(tool_results)}]
