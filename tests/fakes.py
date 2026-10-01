from strine.providers.base import NormalizedResponse, Provider, ProviderError


class FakeProvider(Provider):
    """Test provider: returns programmed responses in sequence.

    Usage:
        provider = FakeProvider(responses=[
            NormalizedResponse(stop_reason="end_turn", text="hi", tool_calls=[]),
        ])
        provider.create_message(...)  # returns the first response in the list

    If `raise_error` is passed, create_message raises ProviderError
    instead of returning a response (to test the error path).
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, responses=None, raise_error=None, name_override=None):
        self._responses = list(responses or [])
        self._raise_error = raise_error
        self.calls = []  # every create_message call is recorded here
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
            raise AssertionError("FakeProvider ran out of programmed responses")
        return self._responses.pop(0)

    def build_assistant_message(self, response: NormalizedResponse) -> dict:
        return {"role": "assistant", "tool_calls": list(response.tool_calls), "text": response.text}

    def build_tool_result_messages(self, tool_results: list) -> list:
        return [{"role": "tool_result", "results": list(tool_results)}]
