from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class NormalizedResponse:
    stop_reason: str  # "tool_use" | "end_turn"
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)


class ProviderError(RuntimeError):
    """API call error, independent of which provider is behind it."""


class Provider(ABC):
    name: str
    # Model actually in use (the provider's default, or what came via
    # --model). Public because agent.json stores this resolved value,
    # keeping a saved agent reproducible even if the default changes later.
    model: str

    @abstractmethod
    def build_user_message(self, text: str) -> Any:
        """Builds the first message (user turn) in the provider's native format."""

    @abstractmethod
    def create_message(
        self,
        system_prompt: str,
        messages: list,
        tools: Optional[list] = None,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        """Makes an API call. tools uses the generic schema
        {"name", "description", "input_schema"}. force_tool, if passed,
        forces that specific tool to be called. Must catch the SDK's
        native error and raise ProviderError."""

    @abstractmethod
    def build_assistant_message(self, response: NormalizedResponse) -> Any:
        """Builds the 'assistant' turn (with the requested tool_calls) in
        the provider's native format, to be appended to the history."""

    @abstractmethod
    def build_tool_result_messages(self, tool_results: list) -> List[Any]:
        """tool_results: list of {"tool_call_id": str, "name": str,
        "content": str}. Builds the tool-result turn(s) in the provider's
        native format.

        Returns a LIST because formats diverge: Claude and Gemini group
        all results into a single turn, while the OpenAI format requires
        a separate {"role": "tool"} message per result. Callers extend(),
        not append()."""
