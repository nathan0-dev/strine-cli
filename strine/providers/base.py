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
    """Erro de chamada de API, independente de qual provider está por trás."""


class Provider(ABC):
    name: str

    @abstractmethod
    def build_user_message(self, text: str) -> Any:
        """Monta a primeira mensagem (turno do usuário) no formato nativo do provider."""

    @abstractmethod
    def create_message(
        self,
        system_prompt: str,
        messages: list,
        tools: Optional[list] = None,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        """Faz uma chamada de API. tools usa o schema genérico
        {"name", "description", "input_schema"}. force_tool, se passado,
        força a chamada daquela tool específica. Deve capturar o erro
        nativo da SDK e levantar ProviderError."""

    @abstractmethod
    def build_assistant_message(self, response: NormalizedResponse) -> Any:
        """Monta o turno 'assistant' (com os tool_calls pedidos) no formato
        nativo do provider, pra ser anexado ao histórico."""

    @abstractmethod
    def build_tool_result_message(self, tool_results: list) -> Any:
        """tool_results: lista de {"tool_call_id": str, "name": str,
        "content": str}. Monta o turno de resultado de tool no formato
        nativo do provider."""
