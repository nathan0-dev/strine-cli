# Multi-Provider Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o Strine falar com Claude ou Gemini por trás de uma interface comum (`Provider`), escolhido por `--provider` na criação do agent e salvo no `agent.json`, sem duplicar o loop de tool-calling que já existe.

**Architecture:** Um pacote novo `strine/providers/` com uma interface fina (`Provider`) — uma chamada de API (`create_message`) que normaliza a resposta (`NormalizedResponse`), mais três métodos pra montar mensagens no formato nativo de cada provider (`build_user_message`/`build_assistant_message`/`build_tool_result_message`). `planner.py`, `custom_tools.py` e `runtime.py` passam a receber um `Provider` já pronto em vez de instanciar `anthropic.Anthropic()` direto — o loop de round-trips em `runtime.py` continua com a mesma estrutura de hoje, só troca as chamadas diretas à SDK por chamadas genéricas ao provider.

**Tech Stack:** Python, SDK `anthropic` (já em uso), SDK `google-genai` (nova, pacote atual do Google — não a `google-generativeai` descontinuada), `pytest` + `unittest.mock`.

**Spec:** [docs/specs/2026-09-21-multi-provider-design.md](../specs/2026-09-21-multi-provider-design.md)

## Global Constraints

- `Provider.create_message()` deve capturar o erro nativo da SDK do provider e levantar `ProviderError` — nunca deixar a exceção nativa escapar pro chamador.
- `messages` (histórico de conversa) fica sempre no formato nativo de cada provider — nenhum código fora de `strine/providers/*` deve inspecionar ou construir esse formato diretamente.
- O schema de tool genérico (`{"name", "description", "input_schema"}`) não muda — cada provider traduz isso pro formato da própria SDK internamente.
- `ToolCall.id`: quando a SDK do provider não fornece um id nativo por chamada, o adapter gera um id sintético (ex: índice da chamada) só pra uso interno — nunca reenviado pra API se ela não esperar por isso.
- `AgentConfig.provider` default `"claude"` — um `agent.json` sem essa chave deve continuar funcionando (`.get("provider", "claude")`).
- Nenhuma chamada de API real (Anthropic, Gemini) em teste automatizado — sempre mockada.
- Mesmo provider serve tanto o planner quanto o agent gerado — nenhuma opção de provider diferente pra cada um.

---

## File Structure

```
strine/
├── config.py                    # MODIFICAR — remove load_api_key() (vira get_provider())
├── planner.py                    # MODIFICAR — plan_agent(description, provider), AgentConfig.provider
├── custom_tools.py                # MODIFICAR — generate_custom_tool(description, provider)
├── runtime.py                      # MODIFICAR — run_agent(..., provider, ...) em vez de api_key
├── cli.py                           # MODIFICAR — --provider no describe, get_provider() em run
└── providers/
    ├── __init__.py                  # CRIAR — PROVIDERS, get_provider(), UnknownProviderError
    ├── base.py                       # CRIAR — Provider, NormalizedResponse, ToolCall, ProviderError
    ├── claude.py                      # CRIAR — ClaudeProvider
    └── gemini.py                       # CRIAR — GeminiProvider

tests/
├── fakes.py                          # CRIAR — FakeProvider (test double compartilhado)
├── test_config.py                     # DELETAR — comportamento migra pra test_registry.py
├── test_planner.py                     # MODIFICAR — usa FakeProvider em vez de mockar anthropic
├── test_custom_tools.py                 # MODIFICAR — idem
├── test_runtime.py                       # MODIFICAR — idem
├── test_cli_describe.py                   # MODIFICAR — mocka get_provider em vez de load_api_key
├── test_cli_run.py                         # MODIFICAR — idem
└── providers/
    ├── __init__.py                          # CRIAR (vazio)
    ├── test_claude.py                        # CRIAR
    ├── test_gemini.py                         # CRIAR
    └── test_registry.py                        # CRIAR

pyproject.toml                        # MODIFICAR — adiciona google-genai
.env.example                           # MODIFICAR — GEMINI_API_KEY, STRINE_DEFAULT_PROVIDER
```

---

### Task 1: `strine/providers/base.py` + `FakeProvider` de teste

**Files:**
- Create: `strine/providers/__init__.py` (vazio nesta task — populado na Task 4)
- Create: `strine/providers/base.py`
- Create: `tests/fakes.py`
- Test: `tests/providers/__init__.py` (vazio)

**Interfaces:**
- Produces: `strine.providers.base.ToolCall` (dataclass: `id: str`, `name: str`, `input: dict`), `strine.providers.base.NormalizedResponse` (dataclass: `stop_reason: str`, `text: str`, `tool_calls: List[ToolCall]`), `strine.providers.base.ProviderError` (exceção), `strine.providers.base.Provider` (ABC com `name: str`, `build_user_message(text: str)`, `create_message(system_prompt, messages, tools=None, force_tool=None) -> NormalizedResponse`, `build_assistant_message(response: NormalizedResponse)`, `build_tool_result_message(tool_results: list[dict])`). `tests.fakes.FakeProvider` (test double completo, ver Step 3).

- [ ] **Step 1: Criar a estrutura de diretórios**

```bash
mkdir -p strine/providers tests/providers
touch strine/providers/__init__.py tests/providers/__init__.py
```

- [ ] **Step 2: Implementar `strine/providers/base.py`**

```python
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
```

- [ ] **Step 3: Criar `tests/fakes.py` com `FakeProvider`**

Esse test double é usado pelas Tasks 5, 6, 7 e 8 pra testar `planner.py`,
`custom_tools.py`, `runtime.py` e `cli.py` sem precisar mockar nenhuma SDK
real — só injeta respostas programadas.

```python
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

    def __init__(self, responses=None, raise_error=None):
        self._responses = list(responses or [])
        self._raise_error = raise_error
        self.calls = []  # cada chamada a create_message fica registrada aqui

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

    def build_tool_result_message(self, tool_results: list) -> dict:
        return {"role": "tool_result", "results": list(tool_results)}
```

- [ ] **Step 4: Escrever um teste mínimo confirmando que `FakeProvider` satisfaz a interface**

Crie `tests/providers/test_fakes.py`:

```python
from strine.providers.base import NormalizedResponse, Provider, ProviderError
from tests.fakes import FakeProvider


def test_fake_provider_is_a_provider():
    provider = FakeProvider(responses=[])
    assert isinstance(provider, Provider)
    assert provider.name == "fake"


def test_fake_provider_returns_programmed_responses_in_order():
    r1 = NormalizedResponse(stop_reason="end_turn", text="primeira", tool_calls=[])
    r2 = NormalizedResponse(stop_reason="end_turn", text="segunda", tool_calls=[])
    provider = FakeProvider(responses=[r1, r2])

    msg = provider.build_user_message("oi")
    first = provider.create_message("system", [msg])
    second = provider.create_message("system", [msg])

    assert first.text == "primeira"
    assert second.text == "segunda"
    assert len(provider.calls) == 2


def test_fake_provider_raises_provider_error_when_configured():
    provider = FakeProvider(raise_error="boom")
    try:
        provider.create_message("system", [])
        assert False, "deveria ter levantado ProviderError"
    except ProviderError as exc:
        assert "boom" in str(exc)
```

- [ ] **Step 5: Rodar os testes**

Run: `.venv/bin/pytest tests/providers/test_fakes.py -v`
Expected: PASS (3 testes)

- [ ] **Step 6: Commit**

```bash
git add strine/providers/__init__.py strine/providers/base.py tests/fakes.py tests/providers/__init__.py tests/providers/test_fakes.py
git commit -m "feat: add Provider interface + FakeProvider test double"
```

---

### Task 2: `strine/providers/claude.py`

**Files:**
- Create: `strine/providers/claude.py`
- Test: `tests/providers/test_claude.py`

**Interfaces:**
- Consumes: `strine.providers.base.{Provider, NormalizedResponse, ToolCall, ProviderError}` (Task 1), `strine.config.DEFAULT_MODEL` (já existe: `"claude-sonnet-5"`).
- Produces: `strine.providers.claude.ClaudeProvider(api_key: str, model: Optional[str] = None)`, com `.name == "claude"`.

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/providers/test_claude.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.providers.base import ProviderError
from strine.providers.claude import ClaudeProvider


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id, name, input_dict):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_dict)


def _fake_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def test_build_user_message_returns_anthropic_shape():
    with patch("strine.providers.claude.anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="sk-ant-fake")

    assert provider.build_user_message("oi") == {"role": "user", "content": "oi"}


def test_create_message_returns_text_when_no_tool_call():
    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _fake_response(
            _text_block("Olá!")
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        response = provider.create_message("system", [provider.build_user_message("oi")])

    assert response.stop_reason == "end_turn"
    assert response.text == "Olá!"
    assert response.tool_calls == []


def test_create_message_returns_tool_calls():
    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _fake_response(
            _tool_use_block("toolu_1", "query_database", {"query": "SELECT 1"})
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "toolu_1"
    assert response.tool_calls[0].name == "query_database"
    assert response.tool_calls[0].input == {"query": "SELECT 1"}


def test_create_message_with_force_tool_passes_tool_choice():
    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _fake_response(
            _tool_use_block("toolu_1", "create_agent_plan", {})
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "create_agent_plan", "description": "...", "input_schema": {}}],
            force_tool="create_agent_plan",
        )

    call_kwargs = MockAnthropic.return_value.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "create_agent_plan"}


def test_create_message_wraps_api_error():
    import anthropic

    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = anthropic.APIError(
            "boom", request=SimpleNamespace(), body=None
        )
        provider = ClaudeProvider(api_key="sk-ant-fake")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_build_assistant_message_includes_text_and_tool_calls():
    from strine.providers.base import NormalizedResponse, ToolCall

    with patch("strine.providers.claude.anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="sk-ant-fake")

    response = NormalizedResponse(
        stop_reason="tool_use",
        text="deixa eu checar",
        tool_calls=[ToolCall(id="toolu_1", name="query_database", input={"query": "SELECT 1"})],
    )
    message = provider.build_assistant_message(response)

    assert message["role"] == "assistant"
    assert {"type": "text", "text": "deixa eu checar"} in message["content"]
    assert {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "query_database",
        "input": {"query": "SELECT 1"},
    } in message["content"]


def test_build_tool_result_message_shape():
    with patch("strine.providers.claude.anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="sk-ant-fake")

    message = provider.build_tool_result_message(
        [{"tool_call_id": "toolu_1", "name": "query_database", "content": "42"}]
    )

    assert message == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}],
    }
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/providers/test_claude.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strine.providers.claude'`

- [ ] **Step 3: Implementar `strine/providers/claude.py`**

```python
from typing import Optional

import anthropic

from strine.config import DEFAULT_MODEL
from strine.providers.base import NormalizedResponse, Provider, ProviderError, ToolCall


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or DEFAULT_MODEL

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
            "model": self._model,
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

    def build_tool_result_message(self, tool_results: list) -> dict:
        return {
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
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/providers/test_claude.py -v`
Expected: PASS (7 testes)

- [ ] **Step 5: Commit**

```bash
git add strine/providers/claude.py tests/providers/test_claude.py
git commit -m "feat: add ClaudeProvider"
```

---

### Task 3: `strine/providers/gemini.py`

**Files:**
- Create: `strine/providers/gemini.py`
- Test: `tests/providers/test_gemini.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `strine.providers.base.{Provider, NormalizedResponse, ToolCall, ProviderError}` (Task 1).
- Produces: `strine.providers.gemini.GeminiProvider(api_key: str, model: Optional[str] = None)`, com `.name == "gemini"`. Default model: `"gemini-2.5-flash"`.

**Nota sobre a SDK:** usa o pacote `google-genai` (SDK unificada atual do
Google — `pip install google-genai`, importada como `from google import genai`
e `from google.genai import types, errors`). As chamadas usadas aqui foram
confirmadas por introspecção direta da versão instalada (2.24.0):
`FunctionDeclaration` aceita `parameters_json_schema` (JSON schema puro, sem
precisar converter pro formato `Schema` da Google); `FunctionCall.id` é
opcional (nem toda chamada vem com id, daí o fallback sintético);
`FunctionResponse.response` espera um `dict`, convenção `{"output": ...}`
pra sucesso; erros de API são `google.genai.errors.APIError`.

- [ ] **Step 1: Adicionar a dependência**

Em `pyproject.toml`, no array `dependencies`, adicione depois de `"rich>=13.0",`:

```toml
    "google-genai>=2.0",
```

Instale: `.venv/bin/pip install -e .`

- [ ] **Step 2: Escrever o teste falhando**

Crie `tests/providers/test_gemini.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.providers.base import ProviderError
from strine.providers.gemini import GeminiProvider


def _text_part(text):
    return SimpleNamespace(text=text, function_call=None)


def _function_call_part(name, args, call_id=None):
    call = SimpleNamespace(id=call_id, name=name, args=args)
    return SimpleNamespace(text=None, function_call=call)


def _fake_response(*parts):
    candidate = SimpleNamespace(content=SimpleNamespace(parts=list(parts)))
    return SimpleNamespace(candidates=[candidate])


def test_create_message_returns_text_when_no_function_call():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _text_part("Olá!")
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message("system", [provider.build_user_message("oi")])

    assert response.stop_reason == "end_turn"
    assert response.text == "Olá!"
    assert response.tool_calls == []


def test_create_message_returns_tool_calls_with_native_id():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("query_database", {"query": "SELECT 1"}, call_id="call_abc")
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {"type": "object", "properties": {}}}],
        )

    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_abc"
    assert response.tool_calls[0].name == "query_database"
    assert response.tool_calls[0].input == {"query": "SELECT 1"}


def test_create_message_generates_synthetic_id_when_missing():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("query_database", {"query": "SELECT 1"}, call_id=None)
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        response = provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "query_database", "description": "...", "input_schema": {}}],
        )

    assert response.tool_calls[0].id == "call_0"


def test_create_message_with_force_tool_sets_tool_config():
    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = _fake_response(
            _function_call_part("create_agent_plan", {})
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        provider.create_message(
            "system",
            [provider.build_user_message("oi")],
            tools=[{"name": "create_agent_plan", "description": "...", "input_schema": {}}],
            force_tool="create_agent_plan",
        )

    call_kwargs = MockClient.return_value.models.generate_content.call_args.kwargs
    tool_config = call_kwargs["config"].tool_config
    assert tool_config.function_calling_config.mode == "ANY"
    assert tool_config.function_calling_config.allowed_function_names == ["create_agent_plan"]


def test_create_message_wraps_api_error():
    from google.genai import errors as genai_errors

    with patch("strine.providers.gemini.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.side_effect = genai_errors.ClientError(
            code=401, response_json={"error": {"message": "boom"}}, response=SimpleNamespace()
        )
        provider = GeminiProvider(api_key="fake-gemini-key")
        with pytest.raises(ProviderError):
            provider.create_message("system", [provider.build_user_message("oi")])


def test_build_tool_result_message_wraps_content_in_output_dict():
    with patch("strine.providers.gemini.genai.Client"):
        provider = GeminiProvider(api_key="fake-gemini-key")

    message = provider.build_tool_result_message(
        [{"tool_call_id": "call_abc", "name": "query_database", "content": "42"}]
    )

    part = message.parts[0]
    assert part.function_response.id == "call_abc"
    assert part.function_response.name == "query_database"
    assert part.function_response.response == {"output": "42"}
```

- [ ] **Step 3: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/providers/test_gemini.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strine.providers.gemini'`

- [ ] **Step 4: Implementar `strine/providers/gemini.py`**

```python
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
```

- [ ] **Step 5: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/providers/test_gemini.py -v`
Expected: PASS (6 testes)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml strine/providers/gemini.py tests/providers/test_gemini.py
git commit -m "feat: add GeminiProvider"
```

---

### Task 4: `strine/providers/__init__.py` — registry + `get_provider()`

**Files:**
- Modify: `strine/providers/__init__.py`
- Modify: `strine/config.py`
- Delete: `tests/test_config.py`
- Test: `tests/providers/test_registry.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `strine.providers.claude.ClaudeProvider`, `strine.providers.gemini.GeminiProvider` (Tasks 2-3), `strine.config.MissingAPIKeyError` (já existe).
- Produces: `strine.providers.PROVIDERS` (dict `{"claude": ClaudeProvider, "gemini": GeminiProvider}`), `strine.providers.DEFAULT_PROVIDER` (str, do env `STRINE_DEFAULT_PROVIDER` ou `"claude"`), `strine.providers.get_provider(name: str) -> Provider`, `strine.providers.UnknownProviderError`.

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/providers/test_registry.py`:

```python
from unittest.mock import patch

import pytest

from strine.config import MissingAPIKeyError
from strine.providers import PROVIDERS, UnknownProviderError, get_provider
from strine.providers.claude import ClaudeProvider
from strine.providers.gemini import GeminiProvider


def test_providers_registry_has_claude_and_gemini():
    assert PROVIDERS == {"claude": ClaudeProvider, "gemini": GeminiProvider}


def test_get_provider_unknown_name_raises_friendly_error():
    with pytest.raises(UnknownProviderError) as exc_info:
        get_provider("bogus")

    assert "claude" in str(exc_info.value)
    assert "gemini" in str(exc_info.value)


def test_get_provider_claude_uses_anthropic_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")

    provider = get_provider("claude")

    assert isinstance(provider, ClaudeProvider)
    assert provider.name == "claude"


def test_get_provider_claude_missing_key_raises_missing_api_key_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_provider("claude")

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_get_provider_gemini_missing_key_raises_missing_api_key_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_provider("gemini")

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_get_provider_prefers_dotenv_in_current_working_directory(tmp_path, monkeypatch):
    """Mesma proteção que já existia em load_api_key(): resolve o .env a
    partir do cwd de quem roda `strine`, não de onde o pacote está
    instalado. Mocka o construtor da SDK da Anthropic pra capturar
    exatamente qual api_key foi usada — sem isso o teste não prova nada
    além de "não levantou exceção"."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-project-dotenv\n")

    with patch("strine.providers.claude.anthropic.Anthropic") as MockAnthropic:
        get_provider("claude")

    MockAnthropic.assert_called_once_with(api_key="sk-ant-from-project-dotenv")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/providers/test_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_provider' from 'strine.providers'`

- [ ] **Step 3: Simplificar `strine/config.py`**

Substitua o conteúdo inteiro do arquivo por:

```python
DEFAULT_MODEL = "claude-sonnet-5"


class MissingAPIKeyError(RuntimeError):
    pass
```

`load_api_key()` é removida — sua responsabilidade (resolver `.env` a
partir do cwd e checar a API key) passa a viver em `get_provider()`, que
faz isso por provider.

- [ ] **Step 4: Deletar `tests/test_config.py`**

```bash
rm tests/test_config.py
```

(A cobertura que existia lá — env var funciona, erro amigável quando
ausente, `.env` resolvido a partir do cwd — está recriada em
`tests/providers/test_registry.py`, Step 1 acima.)

- [ ] **Step 5: Implementar `strine/providers/__init__.py`**

```python
import os

from dotenv import find_dotenv, load_dotenv

from strine.config import MissingAPIKeyError
from strine.providers.base import Provider
from strine.providers.claude import ClaudeProvider
from strine.providers.gemini import GeminiProvider

PROVIDERS = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
}

_PROVIDER_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

DEFAULT_PROVIDER = os.getenv("STRINE_DEFAULT_PROVIDER", "claude")


class UnknownProviderError(RuntimeError):
    pass


def get_provider(name: str) -> Provider:
    """Instancia o provider certo, já com a API key carregada do .env.

    usecwd=True: resolve o .env a partir do diretório onde o usuário RODOU
    o strine, não de onde o pacote está instalado (mesma proteção que
    load_api_key() já tinha). override=True: o .env do projeto tem
    prioridade sobre uma variável de ambiente já exportada no shell.
    """
    if name not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"Provider '{name}' não é reconhecido. Providers disponíveis: {valid}."
        )

    load_dotenv(find_dotenv(usecwd=True), override=True)

    env_var = _PROVIDER_ENV_VARS[name]
    api_key = os.getenv(env_var)
    if not api_key:
        raise MissingAPIKeyError(
            f"{env_var} não configurada.\n\n"
            f"Configure sua chave da API antes de usar o provider '{name}':\n"
            f"  1. Copie .env.example para .env:  cp .env.example .env\n"
            f"  2. Edite .env e adicione:          {env_var}=...\n"
            f"  3. Ou exporte diretamente:          export {env_var}=...\n"
        )

    provider_cls = PROVIDERS[name]
    return provider_cls(api_key=api_key)
```

- [ ] **Step 6: Atualizar `.env.example`**

Adicione, no bloco de variáveis opcionais:

```
# GEMINI_API_KEY=...
# STRINE_DEFAULT_PROVIDER=claude
```

- [ ] **Step 7: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/providers/ -v`
Expected: PASS (todos os testes de Tasks 1-4)

- [ ] **Step 8: Commit**

```bash
git add strine/providers/__init__.py strine/config.py .env.example tests/providers/test_registry.py
git rm tests/test_config.py
git commit -m "feat: add provider registry and get_provider(), simplify config.py"
```

---

### Task 5: Refatorar `strine/planner.py`

**Files:**
- Modify: `strine/planner.py`
- Modify: `tests/test_planner.py`

**Interfaces:**
- Consumes: `strine.providers.base.{Provider, NormalizedResponse, ToolCall, ProviderError}` (Task 1), `tests.fakes.FakeProvider` (Task 1).
- Produces: `strine.planner.plan_agent(description: str, provider: Provider) -> AgentConfig`, `strine.planner.AgentConfig` agora com campo `provider: str = "claude"` e `to_dict()` incluindo essa chave.

- [ ] **Step 1: Reescrever `tests/test_planner.py`**

```python
import pytest

from strine.planner import VALID_TOOLS, AgentConfig, PlannerError, plan_agent
from strine.providers.base import NormalizedResponse, ToolCall
from tests.fakes import FakeProvider


def test_valid_tools_includes_all_eight():
    assert VALID_TOOLS == {
        "sql",
        "slack",
        "webhook",
        "http_request",
        "web_search",
        "send_email",
        "file_read",
        "file_write",
    }


def test_agent_config_to_dict_includes_provider_and_custom_tools_defaults():
    config = AgentConfig(name="foo", prompt="bar", tools=["sql"])
    assert config.to_dict() == {
        "name": "foo",
        "prompt": "bar",
        "tools": ["sql"],
        "custom_tools": [],
        "provider": "claude",
    }


def _plan_response(name, tools, prompt="You are helpful."):
    return NormalizedResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="create_agent_plan",
                input={"name": name, "prompt": prompt, "tools": tools},
            )
        ],
    )


def test_plan_agent_filters_invalid_tool_names():
    provider = FakeProvider(responses=[_plan_response("my-agent", ["sql", "not-a-real-tool"])])

    config = plan_agent("descrição qualquer", provider)

    assert config.tools == ["sql"]
    assert config.custom_tools == []


def test_plan_agent_stores_provider_name():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    config = plan_agent("descrição qualquer", provider)

    assert config.provider == "fake"


def test_plan_agent_accepts_new_tool_names():
    provider = FakeProvider(
        responses=[_plan_response("researcher", ["web_search", "http_request"])]
    )

    config = plan_agent("descrição qualquer", provider)

    assert set(config.tools) == {"web_search", "http_request"}


def test_plan_agent_rejects_unsafe_name_with_path_separator():
    provider = FakeProvider(responses=[_plan_response("../../evil", [])])

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_rejects_name_with_space():
    provider = FakeProvider(responses=[_plan_response("my agent", [])])

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_rejects_empty_name():
    provider = FakeProvider(responses=[_plan_response("", [])])

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_accepts_valid_slug_name():
    provider = FakeProvider(responses=[_plan_response("sales-analyzer", [])])

    config = plan_agent("descrição qualquer", provider)

    assert config.name == "sales-analyzer"


def test_plan_agent_raises_planner_error_when_no_tool_call_returned():
    provider = FakeProvider(
        responses=[NormalizedResponse(stop_reason="end_turn", text="oi", tool_calls=[])]
    )

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_wraps_provider_error():
    provider = FakeProvider(raise_error="API fora do ar")

    with pytest.raises(PlannerError):
        plan_agent("descrição qualquer", provider)


def test_plan_agent_forces_the_create_agent_plan_tool():
    provider = FakeProvider(responses=[_plan_response("my-agent", [])])

    plan_agent("descrição qualquer", provider)

    assert provider.calls[0]["force_tool"] == "create_agent_plan"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_planner.py -v`
Expected: FAIL — `plan_agent()` ainda espera `api_key: str`, e `AgentConfig.to_dict()` não tem `"provider"`

- [ ] **Step 3: Atualizar `strine/planner.py`**

Substitua as linhas 1-9 (imports + `MODEL`) por:

```python
import re
from dataclasses import dataclass, field
from typing import List

from strine.providers.base import Provider, ProviderError
```

Substitua a definição de `AgentConfig` por:

```python
@dataclass
class AgentConfig:
    name: str
    prompt: str
    tools: List[str] = field(default_factory=list)
    custom_tools: List[dict] = field(default_factory=list)
    provider: str = "claude"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "tools": self.tools,
            "custom_tools": self.custom_tools,
            "provider": self.provider,
        }
```

Substitua a função `plan_agent` inteira por:

```python
def plan_agent(description: str, provider: Provider) -> AgentConfig:
    """Chama o provider pra decidir tools + system prompt + nome do agent.

    Usa tool use forçado (force_tool) em vez de pedir "responda em JSON" e
    fazer parsing manual — o schema garante o formato da resposta.
    """
    try:
        response = provider.create_message(
            system_prompt=SYSTEM_PROMPT,
            messages=[provider.build_user_message(description)],
            tools=[AGENT_PLAN_TOOL],
            force_tool="create_agent_plan",
        )
    except ProviderError as exc:
        raise PlannerError(f"Erro ao chamar a API: {exc}") from exc

    if not response.tool_calls:
        raise PlannerError("O modelo não retornou um plano estruturado.")

    plan = response.tool_calls[0].input
    tools = [t for t in plan.get("tools", []) if t in VALID_TOOLS]

    _validate_name(plan["name"])

    return AgentConfig(
        name=plan["name"], prompt=plan["prompt"], tools=tools, provider=provider.name
    )
```

O restante do arquivo (`VALID_TOOLS`, `SYSTEM_PROMPT`, `AGENT_PLAN_TOOL`,
`PlannerError`, `_validate_name`) não muda.

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_planner.py -v`
Expected: PASS (11 testes)

- [ ] **Step 5: Commit**

```bash
git add strine/planner.py tests/test_planner.py
git commit -m "refactor: plan_agent takes a Provider instead of an api_key"
```

---

### Task 6: Refatorar `strine/custom_tools.py`

**Files:**
- Modify: `strine/custom_tools.py`
- Modify: `tests/test_custom_tools.py`

**Interfaces:**
- Consumes: `strine.providers.base.{Provider, NormalizedResponse, ToolCall, ProviderError}` (Task 1), `tests.fakes.FakeProvider` (Task 1).
- Produces: `strine.custom_tools.generate_custom_tool(description: str, provider: Provider) -> CustomToolSpec` (assinatura muda de `api_key: str` pra `provider: Provider`; `CustomToolSpec` e o resto não mudam).

- [ ] **Step 1: Ler os testes atuais pra manter os casos de validação já cobertos**

Leia `tests/test_custom_tools.py` antes de reescrever — os testes de
`_validate_code`/`_validate_name`/`_validate_input_schema` (sintaxe
inválida, sem função `execute`, nome com path traversal, `input_schema`
malformado) continuam exatamente iguais, só a forma de simular a resposta
do "modelo" muda de mock de SDK pra `FakeProvider`.

- [ ] **Step 2: Reescrever `tests/test_custom_tools.py`**

```python
import pytest

from strine.custom_tools import CustomToolError, CustomToolSpec, generate_custom_tool
from strine.providers.base import NormalizedResponse, ToolCall
from tests.fakes import FakeProvider


def _tool_response(name, description, input_schema, code):
    return NormalizedResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="create_custom_tool",
                input={
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                    "code": code,
                },
            )
        ],
    )


def test_generate_custom_tool_returns_spec_for_valid_code():
    provider = FakeProvider(
        responses=[
            _tool_response(
                "parse_invoice_total",
                "Extrai o valor total de um texto de nota fiscal.",
                {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                "def execute(text):\n    return 'total: 0'\n",
            )
        ]
    )

    spec = generate_custom_tool("extrai o total de uma nota fiscal", provider)

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "parse_invoice_total"
    assert "def execute" in spec.code


def test_generate_custom_tool_forces_the_create_custom_tool_tool():
    provider = FakeProvider(
        responses=[
            _tool_response("valid_name", "desc", {"type": "object", "properties": {}}, "def execute():\n    return 'x'\n")
        ]
    )

    generate_custom_tool("qualquer coisa", provider)

    assert provider.calls[0]["force_tool"] == "create_custom_tool"


def test_generate_custom_tool_rejects_code_with_syntax_error():
    provider = FakeProvider(
        responses=[
            _tool_response("broken_tool", "Tool quebrada.", {"type": "object", "properties": {}}, "def execute(:\n    pass")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_code_without_execute_function():
    provider = FakeProvider(
        responses=[
            _tool_response("no_execute_tool", "Tool sem execute.", {"type": "object", "properties": {}}, "def other_function():\n    pass")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_execute_as_class_method():
    provider = FakeProvider(
        responses=[
            _tool_response(
                "bad_scope",
                "desc",
                {"type": "object", "properties": {}},
                "class Foo:\n    def execute(self, **kwargs):\n        return 'x'\n",
            )
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_name_with_path_separator():
    provider = FakeProvider(
        responses=[
            _tool_response("../../evil", "desc", {"type": "object", "properties": {}}, "def execute():\n    return 'x'\n")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_rejects_input_schema_not_object_type():
    provider = FakeProvider(
        responses=[
            _tool_response("valid_name", "desc", {"type": "array"}, "def execute():\n    return 'x'\n")
        ]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_wraps_provider_error():
    provider = FakeProvider(raise_error="boom")

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)


def test_generate_custom_tool_raises_when_no_tool_call_returned():
    provider = FakeProvider(
        responses=[NormalizedResponse(stop_reason="end_turn", text="oi", tool_calls=[])]
    )

    with pytest.raises(CustomToolError):
        generate_custom_tool("qualquer coisa", provider)
```

- [ ] **Step 3: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_custom_tools.py -v`
Expected: FAIL — `generate_custom_tool()` ainda espera `api_key: str`

- [ ] **Step 4: Atualizar `strine/custom_tools.py`**

Substitua as linhas 1-9 (imports + `MODEL`) por:

```python
import ast
import re
from dataclasses import dataclass

from strine.providers.base import Provider, ProviderError
```

Substitua a função `generate_custom_tool` inteira por:

```python
def generate_custom_tool(description: str, provider: Provider) -> CustomToolSpec:
    try:
        response = provider.create_message(
            system_prompt=SYSTEM_PROMPT,
            messages=[provider.build_user_message(description)],
            tools=[CREATE_CUSTOM_TOOL_TOOL],
            force_tool="create_custom_tool",
        )
    except ProviderError as exc:
        raise CustomToolError(f"Erro ao chamar a API: {exc}") from exc

    if not response.tool_calls:
        raise CustomToolError("O modelo não retornou uma tool customizada estruturada.")

    plan = response.tool_calls[0].input
    _validate_code(plan["code"])
    _validate_name(plan["name"])
    _validate_input_schema(plan["input_schema"])

    return CustomToolSpec(
        name=plan["name"],
        description=plan["description"],
        input_schema=plan["input_schema"],
        code=plan["code"],
    )
```

O restante do arquivo (`SYSTEM_PROMPT`, `CREATE_CUSTOM_TOOL_TOOL`,
`CustomToolSpec`, `CustomToolError`, `_validate_code`, `_validate_name`,
`_validate_input_schema`) não muda.

- [ ] **Step 5: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_custom_tools.py -v`
Expected: PASS (9 testes)

- [ ] **Step 6: Commit**

```bash
git add strine/custom_tools.py tests/test_custom_tools.py
git commit -m "refactor: generate_custom_tool takes a Provider instead of an api_key"
```

---

### Task 7: Refatorar `strine/runtime.py`

**Files:**
- Modify: `strine/runtime.py`
- Modify: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `strine.providers.base.{Provider, NormalizedResponse, ToolCall, ProviderError}` (Task 1), `tests.fakes.FakeProvider` (Task 1). `prepare_agent_tools()` não muda (Day 4).
- Produces: `strine.runtime.run_agent(system_prompt, tool_schemas, executors, user_input, provider, on_tool_call=None) -> str` (parâmetro `api_key: str` vira `provider: Provider`, na mesma posição).

- [ ] **Step 1: Reescrever a seção de `run_agent` em `tests/test_runtime.py`**

Os testes de `prepare_agent_tools` (linhas 1-90 do arquivo atual) **não
mudam** — essa função continua igual. Substitua toda a seção `# ---
run_agent ---` em diante por:

```python
# --- run_agent ---


def _text_response(text):
    return NormalizedResponse(stop_reason="end_turn", text=text, tool_calls=[])


def _tool_use_response(name, tool_input, call_id="toolu_1"):
    return NormalizedResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, input=tool_input)],
    )


def test_run_agent_returns_text_when_no_tool_needed():
    provider = FakeProvider(responses=[_text_response("Olá! Como posso ajudar?")])

    result = run_agent(
        system_prompt="You are helpful.",
        tool_schemas=[],
        executors={},
        user_input="oi",
        provider=provider,
    )

    assert result == "Olá! Como posso ajudar?"


def test_run_agent_executes_tool_and_returns_final_text():
    executed = {}

    def fake_execute(query):
        executed["query"] = query
        return "42"

    provider = FakeProvider(
        responses=[
            _tool_use_response("query_database", {"query": "SELECT 1"}),
            _text_response("O resultado é 42."),
        ]
    )

    result = run_agent(
        system_prompt="You are helpful.",
        tool_schemas=[{"name": "query_database", "description": "...", "input_schema": {}}],
        executors={"query_database": fake_execute},
        user_input="quantos usuários temos?",
        provider=provider,
    )

    assert executed["query"] == "SELECT 1"
    assert result == "O resultado é 42."


def test_run_agent_handles_tool_execution_error_gracefully():
    def failing_execute(**kwargs):
        raise ValueError("boom")

    provider = FakeProvider(
        responses=[
            _tool_use_response("broken_tool", {}),
            _text_response("Tive um problema, mas seguimos."),
        ]
    )

    result = run_agent(
        system_prompt="You are helpful.",
        tool_schemas=[{"name": "broken_tool", "description": "...", "input_schema": {}}],
        executors={"broken_tool": failing_execute},
        user_input="usa a tool quebrada",
        provider=provider,
    )

    assert result == "Tive um problema, mas seguimos."


def test_run_agent_stops_after_max_tool_rounds():
    provider = FakeProvider(
        responses=[_tool_use_response("loopy_tool", {}) for _ in range(6)]
    )

    result = run_agent(
        system_prompt="You are helpful.",
        tool_schemas=[{"name": "loopy_tool", "description": "...", "input_schema": {}}],
        executors={"loopy_tool": lambda **kwargs: "still going"},
        user_input="loop forever",
        provider=provider,
    )

    # initial call + 5 tool rounds = 6 chamadas totais, nunca ilimitado
    assert len(provider.calls) == 6
    assert "limite" in result.lower()


def test_run_agent_calls_on_tool_call_callback_before_executing_tool():
    calls = []

    provider = FakeProvider(
        responses=[
            _tool_use_response("query_database", {"query": "SELECT 1"}),
            _text_response("O resultado é 42."),
        ]
    )

    result = run_agent(
        system_prompt="You are helpful.",
        tool_schemas=[{"name": "query_database", "description": "...", "input_schema": {}}],
        executors={"query_database": lambda query: "42"},
        user_input="quantos usuários temos?",
        provider=provider,
        on_tool_call=lambda name: calls.append(name),
    )

    assert calls == ["query_database"]
    assert result == "O resultado é 42."


def test_run_agent_works_without_on_tool_call_callback():
    provider = FakeProvider(
        responses=[
            _tool_use_response("query_database", {"query": "SELECT 1"}),
            _text_response("O resultado é 42."),
        ]
    )

    result = run_agent(
        system_prompt="You are helpful.",
        tool_schemas=[{"name": "query_database", "description": "...", "input_schema": {}}],
        executors={"query_database": lambda query: "42"},
        user_input="quantos usuários temos?",
        provider=provider,
    )

    assert result == "O resultado é 42."


def test_run_agent_raises_runtime_error_on_provider_error():
    provider = FakeProvider(raise_error="boom")

    with pytest.raises(AgentRuntimeError):
        run_agent(
            system_prompt="You are helpful.",
            tool_schemas=[],
            executors={},
            user_input="oi",
            provider=provider,
        )
```

E ajuste os imports no topo do arquivo (mantenha `prepare_agent_tools` e o
que já for usado pelos testes que não mudam):

```python
import pytest

from strine.providers.base import NormalizedResponse, ToolCall
from strine.runtime import AgentRuntimeError, prepare_agent_tools, run_agent
from tests.fakes import FakeProvider
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_runtime.py -v`
Expected: FAIL — `run_agent()` ainda espera `api_key: str`

- [ ] **Step 3: Atualizar `strine/runtime.py`**

Substitua as linhas 1-15 (imports + `MODEL`/`AgentRuntimeError`) por:

```python
import importlib.util
from pathlib import Path
from typing import Callable, Optional

from strine.providers.base import Provider, ProviderError
from strine.tools import TOOLS

MAX_TOOL_ROUNDS = 5


class AgentRuntimeError(RuntimeError):
    pass
```

`prepare_agent_tools()` não muda. Remova a função `_call_claude` inteira e
substitua a assinatura + corpo de `run_agent` por:

```python
def run_agent(
    system_prompt: str,
    tool_schemas: list,
    executors: dict,
    user_input: str,
    provider: Provider,
    on_tool_call: Optional[Callable[[str], None]] = None,
) -> str:
    """Roda uma pergunta do usuário contra o agent, executando tools de verdade.

    Cada chamada é uma conversa nova (sem memória entre perguntas do REPL).
    Se o modelo pedir uma tool, ela é executada e o resultado volta pra
    ele, num loop limitado a MAX_TOOL_ROUNDS idas-e-voltas, pra nunca
    rodar indefinidamente.

    on_tool_call, se passado, é chamado com o nome da tool logo antes dela
    ser executada — permite ao chamador (cli.py) mostrar feedback visual
    sem que esse módulo precise fazer I/O diretamente.
    """
    messages = [provider.build_user_message(user_input)]

    try:
        response = provider.create_message(system_prompt, messages, tools=tool_schemas)
    except ProviderError as exc:
        raise AgentRuntimeError(f"Erro ao chamar a API: {exc}") from exc

    rounds = 0
    while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
        messages.append(provider.build_assistant_message(response))

        tool_results = []
        for call in response.tool_calls:
            executor = executors.get(call.name)
            if executor is None:
                result_text = f"Tool '{call.name}' não está disponível."
            else:
                if on_tool_call is not None:
                    on_tool_call(call.name)
                try:
                    result_text = executor(**call.input)
                except Exception as exc:
                    result_text = f"Erro ao executar a tool '{call.name}': {exc}"

            tool_results.append(
                {"tool_call_id": call.id, "name": call.name, "content": result_text}
            )

        messages.append(provider.build_tool_result_message(tool_results))
        rounds += 1

        try:
            response = provider.create_message(system_prompt, messages, tools=tool_schemas)
        except ProviderError as exc:
            raise AgentRuntimeError(f"Erro ao chamar a API: {exc}") from exc

    final_text = response.text

    if response.stop_reason == "tool_use" and rounds >= MAX_TOOL_ROUNDS:
        note = (
            "(O agent atingiu o limite de chamadas de tools nesta pergunta; "
            "a resposta pode estar incompleta.)"
        )
        final_text = f"{final_text}\n\n{note}" if final_text else note

    return final_text
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_runtime.py -v`
Expected: PASS (todos os testes: `prepare_agent_tools` inalterados + `run_agent` novos)

- [ ] **Step 5: Commit**

```bash
git add strine/runtime.py tests/test_runtime.py
git commit -m "refactor: run_agent takes a Provider instead of an api_key"
```

---

### Task 8: Conectar tudo em `strine/cli.py`

**Files:**
- Modify: `strine/cli.py`
- Modify: `tests/test_cli_describe.py`
- Modify: `tests/test_cli_run.py`
- Modify: `tests/fakes.py` (adiciona `name_override` a `FakeProvider`)

**Interfaces:**
- Consumes: `strine.providers.{get_provider, DEFAULT_PROVIDER, PROVIDERS, UnknownProviderError}` (Task 4), `strine.planner.plan_agent(description, provider)` (Task 5), `strine.custom_tools.generate_custom_tool(description, provider)` (Task 6), `strine.runtime.run_agent(..., provider, ...)` (Task 7).

- [ ] **Step 1: Atualizar os mocks em `tests/test_cli_describe.py`**

O helper `_patch_common()` hoje mocka `strine.cli.load_api_key` (que não
existe mais) e `strine.cli.plan_agent`. Substitua por:

```python
def _patch_common():
    return (
        patch("strine.cli.get_provider", return_value=FakeProvider(name_override="claude")),
        patch(
            "strine.cli.plan_agent",
            return_value=AgentConfig(name="test-agent", prompt="You help.", tools=["sql"]),
        ),
    )
```

Isso exige um pequeno ajuste no `FakeProvider` (Task 1) pra aceitar
`name_override` — adicione ao `__init__`:

```python
    def __init__(self, responses=None, raise_error=None, name_override=None):
        self._responses = list(responses or [])
        self._raise_error = raise_error
        self.calls = []
        if name_override is not None:
            self.name = name_override
```

(Isso não quebra nenhum uso anterior de `FakeProvider`, já que
`name_override` é opcional.)

Adicione os imports necessários no topo de `tests/test_cli_describe.py`:

```python
from tests.fakes import FakeProvider
```

Todos os outros testes desse arquivo continuam iguais — eles não
inspecionam `provider` diretamente, só o comportamento de
`describe_agent`.

- [ ] **Step 2: Adicionar teste do `--provider` em `tests/test_cli_describe.py`**

```python
def test_describe_passes_selected_provider_to_get_provider():
    with isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key as mock_get_provider, patch_plan:
            result = runner.invoke(
                app, ["describe", "--provider", "gemini", "um", "agent", "qualquer"], input="\n"
            )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("gemini")


def test_describe_unknown_provider_shows_friendly_error():
    from strine.providers import UnknownProviderError

    with isolated_filesystem():
        with patch("strine.cli.get_provider", side_effect=UnknownProviderError("Provider 'bogus' não é reconhecido.")):
            result = runner.invoke(
                app, ["describe", "--provider", "bogus", "um", "agent", "qualquer"]
            )

        assert result.exit_code == 1
        assert "bogus" in result.output
```

- [ ] **Step 3: Atualizar `tests/test_cli_run.py`**

Troque todos os `patch("strine.cli.load_api_key", return_value="sk-ant-fake")`
por `patch("strine.cli.get_provider", return_value=FakeProvider())`, e
ajuste as chamadas de `run_agent` mockadas — `mock_run_agent.assert_called_once()`
e a checagem de `kwargs.get("user_input")` continuam funcionando sem
mudança (a assinatura mockada não é verificada em detalhe nesses testes).
Adicione `from tests.fakes import FakeProvider` no topo.

Em `_write_agent_json`, adicione `"provider": "claude"` ao dict
`_AGENT_JSON` (já é o default, mas deixa explícito no fixture).

Adicione um teste novo confirmando que `strine run` lê o provider do
`agent.json`:

```python
def test_run_uses_provider_stored_in_agent_json():
    agent_with_gemini = dict(_AGENT_JSON, provider="gemini")
    with isolated_filesystem():
        with open("test-agent.json", "w") as f:
            json.dump(agent_with_gemini, f)

        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()) as mock_get_provider,
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch("strine.cli.run_agent", return_value="oi"),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("gemini")


def test_run_defaults_to_claude_when_agent_json_has_no_provider_key():
    agent_without_provider = {k: v for k, v in _AGENT_JSON.items() if k != "provider"}
    with isolated_filesystem():
        with open("test-agent.json", "w") as f:
            json.dump(agent_without_provider, f)

        with (
            patch("strine.cli.get_provider", return_value=FakeProvider()) as mock_get_provider,
            patch("strine.cli.prepare_agent_tools", return_value=([], {}, [])),
            patch("strine.cli.run_agent", return_value="oi"),
        ):
            result = runner.invoke(app, ["run", "./test-agent.json"], input="sair\n")

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("claude")
```

- [ ] **Step 4: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_cli_describe.py tests/test_cli_run.py -v`
Expected: FAIL — `cli.py` ainda importa `load_api_key`, não tem `--provider`

- [ ] **Step 5: Atualizar `strine/cli.py`**

Substitua os imports do topo do arquivo por:

```python
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from strine.config import MissingAPIKeyError
from strine.custom_tools import CustomToolError, generate_custom_tool
from strine.planner import PlannerError, plan_agent
from strine.providers import DEFAULT_PROVIDER, UnknownProviderError, get_provider
from strine.runtime import AgentRuntimeError, prepare_agent_tools, run_agent
from strine.tools import TOOLS
```

Na assinatura de `describe_agent`, adicione o parâmetro `--provider` (mantendo
o parâmetro `description` como está):

```python
@app.command(name="describe", hidden=True)
def describe_agent(
    description: List[str] = typer.Argument(
        ...,
        help="Descrição em linguagem natural do agent que você quer criar.",
    ),
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER,
        "--provider",
        help="Provider de IA a usar (claude, gemini).",
    ),
) -> None:
```

Substitua o bloco que hoje chama `load_api_key()` (as primeiras linhas do
corpo da função) por:

```python
    console = Console()
    text = " ".join(description)

    try:
        provider = get_provider(provider_name)
    except UnknownProviderError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    try:
        with console.status("[bold cyan]Planejando o agent...[/bold cyan]"):
            agent_config = plan_agent(text, provider)
    except PlannerError as exc:
        console.print(f"[red]Erro ao planejar o agent: {exc}[/red]")
        raise typer.Exit(code=1)
```

Mais adiante na mesma função, troque `generate_custom_tool(custom_description, api_key)`
por `generate_custom_tool(custom_description, provider)` (mesmo `provider`
já resolvido acima — planner e agent gerado usam o mesmo, conforme
decidido).

No comando `run`, troque o bloco que hoje chama `load_api_key()` por:

```python
    provider_name = agent_config.get("provider", "claude")
    try:
        provider = get_provider(provider_name)
    except (UnknownProviderError, MissingAPIKeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
```

E na chamada de `run_agent` dentro do loop do REPL, troque `api_key=api_key`
por `provider=provider`.

- [ ] **Step 6: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_cli_describe.py tests/test_cli_run.py -v`
Expected: PASS

- [ ] **Step 7: Rodar a suíte inteira**

Run: `.venv/bin/pytest -v`
Expected: PASS — todos os testes do projeto, Tasks 1-8

- [ ] **Step 8: Commit**

```bash
git add strine/cli.py tests/test_cli_describe.py tests/test_cli_run.py tests/fakes.py
git commit -m "feat: wire --provider flag into describe, read provider from agent.json in run"
```

---

## Verificação final (manual, com API keys reais)

Não automatizável sem gastar crédito de API de verdade — validar quando
tiver saldo em pelo menos uma das duas contas (Claude e/ou Gemini):

1. `strine "quero um agent que responde perguntas gerais" --provider gemini` —
   confirmar que o `agent.json` gerado tem `"provider": "gemini"`.
2. `strine run ./<agent>.json` nesse agent — confirmar que ele conversa de
   verdade usando o Gemini (sem precisar passar `--provider` de novo).
3. Repetir 1-2 com `--provider claude` (ou sem a flag, testando o default).
4. `strine "descrição" --provider bogus` — confirmar erro amigável listando
   os providers válidos, exit code 1.
5. Gerar um agent com uma tool que force tool-calling de verdade (ex: "agent
   que consulta a web") em cada provider, confirmar que o loop de
   round-trip funciona igual nos dois.
