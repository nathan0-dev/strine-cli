# Spec: Suporte a múltiplos providers de LLM (Claude + Gemini)

**Data:** 2026-09-21
**Status:** Aprovado, aguardando plano de implementação

## Contexto

O Strine hoje fala direto com a API da Anthropic em três lugares (`strine/planner.py`,
`strine/custom_tools.py`, `strine/runtime.py`), cada um instanciando
`anthropic.Anthropic()` e usando o formato de tool-use específico da
Anthropic (`tool_choice`, blocos `tool_use`/`text`, etc). Essa decisão fazia
sentido pro escopo original (validar a ideia central com um provider só),
mas o próximo passo do projeto é suportar outros providers — começando por
Gemini, com GPT/Groq/OpenRouter planejados pra depois.

GPT, Groq e OpenRouter seguem o formato de function-calling da OpenAI (só
mudam URL base e API key) — na prática, suportar os 5 providers do roadmap
exige só 2 integrações de verdade: um adapter "estilo OpenAI" (cobre os
três de uma vez) e um adapter específico do Gemini (formato de tool-calling
genuinamente diferente). Esta spec cobre a abstração completa (pensada pros
5) mas implementa e valida só **Claude + Gemini** — o adapter estilo OpenAI
fica pra uma rodada seguinte.

## Objetivo

Tornar o Strine capaz de rodar tanto o "planner" quanto o agent gerado
usando qualquer provider suportado, escolhido por flag no momento da
criação do agent, sem duplicar a lógica de loop de tool-calling que já
existe e está testada.

## Não-objetivos (fora de escopo desta spec)

- Implementação do adapter "estilo OpenAI" (GPT/Groq/OpenRouter) — a
  interface fica pronta pra receber esse adapter depois, mas ele não é
  escrito nesta spec.
- Provider diferente para o planner vs. o agent gerado — os dois sempre
  usam o mesmo provider nessa versão (decisão já tomada no brainstorm).
- Flag de override de provider no `strine run` — o provider vem sempre do
  que está salvo no `agent.json`, criado no momento do `describe`.
- Comparação de custo/qualidade entre providers, benchmarks, fallback
  automático entre providers — nada disso está em escopo.

## 1. Arquitetura: pacote `strine/providers/`

```
strine/providers/
├── __init__.py    # PROVIDERS registry + get_provider(name) -> Provider
├── base.py         # Provider (ABC), NormalizedResponse, ToolCall, ProviderError
├── claude.py       # ClaudeProvider — refatoração do que já existe
└── gemini.py       # GeminiProvider — novo
```

### `strine/providers/base.py`

```python
@dataclass
class ToolCall:
    id: str  # identificador único da chamada, pra correlacionar com o
              # resultado depois. Nem toda SDK dá um id nativo por tool
              # call (a Anthropic dá; o Gemini pode não dar da mesma
              # forma) — quando a SDK não fornecer um, é responsabilidade
              # do adapter gerar um synthetic id (ex: índice da chamada
              # na resposta) só pra uso interno do Strine, nunca enviado
              # de volta pra API se ela não esperar por isso.
    name: str
    input: dict


@dataclass
class NormalizedResponse:
    stop_reason: str  # "tool_use" | "end_turn"
    text: str
    tool_calls: List[ToolCall]


class ProviderError(RuntimeError):
    """Erro genérico de chamada de API, independente do provider por trás."""


class Provider(ABC):
    name: str  # "claude" | "gemini" — usado no agent.json e na flag --provider

    @abstractmethod
    def build_user_message(self, text: str) -> Any:
        """Monta a primeira mensagem (role user) no formato nativo do provider."""

    @abstractmethod
    def create_message(
        self,
        system_prompt: str,
        messages: list,
        tools: Optional[list[dict]] = None,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        """Faz uma chamada de API. tools usa o schema genérico já existente
        (name/description/input_schema). force_tool, se passado, força a
        chamada daquela tool específica (equivalente ao tool_choice forçado
        da Anthropic). Deve capturar o erro nativo da SDK do provider e
        levantar ProviderError."""

    @abstractmethod
    def build_assistant_message(self, response: NormalizedResponse) -> Any:
        """Monta a mensagem 'assistant' (com os tool_calls que o modelo
        pediu) no formato nativo do provider, pra ser anexada ao histórico."""

    @abstractmethod
    def build_tool_result_message(self, tool_results: list[dict]) -> Any:
        """tool_results: lista de {"tool_call_id": str, "name": str,
        "content": str}. Monta a mensagem de resultado de tool no formato
        nativo do provider."""
```

`messages` é uma lista no formato nativo de cada provider — opaca pra quem
chama (`runtime.py`, `planner.py`, `custom_tools.py`). Cada provider sabe
construir e interpretar seu próprio histórico; ninguém fora do adapter
precisa entender esse formato.

O schema de tool que já existe hoje em todo o projeto
(`{"name", "description", "input_schema"}`, usado no `TOOLS` registry, nos
schemas de tool customizada e no `AGENT_PLAN_TOOL`) **não muda**. Cada
provider traduz esse schema genérico pro formato específico da própria SDK
internamente — é essa tradução que o adapter existe pra fazer.

### `strine/providers/__init__.py`

```python
PROVIDERS = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
}

DEFAULT_PROVIDER = os.getenv("STRINE_DEFAULT_PROVIDER", "claude")


def get_provider(name: str) -> Provider:
    """Instancia o provider certo, já com a API key carregada do .env.
    Levanta MissingAPIKeyError (reaproveitando a exceção existente) se a
    key daquele provider específico não estiver configurada, ou
    UnknownProviderError se o nome não for reconhecido."""
```

## 2. Mudanças nos módulos existentes

### `strine/planner.py`

`plan_agent(description: str, provider: Provider) -> AgentConfig` — recebe
um `Provider` já pronto em vez de uma `api_key` crua. Internamente:

```python
response = provider.create_message(
    system_prompt=SYSTEM_PROMPT,
    messages=[provider.build_user_message(description)],
    tools=[AGENT_PLAN_TOOL],
    force_tool="create_agent_plan",
)
```

`except ProviderError as exc: raise PlannerError(...)` substitui o atual
`except anthropic.APIError`.

`AgentConfig` ganha um campo novo:

```python
@dataclass
class AgentConfig:
    name: str
    prompt: str
    tools: List[str] = field(default_factory=list)
    custom_tools: List[dict] = field(default_factory=list)
    provider: str = "claude"  # novo — default mantém compatibilidade
```

`to_dict()` passa a incluir `"provider"`.

### `strine/custom_tools.py`

Mesma mudança de padrão: `generate_custom_tool(description: str, provider: Provider) -> CustomToolSpec`,
usando `provider.create_message(..., force_tool="create_custom_tool")` e
capturando `ProviderError`.

### `strine/runtime.py`

`run_agent(system_prompt, tool_schemas, executors, user_input, provider, on_tool_call=None) -> str`
— o parâmetro `api_key: str` vira `provider: Provider`. O loop interno
continua com a mesma estrutura de hoje (round-trips limitados a
`MAX_TOOL_ROUNDS`), só trocando as chamadas diretas à SDK da Anthropic por
chamadas genéricas:

```python
messages = [provider.build_user_message(user_input)]
response = provider.create_message(system_prompt, messages, tools=tool_schemas)

while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
    messages.append(provider.build_assistant_message(response))
    tool_results = [...]  # mesma lógica de execução de tool de hoje
    messages.append(provider.build_tool_result_message(tool_results))
    rounds += 1
    response = provider.create_message(system_prompt, messages, tools=tool_schemas)
```

`prepare_agent_tools()` não muda — já produz o schema genérico que os
providers consomem.

## 3. CLI (`strine/cli.py`)

- Comando `describe` ganha uma opção `--provider` (`typer.Option`, default
  `None` → resolve pra `get_provider(DEFAULT_PROVIDER)` se não passada).
  Nome inválido (ex: `--provider bogus`) → erro amigável listando os
  providers válidos (`claude`, `gemini`), exit code 1.
- `strine run agent.json` lê `agent_config.get("provider", "claude")` do
  JSON e chama `get_provider(...)` automaticamente — sem flag nova nesse
  comando, conforme decidido no brainstorm.
- Erro de API key ausente pro provider escolhido reaproveita o padrão já
  existente (`MissingAPIKeyError`), só com a mensagem apontando pra env var
  certa (`ANTHROPIC_API_KEY` ou `GEMINI_API_KEY`).

## 4. Gemini — detalhes de implementação

SDK: `google-genai` (pacote atual e mantido; **não** o `google-generativeai`,
que está em descontinuação). Tool forçado via `tool_config` com
`function_calling_config.mode="ANY"` e `allowed_function_names=[nome]`. O
formato exato de request/response será confirmado contra a documentação
atual no momento da implementação (mesma disciplina que o roadmap já pede
pra qualquer integração nova).

Variável de ambiente: `GEMINI_API_KEY`.

## 5. Formato do `agent.json`

```json
{
  "name": "sales-analyzer",
  "prompt": "...",
  "tools": ["sql", "slack"],
  "custom_tools": [],
  "provider": "gemini"
}
```

**Compatibilidade:** um `agent.json` gerado antes dessa mudança não tem a
chave `"provider"`. `strine run` trata isso com `.get("provider", "claude")`
— continua funcionando como sempre funcionou, sem quebrar agents já
criados.

## 6. Erros e casos de borda

| Caso | Comportamento |
|---|---|
| API key do provider escolhido ausente | `MissingAPIKeyError`, mensagem aponta pra env var certa do provider |
| `--provider` com nome não reconhecido | Erro amigável listando providers válidos, exit code 1 |
| Erro de API (qualquer provider) durante planning/geração/execução | Capturado dentro do adapter como exceção nativa da SDK, relançado como `ProviderError`, e daí como `PlannerError`/`CustomToolError`/`AgentRuntimeError` igual hoje — comportamento visível pro usuário não muda |
| `agent.json` antigo sem campo `"provider"` | Assume `"claude"`, roda normalmente |

## 7. Testes

- `tests/providers/test_claude.py`, `tests/providers/test_gemini.py`: cada
  um mocka a SDK correspondente e verifica a tradução nos dois sentidos —
  schema genérico → formato da SDK, resposta da SDK → `NormalizedResponse`
  — cobrindo: tool forçada, tool não-forçada com `stop_reason="tool_use"`,
  resposta só texto, erro de API vira `ProviderError`.
- Testes existentes de `planner.py`/`custom_tools.py`/`runtime.py` são
  **simplificados**: em vez de mockar `anthropic.Anthropic` no fundo de
  cada módulo, injetam um `Provider` fake/stub diretamente — a interface
  virou a costura de teste certa. Nenhum desses precisa saber que Gemini
  existe.
- Teste de CLI: `--provider gemini` seleciona o provider certo; `agent.json`
  sem `"provider"` funciona (default claude); `--provider` inválido dá erro
  amigável.
- Validação manual (não automatizável sem gastar crédito de API de
  verdade): gerar um agent com `--provider gemini`, confirmar que o
  `agent.json` tem `"provider": "gemini"`, e rodar `strine run` nele.

## Decisões já tomadas (não reabrir sem motivo novo)

- Mesmo provider pro planner e pro agent gerado (não dois providers
  independentes).
- Escolha de provider por flag (`--provider`) no `describe`, não só via
  `.env`.
- Provider fica salvo no `agent.json`; `strine run` não tem flag de
  override.
- Escopo desta rodada: só `ClaudeProvider` + `GeminiProvider`. Adapter
  estilo OpenAI (GPT/Groq/OpenRouter) fica pra depois, mas a interface já
  é desenhada pra recebê-lo sem retrabalho.
