# Tools Expansion + Custom Tool Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expandir o catálogo de tools pré-construídas do Strine de 3 para 7 (`sql`, `slack`, `webhook`, `http_request`, `web_search`, `send_email`, `file_read`/`file_write`) e adicionar um subsistema que permite ao Claude gerar uma tool customizada sob demanda, revisada e confirmada pelo usuário antes de ser salva.

**Architecture:** Cada tool nova segue o padrão já estabelecido em `strine/tools/*.py` — um módulo com `SCHEMA` (dict) + `execute(**kwargs) -> str`, registrado no dict central `TOOLS` em `strine/tools/__init__.py`. A geração de tool customizada é um módulo novo (`strine/custom_tools.py`) que reusa o padrão de tool-use forçado já usado em `planner.py`, mas produz código Python em vez de um plano; esse código é mostrado ao usuário e só é persistido em disco (`<agent-name>_tools/<tool_name>.py` + entrada em `agent.json`) após confirmação explícita. Toda chamada de rede (Anthropic, Tavily, Resend) é mockada nos testes automatizados — não há teste automatizado que dependa de credenciais reais.

**Tech Stack:** Python 3.9+, Typer (CLI), `anthropic` SDK (tool use forçado), `requests` (HTTP), `pytest` (testes, novo nesta plan), `unittest.mock` / `monkeypatch` do pytest (mocks de rede).

**Spec:** [docs/specs/2026-09-20-tools-expansion-design.md](../specs/2026-09-20-tools-expansion-design.md)

## Global Constraints

- Toda tool nova segue o padrão: `SCHEMA` dict + `execute(**kwargs) -> str`, nunca lança exceção pro chamador — captura e retorna string de erro amigável.
- Nenhum código em `strine/tools/*`, `strine/custom_tools.py` ou `strine/planner.py` chama `print()`/`input()`/`typer.echo`/`typer.confirm` — isso é exclusivo de `strine/cli.py`.
- Tool customizada gerada só pode usar a biblioteca padrão do Python e `requests` (nenhum outro pacote de terceiros) — isso é uma instrução no system prompt de `generate_custom_tool`, não uma sandbox técnica.
- Tool customizada gerada é sempre mostrada por inteiro no terminal e exige confirmação `Y/n` antes de ser salva — nunca salva silenciosamente.
- `file_read`/`file_write` rejeitam qualquer `path` que resolva pra fora do diretório de trabalho atual (mesma validação de path traversal que o `DiskMemory` do gpt-engineer usa).
- Nenhuma chamada de API real (Anthropic, Tavily, Resend) em teste automatizado — sempre mockada.

---

## File Structure

```
strine/
├── cli.py                    # MODIFICAR — fluxo interativo de tool customizada
├── planner.py                 # MODIFICAR — VALID_TOOLS expandido, AgentConfig.custom_tools
├── custom_tools.py            # CRIAR — CustomToolSpec + generate_custom_tool()
└── tools/
    ├── __init__.py             # MODIFICAR — registrar as 4 tools novas
    ├── http_request.py          # CRIAR
    ├── web_search.py            # CRIAR
    ├── send_email.py             # CRIAR
    └── file_ops.py                # CRIAR — file_read + file_write

tests/
├── __init__.py                # CRIAR (vazio)
├── tools/
│   ├── __init__.py              # CRIAR (vazio)
│   ├── test_http_request.py      # CRIAR
│   ├── test_web_search.py         # CRIAR
│   ├── test_send_email.py          # CRIAR
│   ├── test_file_ops.py             # CRIAR
│   └── test_registry.py              # CRIAR
├── test_planner.py             # CRIAR
├── test_custom_tools.py         # CRIAR
└── test_cli_describe.py          # CRIAR

pyproject.toml                 # MODIFICAR — dependência de teste (pytest)
.env.example                   # MODIFICAR — TAVILY_API_KEY, RESEND_API_KEY
```

---

### Task 1: Infra de testes + tool `http_request`

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/tools/__init__.py`
- Create: `strine/tools/http_request.py`
- Test: `tests/tools/test_http_request.py`

**Interfaces:**
- Produces: `strine.tools.http_request.HTTP_REQUEST_SCHEMA` (dict), `strine.tools.http_request.execute(method: str, url: str, headers: dict | None = None, body: dict | None = None) -> str`

- [ ] **Step 1: Adicionar pytest como dependência de teste**

Em `pyproject.toml`, adicione a seção depois de `[project.scripts]`:

```toml
[project.optional-dependencies]
test = ["pytest>=8.0"]
```

- [ ] **Step 2: Instalar e criar estrutura de testes**

```bash
.venv/bin/pip install -e ".[test]"
mkdir -p tests/tools
touch tests/__init__.py tests/tools/__init__.py
```

- [ ] **Step 3: Escrever o teste falhando pra `http_request`**

Crie `tests/tools/test_http_request.py`:

```python
from unittest.mock import Mock, patch

from strine.tools.http_request import execute


def test_get_request_returns_status_and_body():
    fake_response = Mock(status_code=200, text='{"ok": true}')
    with patch("strine.tools.http_request.requests.request", return_value=fake_response) as mock_request:
        result = execute(method="GET", url="https://api.example.com/status")

    mock_request.assert_called_once_with(
        "GET",
        "https://api.example.com/status",
        headers=None,
        json=None,
        timeout=10,
    )
    assert "200" in result
    assert '{"ok": true}' in result


def test_post_request_with_headers_and_body():
    fake_response = Mock(status_code=201, text="created")
    with patch("strine.tools.http_request.requests.request", return_value=fake_response) as mock_request:
        result = execute(
            method="POST",
            url="https://api.example.com/items",
            headers={"Authorization": "Bearer xyz"},
            body={"name": "item"},
        )

    mock_request.assert_called_once_with(
        "POST",
        "https://api.example.com/items",
        headers={"Authorization": "Bearer xyz"},
        json={"name": "item"},
        timeout=10,
    )
    assert "201" in result


def test_long_response_body_is_truncated():
    fake_response = Mock(status_code=200, text="x" * 5000)
    with patch("strine.tools.http_request.requests.request", return_value=fake_response):
        result = execute(method="GET", url="https://api.example.com/big")

    assert len(result) < 5000


def test_connection_error_returns_friendly_string():
    import requests

    with patch("strine.tools.http_request.requests.request", side_effect=requests.RequestException("boom")):
        result = execute(method="GET", url="https://api.example.com/down")

    assert "Erro" in result
    assert "boom" in result
```

- [ ] **Step 4: Rodar os testes e confirmar que falham**

Run: `.venv/bin/pytest tests/tools/test_http_request.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strine.tools.http_request'`

- [ ] **Step 5: Implementar `strine/tools/http_request.py`**

```python
from __future__ import annotations

import requests

HTTP_REQUEST_SCHEMA = {
    "name": "http_request",
    "description": (
        "Faz uma chamada HTTP genérica (GET, POST, PUT, PATCH ou DELETE) "
        "para qualquer URL externa, com headers e corpo opcionais. Use "
        "isso para integrar com qualquer API REST."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "Método HTTP da requisição.",
            },
            "url": {
                "type": "string",
                "description": "URL de destino da requisição.",
            },
            "headers": {
                "type": "object",
                "description": "Headers HTTP opcionais, ex: Authorization.",
            },
            "body": {
                "type": "object",
                "description": "Corpo JSON opcional da requisição.",
            },
        },
        "required": ["method", "url"],
    },
}

_MAX_RESPONSE_CHARS = 2000


def execute(
    method: str, url: str, headers: dict | None = None, body: dict | None = None
) -> str:
    try:
        response = requests.request(
            method, url, headers=headers, json=body, timeout=10
        )
    except requests.RequestException as exc:
        return f"Erro ao fazer a requisição HTTP: {exc}"

    text = response.text
    if len(text) > _MAX_RESPONSE_CHARS:
        text = text[:_MAX_RESPONSE_CHARS] + "... (truncado)"

    return f"Status: {response.status_code}\nCorpo: {text}"
```

- [ ] **Step 6: Rodar os testes e confirmar que passam**

Run: `.venv/bin/pytest tests/tools/test_http_request.py -v`
Expected: PASS (4 testes)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/tools/__init__.py tests/tools/test_http_request.py strine/tools/http_request.py
git commit -m "feat: add http_request tool with test infra"
```

---

### Task 2: Tool `web_search` (Tavily)

**Files:**
- Create: `strine/tools/web_search.py`
- Test: `tests/tools/test_web_search.py`

**Interfaces:**
- Produces: `strine.tools.web_search.WEB_SEARCH_SCHEMA` (dict), `strine.tools.web_search.execute(query: str) -> str`

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/tools/test_web_search.py`:

```python
import os
from unittest.mock import Mock, patch

from strine.tools.web_search import execute


def test_missing_api_key_returns_friendly_message(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = execute(query="strine cli")
    assert "TAVILY_API_KEY" in result


def test_successful_search_formats_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    fake_response = Mock()
    fake_response.json.return_value = {
        "results": [
            {
                "title": "Strine CLI",
                "url": "https://example.com/strine",
                "content": "Describe your agent, get it running.",
            }
        ]
    }
    with patch("strine.tools.web_search.requests.post", return_value=fake_response) as mock_post:
        result = execute(query="strine cli")

    assert mock_post.call_args.kwargs["json"]["api_key"] == "tvly-fake-key"
    assert mock_post.call_args.kwargs["json"]["query"] == "strine cli"
    assert "Strine CLI" in result
    assert "https://example.com/strine" in result


def test_request_exception_returns_friendly_message(monkeypatch):
    import requests

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    with patch("strine.tools.web_search.requests.post", side_effect=requests.RequestException("timeout")):
        result = execute(query="strine cli")

    assert "Erro" in result
    assert "timeout" in result
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/tools/test_web_search.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `strine/tools/web_search.py`**

```python
from __future__ import annotations

import os

import requests

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Pesquisa informação atual na web e retorna os resultados mais relevantes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Termo ou pergunta a ser pesquisada.",
            },
        },
        "required": ["query"],
    },
}


def execute(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Tool 'web_search' não configurada: defina TAVILY_API_KEY no "
            "seu .env pra habilitar busca na web."
        )

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query},
            timeout=10,
        )
        data = response.json()
    except requests.RequestException as exc:
        return f"Erro ao pesquisar na web: {exc}"

    results = data.get("results", [])
    if not results:
        return "Nenhum resultado encontrado."

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"{i}. {item.get('title', '')}\n"
            f"   {item.get('content', '')}\n"
            f"   {item.get('url', '')}"
        )
    return "\n\n".join(lines)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/tools/test_web_search.py -v`
Expected: PASS (3 testes)

- [ ] **Step 5: Commit**

```bash
git add strine/tools/web_search.py tests/tools/test_web_search.py
git commit -m "feat: add web_search tool via Tavily"
```

---

### Task 3: Tool `send_email` (Resend)

**Files:**
- Create: `strine/tools/send_email.py`
- Test: `tests/tools/test_send_email.py`

**Interfaces:**
- Produces: `strine.tools.send_email.SEND_EMAIL_SCHEMA` (dict), `strine.tools.send_email.execute(to: str, subject: str, body: str) -> str`

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/tools/test_send_email.py`:

```python
from unittest.mock import Mock, patch

from strine.tools.send_email import execute


def test_missing_api_key_returns_friendly_message(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = execute(to="a@example.com", subject="Oi", body="Teste")
    assert "RESEND_API_KEY" in result


def test_successful_send(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_fake_key")
    fake_response = Mock(status_code=200)
    with patch("strine.tools.send_email.requests.post", return_value=fake_response) as mock_post:
        result = execute(to="a@example.com", subject="Oi", body="Teste")

    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["to"] == ["a@example.com"]
    assert sent_json["subject"] == "Oi"
    assert "sucesso" in result.lower()


def test_api_error_returns_friendly_message(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_fake_key")
    fake_response = Mock(status_code=422, text='{"message": "invalid to address"}')
    with patch("strine.tools.send_email.requests.post", return_value=fake_response):
        result = execute(to="not-an-email", subject="Oi", body="Teste")

    assert "422" in result or "erro" in result.lower()


def test_request_exception_returns_friendly_message(monkeypatch):
    import requests

    monkeypatch.setenv("RESEND_API_KEY", "re_fake_key")
    with patch("strine.tools.send_email.requests.post", side_effect=requests.RequestException("down")):
        result = execute(to="a@example.com", subject="Oi", body="Teste")

    assert "Erro" in result
    assert "down" in result
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/tools/test_send_email.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `strine/tools/send_email.py`**

```python
from __future__ import annotations

import os

import requests

SEND_EMAIL_SCHEMA = {
    "name": "send_email",
    "description": "Envia um email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Endereço de email do destinatário."},
            "subject": {"type": "string", "description": "Assunto do email."},
            "body": {"type": "string", "description": "Corpo do email (texto simples)."},
        },
        "required": ["to", "subject", "body"],
    },
}

_DEFAULT_FROM = "Strine Agent <onboarding@resend.dev>"


def execute(to: str, subject: str, body: str) -> str:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return (
            "Tool 'send_email' não configurada: defina RESEND_API_KEY no "
            "seu .env pra habilitar o envio de emails."
        )

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": os.getenv("RESEND_FROM_EMAIL", _DEFAULT_FROM),
                "to": [to],
                "subject": subject,
                "text": body,
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        return f"Erro ao enviar email: {exc}"

    if response.status_code >= 300:
        return f"Resend retornou um erro (status {response.status_code}): {response.text}"

    return f"Email enviado com sucesso para {to}."
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/tools/test_send_email.py -v`
Expected: PASS (4 testes)

- [ ] **Step 5: Commit**

```bash
git add strine/tools/send_email.py tests/tools/test_send_email.py
git commit -m "feat: add send_email tool via Resend"
```

---

### Task 4: Tools `file_read` / `file_write`

**Files:**
- Create: `strine/tools/file_ops.py`
- Test: `tests/tools/test_file_ops.py`

**Interfaces:**
- Produces: `strine.tools.file_ops.FILE_READ_SCHEMA`, `strine.tools.file_ops.FILE_WRITE_SCHEMA` (dicts), `strine.tools.file_ops.execute_read(path: str) -> str`, `strine.tools.file_ops.execute_write(path: str, content: str) -> str`

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/tools/test_file_ops.py`:

```python
import os

from strine.tools.file_ops import execute_read, execute_write


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    write_result = execute_write(path="notes.txt", content="olá mundo")
    assert "sucesso" in write_result.lower()

    read_result = execute_read(path="notes.txt")
    assert read_result == "olá mundo"


def test_read_missing_file_returns_friendly_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="nao-existe.txt")
    assert "não encontrado" in result.lower() or "not found" in result.lower()


def test_read_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="../../etc/passwd")
    assert "não permitido" in result.lower() or "inválido" in result.lower()


def test_write_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_write(path="../outside.txt", content="x")
    assert "não permitido" in result.lower() or "inválido" in result.lower()
    assert not (tmp_path.parent / "outside.txt").exists()


def test_write_creates_parent_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    execute_write(path="subdir/nested.txt", content="ok")
    assert (tmp_path / "subdir" / "nested.txt").read_text() == "ok"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/tools/test_file_ops.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `strine/tools/file_ops.py`**

```python
from __future__ import annotations

from pathlib import Path

FILE_READ_SCHEMA = {
    "name": "file_read",
    "description": "Lê o conteúdo de um arquivo local, dentro do diretório de trabalho atual.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho relativo do arquivo a ser lido.",
            },
        },
        "required": ["path"],
    },
}

FILE_WRITE_SCHEMA = {
    "name": "file_write",
    "description": "Escreve conteúdo em um arquivo local, dentro do diretório de trabalho atual.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho relativo do arquivo a ser escrito.",
            },
            "content": {
                "type": "string",
                "description": "Conteúdo a ser escrito no arquivo.",
            },
        },
        "required": ["path", "content"],
    },
}


def _resolve_safe_path(path: str) -> Path | None:
    base = Path.cwd().resolve()
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def execute_read(path: str) -> str:
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return f"Caminho não permitido: '{path}' está fora do diretório de trabalho."

    if not resolved.is_file():
        return f"Arquivo não encontrado: '{path}'."

    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Erro ao ler o arquivo: {exc}"


def execute_write(path: str, content: str) -> str:
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return f"Caminho não permitido: '{path}' está fora do diretório de trabalho."

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Erro ao escrever o arquivo: {exc}"

    return f"Arquivo '{path}' escrito com sucesso."
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/tools/test_file_ops.py -v`
Expected: PASS (5 testes)

- [ ] **Step 5: Commit**

```bash
git add strine/tools/file_ops.py tests/tools/test_file_ops.py
git commit -m "feat: add file_read and file_write tools"
```

---

### Task 5: Registrar as 4 tools novas + atualizar `.env.example`

**Files:**
- Modify: `strine/tools/__init__.py`
- Modify: `.env.example`
- Test: `tests/tools/test_registry.py`

**Interfaces:**
- Consumes: todos os `SCHEMA`/`execute`/`execute_read`/`execute_write` das Tasks 1-4.
- Produces: `strine.tools.TOOLS` (dict) com 7 chaves: `sql`, `slack`, `webhook`, `http_request`, `web_search`, `send_email`, `file_read`, `file_write` (8 chaves no total — `file_read` e `file_write` são entradas separadas).

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/tools/test_registry.py`:

```python
from strine.tools import TOOLS

EXPECTED_TOOL_NAMES = {
    "sql",
    "slack",
    "webhook",
    "http_request",
    "web_search",
    "send_email",
    "file_read",
    "file_write",
}


def test_registry_has_all_expected_tools():
    assert set(TOOLS.keys()) == EXPECTED_TOOL_NAMES


def test_every_tool_has_well_formed_schema_and_callable_execute():
    for name, entry in TOOLS.items():
        schema = entry["schema"]
        assert "name" in schema, f"{name}: schema sem 'name'"
        assert "description" in schema, f"{name}: schema sem 'description'"
        assert schema["input_schema"]["type"] == "object", f"{name}: input_schema não é object"
        assert callable(entry["execute"]), f"{name}: execute não é callable"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/tools/test_registry.py -v`
Expected: FAIL — `AssertionError` (registry ainda só tem 3 chaves)

- [ ] **Step 3: Atualizar `strine/tools/__init__.py`**

Substitua o conteúdo inteiro do arquivo por:

```python
from strine.tools import file_ops, http_request, send_email, slack, sql, web_search, webhook

TOOLS = {
    "sql": {"schema": sql.QUERY_DATABASE_SCHEMA, "execute": sql.execute},
    "slack": {"schema": slack.POST_TO_SLACK_SCHEMA, "execute": slack.execute},
    "webhook": {"schema": webhook.SEND_WEBHOOK_SCHEMA, "execute": webhook.execute},
    "http_request": {
        "schema": http_request.HTTP_REQUEST_SCHEMA,
        "execute": http_request.execute,
    },
    "web_search": {
        "schema": web_search.WEB_SEARCH_SCHEMA,
        "execute": web_search.execute,
    },
    "send_email": {
        "schema": send_email.SEND_EMAIL_SCHEMA,
        "execute": send_email.execute,
    },
    "file_read": {
        "schema": file_ops.FILE_READ_SCHEMA,
        "execute": file_ops.execute_read,
    },
    "file_write": {
        "schema": file_ops.FILE_WRITE_SCHEMA,
        "execute": file_ops.execute_write,
    },
}
```

- [ ] **Step 4: Atualizar `.env.example`**

Substitua o bloco de variáveis opcionais no final do arquivo por:

```
# Opcionais — usados por algumas tools do agent:
# DATABASE_URL=postgresql://user:pass@host:5432/dbname
# SLACK_BOT_TOKEN=xoxb-...
# TAVILY_API_KEY=tvly-...
# RESEND_API_KEY=re_...
# RESEND_FROM_EMAIL=Seu Nome <voce@seudominio.com>
```

- [ ] **Step 5: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/tools/ -v`
Expected: PASS (todos os testes de Task 1-5)

- [ ] **Step 6: Commit**

```bash
git add strine/tools/__init__.py .env.example tests/tools/test_registry.py
git commit -m "feat: register 4 new tools in central registry"
```

---

### Task 6: Expandir `planner.py` pra conhecer as 7 tools + campo `custom_tools`

**Files:**
- Modify: `strine/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: nada de tasks anteriores diretamente (só precisa dos *nomes* das tools, que já são conhecidos).
- Produces: `strine.planner.VALID_TOOLS` (set com 8 nomes), `strine.planner.AgentConfig` agora com campo `custom_tools: List[dict] = field(default_factory=list)`, e `AgentConfig.to_dict()` incluindo essa chave.

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/test_planner.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

from strine.planner import VALID_TOOLS, AgentConfig, plan_agent


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


def test_agent_config_to_dict_includes_custom_tools_default_empty():
    config = AgentConfig(name="foo", prompt="bar", tools=["sql"])
    assert config.to_dict() == {
        "name": "foo",
        "prompt": "bar",
        "tools": ["sql"],
        "custom_tools": [],
    }


def _fake_response(input_dict):
    tool_use_block = SimpleNamespace(type="tool_use", input=input_dict)
    return SimpleNamespace(content=[tool_use_block])


def test_plan_agent_filters_invalid_tool_names():
    fake = _fake_response(
        {"name": "my-agent", "prompt": "You are helpful.", "tools": ["sql", "not-a-real-tool"]}
    )
    with patch("strine.planner.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        config = plan_agent("descrição qualquer", api_key="sk-ant-fake")

    assert config.tools == ["sql"]
    assert config.custom_tools == []


def test_plan_agent_accepts_new_tool_names():
    fake = _fake_response(
        {
            "name": "researcher",
            "prompt": "You research things.",
            "tools": ["web_search", "http_request"],
        }
    )
    with patch("strine.planner.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        config = plan_agent("descrição qualquer", api_key="sk-ant-fake")

    assert set(config.tools) == {"web_search", "http_request"}
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_planner.py -v`
Expected: FAIL — `VALID_TOOLS` ainda tem só 3 nomes; `to_dict()` sem `custom_tools`

- [ ] **Step 3: Atualizar `strine/planner.py`**

Substitua as linhas 8-25 (definição de `VALID_TOOLS` e `SYSTEM_PROMPT`):

```python
VALID_TOOLS = {
    "sql",
    "slack",
    "webhook",
    "http_request",
    "web_search",
    "send_email",
    "file_read",
    "file_write",
}

SYSTEM_PROMPT = """Você é um "agent architect". Dado o pedido de um usuário em
linguagem natural, decida como montar um agent de IA:

1. Quais tools esse agent precisa, escolhendo apenas entre: "sql", "slack",
   "webhook", "http_request", "web_search", "send_email", "file_read",
   "file_write". Se nenhuma for necessária, retorne uma lista vazia.
2. Um system prompt claro e específico para o agent que vai ser criado,
   descrevendo seu papel, escopo e como deve se comportar.
3. Um nome curto e descritivo em formato slug (minúsculas, hífens, sem
   espaços), ex: "sales-analyzer", "support-bot".

Escolha uma tool apenas quando o pedido do usuário claramente precisar dela:
- "sql": o agent precisa consultar um banco de dados.
- "slack": o agent precisa postar mensagens no Slack.
- "webhook": o agent precisa disparar uma chamada HTTP simples (POST com payload fixo).
- "http_request": o agent precisa chamar uma API HTTP externa de forma mais flexível (qualquer método, headers, auth).
- "web_search": o agent precisa pesquisar informação atual na internet.
- "send_email": o agent precisa enviar emails.
- "file_read": o agent precisa ler arquivos locais.
- "file_write": o agent precisa escrever/salvar arquivos locais.

Prefira "http_request" a "webhook" para integrações novas, a menos que o
pedido seja literalmente só um POST simples. Não invente necessidade de
tools que o usuário não pediu."""
```

Substitua a definição de `AgentConfig` (linhas 52-59 no arquivo original) por:

```python
@dataclass
class AgentConfig:
    name: str
    prompt: str
    tools: List[str] = field(default_factory=list)
    custom_tools: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "tools": self.tools,
            "custom_tools": self.custom_tools,
        }
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_planner.py -v`
Expected: PASS (4 testes)

- [ ] **Step 5: Rodar a suíte inteira até aqui**

Run: `.venv/bin/pytest -v`
Expected: PASS (todos os testes de Task 1-6)

- [ ] **Step 6: Commit**

```bash
git add strine/planner.py tests/test_planner.py
git commit -m "feat: expand planner to 8 built-in tools, add custom_tools field"
```

---

### Task 7: `strine/custom_tools.py` — geração de tool customizada

**Files:**
- Create: `strine/custom_tools.py`
- Test: `tests/test_custom_tools.py`

**Interfaces:**
- Produces: `strine.custom_tools.CustomToolSpec` (dataclass com `name: str`, `description: str`, `input_schema: dict`, `code: str`), `strine.custom_tools.CustomToolError` (exceção), `strine.custom_tools.generate_custom_tool(description: str, api_key: str) -> CustomToolSpec`.

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/test_custom_tools.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strine.custom_tools import CustomToolError, CustomToolSpec, generate_custom_tool


def _fake_response(input_dict):
    tool_use_block = SimpleNamespace(type="tool_use", input=input_dict)
    return SimpleNamespace(content=[tool_use_block])


def test_generate_custom_tool_returns_spec_for_valid_code():
    fake = _fake_response(
        {
            "name": "parse_invoice_total",
            "description": "Extrai o valor total de um texto de nota fiscal.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "code": "def execute(text):\n    return 'total: 0'\n",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        spec = generate_custom_tool("extrai o total de uma nota fiscal", api_key="sk-ant-fake")

    assert isinstance(spec, CustomToolSpec)
    assert spec.name == "parse_invoice_total"
    assert "def execute" in spec.code


def test_generate_custom_tool_rejects_code_with_syntax_error():
    fake = _fake_response(
        {
            "name": "broken_tool",
            "description": "Tool quebrada.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def execute(:\n    pass",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_rejects_code_without_execute_function():
    fake = _fake_response(
        {
            "name": "no_execute_tool",
            "description": "Tool sem execute.",
            "input_schema": {"type": "object", "properties": {}},
            "code": "def other_function():\n    pass",
        }
    )
    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")


def test_generate_custom_tool_wraps_api_error():
    import anthropic

    with patch("strine.custom_tools.anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = anthropic.APIError(
            "boom", request=SimpleNamespace(), body=None
        )
        with pytest.raises(CustomToolError):
            generate_custom_tool("qualquer coisa", api_key="sk-ant-fake")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_custom_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strine.custom_tools'`

- [ ] **Step 3: Implementar `strine/custom_tools.py`**

```python
import ast
from dataclasses import dataclass

import anthropic

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """Você escreve uma tool Python customizada pra um agent de IA,
a partir da descrição de um usuário.

Regras obrigatórias do código gerado:
- Defina exatamente uma função chamada `execute` que recebe os parâmetros
  descritos no seu próprio input_schema como keyword arguments e retorna
  uma string.
- Só pode importar a biblioteca padrão do Python ou o pacote `requests`.
  Nenhum outro pacote de terceiros está disponível.
- Nunca deixe uma exceção não tratada escapar de `execute` — capture erros
  internamente (try/except) e retorne uma mensagem de erro como string,
  igual o resto do código faria.
- O código deve ser completo e correto, sem placeholders."""

CREATE_CUSTOM_TOOL_TOOL = {
    "name": "create_custom_tool",
    "description": "Registra a tool customizada gerada: nome, descrição, schema de input e código Python.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Nome curto em formato slug (minúsculas, underscores), ex: parse_invoice_total.",
            },
            "description": {
                "type": "string",
                "description": "Descrição curta do que a tool faz, pro agent entender quando usá-la.",
            },
            "input_schema": {
                "type": "object",
                "description": "JSON schema (formato input_schema da Anthropic) dos parâmetros que a tool recebe.",
            },
            "code": {
                "type": "string",
                "description": "Código Python completo definindo a função execute(**kwargs) -> str.",
            },
        },
        "required": ["name", "description", "input_schema", "code"],
    },
}


@dataclass
class CustomToolSpec:
    name: str
    description: str
    input_schema: dict
    code: str


class CustomToolError(RuntimeError):
    pass


def _validate_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CustomToolError(f"O código gerado tem um erro de sintaxe: {exc}") from exc

    has_execute = any(
        isinstance(node, ast.FunctionDef) and node.name == "execute"
        for node in ast.walk(tree)
    )
    if not has_execute:
        raise CustomToolError(
            "O código gerado não define uma função 'execute'."
        )


def generate_custom_tool(description: str, api_key: str) -> CustomToolSpec:
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[CREATE_CUSTOM_TOOL_TOOL],
            tool_choice={"type": "tool", "name": "create_custom_tool"},
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.APIError as exc:
        raise CustomToolError(f"Erro ao chamar a API da Anthropic: {exc}") from exc

    tool_use = next(
        (block for block in response.content if block.type == "tool_use"), None
    )
    if tool_use is None:
        raise CustomToolError("O modelo não retornou uma tool customizada estruturada.")

    plan = tool_use.input
    _validate_code(plan["code"])

    return CustomToolSpec(
        name=plan["name"],
        description=plan["description"],
        input_schema=plan["input_schema"],
        code=plan["code"],
    )
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_custom_tools.py -v`
Expected: PASS (4 testes)

- [ ] **Step 5: Commit**

```bash
git add strine/custom_tools.py tests/test_custom_tools.py
git commit -m "feat: add custom tool generation via forced tool use"
```

---

### Task 8: Fluxo interativo no `cli.py` — oferecer e persistir tool customizada

**Files:**
- Modify: `strine/cli.py`
- Test: `tests/test_cli_describe.py`

**Interfaces:**
- Consumes: `strine.planner.plan_agent` (Task 6), `strine.custom_tools.generate_custom_tool` / `CustomToolError` / `CustomToolSpec` (Task 7).
- Produces: comportamento novo do comando `describe` — sem mudar sua assinatura pública (`strine "descrição"` continua funcionando igual).

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/test_cli_describe.py`:

```python
import json
from unittest.mock import patch

from typer.testing import CliRunner

from strine.cli import app
from strine.custom_tools import CustomToolSpec
from strine.planner import AgentConfig

runner = CliRunner()


def _patch_common():
    return (
        patch("strine.cli.load_api_key", return_value="sk-ant-fake"),
        patch(
            "strine.cli.plan_agent",
            return_value=AgentConfig(name="test-agent", prompt="You help.", tools=["sql"]),
        ),
    )


def test_describe_skips_custom_tool_when_user_presses_enter():
    with runner.isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan:
            result = runner.invoke(app, ["describe", "um", "agent", "qualquer"], input="\n")

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []


def test_describe_saves_custom_tool_when_user_confirms():
    custom_spec = CustomToolSpec(
        name="parse_thing",
        description="Faz algo customizado.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        code="def execute(x):\n    return x\n",
    )
    with runner.isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan, patch("strine.cli.generate_custom_tool", return_value=custom_spec):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="parseia uma coisa\ny\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert len(saved["custom_tools"]) == 1
        assert saved["custom_tools"][0]["name"] == "parse_thing"
        assert saved["custom_tools"][0]["module_path"] == "test-agent_tools/parse_thing.py"

        with open("test-agent_tools/parse_thing.py") as f:
            assert "def execute(x):" in f.read()


def test_describe_discards_custom_tool_when_user_declines():
    custom_spec = CustomToolSpec(
        name="parse_thing",
        description="Faz algo customizado.",
        input_schema={"type": "object", "properties": {}},
        code="def execute():\n    return 'x'\n",
    )
    with runner.isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with patch_key, patch_plan, patch("strine.cli.generate_custom_tool", return_value=custom_spec):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="parseia uma coisa\nn\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []
        import os

        assert not os.path.exists("test-agent_tools")


def test_describe_handles_custom_tool_generation_failure_gracefully():
    from strine.custom_tools import CustomToolError

    with runner.isolated_filesystem():
        patch_key, patch_plan = _patch_common()
        with (
            patch_key,
            patch_plan,
            patch("strine.cli.generate_custom_tool", side_effect=CustomToolError("falhou")),
        ):
            result = runner.invoke(
                app,
                ["describe", "um", "agent", "qualquer"],
                input="parseia uma coisa\n",
            )

        assert result.exit_code == 0
        saved = json.loads(open("test-agent.json").read())
        assert saved["custom_tools"] == []
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.venv/bin/pytest tests/test_cli_describe.py -v`
Expected: FAIL — comando ainda não faz a pergunta de tool customizada, `input="\n"` sobra sem ser consumido ou o teste de confirmação falha

- [ ] **Step 3: Atualizar `strine/cli.py`**

Adicione os imports novos no topo do arquivo (depois da linha `from strine.planner import PlannerError, plan_agent`):

```python
from strine.custom_tools import CustomToolError, generate_custom_tool
```

Substitua o corpo de `describe_agent` (do `tools_label = ...` até o fim da função) por:

```python
    tools_label = ", ".join(t.upper() for t in agent_config.tools) or "nenhuma"
    typer.echo(f"✓ Tools escolhidas: {tools_label}")
    typer.echo(f"✓ Prompt gerado: {agent_config.prompt}")

    custom_description = typer.prompt(
        "\nQuer adicionar uma tool customizada? Descreva o que ela precisa "
        "fazer (Enter pra pular)",
        default="",
        show_default=False,
    )

    if custom_description.strip():
        try:
            custom_spec = generate_custom_tool(custom_description, api_key)
        except CustomToolError as exc:
            typer.echo(f"Não foi possível gerar a tool customizada: {exc}", err=True)
        else:
            typer.echo("\nCódigo gerado para a tool customizada:\n")
            typer.echo(custom_spec.code)
            if typer.confirm("\nUsar essa tool?", default=True):
                tools_dir = Path(f"{agent_config.name}_tools")
                tools_dir.mkdir(parents=True, exist_ok=True)
                module_path = tools_dir / f"{custom_spec.name}.py"
                module_path.write_text(custom_spec.code)

                agent_config.custom_tools.append(
                    {
                        "name": custom_spec.name,
                        "description": custom_spec.description,
                        "input_schema": custom_spec.input_schema,
                        "module_path": str(module_path),
                    }
                )
                typer.echo(f"✓ Tool customizada salva em {module_path}")
            else:
                typer.echo("Ok, seguindo sem essa tool.")

    output_path = Path(f"{agent_config.name}.json")
    output_path.write_text(
        json.dumps(agent_config.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )

    typer.echo(
        f"\nAgent salvo em ./{output_path.name}. "
        f"Rode com: strine run ./{output_path.name}"
    )
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.venv/bin/pytest tests/test_cli_describe.py -v`
Expected: PASS (4 testes)

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv/bin/pytest -v`
Expected: PASS — todos os testes de Task 1-8

- [ ] **Step 6: Commit**

```bash
git add strine/cli.py tests/test_cli_describe.py
git commit -m "feat: offer and persist custom tool generation in describe flow"
```

---

## Verificação final (manual, com API key real)

Depois de todas as tasks completas e a suíte de testes passando, valide manualmente (isso não é automatizável sem gastar créditos de API de verdade — fica pro Day 7 do roadmap original):

1. `strine "quero um agent que pesquisa notícias na web e me manda um resumo por email"` — confirmar que escolhe `web_search` + `send_email`.
2. Quando perguntado sobre tool customizada, digitar uma descrição de algo fora do catálogo (ex: "lê um PDF de nota fiscal e extrai o valor total"), confirmar que o código gerado faz sentido, aceitar (Y), e verificar que `<agent-name>_tools/<tool>.py` foi criado e `custom_tools` está no `agent.json`.
3. Repetir o passo 2 recusando (n) e confirmar que nada foi salvo.
4. Rodar `strine "descrição qualquer"` e simplesmente apertar Enter na pergunta de tool customizada — confirmar que segue normal, sem erro.
