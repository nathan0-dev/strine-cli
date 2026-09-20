# Spec: Expansão de tools + geração de tool customizada

**Data:** 2026-09-20
**Status:** Aprovado, aguardando plano de implementação

## Contexto

O Day 3 do roadmap original implementou 3 tools pré-construídas (`sql`,
`slack`, `webhook`). Ao revisar, ficou claro que esse catálogo fixo é
insuficiente pra atrair público real — "dá pra fazer pouca coisa" com
apenas essas 3. Durante a pesquisa de referência do gpt-engineer
([day0.5-gpt-engineer-research.md](../../day0.5-gpt-engineer-research.md)),
identificamos que a diferença central de capacidade entre os dois projetos
é que o gpt-engineer **gera código arbitrário** (sem catálogo fixo),
enquanto o Strine só permite escolher entre tools pré-definidas.

Esta spec expande o catálogo de tools pré-construídas e adiciona um
subsistema — inspirado no gpt-engineer, mas com escopo bem mais estreito e
seguro — que permite ao Claude **gerar uma tool customizada sob demanda**
quando o catálogo fixo não cobre o que o usuário precisa.

## Objetivo

Dar ao Strine capacidade prática de resolver a maioria dos pedidos reais de
"agent que faz X", sem exigir sandbox de execução nem infraestrutura nova,
mantendo a filosofia "é local, é sua máquina, sua API key" do roadmap.

## Não-objetivos (fora de escopo desta spec)

- Sandboxing/isolamento de execução de código gerado (aceito como risco,
  igual ao próprio gpt-engineer — mitigado por revisão manual, não por
  isolamento técnico).
- Retry automático quando o usuário recusa uma tool customizada gerada —
  v1 simplesmente não inclui a tool; o usuário pode rodar `strine` de novo
  com uma descrição diferente.
- Marketplace ou compartilhamento de tools customizadas entre usuários.
- Qualquer mudança na CLI de `strine run` (isso é escopo do Day 4).

## 1. Tools pré-construídas novas

Quatro tools novas se somam às 3 existentes, todas registradas no mesmo
`TOOLS` central (`strine/tools/__init__.py`), seguindo o padrão já
estabelecido: um módulo por tool, `SCHEMA` + `execute(**kwargs) -> str`,
erro amigável em string (nunca exceção crua) quando a credencial não está
configurada.

### `http_request` (`strine/tools/http_request.py`)

Chamador de API HTTP genérico — substitui o `webhook` como a tool "coringa"
pra conectar em qualquer serviço externo via REST.

- **Schema:** `method` (enum: GET/POST/PUT/PATCH/DELETE), `url`, `headers`
  (objeto, opcional), `body` (objeto, opcional).
- **Execução:** `requests.request(method, url, headers=headers, json=body,
  timeout=10)`. Retorna status code + corpo da resposta (truncado se muito
  grande, ex: 2000 caracteres) como string.
- Sem credencial obrigatória — auth (se precisar) vai nos `headers` que o
  próprio agent monta a partir do seu system prompt/config.

O `webhook` existente (`strine/tools/webhook.py`) é mantido como está —
não será removido nesta spec, para não quebrar agents já gerados. O
planner passa a preferir `http_request` para casos novos.

### `web_search` (`strine/tools/web_search.py`)

Busca na web via [Tavily](https://tavily.com) — API feita especificamente
pra consumo por agentes de IA (resultado já vem resumido/limpo).

- **Schema:** `query` (string).
- **Credencial:** `TAVILY_API_KEY` (env var opcional, mesmo padrão das
  outras).
- **Execução:** POST em `https://api.tavily.com/search` com a query,
  retorna os top resultados formatados como string (título + snippet +
  URL por resultado).

### `send_email` (`strine/tools/send_email.py`)

Envia email via [Resend](https://resend.com) — 1 API key, 1 chamada HTTP,
sem configuração de SMTP.

- **Schema:** `to`, `subject`, `body`.
- **Credencial:** `RESEND_API_KEY`.
- **Execução:** POST em `https://api.resend.com/emails`.

### `file_read` / `file_write` (`strine/tools/file_ops.py`)

Lê e escreve arquivos locais, no diretório de trabalho de onde o `strine
run` foi executado.

- **Schemas:**
  - `file_read`: `path` (string).
  - `file_write`: `path` (string), `content` (string).
- **Sem credencial.**
- **Validação de segurança:** ambas rejeitam paths que tentem escapar do
  diretório de trabalho (`../`, paths absolutos fora do cwd) — mesmo
  padrão de validação do `DiskMemory.__setitem__` do gpt-engineer
  (`if str(key).startswith("../"): raise ValueError(...)`).

## 2. Geração de tool customizada

### Fluxo (dentro do comando `strine "descrição"` / `describe_agent`)

```
1. plan_agent() roda normalmente, escolhe entre as 7 tools pré-construídas.
2. CLI mostra as tools escolhidas (como já faz hoje).
3. CLI sempre pergunta (input interativo):
   "Quer adicionar uma tool customizada? Descreva o que ela precisa
   fazer (Enter pra pular): "
4. Se o usuário digitar algo (não vazio):
   a. Chama generate_custom_tool(description, api_key) — tool use
      forçado, mesmo padrão do planner.
   b. CLI mostra o código gerado no terminal.
   c. CLI pergunta "Usar essa tool? (Y/n)".
   d. Se Y: salva o arquivo + registra no agent.json.
      Se n: não inclui a tool, segue sem ela (sem retry automático).
5. Se o usuário só apertar Enter: segue sem tool customizada.
```

### `generate_custom_tool()` — novo, em `strine/custom_tools.py`

```python
def generate_custom_tool(description: str, api_key: str) -> CustomToolSpec:
    ...
```

Usa tool use forçado (mesmo padrão de `plan_agent`) com uma tool
`create_custom_tool` cujo schema exige:

- `name`: slug (minúsculas, hífens/underscores).
- `description`: descrição curta da tool (pro schema que vai pro Claude
  em runtime).
- `input_schema`: JSON schema dos parâmetros que a tool recebe (objeto
  livre, mas validado como JSON schema válido antes de aceitar).
- `code`: string contendo o corpo completo de uma função Python
  `execute(**kwargs) -> str`. O prompt de sistema deixa explícito: só
  pode usar `requests` e a biblioteca padrão do Python (sem imports de
  pacotes não instalados), deve sempre retornar string, nunca lançar
  exceção não tratada (deve fazer seu próprio try/except e retornar uma
  mensagem de erro como string, igual as outras tools).

### `CustomToolSpec` (dataclass)

```python
@dataclass
class CustomToolSpec:
    name: str
    description: str
    input_schema: dict
    code: str
```

### Validação antes de mostrar pro usuário

- `compile(code, "<custom_tool>", "exec")` — captura `SyntaxError` antes
  de mostrar/salvar. Se falhar, trata como geração malsucedida (mostra
  erro, não oferece Y/n, não salva).
- Confirma que o código define uma função chamada `execute` (checagem
  simples via regex/AST, não execução).

### Persistência

Ao confirmar (Y):

- Código salvo em `<agent-name>_tools/<tool_name>.py` (pasta ao lado do
  `agent.json`, criada se não existir).
- `agent.json` ganha um campo novo:

```json
{
  "name": "sales-analyzer",
  "prompt": "...",
  "tools": ["sql", "slack"],
  "custom_tools": [
    {
      "name": "parse_invoice_pdf",
      "description": "Extrai o valor total de um PDF de nota fiscal.",
      "input_schema": { "type": "object", "properties": { "path": {"type": "string"} }, "required": ["path"] },
      "module_path": "sales-analyzer_tools/parse_invoice_pdf.py"
    }
  ]
}
```

`tools` continua só com os nomes das pré-construídas (chave no `TOOLS`
registry); `custom_tools` carrega tudo que o runtime precisa pra montar o
schema e importar o módulo, sem precisar re-gerar nada.

### Integração com o runtime (Day 4 — ainda não implementado)

Quando o `runtime.py` for escrito, ele monta a lista de tools disponíveis
assim:

1. Para cada nome em `agent_config["tools"]`: busca `TOOLS[nome]["schema"]`
   e `TOOLS[nome]["execute"]`.
2. Para cada entrada em `agent_config["custom_tools"]`: importa o módulo
   dinamicamente (`importlib.util.spec_from_file_location`), pega a
   função `execute` de lá, usa o `schema`/`description`/`input_schema`
   já salvos no `agent.json` (não precisa reconsultar o Claude).

Isso não exige nenhuma mudança em código já escrito — encaixa no que o
Day 4 já ia precisar fazer de qualquer forma.

## Erros e casos de borda

| Caso | Comportamento |
|---|---|
| Credencial de tool nova ausente (`TAVILY_API_KEY`, `RESEND_API_KEY`) | Mesmo padrão das tools existentes: `execute()` retorna string amigável explicando o que configurar, nunca lança exceção. |
| `file_read`/`file_write` com path tentando escapar do cwd | Rejeitado com `ValueError`/mensagem amigável, mesmo padrão do `DiskMemory`. |
| Código customizado gerado não compila | Mostra o erro de sintaxe, não oferece confirmação, tool não é incluída. |
| Usuário recusa (`n`) a tool customizada gerada | Tool não é incluída, agent é salvo sem ela, sem retry automático. |
| Usuário aperta Enter (pula) | Segue normalmente, nenhuma tool customizada, comportamento idêntico ao fluxo atual. |

## Segurança

Este subsistema executa código Python escrito pelo modelo, sem sandbox.
Isso é uma decisão consciente, não um descuido — mesma postura do
gpt-engineer (que também não faz sandbox, só confirmação visual). A
mitigação é: (1) o código é gerado uma única vez, na criação do agent, não
a cada execução; (2) é sempre mostrado por inteiro antes de ser salvo; (3)
exige confirmação explícita (Y/n); (4) fica salvo como arquivo `.py`
comum, revisável a qualquer momento depois. Não há garantia técnica contra
código malicioso — a garantia é a revisão humana no momento da criação.

## Teste (cenário novo pro Day 7)

Gerar um agent com um pedido que claramente não é coberto pelas 7 tools
prontas (ex: "agent que lê um PDF de nota fiscal e extrai o valor total"),
confirmar que:
1. O Strine oferece a opção de tool customizada.
2. O código gerado é sintaticamente válido e faz sentido pro pedido.
3. Ao confirmar (Y), o arquivo é salvo no lugar certo e o `agent.json`
   tem `custom_tools` populado corretamente.
4. Ao recusar (n), o agent é salvo sem a tool, sem erro.

## Decisões já tomadas (não reabrir sem motivo novo)

- Email via Resend, não SMTP genérico.
- Custom tool é gerada na criação do agent (`describe`), nunca em tempo de
  execução (`run`).
- A opção de tool customizada é sempre oferecida (opt-in do usuário), não
  depende do planner detectar automaticamente uma lacuna.
- Confirmação Y/n sempre acontece antes de salvar código gerado.
- Sem retry automático quando a tool é recusada — escopo v1.
