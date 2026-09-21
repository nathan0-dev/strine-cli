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
    except requests.RequestException as exc:
        return f"Erro ao pesquisar na web: {exc}"

    if response.status_code >= 400:
        return f"Tavily retornou um erro (status {response.status_code}): {response.text}"

    data = response.json()
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
