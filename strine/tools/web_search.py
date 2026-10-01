from __future__ import annotations

import os

import requests

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Searches the web for current information and returns the most relevant results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Term or question to search for.",
            },
        },
        "required": ["query"],
    },
}


def execute(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Tool 'web_search' not configured: set TAVILY_API_KEY in your "
            ".env to enable web search."
        )

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query},
            timeout=10,
        )
    except requests.RequestException as exc:
        return f"Error searching the web: {exc}"

    if response.status_code >= 400:
        return f"Tavily returned an error (status {response.status_code}): {response.text}"

    data = response.json()
    results = data.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"{i}. {item.get('title', '')}\n"
            f"   {item.get('content', '')}\n"
            f"   {item.get('url', '')}"
        )
    return "\n\n".join(lines)
