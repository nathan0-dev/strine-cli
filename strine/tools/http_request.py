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

_MAX_RESPONSE_CHARS = 8000


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
