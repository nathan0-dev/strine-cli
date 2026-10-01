from __future__ import annotations

import requests

HTTP_REQUEST_SCHEMA = {
    "name": "http_request",
    "description": (
        "Makes a generic HTTP call (GET, POST, PUT, PATCH, or DELETE) to "
        "any external URL, with optional headers and body. Use this to "
        "integrate with any REST API."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method for the request.",
            },
            "url": {
                "type": "string",
                "description": "Destination URL for the request.",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers, e.g. Authorization.",
            },
            "body": {
                "type": "object",
                "description": "Optional JSON body for the request.",
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
        return f"Error making the HTTP request: {exc}"

    text = response.text
    if len(text) > _MAX_RESPONSE_CHARS:
        text = text[:_MAX_RESPONSE_CHARS] + "... (truncated)"

    return f"Status: {response.status_code}\nBody: {text}"
