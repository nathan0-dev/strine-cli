from __future__ import annotations

import requests

SEND_WEBHOOK_SCHEMA = {
    "name": "send_webhook",
    "description": "Envia um POST HTTP com um payload JSON para uma URL externa.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL de destino do webhook.",
            },
            "payload": {
                "type": "object",
                "description": "Corpo JSON a ser enviado no POST.",
            },
        },
        "required": ["url", "payload"],
    },
}


def execute(url: str, payload: dict) -> str:
    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        return f"Erro ao enviar webhook: {exc}"

    return f"Webhook enviado. Status: {response.status_code}."
