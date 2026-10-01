from __future__ import annotations

import requests

SEND_WEBHOOK_SCHEMA = {
    "name": "send_webhook",
    "description": "Sends an HTTP POST with a JSON payload to an external URL.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Destination URL for the webhook.",
            },
            "payload": {
                "type": "object",
                "description": "JSON body to send in the POST.",
            },
        },
        "required": ["url", "payload"],
    },
}


def execute(url: str, payload: dict) -> str:
    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        return f"Error sending webhook: {exc}"

    return f"Webhook sent. Status: {response.status_code}."
