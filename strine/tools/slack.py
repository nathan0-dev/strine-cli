from __future__ import annotations

import os

import requests

POST_TO_SLACK_SCHEMA = {
    "name": "post_to_slack",
    "description": "Posta uma mensagem em um canal do Slack.",
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Canal do Slack onde a mensagem será postada, ex: #vendas.",
            },
            "message": {
                "type": "string",
                "description": "Conteúdo da mensagem a ser postada.",
            },
        },
        "required": ["channel", "message"],
    },
}


def execute(channel: str, message: str) -> str:
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return (
            "Tool 'slack' não configurada: defina SLACK_BOT_TOKEN no seu .env "
            "pra habilitar o envio de mensagens ao Slack."
        )

    try:
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel, "text": message},
            timeout=10,
        )
        data = response.json()
    except requests.RequestException as exc:
        return f"Erro ao conectar com o Slack: {exc}"

    if not data.get("ok"):
        return f"Slack retornou um erro: {data.get('error', 'desconhecido')}"

    return f"Mensagem postada em {channel} com sucesso."
