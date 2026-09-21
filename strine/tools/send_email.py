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
