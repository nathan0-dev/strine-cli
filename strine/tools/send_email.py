from __future__ import annotations

import os

import requests

SEND_EMAIL_SCHEMA = {
    "name": "send_email",
    "description": "Sends an email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient's email address."},
            "subject": {"type": "string", "description": "Email subject."},
            "body": {"type": "string", "description": "Email body (plain text)."},
        },
        "required": ["to", "subject", "body"],
    },
}

_DEFAULT_FROM = "Strine Agent <onboarding@resend.dev>"


def execute(to: str, subject: str, body: str) -> str:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return (
            "Tool 'send_email' not configured: set RESEND_API_KEY in your "
            ".env to enable sending emails."
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
        return f"Error sending email: {exc}"

    if response.status_code >= 300:
        return f"Resend returned an error (status {response.status_code}): {response.text}"

    return f"Email sent successfully to {to}."
