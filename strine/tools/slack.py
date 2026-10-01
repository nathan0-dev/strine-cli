from __future__ import annotations

import os

import requests

POST_TO_SLACK_SCHEMA = {
    "name": "post_to_slack",
    "description": "Posts a message to a Slack channel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Slack channel where the message will be posted, e.g. #sales.",
            },
            "message": {
                "type": "string",
                "description": "Content of the message to post.",
            },
        },
        "required": ["channel", "message"],
    },
}


def execute(channel: str, message: str) -> str:
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return (
            "Tool 'slack' not configured: set SLACK_BOT_TOKEN in your .env "
            "to enable sending messages to Slack."
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
        return f"Error connecting to Slack: {exc}"

    if not data.get("ok"):
        return f"Slack returned an error: {data.get('error', 'unknown')}"

    return f"Message posted to {channel} successfully."
