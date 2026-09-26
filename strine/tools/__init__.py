from strine.tools import file_ops, http_request, send_email, slack, sql, web_search, webhook

# requires_env: variáveis de ambiente sem as quais a tool não funciona.
# O catálogo (strine describe / strine tools) usa isso pra avisar o que falta.
TOOLS = {
    "sql": {
        "schema": sql.QUERY_DATABASE_SCHEMA,
        "execute": sql.execute,
        "requires_env": ["DATABASE_URL"],
    },
    "slack": {
        "schema": slack.POST_TO_SLACK_SCHEMA,
        "execute": slack.execute,
        "requires_env": ["SLACK_BOT_TOKEN"],
    },
    "webhook": {
        "schema": webhook.SEND_WEBHOOK_SCHEMA,
        "execute": webhook.execute,
        "requires_env": [],
    },
    "http_request": {
        "schema": http_request.HTTP_REQUEST_SCHEMA,
        "execute": http_request.execute,
        "requires_env": [],
    },
    "web_search": {
        "schema": web_search.WEB_SEARCH_SCHEMA,
        "execute": web_search.execute,
        "requires_env": ["TAVILY_API_KEY"],
    },
    "send_email": {
        "schema": send_email.SEND_EMAIL_SCHEMA,
        "execute": send_email.execute,
        "requires_env": ["RESEND_API_KEY"],
    },
    "file_read": {
        "schema": file_ops.FILE_READ_SCHEMA,
        "execute": file_ops.execute_read,
        "requires_env": [],
    },
    "file_write": {
        "schema": file_ops.FILE_WRITE_SCHEMA,
        "execute": file_ops.execute_write,
        "requires_env": [],
    },
}
