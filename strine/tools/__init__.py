from strine.tools import file_ops, http_request, send_email, slack, sql, web_search, webhook

TOOLS = {
    "sql": {"schema": sql.QUERY_DATABASE_SCHEMA, "execute": sql.execute},
    "slack": {"schema": slack.POST_TO_SLACK_SCHEMA, "execute": slack.execute},
    "webhook": {"schema": webhook.SEND_WEBHOOK_SCHEMA, "execute": webhook.execute},
    "http_request": {
        "schema": http_request.HTTP_REQUEST_SCHEMA,
        "execute": http_request.execute,
    },
    "web_search": {
        "schema": web_search.WEB_SEARCH_SCHEMA,
        "execute": web_search.execute,
    },
    "send_email": {
        "schema": send_email.SEND_EMAIL_SCHEMA,
        "execute": send_email.execute,
    },
    "file_read": {
        "schema": file_ops.FILE_READ_SCHEMA,
        "execute": file_ops.execute_read,
    },
    "file_write": {
        "schema": file_ops.FILE_WRITE_SCHEMA,
        "execute": file_ops.execute_write,
    },
}
