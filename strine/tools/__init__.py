from strine.tools import slack, sql, webhook

TOOLS = {
    "sql": {"schema": sql.QUERY_DATABASE_SCHEMA, "execute": sql.execute},
    "slack": {"schema": slack.POST_TO_SLACK_SCHEMA, "execute": slack.execute},
    "webhook": {"schema": webhook.SEND_WEBHOOK_SCHEMA, "execute": webhook.execute},
}
