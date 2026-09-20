from __future__ import annotations

import os
import re

from sqlalchemy import create_engine, text

QUERY_DATABASE_SCHEMA = {
    "name": "query_database",
    "description": "Executa uma query SQL de leitura (SELECT) contra o banco de dados configurado.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query SQL a ser executada. Apenas SELECT é permitido.",
            },
        },
        "required": ["query"],
    },
}

_DISALLOWED_KEYWORDS = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE")


class SQLValidationError(RuntimeError):
    pass


def _validate_query(query: str) -> None:
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        raise SQLValidationError("Apenas queries SELECT são permitidas.")
    for keyword in _DISALLOWED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            raise SQLValidationError(
                f"Query contém uma operação não permitida: {keyword}."
            )


def execute(query: str) -> str:
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        return (
            "Tool 'sql' não configurada: defina DATABASE_URL no seu .env "
            "pra habilitar consultas ao banco de dados."
        )

    try:
        _validate_query(query)
    except SQLValidationError as exc:
        return f"Query rejeitada: {exc}"

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = [dict(row._mapping) for row in result]
    except Exception as exc:  # driver/conexão podem falhar de várias formas
        return f"Erro ao executar query: {exc}"

    return str(rows)
