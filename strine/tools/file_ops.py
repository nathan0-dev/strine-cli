from __future__ import annotations

from pathlib import Path

FILE_READ_SCHEMA = {
    "name": "file_read",
    "description": "Lê o conteúdo de um arquivo local, dentro do diretório de trabalho atual.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho relativo do arquivo a ser lido.",
            },
        },
        "required": ["path"],
    },
}

FILE_WRITE_SCHEMA = {
    "name": "file_write",
    "description": "Escreve conteúdo em um arquivo local, dentro do diretório de trabalho atual.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho relativo do arquivo a ser escrito.",
            },
            "content": {
                "type": "string",
                "description": "Conteúdo a ser escrito no arquivo.",
            },
        },
        "required": ["path", "content"],
    },
}


def _resolve_safe_path(path: str) -> Path | None:
    try:
        base = Path.cwd().resolve()
        candidate = (base / path).resolve()
    except (OSError, ValueError):
        return None
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def execute_read(path: str) -> str:
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return f"Caminho não permitido: '{path}' está fora do diretório de trabalho."

    try:
        if not resolved.is_file():
            return f"Arquivo não encontrado: '{path}'."
    except OSError as exc:
        return f"Caminho não permitido: {exc}"

    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Erro ao ler o arquivo: {exc}"


def execute_write(path: str, content: str) -> str:
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return f"Caminho não permitido: '{path}' está fora do diretório de trabalho."

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Erro ao escrever o arquivo: {exc}"

    return f"Arquivo '{path}' escrito com sucesso."
