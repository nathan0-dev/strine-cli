from __future__ import annotations

from pathlib import Path

FILE_READ_SCHEMA = {
    "name": "file_read",
    "description": "Reads the content of a local file, within the current working directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file to read.",
            },
        },
        "required": ["path"],
    },
}

FILE_WRITE_SCHEMA = {
    "name": "file_write",
    "description": "Writes content to a local file, within the current working directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
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
        return f"Path not allowed: '{path}' is outside the working directory."

    try:
        if not resolved.is_file():
            return f"File not found: '{path}'."
    except OSError as exc:
        return f"Path not allowed: {exc}"

    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading the file: {exc}"


def execute_write(path: str, content: str) -> str:
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return f"Path not allowed: '{path}' is outside the working directory."

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error writing the file: {exc}"

    return f"File '{path}' written successfully."
