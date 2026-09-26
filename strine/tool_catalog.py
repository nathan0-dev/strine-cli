import os
import re
from typing import Iterable, List, Tuple

from strine.tools import TOOLS

_NO_TOOLS_KEYWORD = "nenhuma"


def required_env_vars(tool_key: str) -> List[str]:
    return list(TOOLS[tool_key].get("requires_env", []))


def missing_env_vars(tool_key: str) -> List[str]:
    """Variáveis que a tool exige e que NÃO estão configuradas (vazia conta
    como ausente — uma linha `X=` no .env não é uma credencial)."""
    return [name for name in required_env_vars(tool_key) if not os.getenv(name)]


def parse_tool_selection(
    raw: str, valid_keys: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """Interpreta o que o usuário digitou pra escolher tools.

    Devolve (selecionadas, desconhecidas). Aceita vírgulas e/ou espaços,
    ignora maiúsculas e duplicatas (mantendo a ordem digitada). Só a palavra
    'nenhuma' sozinha significa "nenhuma tool"; misturada com nomes é tratada
    como nome desconhecido — melhor apontar a ambiguidade do que adivinhar.

    Entrada vazia devolve ([], []): quem chama decide o que Enter significa.
    """
    valid = set(valid_keys)
    tokens = [t for t in re.split(r"[,\s]+", raw.strip().lower()) if t]

    if tokens == [_NO_TOOLS_KEYWORD]:
        return [], []

    selected: List[str] = []
    invalid: List[str] = []
    for token in tokens:
        bucket = selected if token in valid else invalid
        if token not in bucket:
            bucket.append(token)
    return selected, invalid
