import os
import re
from typing import Iterable, List, Tuple

from strine.tools import TOOLS

_NO_TOOLS_KEYWORD = "none"


def required_env_vars(tool_key: str) -> List[str]:
    return list(TOOLS[tool_key].get("requires_env", []))


def missing_env_vars(tool_key: str) -> List[str]:
    """Variables the tool requires that are NOT configured (empty counts
    as missing — an `X=` line in .env is not a credential)."""
    return [name for name in required_env_vars(tool_key) if not os.getenv(name)]


def parse_tool_selection(
    raw: str, valid_keys: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """Parses what the user typed to choose tools.

    Returns (selected, unknown). Accepts commas and/or spaces, ignores
    case and duplicates (keeping the order typed). Only the word 'none'
    by itself means "no tools"; mixed with names it's treated as an
    unknown name — better to flag the ambiguity than guess.

    Empty input returns ([], []): the caller decides what Enter means.
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
