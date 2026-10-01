import pytest

from strine.tool_catalog import missing_env_vars, parse_tool_selection, required_env_vars

VALID = {"sql", "web_search", "http_request", "file_read"}


# --- credentials ---


def test_required_env_vars_for_tool_that_needs_a_credential():
    assert required_env_vars("web_search") == ["TAVILY_API_KEY"]


def test_required_env_vars_is_empty_for_tool_without_credentials():
    assert required_env_vars("http_request") == []


def test_missing_env_vars_reports_unset_variable(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert missing_env_vars("web_search") == ["TAVILY_API_KEY"]


def test_missing_env_vars_is_empty_when_variable_is_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    assert missing_env_vars("web_search") == []


def test_missing_env_vars_treats_empty_string_as_missing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")

    assert missing_env_vars("web_search") == ["TAVILY_API_KEY"]


def test_missing_env_vars_is_empty_for_tool_without_credentials():
    assert missing_env_vars("file_read") == []


# --- parse_tool_selection ---


def test_parse_accepts_comma_separated_names():
    assert parse_tool_selection("sql, web_search", VALID) == (["sql", "web_search"], [])


def test_parse_accepts_space_separated_names():
    assert parse_tool_selection("sql web_search", VALID) == (["sql", "web_search"], [])


def test_parse_preserves_the_order_the_user_typed():
    selected, _ = parse_tool_selection("web_search, sql, file_read", VALID)

    assert selected == ["web_search", "sql", "file_read"]


def test_parse_ignores_case_and_extra_whitespace():
    assert parse_tool_selection("  SQL ,  Web_Search  ", VALID) == (
        ["sql", "web_search"],
        [],
    )


def test_parse_removes_duplicates_keeping_first_occurrence():
    assert parse_tool_selection("sql, web_search, sql", VALID) == (
        ["sql", "web_search"],
        [],
    )


def test_parse_reports_unknown_names_separately():
    selected, invalid = parse_tool_selection("sql, bogus, web_search, nope", VALID)

    assert selected == ["sql", "web_search"]
    assert invalid == ["bogus", "nope"]


def test_parse_none_means_an_empty_selection():
    assert parse_tool_selection("none", VALID) == ([], [])
    assert parse_tool_selection("  None  ", VALID) == ([], [])


def test_parse_none_mixed_with_names_is_treated_as_an_unknown_name():
    """'none, sql' is ambiguous — better to flag it than guess."""
    selected, invalid = parse_tool_selection("none, sql", VALID)

    assert selected == ["sql"]
    assert invalid == ["none"]


@pytest.mark.parametrize("raw", ["", "   ", ",,", " , "])
def test_parse_blank_input_yields_nothing(raw):
    assert parse_tool_selection(raw, VALID) == ([], [])
