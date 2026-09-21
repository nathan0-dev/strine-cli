from strine.tools import TOOLS

EXPECTED_TOOL_NAMES = {
    "sql",
    "slack",
    "webhook",
    "http_request",
    "web_search",
    "send_email",
    "file_read",
    "file_write",
}


def test_registry_has_all_expected_tools():
    assert set(TOOLS.keys()) == EXPECTED_TOOL_NAMES


def test_every_tool_has_well_formed_schema_and_callable_execute():
    for name, entry in TOOLS.items():
        schema = entry["schema"]
        assert "name" in schema, f"{name}: schema sem 'name'"
        assert "description" in schema, f"{name}: schema sem 'description'"
        assert schema["input_schema"]["type"] == "object", f"{name}: input_schema não é object"
        assert callable(entry["execute"]), f"{name}: execute não é callable"


def test_schema_name_vs_registry_key_documented_reality():
    """Documents which TOOLS registry keys match their schema['name'].

    Nothing in the codebase currently depends on schema['name'] matching
    the registry key — this test exists purely to lock in and document the
    current, inconsistent reality so a future implementer (e.g. of the Day
    4 runtime that will actually dispatch tool calls to `execute`) doesn't
    silently assume one relationship holds everywhere.

    The three original tools ("sql", "slack", "webhook") predate the
    convention and use a different schema name than their registry key.
    All 5 tools added later ("http_request", "web_search", "send_email",
    "file_read", "file_write") use a schema name equal to the registry key.
    """
    matching = {
        key for key, entry in TOOLS.items() if entry["schema"]["name"] == key
    }
    mismatched = set(TOOLS.keys()) - matching

    assert mismatched == {"sql", "slack", "webhook"}
    assert matching == {
        "http_request",
        "web_search",
        "send_email",
        "file_read",
        "file_write",
    }
