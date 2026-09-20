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
