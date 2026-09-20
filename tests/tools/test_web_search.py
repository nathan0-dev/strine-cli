import os
from unittest.mock import Mock, patch

from strine.tools.web_search import execute


def test_missing_api_key_returns_friendly_message(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = execute(query="strine cli")
    assert "TAVILY_API_KEY" in result


def test_successful_search_formats_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    fake_response = Mock()
    fake_response.json.return_value = {
        "results": [
            {
                "title": "Strine CLI",
                "url": "https://example.com/strine",
                "content": "Describe your agent, get it running.",
            }
        ]
    }
    with patch("strine.tools.web_search.requests.post", return_value=fake_response) as mock_post:
        result = execute(query="strine cli")

    assert mock_post.call_args.kwargs["json"]["api_key"] == "tvly-fake-key"
    assert mock_post.call_args.kwargs["json"]["query"] == "strine cli"
    assert "Strine CLI" in result
    assert "https://example.com/strine" in result


def test_request_exception_returns_friendly_message(monkeypatch):
    import requests

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    with patch("strine.tools.web_search.requests.post", side_effect=requests.RequestException("timeout")):
        result = execute(query="strine cli")

    assert "Erro" in result
    assert "timeout" in result
