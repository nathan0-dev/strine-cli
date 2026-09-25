from unittest.mock import Mock, patch

from strine.tools.http_request import execute


def test_get_request_returns_status_and_body():
    fake_response = Mock(status_code=200, text='{"ok": true}')
    with patch("strine.tools.http_request.requests.request", return_value=fake_response) as mock_request:
        result = execute(method="GET", url="https://api.example.com/status")

    mock_request.assert_called_once_with(
        "GET",
        "https://api.example.com/status",
        headers=None,
        json=None,
        timeout=10,
    )
    assert "200" in result
    assert '{"ok": true}' in result


def test_post_request_with_headers_and_body():
    fake_response = Mock(status_code=201, text="created")
    with patch("strine.tools.http_request.requests.request", return_value=fake_response) as mock_request:
        result = execute(
            method="POST",
            url="https://api.example.com/items",
            headers={"Authorization": "Bearer xyz"},
            body={"name": "item"},
        )

    mock_request.assert_called_once_with(
        "POST",
        "https://api.example.com/items",
        headers={"Authorization": "Bearer xyz"},
        json={"name": "item"},
        timeout=10,
    )
    assert "201" in result


def test_long_response_body_is_truncated():
    fake_response = Mock(status_code=200, text="x" * 20000)
    with patch("strine.tools.http_request.requests.request", return_value=fake_response):
        result = execute(method="GET", url="https://api.example.com/big")

    assert len(result) < 20000


def test_connection_error_returns_friendly_string():
    import requests

    with patch("strine.tools.http_request.requests.request", side_effect=requests.RequestException("boom")):
        result = execute(method="GET", url="https://api.example.com/down")

    assert "Erro" in result
    assert "boom" in result
