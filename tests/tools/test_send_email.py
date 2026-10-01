from unittest.mock import Mock, patch

from strine.tools.send_email import execute


def test_missing_api_key_returns_friendly_message(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = execute(to="a@example.com", subject="Oi", body="Teste")
    assert "RESEND_API_KEY" in result


def test_successful_send(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_fake_key")
    fake_response = Mock(status_code=200)
    with patch("strine.tools.send_email.requests.post", return_value=fake_response) as mock_post:
        result = execute(to="a@example.com", subject="Oi", body="Teste")

    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["to"] == ["a@example.com"]
    assert sent_json["subject"] == "Oi"
    assert "successfully" in result.lower()


def test_api_error_returns_friendly_message(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_fake_key")
    fake_response = Mock(status_code=422, text='{"message": "invalid to address"}')
    with patch("strine.tools.send_email.requests.post", return_value=fake_response):
        result = execute(to="not-an-email", subject="Oi", body="Teste")

    assert "422" in result or "error" in result.lower()


def test_request_exception_returns_friendly_message(monkeypatch):
    import requests

    monkeypatch.setenv("RESEND_API_KEY", "re_fake_key")
    with patch("strine.tools.send_email.requests.post", side_effect=requests.RequestException("down")):
        result = execute(to="a@example.com", subject="Oi", body="Teste")

    assert "Error" in result
    assert "down" in result
