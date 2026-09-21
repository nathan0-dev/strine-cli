from strine.providers.base import NormalizedResponse, Provider, ProviderError
from tests.fakes import FakeProvider


def test_fake_provider_is_a_provider():
    provider = FakeProvider(responses=[])
    assert isinstance(provider, Provider)
    assert provider.name == "fake"


def test_fake_provider_returns_programmed_responses_in_order():
    r1 = NormalizedResponse(stop_reason="end_turn", text="primeira", tool_calls=[])
    r2 = NormalizedResponse(stop_reason="end_turn", text="segunda", tool_calls=[])
    provider = FakeProvider(responses=[r1, r2])

    msg = provider.build_user_message("oi")
    first = provider.create_message("system", [msg])
    second = provider.create_message("system", [msg])

    assert first.text == "primeira"
    assert second.text == "segunda"
    assert len(provider.calls) == 2


def test_fake_provider_raises_provider_error_when_configured():
    provider = FakeProvider(raise_error="boom")
    try:
        provider.create_message("system", [])
        assert False, "deveria ter levantado ProviderError"
    except ProviderError as exc:
        assert "boom" in str(exc)
