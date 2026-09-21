import pytest

from strine.config import MissingAPIKeyError, load_api_key


def test_load_api_key_uses_env_var_when_no_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")

    assert load_api_key() == "sk-ant-from-shell"


def test_load_api_key_raises_when_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        load_api_key()


def test_load_api_key_prefers_dotenv_in_current_working_directory(tmp_path, monkeypatch):
    """.env deve ser resolvido a partir do cwd de quem roda `strine`, não do
    diretório onde o pacote strine está instalado — senão um .env que exista
    por acaso ao lado do código-fonte (ex: checkout de desenvolvimento)
    vazaria pra qualquer execução do strine, em qualquer diretório."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-project-dotenv\n")

    assert load_api_key() == "sk-ant-from-project-dotenv"
