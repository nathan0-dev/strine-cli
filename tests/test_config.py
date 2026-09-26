import os

from strine.config import load_project_env


def test_load_project_env_reads_dotenv_from_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRINE_TEST_VAR", "valor-antigo")
    (tmp_path / ".env").write_text("STRINE_TEST_VAR=valor-do-dotenv\n")

    load_project_env()

    assert os.environ["STRINE_TEST_VAR"] == "valor-do-dotenv"


def test_load_project_env_without_a_dotenv_file_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRINE_TEST_VAR", "valor-do-shell")

    load_project_env()

    assert os.environ["STRINE_TEST_VAR"] == "valor-do-shell"
