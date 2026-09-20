from strine.tools.file_ops import execute_read, execute_write


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    write_result = execute_write(path="notes.txt", content="olá mundo")
    assert "sucesso" in write_result.lower()

    read_result = execute_read(path="notes.txt")
    assert read_result == "olá mundo"


def test_read_missing_file_returns_friendly_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="nao-existe.txt")
    assert "não encontrado" in result.lower() or "not found" in result.lower()


def test_read_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="../../etc/passwd")
    assert "não permitido" in result.lower() or "inválido" in result.lower()


def test_write_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_write(path="../outside.txt", content="x")
    assert "não permitido" in result.lower() or "inválido" in result.lower()
    assert not (tmp_path.parent / "outside.txt").exists()


def test_write_creates_parent_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    execute_write(path="subdir/nested.txt", content="ok")
    assert (tmp_path / "subdir" / "nested.txt").read_text() == "ok"


def test_read_with_null_byte_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="foo\x00bar.txt")
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "não permitido" in result.lower() or "erro" in result.lower()


def test_write_with_null_byte_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_write(path="foo\x00bar.txt", content="test")
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "não permitido" in result.lower() or "erro" in result.lower()


def test_read_with_excessive_filename_length_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    long_filename = "a" * 300 + ".txt"
    result = execute_read(path=long_filename)
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "não permitido" in result.lower() or "erro" in result.lower()


def test_write_with_excessive_filename_length_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    long_filename = "a" * 300 + ".txt"
    result = execute_write(path=long_filename, content="test")
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "não permitido" in result.lower() or "erro" in result.lower()
