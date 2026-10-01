from strine.tools.file_ops import execute_read, execute_write


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    write_result = execute_write(path="notes.txt", content="hello world")
    assert "successfully" in write_result.lower()

    read_result = execute_read(path="notes.txt")
    assert read_result == "hello world"


def test_read_missing_file_returns_friendly_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="does-not-exist.txt")
    assert "not found" in result.lower()


def test_read_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_read(path="../../etc/passwd")
    assert "not allowed" in result.lower()


def test_write_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_write(path="../outside.txt", content="x")
    assert "not allowed" in result.lower()
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
    assert "not allowed" in result.lower() or "error" in result.lower()


def test_write_with_null_byte_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = execute_write(path="foo\x00bar.txt", content="test")
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "not allowed" in result.lower() or "error" in result.lower()


def test_read_with_excessive_filename_length_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    long_filename = "a" * 300 + ".txt"
    result = execute_read(path=long_filename)
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "not allowed" in result.lower() or "error" in result.lower()


def test_write_with_excessive_filename_length_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    long_filename = "a" * 300 + ".txt"
    result = execute_write(path=long_filename, content="test")
    assert isinstance(result, str)
    assert not result.startswith("Traceback")
    assert "not allowed" in result.lower() or "error" in result.lower()
