"""Tests for file indexability detection."""

import tempfile
from pathlib import Path

from index import is_indexable_file


def test_text_file_is_indexable():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello, world!\nThis is a log file.\n")
        f.flush()
        assert is_indexable_file(Path(f.name)) is True


def test_empty_file_not_indexable():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.flush()
        assert is_indexable_file(Path(f.name)) is False


def test_binary_file_not_indexable():
    with tempfile.NamedTemporaryFile(suffix=".bin", mode="wb", delete=False) as f:
        f.write(b"\x00\x01\x02\x03binary content")
        f.flush()
        assert is_indexable_file(Path(f.name)) is False


def test_pdf_extension_is_indexable():
    with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb", delete=False) as f:
        f.write(b"%PDF-1.4 fake pdf")
        f.flush()
        assert is_indexable_file(Path(f.name)) is True


def test_nonexistent_file():
    assert is_indexable_file(Path("/tmp/does_not_exist_abc123.txt")) is False
