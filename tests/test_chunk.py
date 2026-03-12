"""Tests for text chunking logic."""

from index import chunk_text


def test_single_chunk_when_content_fits():
    content = "line one\nline two\nline three"
    chunks = chunk_text(content, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == content


def test_splits_on_chunk_boundary():
    # Each line is 6 chars ("AAAAA\n"), chunk_size=12 should fit 2 lines
    content = "\n".join(["AAAAA"] * 6)
    chunks = chunk_text(content, chunk_size=12, overlap=0)
    assert len(chunks) > 1
    # All original content should be present across chunks
    reassembled = set()
    for c in chunks:
        for line in c.split("\n"):
            if line:
                reassembled.add(line)
    assert reassembled == {"AAAAA"}


def test_overlap_repeats_trailing_lines():
    lines = [f"line{i}" for i in range(10)]
    content = "\n".join(lines)
    # Small chunks so we get multiple, with overlap
    chunks = chunk_text(content, chunk_size=20, overlap=10)
    assert len(chunks) > 1

    # With overlap, later chunks should start with content that was at the
    # end of the previous chunk
    for i in range(1, len(chunks)):
        prev_lines = chunks[i - 1].split("\n")
        curr_lines = chunks[i].split("\n")
        # At least one line from the end of the previous chunk should appear
        # at the start of the current chunk
        overlap_found = any(line in prev_lines for line in curr_lines[:3] if line)
        assert overlap_found, (
            f"No overlap found between chunk {i-1} and {i}:\n"
            f"  prev tail: {prev_lines[-3:]}\n"
            f"  curr head: {curr_lines[:3]}"
        )


def test_empty_content():
    chunks = chunk_text("", chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == ""


def test_zero_overlap_no_duplication():
    content = "a\nb\nc\nd\ne"
    chunks = chunk_text(content, chunk_size=4, overlap=0)
    # With zero overlap each line should appear exactly once across all chunks
    all_lines = []
    for c in chunks:
        all_lines.extend(c.split("\n"))
    assert all_lines == ["a", "b", "c", "d", "e"]
