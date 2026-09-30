from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import OutputError
from app.services.outputs import (
    allocate_stem,
    create_output_files,
    ensure_output_dir,
    release_unwritten,
    sanitize_basename,
    write_text_atomic,
)


def test_sanitize_replaces_path_separators():
    assert sanitize_basename("OS: Lecture 1/2?.mp4") == "OS- Lecture 1-2?"


def test_sanitize_strips_unsafe_edges():
    assert sanitize_basename("...lecture...mp4") == "lecture"


def test_sanitize_preserves_unicode():
    assert sanitize_basename("Lezione àèìòù 01.mp4") == "Lezione àèìòù 01"


def test_sanitize_collapses_whitespace():
    assert sanitize_basename("a    b\tc.mp4") == "a b c"


def test_sanitize_empty_fallback():
    assert sanitize_basename(".....") == "transcript"


def test_sanitize_limits_length():
    name = "x" * 400 + ".mp4"
    assert len(sanitize_basename(name)) <= 120


def test_ensure_output_dir_creates_nested(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    ensure_output_dir(target)
    assert target.is_dir()


def test_create_output_files_uses_shared_stem(tmp_path: Path):
    stem, paths = create_output_files(tmp_path, "Lecture 1.mp4")
    assert stem == "Lecture 1"
    assert {p.suffix for p in paths.values()} == {".txt", ".srt", ".vtt", ".json"}
    for path in paths.values():
        assert path.exists()


def test_duplicate_output_filenames(tmp_path: Path):
    first_stem, _ = create_output_files(tmp_path, "Lecture 1.mp4")
    second_stem, second_paths = create_output_files(tmp_path, "Lecture 1.mp4")
    assert first_stem == "Lecture 1"
    assert second_stem == "Lecture 1 (2)"
    assert all(path.exists() for path in second_paths.values())
    third_stem, _ = create_output_files(tmp_path, "Lecture 1.mp4")
    assert third_stem == "Lecture 1 (3)"


def test_unicode_filenames(tmp_path: Path):
    stem, paths = create_output_files(tmp_path, "Lezione àèìòù.mp4")
    assert stem == "Lezione àèìòù"
    assert all(path.exists() for path in paths.values())


def test_write_text_atomic_and_cleanup(tmp_path: Path):
    target = tmp_path / "x.txt"
    write_text_atomic(target, "hello")
    assert target.read_text() == "hello"
    write_text_atomic(target, "replaced")
    assert target.read_text() == "replaced"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_release_unwritten(tmp_path: Path):
    stem, paths = create_output_files(tmp_path, "x.mp4")
    release_unwritten(paths, written={"txt"})
    assert paths["txt"].exists()
    assert not paths["srt"].exists()
    assert not paths["json"].exists()


def test_allocate_stem_subset(tmp_path: Path):
    stem, paths = allocate_stem(tmp_path, "regen - regenerated", ("srt", "vtt"))
    assert set(paths) == {"srt", "vtt"}
    assert paths["srt"].exists()


def test_unwritable_directory(tmp_path: Path, monkeypatch):
    import os

    target = tmp_path / "locked"
    target.mkdir()
    target.chmod(0o500)
    try:
        if os.access(target, os.W_OK):
            pytest.skip("running as privileged user")
        with pytest.raises(OutputError):
            ensure_output_dir(target)
    finally:
        target.chmod(0o700)
