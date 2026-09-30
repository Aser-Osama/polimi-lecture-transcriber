"""Output file naming and safe writing.

Rules:
- never overwrite an existing transcript; bump to "Name (2).txt" etc.
- the same stem is used for all four output files of a job
- files are created exclusively and written atomically
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
import uuid
from pathlib import Path

from app.core.errors import OutputError

log = logging.getLogger(__name__)

OUTPUT_EXTENSIONS = ("txt", "srt", "vtt", "json")
# Path separators, colon (Apple maps it to "/" in Finder) and control chars.
# Tab/newline are intentionally excluded: they are whitespace and get collapsed.
_UNSAFE_CHARS = re.compile(r"[/\\:\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_STEM_LENGTH = 120
_MAX_UNIQUE_ATTEMPTS = 2000


def sanitize_basename(original_filename: str) -> str:
    # Sanitize before stem extraction so separators cannot truncate the name.
    name = unicodedata.normalize("NFC", original_filename)
    name = _UNSAFE_CHARS.sub("-", name)
    stem = Path(name).stem
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if len(stem) > _MAX_STEM_LENGTH:
        stem = stem[:_MAX_STEM_LENGTH].strip(" .")
    return stem or "transcript"


def ensure_output_dir(directory: Path) -> Path:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(
            f"Cannot create output directory {directory}: {exc}",
            user_message="The output folder could not be created. Choose a different one in Settings.",
        ) from exc
    if not os.access(directory, os.W_OK):
        raise OutputError(
            f"Output directory not writable: {directory}",
            user_message="The output folder is not writable. Choose a different one in Settings.",
        )
    return directory


def allocate_stem(
    output_dir: Path, base: str, extensions: tuple[str, ...]
) -> tuple[str, dict[str, Path]]:
    """Reserve a unique stem + file set, never overwriting existing files."""
    ensure_output_dir(output_dir)
    for counter in range(1, _MAX_UNIQUE_ATTEMPTS + 1):
        stem = base if counter == 1 else f"{base} ({counter})"
        paths = {ext: output_dir / f"{stem}.{ext}" for ext in extensions}
        if any(path.exists() for path in paths.values()):
            continue
        created: list[Path] = []
        try:
            for path in paths.values():
                with open(path, "x", encoding="utf-8"):
                    pass
                created.append(path)
            return stem, paths
        except FileExistsError:
            for path in created:
                path.unlink(missing_ok=True)
            continue
    raise OutputError(
        f"Could not allocate unique output names for {base!r} in {output_dir}",
        user_message="Too many files with this name already exist. Rename the source file.",
    )


def create_output_files(output_dir: Path, original_filename: str) -> tuple[str, dict[str, Path]]:
    """Reserve a unique set of the four output files. Returns (stem, paths)."""
    base = sanitize_basename(original_filename)
    return allocate_stem(output_dir, base, OUTPUT_EXTENSIONS)


def write_text_atomic(path: Path, content: str) -> None:
    """Write via a temp file + replace so readers never see partial output."""
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise OutputError(
            f"Failed writing {path}: {exc}",
            user_message="Results could not be saved to disk.",
        ) from exc


def release_unwritten(paths: dict[str, Path], written: set[str]) -> None:
    """Remove reserved-but-unwritten placeholders after a failure."""
    for ext, path in paths.items():
        if ext not in written:
            path.unlink(missing_ok=True)
