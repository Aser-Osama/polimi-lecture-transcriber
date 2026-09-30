"""Hugging Face cache inspection and model download.

Uses the normal Hugging Face hub cache (HF_HOME / ~/.cache/huggingface).
"Downloaded" means: a snapshot directory exists that contains config.json and
either weights.npz or a *.safetensors file, with no incomplete downloads.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from app.core.errors import CancelledError, ModelDownloadError
from app.providers.base import CancellationToken

log = logging.getLogger(__name__)

ProgressReporter = Callable[[float | None, str], None]  # (fraction|None, message)


def hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_cache_dir(repo_id: str, cache_dir: Path | None = None) -> Path:
    folder = "models--" + repo_id.replace("/", "--")
    return (cache_dir or hf_cache_dir()) / folder


def is_model_cached(repo_id: str, cache_dir: Path | None = None) -> bool:
    snapshots = repo_cache_dir(repo_id, cache_dir) / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            has_config = (snapshot / "config.json").is_file()
            if not has_config:
                continue
            weight_files = [snapshot / "weights.npz", *snapshot.glob("*.safetensors")]
            for weight in weight_files:
                if weight.is_file() and weight.stat().st_size > 0:
                    return True
    except OSError:
        log.exception("Failed to inspect model cache at %s", snapshots)
        return False
    return False


class _ReportingTqdm:
    """tqdm-compatible class hooked into huggingface_hub downloads.

    Aggregates byte progress across all active per-file bars and reports an
    honest overall fraction. Cancellation raises out of ``update`` which
    aborts the download.
    """

    _lock = threading.Lock()
    _bars: dict[int, tuple[int, int, str]] = {}  # id -> (done, total, unit)

    def __init__(self, *args, **kwargs):
        if args and args[0] is not None:
            self.iterable = args[0]
        else:
            self.iterable = kwargs.get("iterable")
        self.total = int(kwargs.get("total") or 0)
        self.unit = kwargs.get("unit", "it")
        self.n = 0
        self.desc = kwargs.get("desc") or ""

    def __enter__(self):
        with self._lock:
            type(self)._bars[id(self)] = (0, self.total, self.unit)
            type(self)._report()
        return self

    def __exit__(self, *exc):
        with self._lock:
            type(self)._bars[id(self)] = (self.total, self.total, self.unit)
            type(self)._report()
        return False

    def update(self, n=1):
        with self._lock:
            done, total, unit = type(self)._bars.get(id(self), (0, self.total, self.unit))
            type(self)._bars[id(self)] = (done + int(n), total, unit)
            type(self)._report()
        return True

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else [])

    @classmethod
    def _report(cls):
        reporter = getattr(cls, "_reporter", None)
        token = getattr(cls, "_cancel", None)
        if token is not None and token.cancelled:
            raise CancelledError("Model download cancelled")
        if reporter is None:
            return
        byte_done = byte_total = 0
        file_done = file_total = 0
        for _, (done, total, unit) in cls._bars.items():
            if unit == "B":
                byte_done += done
                byte_total += total
            else:
                file_done += done
                file_total += total
        fraction: float | None = None
        if byte_total > 0:
            fraction = min(1.0, byte_done / byte_total)
        elif file_total > 0:
            fraction = min(1.0, file_done / file_total)
        try:
            reporter(fraction, "Downloading model")
        except Exception:  # progress reporting must never break downloads
            log.exception("Progress reporter failed")


def download_model(
    repo_id: str,
    progress: ProgressReporter | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    """Download a model into the standard HF cache. Returns the snapshot path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ModelDownloadError(f"huggingface_hub unavailable: {exc}") from exc

    _ReportingTqdm._reporter = progress  # type: ignore[attr-defined]
    _ReportingTqdm._cancel = cancel  # type: ignore[attr-defined]
    _ReportingTqdm._bars = {}

    if is_model_cached(repo_id):
        if progress:
            progress(1.0, "Model already downloaded")
        return repo_cache_dir(repo_id)

    if progress:
        progress(None, "Downloading model")
    try:
        path = snapshot_download(repo_id=repo_id, tqdm_class=_ReportingTqdm)
    except CancelledError:
        raise
    except Exception as exc:
        log.exception("Model download failed for %s", repo_id)
        raise ModelDownloadError(f"Failed to download {repo_id}: {exc}") from exc
    if progress:
        progress(1.0, "Model downloaded")
    return Path(path)
