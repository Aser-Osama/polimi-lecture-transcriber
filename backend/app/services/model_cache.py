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

import tqdm as tqdm_module

from app.core.errors import CancelledError, ModelDownloadError
from app.core.progress import CancellationToken

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


class _ReportingTqdm(tqdm_module.tqdm):  # type: ignore[misc]
    """Real tqdm subclass hooked into huggingface_hub downloads.

    Subclassing keeps every method huggingface_hub may call. The bar is
    silenced (disable=True) but still counts; byte progress across all active
    per-file bars is aggregated into an honest overall fraction. Cancellation
    raises out of ``update`` which aborts the download.
    """

    _lock = threading.Lock()
    _bars: dict[int, tuple[int, int, str]] = {}  # id -> (done, total, unit)

    def __init__(self, *args, **kwargs):
        self._report_unit = kwargs.get("unit", "it")
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
        with type(self)._lock:
            type(self)._bars[id(self)] = (0, int(self.total or 0), self._report_unit)

    def update(self, n: int = 1):
        if self.disable:
            self.n += int(n)
        else:
            super().update(n)
        with type(self)._lock:
            done, total, unit = type(self)._bars.get(
                id(self), (int(self.n), int(self.total or 0), self._report_unit)
            )
            type(self)._bars[id(self)] = (done + int(n), total, unit)
            type(self)._report()
        return True

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
