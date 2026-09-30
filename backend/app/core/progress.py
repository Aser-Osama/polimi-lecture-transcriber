"""Cancellation and progress primitives shared across layers.

These live in app.core so low-level services (media, chunking) can use them
without importing the provider package, which would create a cycle.
"""

from __future__ import annotations

import threading

from app.core.errors import CancelledError


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("Cancelled by user")


class ProgressCallback:
    """Callable reporting a stage transition.

    ``fraction`` is an optional honest sub-progress value in [0, 1] within the
    stage (e.g. byte progress while downloading a model, processed-audio
    fraction while transcribing). ``None`` means the stage is indeterminate.
    """

    def __call__(
        self, stage: str, message: str | None = None, fraction: float | None = None
    ) -> None:
        raise NotImplementedError


def noop_progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
    pass
