"""Hook tqdm progress bars inside mlx-whisper.

mlx-whisper drives its transcription loop with ``tqdm`` over mel frames
(``pbar.update(processed_frames)``). We temporarily replace ``tqdm.tqdm`` with
a subclass that stays silent on the console but forwards the genuine fraction
of processed audio to a reporter. This is real progress from the library's own
accounting, not an estimate.

Note: ``tqdm.update()`` returns early when ``disable=True`` without counting,
so the subclass increments ``n`` itself in that case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import tqdm as tqdm_module

MIN_FRACTION_STEP = 0.002


def make_progress_tqdm(reporter: Callable[[float], None]) -> type:
    class ProgressTqdm(tqdm_module.tqdm):  # type: ignore[misc]
        _reporter = reporter
        _last_fraction = 0.0

        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # keep the worker console quiet
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1):
            if self.disable:
                self.n += int(n)
            else:
                super().update(n)
            total = self.total or 0
            reporter_fn = type(self)._reporter
            if total > 0 and reporter_fn is not None:
                fraction = max(0.0, min(1.0, self.n / total))
                last = type(self)._last_fraction
                if fraction - last >= MIN_FRACTION_STEP or (fraction >= 1.0 and last < 1.0):
                    type(self)._last_fraction = fraction
                    reporter_fn(fraction)
            return True

    return ProgressTqdm


@contextmanager
def hook_tqdm(reporter: Callable[[float], None]) -> Iterator[None]:
    """Temporarily route tqdm bars to ``reporter`` (worker process only)."""
    original = tqdm_module.tqdm
    tqdm_module.tqdm = make_progress_tqdm(reporter)
    try:
        yield
    finally:
        tqdm_module.tqdm = original
