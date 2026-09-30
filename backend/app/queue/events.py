"""In-process pub/sub for Server-Sent Events."""

from __future__ import annotations

import asyncio
import contextlib
import logging

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self._subscribers: set[asyncio.Queue] = set()
        self._loop = loop

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: dict) -> None:
        """Publish from the event loop thread."""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("SSE subscriber too slow; dropping event %s", event.get("type"))

    def publish_threadsafe(self, event: dict) -> None:
        """Publish from any thread (e.g. model download threads)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self.publish, event)

    def close_all(self) -> None:
        """Tell every SSE stream to finish so shutdown never waits on them."""
        log.info("Closing %d open event stream(s)", len(self._subscribers))
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait({"type": "shutdown"})
        self._subscribers.clear()
