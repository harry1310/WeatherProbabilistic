"""Heartbeat thread for visibility into long-running sampling.

nutpie's progressbar doesn't render usefully when stdout/stderr are redirected
to a file (no TTY), so during a 20+ min `pm.sample` call the log otherwise goes
silent. This is a tiny daemon thread that prints a timestamped "still running"
line every `interval` seconds with elapsed wall time, so you can see at a
glance whether sampling is making progress vs. hung.

Usage:
    with Heartbeat("Model A lead 24h", interval=30):
        idata = pm.sample(...)
"""

from __future__ import annotations

import threading
import time


class Heartbeat:
    def __init__(self, label: str, interval: float = 30.0) -> None:
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._start = 0.0

    def __enter__(self) -> "Heartbeat":
        self._start = time.time()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._start
            print(
                f"      [heartbeat {time.strftime('%H:%M:%S')} "
                f"+{elapsed:.0f}s] {self.label} still sampling",
                flush=True,
            )
