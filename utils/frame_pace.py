"""Keep capture loops near a target FPS so MP4 duration matches wall-clock time."""

import time


class FramePacer:
    def __init__(self, fps: float):
        self._period = 1.0 / max(fps, 1e-6)
        self._last = time.perf_counter()

    def tick(self) -> None:
        now = time.perf_counter()
        wait = self._period - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.perf_counter()
