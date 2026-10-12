"""FAR-FALCON-Terrains ``sim2real/utils/rate_limiter.py``: the sim loop's ``RateLimiter`` (``PreciseRateLimiter``).

Absolute schedule on ``time.perf_counter``: sleeps in chunks of at most ``max_sleep_time`` until the
next tick, and when a step overruns it restarts the schedule from now instead of sprinting.
"""

from __future__ import annotations

import threading
import time


class PreciseRateLimiter:
    def __init__(self, frequency: float, max_sleep_time: float = 0.1):
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.max_sleep_time = max_sleep_time
        self._next_time: float | None = None
        self._lock = threading.Lock()

    def sleep(self) -> float:
        """Sleep until the next tick; returns the time slept (s)."""
        with self._lock:
            current_time = time.perf_counter()
            if self._next_time is None:
                self._next_time = current_time + self.period
            sleep_time = self._next_time - current_time
            if sleep_time <= 0:  # overran: resynchronize
                self._next_time = current_time + self.period
                return 0.0
            actual_sleep_time = 0.0
            while sleep_time > 0:
                this_sleep = min(sleep_time, self.max_sleep_time)
                time.sleep(this_sleep)
                actual_sleep_time += this_sleep
                sleep_time -= this_sleep
                if sleep_time > 0:
                    sleep_time = self._next_time - time.perf_counter()
            self._next_time += self.period
            return actual_sleep_time


RateLimiter = PreciseRateLimiter
