"""In-process sliding-window rate limiter.

State lives in this process only: with several instances, each enforces its own
limit. That is enough for a single Render service; use Redis for more.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_events: int, window_s: float = 60.0, clock=time.monotonic):
        self.max_events = max_events
        self.window_s = window_s
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> float:
        """Record an event for `key`. Returns 0 if allowed, else seconds to wait."""
        now = self._clock()
        events = self._events[key]
        while events and now - events[0] >= self.window_s:
            events.popleft()
        if len(events) >= self.max_events:
            return self.window_s - (now - events[0])
        events.append(now)
        return 0.0
