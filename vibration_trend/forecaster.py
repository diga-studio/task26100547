"""In-process 7-day vibration trend forecasting via ordinary least squares."""

from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple, Optional

MICROS_PER_SECOND = 1_000_000
WINDOW_MICROS = 7 * 24 * 3600 * MICROS_PER_SECOND  # 7 days
URGENT_THRESHOLD_SECONDS = 259_200  # 72 hours


class Forecast(NamedTuple):
    """Result of a forecast query.

    seconds: predicted seconds until the fitted line reaches the threshold,
             or None when no such prediction exists.
    urgent:  True only when a prediction exists and is strictly below 72h.
    """

    seconds: Optional[float]
    urgent: bool


class TrendForecaster:
    """Tracks one sensor stream against a fixed maintenance threshold."""

    def __init__(self, threshold: float) -> None:
        if not isinstance(threshold, float) or isinstance(threshold, bool):
            raise TypeError("threshold must be a float (binary64)")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        self._threshold = threshold
        self._samples: deque[tuple[int, float]] = deque()
        self._clock: Optional[int] = None
        self._last_sample_t: Optional[int] = None

    def append(self, t: int, value: float) -> None:
        """Add a sample at integer microsecond timestamp ``t``."""
        if not isinstance(t, int) or isinstance(t, bool):
            raise TypeError("t must be an integer microsecond timestamp")
        if not isinstance(value, float) or isinstance(value, bool):
            raise TypeError("value must be a float (binary64)")
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if self._clock is not None and t < self._clock:
            raise ValueError("time must not move backwards")
        if self._last_sample_t is not None and t <= self._last_sample_t:
            raise ValueError("sample timestamps must be strictly increasing")
        self._clock = t
        self._last_sample_t = t
        self._samples.append((t, value))
        self._evict()

    def forecast(self, now: int) -> Forecast:
        """Advance the logical clock to ``now`` and predict time to threshold."""
        if not isinstance(now, int) or isinstance(now, bool):
            raise TypeError("now must be an integer microsecond timestamp")
        if self._clock is not None and now < self._clock:
            raise ValueError("time must not move backwards")
        self._clock = now
        self._evict()

        n = len(self._samples)
        if n < 2:
            return Forecast(seconds=None, urgent=False)

        # Ordinary least squares on (t, value), equal weights. Timestamps are
        # centered for numerical stability; slope is in value per second.
        t_mean = sum(t for t, _ in self._samples) / n
        v_mean = sum(v for _, v in self._samples) / n
        s_tt = 0.0
        s_tv = 0.0
        for t, v in self._samples:
            dt = (t - t_mean) / MICROS_PER_SECOND
            dv = v - v_mean
            s_tt += dt * dt
            s_tv += dt * dv
        slope = s_tv / s_tt
        fitted_now = v_mean + slope * ((now - t_mean) / MICROS_PER_SECOND)

        if fitted_now >= self._threshold:
            seconds: Optional[float] = 0.0
        elif slope > 0.0:
            seconds = (self._threshold - fitted_now) / slope
        else:
            seconds = None

        urgent = seconds is not None and seconds < URGENT_THRESHOLD_SECONDS
        return Forecast(seconds=seconds, urgent=urgent)

    def _evict(self) -> None:
        """Drop samples at or before the left edge of the (open] window."""
        cutoff = self._clock - WINDOW_MICROS
        while self._samples and self._samples[0][0] <= cutoff:
            self._samples.popleft()
